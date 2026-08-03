---
title: "Porting a 128-expert MoE (Gemma-4 26B-A4B) to AWS Inferentia2 — where every rank weighted the wrong experts"
published: false
description: "The MoE was the hard one: a dual-path FFN, a sparse expert loop that won't trace, and a bug where the device output was empty while the CPU reference was perfect and every unit test passed."
tags: machinelearning, aws, ai, python
# canonical_url: https://your-blog.example/gemma4-26b-moe-inferentia
# cover_image: https://.../cover.png                            # 1000x420 recommended
# series: "Gemma-4 on Inferentia"
---

> Of the five Gemma-4 models I ported to AWS Inferentia2, the 26B-A4B was the one that was supposed to
> be impossible: a **Mixture of Experts**. It compiled on the first try — then produced *nothing*. The
> device output was empty. The CPU reference was perfect. My unit tests all passed. The bug was in a
> place I'd have sworn couldn't have one.

This is the MoE entry in the series. The dense models (E2B/E4B/12B/31B) are hard because of
architecture; the 26B is hard because it's **sparse** — 128 experts, top-8 routing — and none of the
AWS vendor stack has ever traced an MoE for Gemma-4.

| | |
|---|---|
| Model | `google/gemma-4-26B-A4B-it` — MoE, ~4B active / 26B total, `model_type: gemma4` |
| Hardware | `inf2.24xlarge` — 12 NeuronCores / 192 GB HBM, **TP=8** |
| Result | Device greedy decode **== CPU fp32 reference** (SEQ_MATCH True); prefill 77 ms; `"The capital of France is Paris."` |
| Artifacts | Docker Hub `xbill9/gemma4-optb-26b` · HF `xbill9/gemma-4-26B-A4B-it-inferentia2` |

---

## "A4B" is a lie your memory budget will not forgive

The name says 26B-**A4B**: ~4 billion **active** parameters per token. That sounds like a 4B model. It
is not. All **128 experts (~49 GB) must be resident in HBM** — top-8 routing only reduces *compute*, not
*footprint*. So you budget for a 26B, not a 4B, and it needs the 24xlarge's 192 GB, same class as the
2×-larger dense 31B. (This bites again later when we try to shrink it onto a smaller box.)

## The architecture: a dual-path FFN nobody warned me about

I inspected the transformers-5.13 module on a meta device before writing a line of port code, and it's
weirder than "MLP replaced by MoE." **Every one of the 30 layers runs a shared dense MLP *in parallel*
with the 128-expert MoE**, combined and passed through *four* feed-forward layernorms:

```
residual (post-attention)
├─ dense: pre_feedforward_layernorm  → mlp (2112)                → post_feedforward_layernorm_1  ─┐
└─ moe:   pre_feedforward_layernorm_2 → router → 128 experts (704) → post_feedforward_layernorm_2 ─┤
                                            hidden = dense + moe ────────────────────────────────────┘
   → post_feedforward_layernorm → + residual → × layer_scalar
```

- **Router** (`Gemma4TextRouter`): its *own* RMSNorm + a `scale` + a `per_expert_scale`, then
  `softmax(128, fp32) → top-8 → renormalize → × per_expert_scale`.
- **Experts** (`Gemma4TextExperts`): a **fused** `gate_up_proj [128,1408,2816]` + `down_proj
  [128,2816,704]`, and its forward is a **sparse gather/scatter loop** (`torch.where`, `index_add_`).
- Attention is the same mixed sliding/global layout as the 31B (which I'd already solved).

That sparse expert loop is the crux: it's **data-dependent** and **will not trace** to a static Neuron
graph.

## The traceable trick: compute *all* experts, mask to top-8

You can't put a per-token, per-expert gather/scatter in a static HLO graph. So don't. Compute **all 128
experts on every token**, weight each by its router weight — which is **zero for the non-top-8** — and
sum. Because a non-selected expert contributes `0 × expert(x) = 0`, this is **mathematically identical**
to HF's sparse top-8, but it's a fixed-shape sequence of matmuls the compiler loves. Wasteful in FLOPs
(you compute 128 experts to use 8), correct in every bit, and traceable.

And the beautiful part: the whole thing collapses onto **two standard parallel linears** that
ModelBuilder already knows how to shard — no custom 3D-parameter sharding:

- `gate_up_proj [128,1408,2816] → [180224, 2816]` as one **`ColumnParallelLinear`** (rank *r* gets
  experts 16*r*…16*r*+15)
- `down_proj [128,2816,704] → [2816, 90112]` as one **`RowParallelLinear`** (input-sharded → all-reduce)

I keep HF's dual-path layer forward and its router **unchanged** (router replicated; its `top_k_weights`
already carry the renorm + per-expert scale) and swap only `self.experts` → my `DenseExperts`. Before
spending a cent on hardware, I unit-tested the math and the TP=8 sharding against HF on CPU:
**MAXDIFF ≈ 2e-6, cosine 1.0.** The math was right.

## The bug: SEQ_MATCH-perfect math, empty device output

First device run: **it compiled** (the first-of-its-kind MoE trace on Neuron — 30 MoE layers,
`MB_TRACED`). And the output was `''`. Empty. First token = end-of-turn, immediately.

- CPU reference: `"The capital of France is Paris."` ✅
- My all-experts-dense math: unit-tested identical to HF ✅
- My TP=8 sharding *logic*: unit-tested identical to HF ✅
- Device: nothing ❌

When your math is proven, your sharding logic is proven, and the device is still wrong, the bug is in
the gap between "how I think NxD's primitives behave" and "how they actually behave under the trace."

Here it was. To weight rank *r*'s experts by the router, I scattered the dense weight matrix per-rank
with `scatter_to_tensor_model_parallel_region`. That function picks its slice using
`get_tensor_model_parallel_rank()` — **a Python int**. ModelBuilder compiles **one rank** and replicates
the graph across all 8 at runtime. So the trace **baked rank 0's slice** (`Wd[:, 0:16]`) into the graph,
and at runtime **every rank weighted experts 0–15** while its `gate_up` computed its *own* experts
(16–31, 32–47, …). Total misalignment → garbage → the model confidently ends the turn.

The auto-sharded `ColumnParallel`/`RowParallel` don't have this problem because their **weights are
pre-sliced at load time** — the forward graph is rank-agnostic. A standalone collective on a **runtime
activation** is not.

The fix is the exact mechanism NxD's own MoE module uses (`enable_spmd_rank`): register an `SPMDRank`
module (a `[1]` int32 parameter checkpoint-loaded from `arange(TP)`, so each rank gets its own number)
and scatter with the **runtime-rank-aware** variant:

```python
Wl = scatter_to_process_group_spmd(Wd, 1, self.spmd_rank.get_rank())  # each rank gets ITS experts
```

Recompile. `CPU GEN` == `DEV GEN` == `"The capital of France is Paris."`, **SEQ_MATCH True**, 77 ms
prefill. The last and hardest of the five Gemma-4 variants was on Inferentia.

**The general lesson, now taped to my monitor:** *any standalone tensor-parallel collective on a
runtime activation needs the SPMD-rank variant under ModelBuilder — auto-sharded parallel linears are
fine because their weights are pre-sliced; runtime tensors are not.*

## A packaging trap worth 20 wasted minutes

Publishing, the server wouldn't load the model: `PytorchStreamReader ... failed finding central
directory`. My background "mirror to S3" loop had uploaded the 64.6 GB neff **while `torch.jit.save` was
still writing it** — a truncated 21.9 GB partial — and I'd terminated the box before the final clean
upload finished. Fix: never mirror an artifact mid-write; `assert zipfile.is_zipfile(...)` then do **one**
clean upload. (The "real" neff being 3× the size of the corrupt partial was the tell.)

## Can it run on a smaller box? Yes, no, and "deeper than it looks"

The experts are ~93% of the weight, so quantizing them is the obvious lever. I built an **int8-expert**
variant (`QuantizedColumnParallel`/`QuantizedRowParallel`, per-channel symmetric) — two NxD autograd
quirks to fix on the way (`.clone()` the down output; monkeypatch `scale_dequantize` to be out-of-place,
it does `x *= scale` in place on an async-comm view). The result was genuinely good:

- **int8 is numerically PERFECT** — `SEQ_MATCH True`, token-for-token identical to fp32.
- **int8 is genuinely smaller** — neff **41.8 GB vs 64.6 GB** bf16, exactly the 22.8 GB expert saving
  (the compiler keeps it int8, doesn't upcast).

But it fits only the 24xlarge. On a **2-core box** (`inf2.8xlarge`/`inf2.xlarge`, 32 GB HBM, TP=2) it
`Allocation Failure`s — experts drop to 11.4 GB/rank, but `lm_head` + the compiled graph + Neuron's
runtime reserve push it ~3–4 GB over the 16 GB core. And `inf2` has **no 4-core SKU** to split the
difference. Fitting a 2-core box needs **fp4** experts — which, unlike int8's five-line quantizer, is a
deep NxD microscaling integration (packed `uint16`, `from_float` hooks) with real accuracy risk. That's
a separate expedition.

So the shipped set is **bf16 (24xlarge)** and a validated **int8 (24xlarge, smaller/faster, bf16-exact
quality)** — both published — with fp4-for-2-cores documented as the next frontier.

## Takeaways

1. **A data-dependent op won't trace? Make it fixed-shape.** All-experts-dense trades FLOPs for a static
   graph and *exact* correctness. Optimize routing later.
2. **Unit-test math AND sharding logic on CPU before you pay for a NeuronCore.** It's what let me say
   with certainty "the bug is in the primitives, not my code."
3. **Runtime-rank is the MoE-on-ModelBuilder gotcha.** Weights get pre-sliced; runtime activations need
   the SPMD-rank scatter. If the device is empty while CPU + unit tests are perfect, look here.
4. **"Active params" is a compute number, not a memory number.** MoE saves FLOPs, not HBM.
5. **int8 per-channel was free accuracy-wise** — a smaller, faster serving artifact at zero quality cost.
   The sub-8-bit step is where the real work (and risk) begins.

## Artifacts

- Docker Hub: `xbill9/gemma4-optb-26b` (bf16) · `xbill9/gemma4-optb-26b-int8`
- HF: `xbill9/gemma-4-26B-A4B-it-inferentia2` (bf16) · `-int8`
- Recipe: `tp_mb_moe.py` (the `DenseExperts` + `SPMDRank` scatter) · `tp_mb_moe_int8.py`

*Written with AI assistance (the port, the debugging, and this write-up were done in a Claude Code
session); every log line quoted is from a real run on real hardware.*
