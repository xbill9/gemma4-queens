---
title: "Five Gemma-4 models, one accelerator: what porting E2B → 31B to AWS Inferentia2 taught me"
published: false
description: "A side-by-side of all five Gemma-4 variants on Inferentia2 — PLE, KV-sharing, MatFormer, mixed attention, and a 128-expert MoE — the recipe that evolved to carry all of them, and the one bug that shows up in every single one."
tags: machinelearning, aws, ai, python
# canonical_url: https://your-blog.example/gemma4-family-inferentia
# cover_image: https://.../cover.png                            # 1000x420 recommended
# series: "Gemma-4 on Inferentia"
---

*I ported the whole Gemma-4 family — E2B, E4B, 12B, 31B, and the 26B-A4B MoE — to run on AWS
Inferentia2. Each has its own write-up in this series; this is the map. What's shared, what's different,
how the recipe evolved from "trace it and pray" to a single-rank-compile pipeline that carries a
30-billion-parameter dense model and a 128-expert MoE — and the one bug that appears, in some costume, in
**every** port.*

## The whole family at a glance

| | **E2B** | **E4B** | **12B** | **31B** | **26B-A4B** |
|---|---|---|---|---|---|
| Params (total / active) | ~5B / 2B | ~8B / 4B | 12B | 31B | 26B / **~4B** |
| Dense or sparse | dense | dense | dense | dense | **MoE (128 exp, top-8)** |
| Per-Layer Embeddings | **✅** | **✅** | — | — | — |
| Cross-layer KV-sharing | **✅** | **✅** | — | — | — |
| MatFormer ("effective" size) | **✅** | **✅** | — | — | — |
| Model class | `Gemma4ForConditionalGeneration` | same | `…Unified` (encoder-free) | `Gemma4ForConditionalGeneration` | same |
| Hardware | 1 core (`inf2.xlarge`) | `inf2.8xlarge` | `inf2.8xlarge` | `inf2.24xlarge` | `inf2.24xlarge` |
| Tensor parallelism | **TP=1** | **TP=2** | **TP=2** | **TP=8** | **TP=8** |
| Recipe | `torch_neuronx.trace` | `tp_alias` | **ModelBuilder** | **ModelBuilder** | **ModelBuilder + MoE** |
| Correctness | SEQ_MATCH ✅ | SEQ_MATCH ✅ | SEQ_MATCH ✅ | SEQ_MATCH ✅ | SEQ_MATCH ✅ |
| Speed | ~44 tok/s | ~34 tok/s | prefill ~101 ms | prefill ~115 ms | prefill ~77 ms |

All five decode **token-for-token identically to their CPU fp32 reference**, and all five answer *"The
capital of France is Paris."* That equality bar — not "looks fluent" — is the thread that ties the
series together, for a reason I'll come back to.

## What every port shares

**1. The vendor stack can't do any of them.** `optimum-neuron` and the Neuron vLLM backend have **no
Gemma-4 model class at all**, and their graph builders can't express the architecture (KV-sharing on the
E-family, MoE on the 26B). The public endpoint I started from loaded *something* and served
fluent-looking gibberish. Every one of these is a hand-rolled port of `transformers`-5.13's own forward.

**2. The same skeleton underneath.** Softcap 30, a **262,144-token** vocabulary, tied embeddings, GQA,
and interleaved **sliding + global** attention show up across the family. So does the serving shape:
**host-side word embeddings**, a **device prefill + decode** with an **on-device KV cache**, greedy
decode compared against a host fp32 reference.

**3. The same environment quirks.** The `aws_neuronx_venv_pytorch_2_8_nxd_inference` venv +
`pip install transformers==5.13.0`, a `transformers.utils.fx` shim so `neuronx_distributed` imports under
the new transformers, and `export PATH=$VENV/bin:/opt/aws/neuron/bin` (that PATH bit fixes a
`libneuronpjrt-path` import error that will waste your afternoon).

**4. Softcap is a host-only concern.** `30·tanh(x/30)` is monotonic, so it never changes the greedy
`argmax`. Every port **drops it from the device graph** (no `tanh` over 262K logits) and re-applies it
host-side only when sampling needs real probabilities.

## What makes each one different

**E2B / E4B — the "effective" models.** These are the hard-*architecture* ports. **Per-Layer
Embeddings** (a separate embedding contribution per layer), **cross-layer KV-sharing** (many layers don't
compute their own K/V — they borrow a neighbor's), and **MatFormer** nesting (the "2B/4B" is a
sub-network of a bigger model). The whole reason this project started: KV-sharing is exactly what the
vendor graph builder *cannot* express — but it **traces fine as live graph dependencies** when you just
`torch_neuronx.trace` the real forward. E2B fits one 16 GB core; E4B is ~2× and needs TP=2, which forces
the GQA question below.

**12B — the encoder-free multimodal one.** Ships as `Gemma4UnifiedForConditionalGeneration` (multimodal
class) with **no audio/vision encoder loaded** — you take `model.language_model` and move on. Drops PLE
and KV-sharing, but its `sliding_window=1024` attention **overflows Neuron's fused-attention SBUF**, so
you **force the eager attention implementation** to make it compile.

**31B — the dense giant.** No PLE, no KV-sharing — architecturally *simple*. The hard part is **scale**:
it doesn't fit fewer than 8 cores, and the hand-driven tensor-parallel tracer that served E4B **OOMs or
deadlocks** at 31B. This is where **NxD `ModelBuilder`** becomes mandatory (compile *one* rank, shard
weights per-rank). A ~39-minute compile, 182 GB host peak, and a mixed-attention layout where the 10
global layers (`nkv=4`) get **replicated** instead of sharded.

**26B-A4B — the MoE.** The only sparse one, and the weirdest: a **shared dense MLP running in parallel
with a 128-expert MoE**, four feed-forward layernorms, a router with its own scale + per-expert scale.
The expert forward is a data-dependent gather/scatter that **won't trace** — so you compute **all 128
experts, mask to the top-8** (identical math, static shape). "A4B" means ~4B *active* params but all 128
experts (~49 GB) are **resident** — it saves compute, not memory, and needs the same 24xlarge as the
2×-larger 31B.

## The recipe evolved — and that's the real story

You can read the family as one recipe maturing under pressure:

1. **`torch_neuronx.trace`** (E2B). Trace the real HF forward on one core. Proves KV-sharing/PLE compile
   at all. Doesn't scale past one core.
2. **`tp_alias`** (E4B). Hand-driven tensor parallelism: on-device **aliased** KV buffers + a CPU-seed
   prefill, sharding GQA down to 1 KV head/rank. Serves two cores. **Doesn't survive contact with 31B**
   (the all-ranks-in-process tracer OOMs; serializing it deadlocks).
3. **NxD `ModelBuilder`** (12B, 31B, 26B). Compile **one** rank, load weights **per-rank** from a
   `checkpoint_loader`. This is the recipe that finally carries everything — the dense giant *and* the
   MoE — because the per-rank compile is what makes 8-way TP tractable.

Each step was forced by the *next* model refusing to run on the *previous* step's recipe. The MoE then
extends ModelBuilder one more notch (all-experts-dense experts as two auto-shardable parallel linears +
an SPMD-rank scatter).

## The bug that's in every port

Here's the thread from the top. Every one of these ports, at some point, produced **output that was
wrong while the CPU reference and my unit tests were perfect** — and every time, the cause was the same
*shape*: **the device faithfully computed a broken input, and only a coherence check caught it.**

- **31B:** a manually-built chat prompt tokenized `<start_of_turn>` into literal characters (Gemma-4's
  turn tokens are actually `<|turn>`/`<turn|>`). Device output matched the CPU reference **token for
  token** — because both faithfully processed the same garbage. `SEQ_MATCH True` on nonsense.
- **26B MoE:** the device output was *empty* while CPU was perfect. A standalone tensor-parallel scatter
  baked **rank 0's slice** into the single-rank ModelBuilder trace, so every rank weighted the wrong
  experts. My math and sharding logic unit-tested identical to HF — the bug lived entirely in "how the
  primitive behaves under the trace."
- **E-family:** the original "gibberish" endpoint — a real model loaded and producing confident nonsense
  because the *harness* around it (PLE path, embedding scale, KV wiring) was wrong, not the accelerator.

**The lesson that survives all five:** `device == reference` proves the accelerator is faithful. It does
**not** prove the reference is correct. A shared upstream bug — chat template, PLE, a runtime-rank scatter
— passes an equality check in dead silence. So you assert the output is *sensible* ("Paris"), not only
that device equals host. It's why every result in this series is quoted as an actual sentence, not just a
green `SEQ_MATCH`.

## If you only take four things

1. **Trace the model's real forward.** The features that break vendor graph builders (KV-sharing, MoE
   routing) often trace fine as ordinary graph dependencies. Don't reimplement what you can trace.
2. **Let the recipe escalate with the model.** trace → hand-rolled TP → ModelBuilder. Reach for the
   single-rank-compile path *before* the manual one OOMs, not after.
3. **"Active params" and "effective size" are compute/marketing numbers, not memory numbers.** MatFormer
   and MoE don't shrink the HBM you must reserve — budget for the whole thing.
4. **Equality with a reference is necessary, not sufficient.** Always also check the output is coherent.
   The most expensive bugs in this whole family were the ones that passed `SEQ_MATCH`.

## Artifacts

- **Docker Hub:** `xbill9/gemma4-optb{,-e4b,-31b,-26b,-26b-int8}`
- **Hugging Face:** `xbill9/gemma-4-{E2B,E4B,31B,26B-A4B}-it-inferentia2` (+ `-int8`)
- Per-model deep-dives: the other posts in this series.

*Written with AI assistance (the ports, the debugging, and this write-up were done across Claude Code
sessions); every number quoted is from a real run on real hardware.*
