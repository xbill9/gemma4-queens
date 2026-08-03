---
title: "My Inferentia port matched its reference token-for-token — and still output garbage"
published: false
description: "Porting Gemma-4 31B (dense) to AWS Inferentia2: the tensor-parallel recipe that worked at 12B collapses at 31B, NxD ModelBuilder saves it — and then a passing validation lies to your face."
tags: machinelearning, aws, ai, python
# canonical_url: https://your-blog.example/gemma4-31b-inferentia
# cover_image: https://.../cover.png                            # 1000x420 recommended
# series: "Gemma-4 on Inferentia"
---

> I ported Google's biggest open Gemma-4 — the 30-billion-parameter dense model — to AWS Inferentia2.
> The compiled model passed my strictest check: its output was **token-for-token identical** to the
> CPU reference. It was also complete gibberish. Both facts were true at the same time, and the reason
> is a lesson worth more than the port.

This is the sequel to porting the smaller Gemma-4 models (E2B/E4B/12B). Those were hard because of
*architecture* — Per-Layer Embeddings, cross-layer KV-sharing, MatFormer nesting. The 31B throws all
of that away and hands you a different kind of hard: **scale**. And a trap.

| | |
|---|---|
| Model | `google/gemma-4-31B-it` — dense, 60 layers, `model_type: gemma4` |
| Hardware | `inf2.24xlarge` — 6 chips / **12 cores** / 384 GB host, **TP=8** |
| Software | `torch-neuronx` 2.8.0 · `neuronx-distributed` · `transformers` 5.13.0 |
| Result | Device greedy decode **== CPU fp32 reference** (SEQ_MATCH True); prefill ~115 ms; `"The capital of France is Paris."` |
| Compile | Single-rank NxD ModelBuilder, **~39 min**, host peak **182 GB**, neffs 108 GB |

---

## The setup: the same model, eight times too big

31B is architecturally *boring*, and that's the point. `enable_moe_block: False` — dense.
`hidden_size_per_layer_input: 0` — no PLE. `num_kv_shared_layers: 0` — every layer owns its KV. It's
a plain decoder. It just happens to be a plain decoder with **30 billion parameters**, which is about
**60 GB in bf16**, which does not fit on one 16 GB NeuronCore. The moment you reach for tensor
parallelism to spread it across 8 cores, everything about how you *build* the graph changes — and the
recipe that served the 4B and 12B models walks straight off a cliff.

## The itch: the recipe that scaled to 12B dies at 31B

For E4B and 12B I hand-drove `neuronx_distributed`'s `parallel_model_trace` — I owned the ranks, the
KV cache, the whole loop. It's clean at 4–12B. At 31B it failed **two** different ways, and both were
instructive:

- **All 8 ranks trace in one process → OOM.** The model is ~15 GB/rank in bf16. Eight *simultaneous*
  fp32 compile graphs blew past **300 GB** and killed the 384 GB box. That's not the model being big.
  That's the *tracer* being 8× parallel.
- **Serialize the ranks (`max_parallel_compilations=1`) → deadlock.** Memory drops to 83 GB, a single
  rank reaches `Compiler status PASS`, and then the multi-rank weight-collection rendezvous **hangs** —
  worker blocked on `pipe_read`, CPU idle, 25 minutes, nothing.

The N=22 of this story. You can route around it — but the thing you route around is the thing that
owns you. The honest read: at 30B, *hand-driving the tracer is the bug.*

## The pivot: stop hand-driving, let ModelBuilder do it

NxD's **`ModelBuilder`** is the designed large-model path. It compiles **one rank**, then loads
weights **per-rank** from a `checkpoint_loader` — no 8× compile, no queue deadlock. The 12B port had
already used it; I should have reached for it *before* the manual recipe fell over, not after.

The whole port is three callables and a wrap module:

```python
mb = ModelBuilder(router=None, tp_degree=8,
                  checkpoint_loader=checkpoint_loader, compiler_workdir="/data/mb_wd")
mb.add("prefill", inst, [prefill_example], compiler_args=CARGS)
mb.add("decode",  inst, [decode_example],  compiler_args=CARGS)
model = mb.trace(initialize_model_weights=True)   # compile ONE rank, shard weights per rank
torch.jit.save(model, "/data/mb_31b_256.pt")      # 108 GB: graph + all 8 ranks' weights
```

## The wrinkle: one attention layout is a lie

The one piece of architecture that *does* survive is nasty. 31B interleaves two attention layouts
with **different head dimensions and KV-head counts**:

| layer type | count | q heads | kv heads | head_dim | v_proj |
|---|---|---|---|---|---|
| `sliding_attention` | 50 | 32 | **16** | 256 | present |
| `full_attention` (global) | 10 | 32 | **4** | **512** | **None** (`attention_k_eq_v` — V reuses K) |

The global layers have **4 KV heads, and 4 < TP=8**. You cannot shard 4 heads across 8 ranks. The
rule that works — **shard the 50 sliding layers, replicate the 10 global layers:**

```python
nkv = a.k_proj.out_features // a.head_dim
if nkv % TP == 0:                       # sliding: 16 kv -> 2/rank, shard everything
    a.q_proj = col(a.q_proj); a.k_proj = col(a.k_proj)
    a.v_proj = col(a.v_proj); a.o_proj = row(a.o_proj)
else:                                    # global: 4 kv < TP=8 -> leave q/k/v/o full on every rank
    pass
```

Because the sharded layers' `o_proj` is row-parallel (it all-reduces back to a full hidden state), and
the replicated layers take a full hidden state in and out, the two layouts compose — the residual
stream is full-width at every boundary. `nkv < TP` is your signal: **replicate, don't shard.**

## The trap under the trap: a buffer that isn't a parameter

This one cost an afternoon and produced *zero* error message — just a model whose cosine similarity to
the reference was ~0. Gemma-4 scales every layer's output by a learned `layer_scalar`. It's registered
as a **buffer**, not a parameter — and ModelBuilder's sharded checkpoint loader moves **parameters
only**. So all 60 `layer_scalar`s silently defaulted to `1.0`, and 60 layers of wrong-by-a-constant
compounded into noise.

```python
# read the buffers straight from the safetensors and copy them in — the loader won't
for i, lyr in enumerate(layers):
    if hasattr(lyr, "layer_scalar") and i in lsv:
        lyr.layer_scalar.copy_(lsv[i])
```

**Moral #1: buffers aren't parameters.** Anything you `register_buffer` — per-layer scalars, some
norms — needs an explicit copy, or your model silently uses config defaults.

## It compiles

```
build_module: 50 sharded-attn layers, 10 replicated-attn (global) layers, TP=8
  loaded 60 layer_scalar buffers
Sharding weights for ranks: 0...7  ->  Done Sharding weights in 172.6s
Finished building model in 2365.3 seconds (~39 min)
MB_TRACED -> MB_SAVED -> RUN_EXIT 0
```

Single-rank compile, all 8 ranks' weights sharded, host peak **182 GB** on the 384 GB box — no OOM.
The manual path's 300 GB was never the model; it was the tracer. 108 GB of neffs, banked to S3.

## The reveal: a perfect match to a broken answer

Reload the neffs, run the prompt, compare to the CPU fp32 reference. Here's the part that stops you
cold:

```
CPU GEN: '<start_of_turn>model\n<start_of_turn>model\n<start_of_turn>model...'
DEV GEN: '<start_of_turn>model\n<start_of_turn>model\n<start_of_turn>model...'
SEQ_MATCH True
```

`SEQ_MATCH True`. The device reproduced the CPU reference **token for token**. My compile was
numerically flawless. And the output was garbage — the model just echoing turn markers forever.

The tell is that *both* were wrong, *identically*. When your device output matches a broken reference
exactly, the bug isn't in the accelerator. It's **upstream of both**, in something they share. Here:
the prompt.

## The hunt: the tokenizer was lying too

Two compounding facts about the Gemma-4 snapshot:

1. **The chat template ships as a separate `chat_template.jinja`** (the "Google Gemma 4 Canonical Chat
   Template", 18.7 KB, with a thinking-token scaffold) — *not* embedded in `tokenizer_config.json`. So
   `apply_chat_template()` raises "no chat template set."
2. **Gemma-4's turn markers are `<|turn>` (105) and `<turn|>` (106)** — *not* `<start_of_turn>`.

So the "obvious" fallback — hand-write `"<start_of_turn>user\n..."` and tokenize — does this:

```
"<start_of_turn>"  ->  ['<', 'start', '_', 'of', '_', 'turn', '>']     # 7 literal chars
convert_tokens_to_ids("<start_of_turn>")  ->  3   # unknown token
```

The model gets a malformed prompt, predicts `<` as the likeliest next token, and loops. The CPU
reference, fed the same malformed prompt, does the exact same thing. `SEQ_MATCH True` the whole time,
on nonsense.

The fix is to fetch and apply the *real* template:

```python
if not getattr(tok, "chat_template", None):
    tok.chat_template = open(hf_hub_download(REPO, "chat_template.jinja", token=hf_tok)).read()
d = tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True)
prompt = d if isinstance(d, list) else d["input_ids"]   # returns a BatchEncoding, not a plain dict
```

## The payoff

Correct prompt, same neffs, same reload:

```
DEV GEN: 'The capital of France is Paris.'
DEVICE_PARIS True
DEVICE PREFILL first-token: 115 ms
```

**Moral #2 — the one this whole post is for:** `SEQ_MATCH == True` proves the accelerator faithfully
reproduces your reference. It does **not** prove your reference is correct. A shared upstream bug
passes an equality check in dead silence. Validate that the output is *sensible*, not only that device
equals host.

There's also a small, load-time gotcha worth its own line: `torch.jit.load` restores the graph and the
weights, but the weights aren't *on the cores* until you initialize them — and the call wants a
start-rank tensor:

```python
model = torch.jit.load("/data/mb_31b_256.pt")
model.nxd_model.initialize_with_saved_weights(torch.tensor([0], dtype=torch.int32))
# without this: RuntimeError "This model is not initialized ..."
```

## The side quest: racing a spot market that had nothing

`inf2.24xlarge` is 96 vCPUs — the *entire* default spot quota. One box, no headroom. During this work,
spot capacity for it was **zero across us-west-2, us-east-1, and us-east-2 for 60+ minutes at a
stretch**, and the boxes that did appear were often reclaimed in ~10–15 minutes. A 39-minute compile
does not fit in a 15-minute window. What made it possible:

- **A multi-region capacity poller** — an `instant` EC2 Fleet firing across every AZ in all three
  regions on a loop, grabbing the first `inf2.24xlarge` that blinked into existence.
- **Durable state in same-region S3.** Weights downloaded from HF *once*, mirrored to S3, never
  fetched again. Neffs pushed to S3 the instant the trace succeeds — so a reclaim *after* compile
  costs nothing.
- **Compile-only / device-only modes** to bank the 39-minute compile immediately and skip the 121 GB
  fp32 CPU-reference generation when I only needed to eyeball the device output.

**Moral #3: design the run for reclaims.** When one instance is your whole quota and capacity is
intermittent, the gap between "impossible" and "done" is *where your durable state lives.*

## The findings, in one table

| # | The thing | Where it bites | How I knew | The fix |
|---|---|---|---|---|
| 1 | Hand-driven `parallel_model_trace` OOMs (300 GB, 8× in-process) | trace | box killed; MPC=1 → deadlock instead | Switch to `ModelBuilder` (compile 1 rank, shard per-rank) |
| 2 | Global layers have `nkv=4 < TP=8` | attention sharding | `4 % 8 != 0` | Replicate the 10 global layers; shard the 50 sliding |
| 3 | `layer_scalar` is a **buffer**, loader skips it | silent | cosine ≈ 0, no error | Copy the buffers from safetensors by hand |
| 4 | `torch.jit.load` leaves weights off the cores | reload | `RuntimeError: not initialized` | `initialize_with_saved_weights(tensor([0]))` |
| 5 | Chat template ships separately; turn tokens are `<\|turn>`/`<turn\|>` | prompt | garbage output, **SEQ_MATCH True** | Fetch `chat_template.jinja`, `apply_chat_template` |
| — | `libneuronpjrt-path` "file not found" on `import neuronx_distributed` | env | — | It's just PATH: `export PATH=$VENV/bin:/opt/aws/neuron/bin:$PATH` |

## What I'm *not* claiming

- **Not a throughput result.** I have first-token prefill latency (~115 ms) and a correctness proof.
  Decode tok/s, batching, and the production 512/128 buckets are open work.
- **Not a server.** The validated model still needs `MB_LOAD` + `initialize_with_saved_weights` + the
  chat-template path wired into a persistent endpoint.
- **The compile is single-rank-fast but wall-clock-slow** (~39 min). That's a real operational cost,
  not something I've optimized away.

## Artifacts

- Compiled TP=8 model + weights: `s3://xbill-gemma4-31b-usw2/` (`neffs/mb_31b_256.pt`, `weights/`).
- Recipe: `tp_mb.py` — one file: ModelBuilder trace, `MB_LOAD` reload, `SKIP_VALIDATE`/`DEVICE_ONLY`
  modes, and the chat-template prompt path.

---

*Environment, for the reproducers: the `aws_neuronx_venv_pytorch_2_8_nxd_inference` venv +
`pip install transformers==5.13.0`, a `transformers.utils.fx` shim for the tfm-5.13 import surface, and
the PATH export above.*

*Written with AI assistance (the debugging, the porting, and this write-up were done in a Claude Code
session); every log line quoted here is from a real run on real hardware.*
