---
title: "Porting Gemma-4 12B (the encoder-free multimodal one) to AWS Inferentia2"
published: false
description: "The 12B ships as a multimodal class with no encoder loaded, and its sliding-window attention overflows Neuron's fused-attention SBUF. Three surgical fixes and it serves Paris at ~15 tok/s on one inf2.8xlarge."
tags: machinelearning, aws, ai, python
# canonical_url: https://your-blog.example/gemma4-12b-inferentia
# cover_image: https://.../cover.png                            # 1000x420 recommended
# series: "Gemma-4 on Inferentia"
---

*The middle child of the series. The 12B is the first Gemma-4 that's neither a MatFormer "effective"
model (E2B/E4B) nor a giant (31B/26B) — but it has its own two surprises: it's packaged as a
**multimodal** model with an encoder that isn't there, and its attention **overflows a Neuron hardware
buffer** in a way the smaller models never did. This is the short, clean port that only needed three
fixes on top of the E4B recipe — once I found them.*

| | |
|---|---|
| Model | `google/gemma-4-12B-it` — dense, `model_type: gemma4_unified` |
| Hardware | AWS Inferentia2 — one `inf2.8xlarge` (1 chip / 2 cores / 32 GB HBM), **TP=2** |
| Result | Greedy decode **token-for-token identical to the CPU fp32 reference** (SEQ_MATCH True); prefill ~101 ms; `"…Paris."` |
| Footprint | ~12 GB / rank, bf16, KV 256/64 |

---

## Surprise #1: a multimodal class with no encoder

`google/gemma-4-12B-it` doesn't load as `Gemma4ForConditionalGeneration` like the E-family — it's
`Gemma4UnifiedForConditionalGeneration`, `model_type: gemma4_unified`. "Unified" means it's the
multimodal architecture… except this checkpoint ships **no audio or vision encoder**. It's an
**encoder-free multimodal model**: all the multimodal *plumbing* in the class, none of the encoders.

For an Inferentia text port that's actually good news — you take **`model.language_model`** and ignore
the rest, exactly as with the other sizes. But you have to *know* to reach past the unified wrapper, and
the config's vision/audio fields are present-but-inert, which makes "is this multimodal or not?" a
five-minute detour before you can even instantiate the text path.

## The base recipe (from E4B → 12B)

The 12B reuses the **E4B ModelBuilder device-prefill recipe** almost wholesale: single-rank compile +
per-rank weight loading, on-device KV cache, host-side word embeddings, a bucketed device prefill, and
greedy decode. What it drops and what it changes:

- **No Per-Layer Embeddings.** `hidden_size_per_layer_input = 0` — feed plain `inputs_embeds`, delete
  the PLE path (same simplification the 31B enjoys).
- **No cross-layer KV-sharing.** Every layer owns its KV.
- Mixed **sliding + global** attention, `sliding_window = 1024`, GQA (16 q / 8 kv), softcap 30.

Then three fixes, each of which cost a compile cycle to find.

## Fix #1: the global layers shard at nkv=1

The global (`full_attention`) layers have **1 KV head** (`num_key_value_heads`-per-global = 1). Under
TP=2 you *can* shard them — but you shard the **query heads** and keep the single KV head grouped
correctly, `groups = nq // TP`. Get the grouping wrong and the global layers silently attend to the
wrong keys. (This is the same class of "how do the global layers shard" problem the 31B has, but the
31B's globals have `nkv=4` and must be *replicated* — 12B's `nkv=1` can shard. Same question, opposite
answer, per model.)

## Fix #2: drop the softcap on-device

Gemma caps its final logits with `30 · tanh(logits / 30)`. On the host reference you apply it. On the
device you **don't** — softcap is monotonic, so it doesn't change the `argmax`, and computing a `tanh`
over a **262,144-token** vocabulary on-device is a lot of transcendental for a no-op. Drop it from the
graph; the server re-applies it host-side only if it needs real probabilities for sampling.

## Fix #3: force eager attention — sliding_window=1024 overflows fused-attn SBUF

This is the one that actually blocks you. Neuron's compiler wants to fuse attention (flash-style) for
speed. At `sliding_window = 1024`, that fused kernel's working set **overflows SBUF** (the on-chip
scratch buffer) and the compile fails. The fix is blunt: **force the eager attention implementation**
(`attn_implementation="eager"`) so attention lowers as plain matmuls that fit. You give up the fused
kernel's speed, but it *compiles and runs correctly*, which beats a fast graph that doesn't exist. (The
smaller-window E-family models don't hit this — it's specifically the 1024-window layers that blow the
buffer.)

## Result

With those three, the 12B compiles and its device greedy decode is **token-for-token identical to the
CPU fp32 reference** — `SEQ_MATCH True`, `"…Paris."`, ~101 ms first-token prefill, ~12 GB/rank in bf16
on a single `inf2.8xlarge` at TP=2. A clean port, hiding behind two non-obvious gotchas.

## Takeaways

1. **"Unified"/multimodal classes may be encoder-free — grab `model.language_model` and move on.** Don't
   let the multimodal wrapper convince you it's a bigger problem than it is.
2. **Softcap is a host-only concern for greedy decode.** Monotonic ⇒ argmax-invariant ⇒ don't pay for it
   on-device over a 262K vocab.
3. **SBUF overflow ⇒ force eager attention.** When a large `sliding_window` fails to compile the fused
   kernel, the eager path is the escape hatch: slower, but it exists.
4. **The same architectural question gets a different answer per model.** "How do the global attention
   layers shard?" is `nkv=1 → shard query heads` here and `nkv=4 → replicate` on the 31B. Re-derive it;
   don't copy it.

## Artifacts

- Recipe/subdir: `nxd-gemma-12b-inf2-work/optb/`
- Compiled model: `s3://xbill-gemma4-patches-2b/optb-12b/mb_12b_256.pt`

*Written with AI assistance (the port, the debugging, and this write-up were done in a Claude Code
session); every result quoted is from a real run on real hardware.*
