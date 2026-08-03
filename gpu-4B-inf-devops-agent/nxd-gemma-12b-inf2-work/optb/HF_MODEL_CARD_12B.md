---
license: apache-2.0
base_model: google/gemma-4-12B-it
tags:
  - gemma
  - gemma-4
  - gemma4-unified
  - aws-inferentia
  - inferentia2
  - neuron
  - torch-neuronx
  - neuronx-distributed
  - tensor-parallel
  - text-generation
library_name: torch-neuronx
pipeline_tag: text-generation
---

# Gemma-4 12B on AWS Inferentia2 — Tensor-Parallel (TP=2), device prefill

Compiled **AWS Neuron** artifacts that run Google's **encoder-free** [`google/gemma-4-12B-it`](https://huggingface.co/google/gemma-4-12B-it)
**across both NeuronCores of a single Inferentia2 device** (`inf2.8xlarge`), with greedy decode
**token-for-token identical to the CPU reference** and the prompt **prefilled on-device** (~0.1 s
to first token).

This is a **runnable inference port**, not a fine-tune — the weights are Google's, unmodified.
What's new is the *tensor-parallel compilation recipe* for the unified (encoder-free) 12B, which
the vendor stack (`optimum-neuron` / NxD) does not currently provide.

## TL;DR

| | |
|---|---|
| Base model | `google/gemma-4-12B-it` — the **unified, encoder-free** Gemma-4 (`model_type: gemma4_unified`), Apache-2.0 |
| Hardware | AWS Inferentia2 — **`inf2.8xlarge`**, both NeuronCores (**TP = 2**, mandatory) |
| Precision | **bf16** sharded weights (~12 GB/core); fp32 does not fit |
| Context | 256 tokens (`KV_MAX=256`, `KV_BUCKET=64`) |
| Prefill | **on-device** (~0.1 s first token) via weight-sharing prefill+decode buckets |
| Correctness | greedy argmax **== CPU float reference**, token-for-token (`SEQ_MATCH True`) |

## Why this is its own port (vs the E2B/E4B builds)

Gemma-4 12B is the **encoder-free unified multimodal** model — it drops the separate audio/vision
encoders and projects raw audio (640 samples / 40 ms) and image patches (48×48) straight into the
LLM embedding space via lightweight linear layers. Its text config differs from the smaller Gemma-4
models in ways that matter for a Neuron port:

- **48 layers**, hidden 3840, 16 attention / 8 KV heads, head_dim 256 (global layers 512),
  intermediate 15360, vocab 262144, softcap 30, tied embeddings.
- **No Per-Layer Embeddings** (`hidden_size_per_layer_input=0`) — unlike E2B/E4B.
- **All 48 layers own their K/V** (`num_kv_shared_layers=0`).
- **`attention_k_eq_v=true`** — global (full-attention) layers reuse K as V.
- **`num_global_key_value_heads=1`** — global layers have a single KV head.
- **`sliding_window=1024`** (2× the E4B window).

## The recipe (what makes it fit + stay correct)

Built with **NxD `ModelBuilder`**: one bf16 sharded weight set is shared across a **prefill bucket**
(seq = 64) and a **decode bucket** (seq = 1), with the KV cache device-resident via I/O aliasing —
so weights are resident once (~12 GB/core) and the KV never leaves the cores. Three details are
specific to the 12B and were required for correctness / to compile:

1. **`layer_scalar` is a buffer** that scales every layer's output; NxD weight-sharding loads
   *parameters* only, never buffers, so it must be copied from the checkpoint by hand — otherwise
   every layer over-scales and the output is garbage.
2. **Global layers (KV heads = 1)** don't divide across TP=2, so their K/V are kept **replicated**
   and `num_key_value_groups` is re-derived to match the per-rank sharded query heads.
3. **Eager attention + no on-device softcap.** The fused attention kernel overflows on-chip SRAM at
   `sliding_window=1024`, so the device model uses eager attention. The softcap (a `tanh` over the
   262144-wide vocab in fp32) also overflows SRAM — but softcap is monotonic, so it's dropped from
   the device graph (greedy `argmax` is unchanged) and applied host-side only when sampling.

## Files

| File | What it is |
|---|---|
| `mb_12b_256.pt` | the device-prefill model — one serialized `NxDModel` (prefill+decode buckets, bf16 sharded weights + aliased KV), load with `torch.jit.load` |
| `tp_mb.py` | compiles + saves the model via NxD `ModelBuilder` (`MB_WDTYPE=bf16 MB_SAVE=... python tp_mb.py`) |
| `optb_server_mb.py` | device-prefill HTTP server (OpenAI routes + `/generate`); host computes only the scaled word embedding |
| `optb_server_slim.py` | **slim** server for `inf2.xlarge` (16 GB RAM) — loads only `embed_tokens.weight`, never the full transformer host-side |
| `Dockerfile.mb` | packages the full build → `xbill9/gemma4-optb-12b:tp2-devprefill-256` |
| `Dockerfile.12b-slim` | packages the slim build → `xbill9/gemma4-optb-12b:slim-devprefill` (`inf2.xlarge`) |

## Run

**Prebuilt Docker image** (bundles the compiled model + weights + server):

```bash
docker run -d --device /dev/neuron0 --ipc=host -p 8080:8080 \
  xbill9/gemma4-optb-12b:tp2-devprefill-256
curl -s localhost:8080/generate -d '{"prompt":"What is AWS Inferentia?","max_tokens":60}'
```

Serves OpenAI-compatible routes (`/v1/chat/completions`, `/v1/completions`) plus `/generate`,
`/health`, `/metrics`.

**Run on a single `inf2.xlarge` (¼ the price) — slim image.** The `inf2.xlarge` has the *same 2
NeuronCores* as the `inf2.8xlarge`, just 16 GB host RAM instead of 128. `gemma4_unified` has no
Per-Layer Embeddings, so the host only needs the scaled word embedding — the **slim** server
(`optb_server_slim.py`) constructs the real `Gemma4UnifiedTextScaledWordEmbedding` and loads *only*
`embed_tokens.weight` from safetensors, instead of materializing the ~24 GB transformer on the host:

```bash
docker run -d --device /dev/neuron0 --ipc=host -p 8080:8080 \
  xbill9/gemma4-optb-12b:slim-devprefill
```

Validated on an `inf2.xlarge`: coherent output ("The capital of France is **Paris**."), device
prefill ~0.1 s, ~15 tok/s decode, **~8 GB host RAM** (the same compiled `mb_12b_256.pt` runs on both
cores). Add swap before first run — the neff load peaks host RAM on a 16 GB box.

**From these files** (on an `inf2.8xlarge` with the Neuron runtime, `transformers==5.13.0`,
`torch-neuronx==2.8.0`, `neuronx-distributed>=0.17`):

```bash
# rebuild the device model (add ≥55 GB swap first — the 2-rank compile peaks past 128 GB RAM)
MODEL_DIR=./gemma-4-12B-it KV_MAX=256 KV_BUCKET=64 MB_WDTYPE=bf16 \
  MB_SAVE=./mb_12b_256.pt python tp_mb.py     # -> SEQ_MATCH True, "The capital of France is Paris."

# serve the saved model
MODEL_DIR=./gemma-4-12B-it MB_PATH=./mb_12b_256.pt KV_MAX=256 KV_BUCKET=64 \
  python optb_server_mb.py
```

## Limitations

- **2 NeuronCores (TP=2) required** — runs on `inf2.8xlarge` (full server) or `inf2.xlarge` (slim server, `:slim-devprefill`); bf16 is mandatory (fp32 = 24 GB/core, won't load).
- **256-token context.** 12B leaves ~4 GB/core after weights, so the window is smaller than the E2B/E4B builds; a larger window needs a recompile and may not fit.
- **Text generation only** in this build — the audio/vision projection paths are not wired to the device model.
- **Batch size 1**, single-stream greedy/sampled decode.
- Compiled specifically for Inferentia2 + the pinned Neuron SDK versions.

## License & attribution

- **Base model:** `google/gemma-4-12B-it`, © Google, **Apache-2.0**.
- **These artifacts** (compiled model + scripts) are Apache-2.0 derivatives; the serialized model
  embeds the base weights in bf16. Redistributed under Apache-2.0 with attribution to Google —
  include the upstream `LICENSE`/`NOTICE` when redistributing.

*Not affiliated with or endorsed by Google or AWS. Provided for research/testing.*
