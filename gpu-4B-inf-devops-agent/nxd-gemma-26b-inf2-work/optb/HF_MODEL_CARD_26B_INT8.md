---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
tags:
  - gemma
  - gemma-4
  - mixture-of-experts
  - moe
  - int8
  - quantized
  - aws-inferentia
  - inferentia2
  - neuron
  - torch-neuronx
  - neuronx-distributed
  - text-generation
library_name: torch-neuronx
pipeline_tag: text-generation
---

# Gemma-4 26B-A4B (MoE, **int8 experts**) on AWS Inferentia2 — TP=8

An **int8-quantized-expert** build of [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it)
for `inf2.24xlarge` (TP=8). The 128 MoE experts are quantized to **per-channel symmetric int8**; every
other weight stays bf16. Greedy decode is **token-for-token identical to the bf16 / CPU fp32 reference**
(`SEQ_MATCH True`, `"The capital of France is Paris."`) — int8 per-channel loses nothing here.

**Why this variant:** the compiled model is **41.8 GB vs 64.6 GB** for the bf16 build (the experts are
~93% of the weight; int8 halves them) — a smaller HBM footprint and faster load/serve on the same box,
at bf16-identical quality. See the bf16 build
[`xbill9/gemma-4-26B-A4B-it-inferentia2`](https://huggingface.co/xbill9/gemma-4-26B-A4B-it-inferentia2)
for the full recipe write-up.

> **Note on smaller boxes:** int8 experts save ~22.8 GB but the model is still ~3–4 GB over a 16 GB
> NeuronCore at TP=2, so this build targets the **24xlarge** (12 cores), not the 2-core `inf2.xlarge`/
> `inf2.8xlarge`. Fitting a 2-core box would need fp4 experts (a further, accuracy-risky step).

## How it works

Reuses the bf16 MoE recipe (`tp_mb_moe.py`): all-experts-dense compute, `SPMDRank` scatter for the
router weighting, mixed-attention shard/replicate. The only change (`tp_mb_moe_int8.py`) swaps the two
expert linears for **`QuantizedColumnParallel`/`QuantizedRowParallel`** (INT8, per-channel-symmetric)
and quantizes the fused expert weights in `checkpoint_loader`. Two NxD autograd quirks were fixed to
compile: `.clone()` the down output, and monkeypatch `scale_dequantize` to be out-of-place.

## Contents

| file | what |
|---|---|
| `mb_26b_int8_tp8.pt` | Compiled TP=8 MoE model, **int8 experts** (~42 GB), KV 256/64 |
| `real-gemma4-26B-A4B-it/` | Google's weights + tokenizer + `chat_template.jinja` |
| `tp_mb_moe_int8.py` | The int8 recipe (QuantizedColumnParallel/RowParallel + q8 in checkpoint_loader) |
| `optb_server_int8.py` | HTTP server (OpenAI-compatible + `/generate`, `/metrics`, streaming) |
| `Dockerfile.int8`, `entrypoint.int8.sh` | Thin image that pulls these artifacts at start and serves |

## Run it

```bash
MODEL_DIR=/data/real-gemma4-26B-A4B-it MB_LOAD=/data/mb_26b_int8_tp8.pt \
  TP_DEGREE=8 KV_MAX=256 KV_BUCKET=64 python optb_server_int8.py
curl -s localhost:8080/generate -d '{"prompt":"What is the capital of France?"}'
# -> {"response":"The capital of France is Paris.", ...}
```

**Docker:** `docker run -d --device /dev/neuron0 ... --device /dev/neuron5 --ipc=host -v gemma26b8:/data -p 8080:8080 xbill9/gemma4-optb-26b-int8:latest`

## Environment

Neuron SDK 2.23 · `torch-neuronx` 2.8.0 · `neuronx-distributed` 0.17.26814 · `transformers` 5.13.0.
License follows the upstream Gemma weights' terms; see `google/gemma-4-26B-A4B-it`.
