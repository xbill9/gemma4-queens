---
license: apache-2.0
base_model: google/gemma-4-31B-it
tags:
  - gemma
  - gemma-4
  - aws-inferentia
  - inferentia2
  - neuron
  - torch-neuronx
  - neuronx-distributed
  - modelbuilder
  - tensor-parallel
  - text-generation
library_name: torch-neuronx
pipeline_tag: text-generation
---

# Gemma-4 31B (dense) on AWS Inferentia2 — Tensor-Parallel (TP=8)

Compiled **AWS Neuron** artifacts that run [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it)
across **8 NeuronCores of a single `inf2.24xlarge`**, with greedy decode **token-for-token identical
to the CPU fp32 reference** (`SEQ_MATCH True`) and coherent output
(`"The capital of France is Paris."`, first-token prefill ~115 ms).

This is a **runnable inference port**, not a fine-tune — the weights are Google's, unmodified. What's
new is the **tensor-parallel compilation recipe** built on **NxD `ModelBuilder`**: a way to compile
and shard the dense 31B across eight 16 GB cores that the vendor stack (`optimum-neuron` / the Neuron
vLLM backend) cannot currently do for Gemma-4.

## Why ModelBuilder (and not the hand-driven tracer)

The manual `parallel_model_trace` recipe that serves the smaller Gemma-4 models (E4B/12B) does **not**
scale to 31B: tracing all 8 ranks in-process OOMs (~300 GB), and serializing them
(`max_parallel_compilations=1`) deadlocks at multi-rank weight collection. **`ModelBuilder` compiles
one rank and shards weights per-rank** — the designed large-model path. On a live `inf2.24xlarge`:
`Finished building model in ~2365 s (~39 min)`, host peak **182 GB** (no OOM), `Compiler status PASS`.

## Architecture notes (the crux)

- **Dense** (`enable_moe_block: false`), 60 layers, `hidden_size` 5376, vocab 262144, softcap 30,
  tied embeddings, **no PLE**, **no cross-layer KV-sharing**.
- **Two attention layouts:** 50 `sliding_attention` layers (head_dim 256, **16 kv** → shard q/k/v/o)
  and 10 `full_attention`/global layers (head_dim **512**, **4 kv**, `v_proj=None` /
  `attention_k_eq_v`). Because `4 < TP=8`, the **global layers are replicated** (full q/k/v/o on every
  rank); the sliding layers are sharded. MLP is sharded everywhere.
- **`layer_scalar` is a buffer, not a parameter** — it must be copied from the checkpoint explicitly,
  or all 60 layers silently use the default `1.0` and the model produces noise.
- **Chat tokens are `<|turn>` (105) / `<turn|>` (106)** — *not* `<start_of_turn>` — and the chat
  template ships as a separate `chat_template.jinja` (bundled here so `AutoTokenizer` auto-loads it).

## Contents

| file | what |
|---|---|
| `mb_31b_256.pt` | Compiled TP=8 model (108 GB) — graph + all 8 ranks' bf16 weights, KV 256/64 |
| `real-gemma4-31B-it/` | Google's weights (bf16 safetensors) + tokenizer + **`chat_template.jinja`** |
| `tp_mb.py` | The full recipe: ModelBuilder trace, `MB_LOAD` reload, `SKIP_VALIDATE`/`DEVICE_ONLY` |
| `optb_server_tp.py` | HTTP server (OpenAI-compatible + `/generate`, `/metrics`, streaming) |
| `Dockerfile`, `entrypoint.sh` | Thin image that pulls these artifacts at start and serves |

## Run it

**Load + reload (the two non-obvious steps):**

```python
import torch
model = torch.jit.load("mb_31b_256.pt")
model.nxd_model.initialize_with_saved_weights(torch.tensor([0], dtype=torch.int32))  # push weights to cores
```

**Server (on an `inf2.24xlarge`):**

```bash
MODEL_DIR=/data/real-gemma4-31B-it MB_LOAD=/data/mb_31b_256.pt \
  TP_DEGREE=8 KV_MAX=256 KV_BUCKET=64 python optb_server_tp.py
curl -s localhost:8080/generate -d '{"prompt":"What is the capital of France?"}'
# -> {"response":"The capital of France is Paris.", ...}
```

**Docker** (pulls artifacts from this repo at first start):

```bash
docker run -d --device /dev/neuron0 ... --device /dev/neuron5 --ipc=host \
  -e HF_TOKEN=hf_xxx -v gemma31b-data:/data -p 8080:8080 xbill9/gemma4-optb-31b:latest
```

## Environment

Neuron SDK 2.23 · `torch-neuronx` 2.8.0 · `neuronx-distributed` (ModelBuilder) · `transformers` 5.13.0.
Validated with the `aws_neuronx_venv_pytorch_2_8_nxd_inference` venv + `transformers==5.13.0` and
`PATH=$VENV/bin:/opt/aws/neuron/bin`.

## Limitations

- **KV buckets 256/64** (first-light). Larger contexts need a recompile with bigger buckets.
- **~39-min single-rank compile** — a one-time cost; the reload path avoids repeating it.
- Throughput beyond prefill latency (decode tok/s, batching) not yet characterized here.
- License: the compiled artifacts follow the upstream Gemma weights' terms; see `google/gemma-4-31B-it`.
