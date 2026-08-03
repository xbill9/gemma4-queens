---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
tags:
  - gemma
  - gemma-4
  - mixture-of-experts
  - moe
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

# Gemma-4 26B-A4B (MoE) on AWS Inferentia2 — Tensor-Parallel (TP=8)

Compiled **AWS Neuron** artifacts that run the **Mixture-of-Experts**
[`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) across **8 NeuronCores
of a single `inf2.24xlarge`**, with greedy decode **token-for-token identical to the CPU fp32
reference** (`SEQ_MATCH True`) and coherent output (`"The capital of France is Paris."`, prefill 77 ms).

This is a **runnable inference port**, not a fine-tune — the weights are Google's, unmodified. What's
new is getting a **128-expert MoE to trace and shard on Neuron**, which the vendor stack
(`optimum-neuron` / the Neuron vLLM backend) cannot do for Gemma-4. This is the first-of-its-kind MoE
port in this series; all five Gemma-4 variants (E2B/E4B/12B/31B/26B-A4B) now run on Inferentia.

## The architecture (a dual-path FFN)

Each of the 30 layers runs a **shared dense MLP in parallel with a 128-expert MoE**, combined and
passed through four feed-forward layernorms:
- `num_experts` 128, `top_k_experts` 8, `moe_intermediate_size` 704 (per-expert), `intermediate_size`
  2112 (shared dense MLP), `hidden_size` 2816, softcap 30, tied embeddings, no PLE.
- Attention: 25 sliding (8 kv, head_dim 256) + 5 global (2 kv, head_dim 512, `attention_k_eq_v`).
- **A4B saves compute, not memory:** ~4B params fire per token, but all 128 experts (~49 GB) are
  resident — needs the `inf2.24xlarge`'s 192 GB HBM, not an 8xlarge.

## How it works (the recipe)

Reuses the 31B `ModelBuilder` recipe (single-rank compile + per-rank weight loading, mixed-attention
shard/replicate, device-resident KV cache, `layer_scalar` buffers, chat-template prompt) and swaps
only the experts:
- **All-experts-dense** compute: all 128 experts on every token, weighted by the top-8 router weight
  (0 for non-selected → exact match to HF's sparse top-8, but static-shape and traceable).
- Expert weights mapped onto two standard parallel linears: `gate_up` ColumnParallelLinear (rank *r*
  gets experts 16*r*…16*r*+15) + `down` RowParallelLinear (input-sharded → all-reduce). ~5.7 GB
  experts/rank.
- **SPMD rank fix (the crux):** the per-expert routing weight must be scattered per-rank with a
  *runtime* rank (`SPMDRank` + `scatter_to_process_group_spmd`) — a plain
  `scatter_to_tensor_model_parallel_region` bakes rank 0's slice into the single-rank trace and every
  rank ends up weighting the wrong experts.

## Contents

| file | what |
|---|---|
| `mb_26b_256.pt` | Compiled TP=8 MoE model (~65 GB), KV 256/64, bf16 |
| `real-gemma4-26B-A4B-it/` | Google's weights + tokenizer + `chat_template.jinja` |
| `tp_mb_moe.py` | The full recipe (DenseExperts + SPMDRank scatter, ModelBuilder trace, `MB_LOAD`) |
| `optb_server_tp.py` | HTTP server (OpenAI-compatible + `/generate`, `/metrics`, streaming) |
| `Dockerfile`, `entrypoint.sh` | Thin image that pulls these artifacts at start and serves |

## Run it

```python
import torch
model = torch.jit.load("mb_26b_256.pt")
model.nxd_model.initialize_with_saved_weights(torch.tensor([0], dtype=torch.int32))
```

```bash
MODEL_DIR=/data/real-gemma4-26B-A4B-it MB_LOAD=/data/mb_26b_256.pt \
  TP_DEGREE=8 KV_MAX=256 KV_BUCKET=64 python optb_server_tp.py
curl -s localhost:8080/generate -d '{"prompt":"What is the capital of France?"}'
# -> {"response":"The capital of France is Paris.", ...}
```

**Docker** (pulls artifacts from this repo at first start):
```bash
docker run -d --device /dev/neuron0 ... --device /dev/neuron5 --ipc=host \
  -v gemma26b-data:/data -p 8080:8080 xbill9/gemma4-optb-26b:latest
```

## Environment

Neuron SDK 2.23 · `torch-neuronx` 2.8.0 · `neuronx-distributed` 0.17.26814 · `transformers` 5.13.0.

## Limitations

- KV buckets 256/64 (first-light); larger contexts need a recompile.
- All-experts-dense trades throughput for a correct, static-shape first light (computes all 128
  experts, not just the routed 8) — expert-routed/blockwise compute is a future optimization.
- License follows the upstream Gemma weights' terms; see `google/gemma-4-26B-A4B-it`.
