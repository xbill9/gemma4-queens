---
license: gemma
language:
- en
base_model: google/gemma-4-26B-A4B-it
tags:
- inferentia2
- aws-neuron
- neuronx-distributed
- gemma-4
- moe
- int8
- inf2.xlarge
pipeline_tag: text-generation
---

# Gemma-4-26B-A4B-it on a **single inf2.xlarge** (int8-squeeze, slim deploy)

This repo runs Google's **Gemma-4-26B-A4B** (26B-parameter Mixture-of-Experts, ~3.2B active, 128 experts
top-8) on the **cheapest AWS Inferentia2 instance — a single `inf2.xlarge` (2 NeuronCores, 16 GB host
RAM)**. It is the all-**int8**-quantized "squeeze" build, compiled with AWS Neuron `neuronx-distributed`
ModelBuilder at tensor-parallel degree 2, plus a **slim host-side deploy** that never instantiates the
full model on the host.

> The 26B MoE does not fit a 2-core box naively (128 experts ≈ 45.6 GB in bf16). This build gets it onto
> a 16 GB-host inf2.xlarge by (1) quantizing **all** the big weight blocks to int8 — experts, the tied
> `lm_head` (sharded), and the shared dense MLP — and (2) deploying the compiled neff via `MB_LOAD` with
> only a host **word-embedding table** resident, so the host never holds the 100 GB+ fp32 model.

## Validated

- `DEV GEN: 'The capital of France is **Paris**.'` — greedy, token-for-token correct.
- Loads on a stock inf2.xlarge (16 GB host) with a **40 GB swapfile** for the one-time neff-load peak
  (host RSS peaks only ~11 GB; swap is barely touched). Compile needs ~180 GB host — deploy does **not**.
- Prefill ~0.25 s; decode ~6 tok/s (all-experts-dense MoE = all 128 experts computed every token at TP=2;
  this is compute-bound on the device, identical on any inf2 — not a swap artifact).

## Artifacts

| file | size | what |
|------|------|------|
| `sqz_neff.pt` | ~26.7 GB | ModelBuilder-traced prefill+decode neff (TP=2, int8 experts+head+MLP, on-device KV) |
| `embed_tokens.pt` | ~1.48 GB | host word-embedding table `[vocab, hidden]` (bf16) |
| `cfg/` | — | `config.json`, `generation_config.json`, `chat_template.jinja`, tokenizer |
| `optb_server_sqz.py` | — | slim HTTP server (OpenAI-compatible + `/generate`, streaming, metrics) |
| `deploy_sqz.py` | — | one-shot deploy/validation script |

## Run (Docker, one inf2.xlarge)

```bash
# 16 GB host needs swap for the neff-load peak:
sudo fallocate -l 40G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile

docker run -d --name gemma-26b-xlarge \
  --device /dev/neuron0 --ipc=host -p 8080:8080 \
  -e HF_TOKEN=hf_xxx \
  -v gemma26b-data:/data \
  xbill9/gemma4-optb-26b:xlarge
```

First start pulls ~28 GB of artifacts from this repo into the `/data` volume, then serves on `:8080`.

## Query (OpenAI-compatible)

```bash
curl http://localhost:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages":[{"role":"user","content":"What is the capital of France?"}],
  "temperature":0.0, "max_tokens":32
}'

# also: POST /generate  {"prompt": "..."}   and streaming with "stream": true
# health: GET /health   metrics: GET /metrics (Prometheus)
```

## Key recipe notes

- **Gemma-4 scaled embeddings:** Gemma-4 uses `Gemma4TextScaledWordEmbedding`, which multiplies word
  embeddings by `embed_scale = hidden_size**0.5` *inside* the embedding; the model forward does not
  re-normalize. Because embedding lookup happens on the host here, the host **must** apply the same
  `×√hidden_size` (a plain `nn.Embedding` without it makes embeds ~√H× too small → garbage output).
- **int8 squeeze:** experts (`gate_up`/`down`) + shared MLP + the tied `lm_head` are all int8
  (per-output-channel symmetric); the head is additionally **sharded** (`gather_output=True`) so no rank
  holds the full 1.48 GB head. int8 is numerically ~exact vs fp32 for greedy decoding on this model.
- **Attention:** mixed sliding (25 layers, sliding_window 1024) + global (5 layers); non-KV-shared
  layers' K/V are sharded to 1 head/rank at TP=2, global layers replicated.

Built with AWS Neuron SDK 2.31 (`neuronx-cc` 2.26, `neuronx-distributed` 0.19, torch 2.9.1,
transformers 5.14.1). Base model © Google, Gemma license. Port © the repo author, Apache-2.0.
