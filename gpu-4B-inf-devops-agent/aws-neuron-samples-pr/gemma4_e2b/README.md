# Gemma-4 (E2B) text inference on Inferentia2

Runs [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it) on a single NeuronCore
(`inf2.xlarge`) with greedy output **token-for-token identical to the CPU reference** (`SEQ_MATCH`),
at ~44 tok/s.

## Files
- `hf_pretrained_gemma4_e2b_inference_on_inf2.ipynb` — the sample (compile → validate → generate).

## Approach
Two graphs traced with `torch_neuronx.trace`:
- **prefill** — runs the padded prompt, returns logits + per-layer K/V.
- **decode** — one token vs a fixed `MAX`-length KV buffer passed as graph I/O, updated with a one-hot
  masked write (no dynamic index ops).

Token embeddings and per-layer inputs are computed on the host and passed as `inputs_embeds` /
`per_layer_inputs`, so the traced graph is free of the Per-Layer-Embeddings lookup.

## Gemma-4 gotchas (see the notebook's final section)
- Compute PLE embeddings on the host; pass them in.
- KV cache as graph I/O is what makes the per-layer KV-sharing trace cleanly.
- `bf16` weights to fit a single 16 GB core.
- `attn_implementation="eager"` + `tanh` GELU + final-logit softcap for exact CPU parity.

## Requirements
Neuron SDK 2.23 — `torch-neuronx==2.8.*`, `neuronx-cc==2.*`, `transformers==5.13.0`. Gated model:
accept the license on the model page and `huggingface-cli login` with an authorized token.

## Scaling up
E2B fits one core. The 4B / 12B Gemma-4 variants exceed 16 GB even in bf16 and need a TP=2 build across
both NeuronCores of an `inf2.8xlarge`.
