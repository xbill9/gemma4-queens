# Gemma-4 (E4B) tensor-parallel (TP=2) inference on Inferentia2

Runs [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it) **across both NeuronCores**
of an `inf2.8xlarge`, with greedy output **token-for-token identical to the CPU reference** and the
prompt **prefilled on-device**. Tensor-parallel follow-up to the single-core
[E2B sample](../hf_pretrained_gemma4_e2b_inference_on_inf2.ipynb) — E4B is ~2× larger and doesn't fit one
16 GB core even in bf16, so it must be sharded across the two cores.

## Files
- `hf_pretrained_gemma4_e4b_tp2_inference_on_inf2.ipynb` — the sample (compile → load → generate).
- `tp2_build.py` — the tensor-parallel compile step (run by the notebook). Shards the model across
  `TP_DEGREE` cores via NxD `ModelBuilder`, traces the prefill + decode buckets, runs an in-process
  `SEQ_MATCH` check, and saves the model with `torch.jit.save`.

## Approach
One weight-sharing `ModelBuilder` trace holds a **prefill** and a **decode** bucket sharing **one bf16
sharded weight set** + a **device-resident KV cache** (aliased as graph I/O). Attention is GQA-sharded
across ranks; token embeddings + per-layer inputs are computed on the host. The tensor-parallel trace
spawns worker processes, so the compile is a script (`tp2_build.py`); load + generate run in-process in
the notebook.

## Gemma-4 TP=2 gotchas (see the notebook's final section)
- GQA head-sharding: shard k/v to `n_kv // TP` heads/rank when divisible (keep `num_key_value_groups`),
  else replicate; global single-KV-head layers are replicated.
- `layer_scalar` is a **buffer** — `ModelBuilder` loads params only, so copy it from the checkpoint by hand.
- Host-side embeddings (PLE); on-device aliased KV; on-device prefill.
- Launcher: `multiprocessing` spawn + `if __name__=="__main__"` guard + a `transformers.utils.fx` shim.

## Requirements
`inf2.8xlarge` (2 NeuronCores), Neuron SDK 2.23 — `torch-neuronx==2.8.*`, `neuronx-cc==2.*`,
`neuronx-distributed>=0.17`, `transformers==5.13.0`. Gated model: accept the license + `huggingface-cli
login`. Add ~55 GB swap before compiling (the 2-rank compile briefly exceeds host RAM).

## Scaling further
The dense 12B / 31B variants use the same recipe at higher TP degrees (`TP_DEGREE=8` on an
`inf2.24xlarge`); bf16 is mandatory (fp32 constants overflow a 16 GB core).
