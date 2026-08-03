# Proposal issue for aws-neuron-samples (paste into the repo's Issues, per CONTRIBUTING.md)

**Title:** [Sample request] Gemma-4 (E2B) text inference on Inf2 via torch-neuronx

**Is your feature request related to a problem?**
There's no `torch-neuronx/inference` sample for Google's **Gemma-4** family. `optimum-neuron` doesn't
support Gemma-4, and the model's per-layer KV-sharing / Per-Layer-Embeddings don't trace through the
vendor stack out of the box, so there's currently no reference for running these models on Inferentia.

**Describe the solution you'd like**
A self-contained notebook — `hf_pretrained_gemma4_e2b_inference_on_inf2.ipynb` — that runs
`google/gemma-4-E2B-it` on a single NeuronCore (`inf2.xlarge`) with greedy output **token-for-token
identical to the CPU reference**. It demonstrates a two-graph KV-cache design (prefill + single-token
decode) with the KV cache passed as graph I/O via `torch_neuronx.trace`, host-side Per-Layer-Embeddings,
and eager attention. Includes a validation cell (`SEQ_MATCH` vs CPU) and a tok/s measurement
(~45 tok/s measured).

**Scope / what I can contribute**
I have this working and validated end-to-end (Neuron SDK 2.23, `torch-neuronx` 2.8, `transformers` 5.13).
I'd start with the **E2B** notebook (single-core, matches the existing `hf_pretrained_*` convention), and
can follow up with a **tensor-parallel (TP=2) E4B/12B** sample if the maintainers want it.

**Question for maintainers:** preferred layout — a single notebook in `torch-neuronx/inference/`, or a
subdirectory (`torch-neuronx/inference/gemma4_e2b/`) with the notebook + a short README documenting the
Gemma-4-specific gotchas (`layer_scalar` buffer loading, eager-attention for `sliding_window`, host-side
PLE, softcap)? And is TP inference in scope for this repo or better suited elsewhere?

I've read CONTRIBUTING.md and will fork, sign the CLA, and ensure the notebook runs clean top-to-bottom
on a fresh Inf2.
