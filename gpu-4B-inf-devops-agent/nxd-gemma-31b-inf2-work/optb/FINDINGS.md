# ✅ RESOLVED (2026-07-16): ModelBuilder compiles 31B on real hardware

The FINDINGS-recommended path **works**. `tp_mb.py` (NxD ModelBuilder, single-rank compile + per-rank
weight sharding) produced a complete TP=8 model on a live inf2.24xlarge (us-west-2d):

```
31B discover: 60 non-shared layers, head_dims=[256, 512], softcap=30.0
build_module: 50 sharded-attn layers, 10 replicated-attn (global) layers, TP=8
  loaded 60 layer_scalar buffers
Sharding weights for ranks: 0...7  ->  Done Sharding weights in 172.6s
Finished building model in 2365.3s (~39 min)
MB_TRACED -> MB_SAVED -> TPMB_OK -> RUN_EXIT 0
```

- **Neffs banked**: `mb_31b_256.pt` (108 GB, TP=8, KV 256/64, bf16) in `s3://xbill-gemma4-31b-usw2/neffs/`.
- **Peak host memory 182 GB** (fp32 checkpoint + fp32 discover model), well under 369 GB — no OOM.
  ModelBuilder compiles ONE rank then shards weights per-rank; none of the non-SPMD 300 GB blow-up.
- **Compile takes ~39 min** — this is why spot windows <15 min kept dying mid-compile. Solution that
  worked: race scarce spot with S3 checkpointing; the weights (62.5 GB) went to S3 once (durable) and
  the compile finally landed in a ~43-min window. `tp_mb.py` now saves neffs immediately post-trace
  (`SKIP_VALIDATE=1` compile-only) and supports `MB_LOAD=` to reload neffs for validation in a short window.
- **Two fixes vs the first live run**: (1) build the Gemma chat prompt manually — the HF snapshot ships
  the chat template as a separate `chat_template.jinja` not embedded in tokenizer_config, so
  `apply_chat_template()` raised; (2) env = `aws_neuronx_venv_pytorch_2_8_nxd_inference` + `pip install
  transformers==5.13.0` + `export PATH=$VENV/bin:/opt/aws/neuron/bin` (the `libneuronpjrt-path`
  import error was just a missing PATH; the `transformers.utils.fx` shim in tp_mb.py handles tfm 5.13).
- **✅ VALIDATED (2026-07-16)**: reloaded neffs via `MB_LOAD=mb_31b_256.pt` (+ `nxd_model.initialize_with_saved_weights(torch.tensor([0]))` — torch.jit.load alone doesn't push weights to cores). Device output **== CPU fp32 reference: SEQ_MATCH True**, device prefill ~115ms. With the correct prompt: **`DEV GEN: 'The capital of France is Paris.'`** The port is numerically and behaviorally correct.
- **Prompt gotcha (cost 4 debug rounds)**: Gemma-4's chat tokens are **`<|turn>`=105 / `<turn|>`=106** (NOT `<start_of_turn>`), and the snapshot ships the chat template as a separate `chat_template.jinja` (18.7KB "Google Gemma 4 Canonical Chat Template", w/ thinking-token scaffold). A manual `<start_of_turn>...` string tokenizes the markers into literal chars → model emits `<start_of_turn>` garbage. Both CPU+device reproduced it identically (that's why SEQ_MATCH was True on garbage). Fix in tp_mb.py: fetch+set `chat_template.jinja` from HF (Secrets-Manager token) then `apply_chat_template`; hardcoded-id fallback `[2,105,2364,107,...,106,107,105,4368,107,100,45518,107,101]`. `apply_chat_template` returns a BatchEncoding (not a plain dict) → extract `["input_ids"]`.
- **✅ SERVER + PUBLISHED (2026-07-16)**: `optb_server_tp.py` rewritten for ModelBuilder (MB_LOAD +
  `initialize_with_saved_weights` + device prefill/decode via `tp_mb._inputs` + chat-template prompt),
  validated live over HTTP: `curl /generate {"prompt":"What is the capital of France?"}` →
  `"The capital of France is Paris."`, READY in ~69s. Server gotcha: **`import
  neuronx_distributed.trace.model_builder` before `torch.jit.load`** or it raises "Unknown type name
  '__torch__.torch.classes.neuron.SPMDModel'" (the import registers the custom TorchScript class).
  Published: **Docker Hub `xbill9/gemma4-optb-31b`** (`:latest` + `:modelbuilder-256-64`, thin image,
  pins `neuronx-distributed==0.17.26814`, pulls artifacts from HF at runtime) + **HF
  `xbill9/gemma-4-31B-it-inferentia2`** (170.8GB: neffs + weights + bundled chat_template.jinja + recipe
  + server + Dockerfile + card). chat_template.jinja now bundled in the weights dir (S3 + HF) so
  AutoTokenizer auto-loads it — no runtime token needed.
- **Remaining**: production 512/128 (and larger) bucket recompiles; throughput/batching characterization.

Infra: spot inf2.24xlarge capacity was ZERO for 60+ min across all 3 regions; a continuous multi-region
poller (`grab.sh`, capacity-optimized EC2 Fleet) eventually caught a us-west-2d window. Weights bucket
`xbill-gemma4-31b-usw2` (us-west-2, same region as capacity → fast pulls). HF token read from Secrets
Manager via scoped inline policy on `aws-elasticbeanstalk-ec2-role`.

---

# Gemma-4 31B Inferentia port — findings (live-hardware run, 2026-07-15)

Ran the scaffold on a real inf2.24xlarge spot box (us-west-2d, 12 Neuron cores, 369 GB host).
Weights `google/gemma-4-31B-it` downloaded + verified (2 shards, 62.5 GB). Got as far as a
**successful single-rank compile** (`Compiler status PASS`), blocked on multi-rank orchestration.

## ✅ Confirmed on real weights

- **Dense**, `enable_moe_block=False`. 60 layers, hidden 5376, vocab 262144, softcap 30, tied embed.
- **Two attention layouts** (the crux — matches the "head size / KV layout" pain):
  - `sliding_attention` × 50: head_dim 256, 32 q-heads, **16 kv-heads**, has v_proj.
  - `full_attention` (global) × 10: head_dim **512**, 32 q-heads, **4 kv-heads**, **v_proj=None** (`attention_k_eq_v=True`, V reuses K).
- No PLE (`hidden_size_per_layer_input=0`), no kv-shared layers (all 60 non-shared).
- **TP=8 locked** (heads 32 / kv 16 divide 8; TP=12 fails).

## ✅ Solved

1. **KV-layout sharding** (`_shard`): shard q/k/v/o for the 50 sliding layers (16 kv → 2/rank);
   **replicate** attention for the 10 global layers (4 kv < 8 ranks, v_proj None) with full 4-head
   KV on every rank. MLP sharded everywhere. Diagnostic prints the distribution.
2. **Host memory**: two fixes — (a) load HF model bf16 not fp32; (b) `col()/row()` pass
   `dtype=o.weight.dtype` so parallel layers stay bf16 (default upcasts to fp32, doubling RAM).
3. **Compile memory**: `max_parallel_compilations=1` serializes per-rank build+compile →
   **peak 83 GB (was 300+)**, `Compiler status PASS`. The 300+ GB was the non-SPMD tracer
   compiling all 8 ranks in-process at once, NOT the model's real footprint (~15 GB/rank bf16).

## ❌ The remaining blocker — non-SPMD orchestration doesn't scale to 31B

- **Default (all 8 ranks compile at once): OOM** (~300 GB > 369 with overhead).
- **`max_parallel_compilations=1`: memory fine, but DEADLOCKS at multi-rank weight collection** —
  worker blocked on `pipe_read`, CPU idle, 25 min no progress. The mp_q/rendezvous collection
  hangs for a model this size.

## ⚠️ SPMD mode has its own obstacle here

`parallel_model_trace(spmd_mode=True, serialization_path=..., checkpoint_loader_callable=...)` is
the designed large-model path (compile 1 rank, serialize weights per-rank — no 8× compile, no
Queue deadlock). BUT `_load_weights` builds the model on **meta device** and injects weights from
the checkpoint via `get_sharded_checkpoint`. **The tp_alias recipe's device-aliased KV buffers are
runtime state, not checkpoint weights**, so they don't map onto the checkpoint-loading flow.
Forcing tp_alias → SPMD is uncertain.

## ➡️ Recommendation: port 31B via ModelBuilder (like the 12B), not tp_alias

The manual tp_alias recipe (from E4B) doesn't scale to 31B: its all-ranks-in-process compile OOMs,
and its aliased-KV design fights SPMD. The **12B port used NxD ModelBuilder**, which handles
single-rank/SPMD compile + per-rank weight loading + KV cache natively. That is the right base for
31B. Reuse the 12B `tp_mb.py` recipe with the 31B config + the global-layer replicate rule above.

## Infra learnings (for the next run)

- **Spot capacity for inf2.24xlarge is near-zero** (placement scores 1–3/10 across us-east-1/2,
  us-west-2). us-east-2 got reclaimed in ~20 min; us-west-2d held. inf2.48xlarge (768 GB) is
  **quota-blocked** (192 vCPU > 96 spot quota).
- **Drive the box via SSM** (no SSH key): use `aws-elasticbeanstalk-ec2-role` (has SSM core + S3).
  HF token via Secrets Manager (`hf_token`, us-east-1) with a scoped role policy.
- **After `pkill -9` of a Neuron process, REBOOT** — SIGKILL leaves cores in a bad state
  (`NRT_FAILURE` / `nrt_infodump` hang on next run). Verified 3×.
- Root disk fills; attach a separate EBS volume for weights (`/data`). fp32 host load of 31B is
  ~121 GB (fits 369), but bf16 is the right choice for tracing anyway.
