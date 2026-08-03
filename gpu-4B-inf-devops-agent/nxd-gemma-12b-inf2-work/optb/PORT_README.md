# 12B port — TP=2 on inf2.8xlarge (device-prefill ModelBuilder stack)

Clean starting point copied from the working **E4B** device-prefill build
(`../nxd-gemma4-inf2-work/optb/`). Same architecture the E4B port landed on:
one weight-sharing `ModelBuilder` trace (prefill + decode buckets share **one** bf16
sharded weight set + on-device aliased KV), device prefill (~0.16 s), served by
`optb_server_mb.py`.

Target: **TP=2 on a single `inf2.8xlarge`** (both cores, 2×16 GB HBM).

## Files (verbatim from E4B, `MODEL_DIR`/`TP_DEGREE` now env-parameterized)
- `tp_mb.py` — build + save the device-prefill model (`MB_WDTYPE=bf16 MB_SAVE=... python tp_mb.py`)
- `optb_server_mb.py` — device-prefill HTTP server (OpenAI routes + `/generate`)
- `Dockerfile.mb` — package the built model + server

## What carries over unchanged (all the hard-won E4B fixes)
- **`layer_scalar` buffer load** — ModelBuilder shards *parameters* only, never buffers;
  `build_module()` copies the per-layer `layer_scalar` from the checkpoint by hand. Required
  again here (config-built module), already in `tp_mb.py`.
- On-device **aliased KV** (device-resident across prefill→decode), one-hot scatter KV write.
- **bf16 sharded weights** (`MB_WDTYPE=bf16`) — correct + fast; halves on-device weights.
- `torch.jit.save`/`load` of the executor + `initialize_with_saved_weights(start_rank)`.

## TARGET RESOLVED: `google/gemma-4-12B-it` (`model_type: gemma4_unified_text`)
The **encoder-free** Gemma 4 12B — projects raw audio (640 samples/40ms) + image patches (48×48)
directly into the embedding space via linear layers, no audio/vision encoder. Config (from HF):

| layers | hidden | nq | nkv | head_dim | global_head_dim | sliding_window | interm | vocab | softcap | tie_emb |
|---|---|---|---|---|---|---|---|---|---|---|
| **48** | 3840 | 16 | 8 | 256 | 512 | 1024 | 15360 | 262144 | 30.0 | true |

Also: `num_kv_shared_layers=0` (ALL layers own K/V), `attention_k_eq_v=true` (V=K everywhere),
`hidden_size_per_layer_input=0` (**no PLE**).

## Port deltas vs E4B (what to change in tp_mb.py)
1. **No PLE** — drop `per_layer_inputs`/`get_per_layer_inputs` from `_inputs`/`embR`/`Wrap.forward`.
   NOTE: different model class (`gemma4_unified_text`) — verify its `.forward` signature + the
   `language_model` submodule path on the box (transformers 5.13 modeling_gemma4_unified*).
2. **`attention_k_eq_v=true`** — V=K on ALL layers → shard only `k_proj`, reuse as V (E4B's
   `v_proj is None` guard already does this for its global layers; here it's every layer).
3. **`num_kv_shared_layers=0`** — `NONSHARED` = all 48 layers (the `is_kv_shared_layer` filter yields all).
4. **GQA is clean at TP=2**: nkv=8 % 2 == 0 → 4 KV heads/rank, q=8/rank, keep `num_key_value_groups=2`.
   `_kv_rank_width` works as-is; no nkv<TP edge.
5. `layer_scalar` buffer fix, aliased KV, bf16 sharding, tied-head mapping, dual head-dim (256/512),
   jit save/load — all UNCHANGED.

## Risk: FIT, not correctness
12B bf16 ≈ 24 GB → **~12 GB/rank** across 2 cores; ~4 GB/rank left for activations + KV on a 16 GB
core. **bf16 is MANDATORY** (fp32 = 24 GB/rank, won't load). Expect `KV_MAX` to cap below E4B's 512 —
start at 256/64, scale up until HBM says no.

## Next step
Stage `google/gemma-4-12B-it` (gated) at `$MODEL_DIR`, adapt `tp_mb.py` per the deltas above, then:
`KV_MAX=256 KV_BUCKET=64 MB_WDTYPE=bf16 MB_SAVE=/workspace/mb_12b_256.pt python tp_mb.py`
→ expect `SEQ_MATCH True`, then push context + wrap with `optb_server_mb.py`.
