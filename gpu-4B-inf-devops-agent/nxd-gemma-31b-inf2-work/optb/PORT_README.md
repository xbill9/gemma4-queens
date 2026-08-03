# Gemma-4 31B → Inferentia2 (inf2.24xlarge, TP=8) — port scaffold

Seeded from the **E4B `tp_alias` recipe** (dense-TP + on-device aliased KV + CPU-seed prefill).
**Recommended first port** — it's a dense scaling problem, not new capability (cf. 26B-A4B MoE).

HF: `google/gemma-4-31B-it` (deploy) / `google/gemma-4-31B` (base). Config saved as
`config.reference.json`. Status: **SCAFFOLD ONLY — not yet traced/validated.**

## Confirmed architecture (from config.json)

| field | value | implication |
|---|---|---|
| `enable_moe_block` | **False** | **DENSE** — tp_alias `_shard()` dense-MLP path is correct as-is ✓ |
| `num_hidden_layers` | 60 | 2× the E4B layer count |
| `hidden_size` | 5376 | |
| `intermediate_size` | 21504 | /8 = 2688 ✓ |
| `num_attention_heads` | 32 | /8 = 4 heads/rank ✓ |
| `num_key_value_heads` | 16 | GQA 2:1; /8 = 2 kv/rank ✓ |
| `head_dim` / `global_head_dim` | 256 / **512** | ⚠️ full-attention layers use head_dim **512** |
| `layer_types` | 50 sliding + 10 full | sliding_window=1024; every 6th layer full |
| `hidden_size_per_layer_input` | **0** | ⚠️ **NO PLE** — strip the per-layer-embeddings path |
| `final_logit_softcapping` | 30.0 | keep (tp_alias `_sc` handles it) ✓ |
| `tie_word_embeddings` | True | embed/lm_head share weights |
| `vocab_size` | 262144 | lm_head = 2.82 GB bf16 |
| `vision_config` | present (gemma4_vision, 27L) | take `model.language_model` only; skip vision |

## TP degree: **8** (locked)

TP=8 divides heads (32) and kv (16). **TP=12 FAILS** (32/12, 16/12 non-integer). Default already
set to `TP_DEGREE=8` in trace/run/server. inf2.24xlarge = 12 cores → TP=8 uses 8, leaves 4 spare.

## Memory (your "weights are heavy" concern — real but manageable)

- ~30.1B params → **bf16 weights ~60 GB**; at TP=8 ≈ **7.5 GB/core**.
- **Watch #1 — per-core budget:** unsharded `lm_head` adds **2.82 GB/core** (replicated) →
  ~10.3 GB/core, under 16 GB but tighter than E4B. **Lever: shard lm_head column-parallel**
  (gather logits) to reclaim ~2.5 GB/core.
- **Watch #2 — host trace peak:** `from_pretrained(torch_dtype=float32)` = **~121 GB host RAM**
  (+ vision tower). inf2.24xlarge host = 384 GB, fits, but load bf16 on host if it thrashes.
- KV cache is not the constraint. Start `KV_MAX=256 KV_BUCKET=64` for first-light anyway.

## Code deltas from the E4B skeleton (the actual work)

1. **Strip PLE.** `hidden_size_per_layer_input=0` → remove `get_per_layer_inputs(...)` and the
   `per_layer_inputs=ple` arg in trace/run/server. Feed only `inputs_embeds`.
2. **KV-shared layers.** E-family used `is_kv_shared_layer` to build `NONSHARED`. Verify the attr
   exists on 31B; if not, ALL 60 layers are non-shared → 60 KV buffers (more per-core KV, still fine).
3. **Variable head_dim.** Full-attention layers use `global_head_dim=512` (10 layers), sliding use
   256. `LINFO[i]=(nkv, a.head_dim)` already keys KV width per-layer — confirm `a.head_dim` reports
   512 on global layers so the aliased KV buffers size correctly.
4. **(Optional) shard lm_head** column-parallel — memory lever above.
5. `TP=8`, `MP=/workspace/real-gemma4-31B-it` — already set.

## Bring-up sequence (on a launched inf2.24xlarge spot box)

1. Download weights → `/workspace/real-gemma4-31B-it`.
2. Apply deltas 1–3 to `tp_alias_trace.py`.
3. `KV_MAX=256 KV_BUCKET=64 TP_DEGREE=8 python tp_alias_trace.py` → saves `/workspace/tpa_dec`.
4. `... python tp_alias_run.py` → expect **SEQ_MATCH True** vs CPU ground truth ("...Paris.").
5. Wrap `optb_server_tp.py`, build `Dockerfile.tp2`, push to ECR/Docker Hub/HF.
