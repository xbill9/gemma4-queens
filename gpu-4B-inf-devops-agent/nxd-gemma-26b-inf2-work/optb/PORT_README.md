# ✅ SOLVED (2026-07-16): MoE runs on Inferentia — SEQ_MATCH True, "Paris"

`tp_mb_moe.py` (NxD ModelBuilder, TP=8) validated on a live inf2.24xlarge:
```
build_module: 25 sharded-attn, 5 replicated-attn, 30 MoE layers, TP=8 (E=128 K=8 I=704)
CPU GEN: 'The capital of France is **Paris**.'
DEV GEN: 'The capital of France is **Paris**.'   SEQ_MATCH True   device prefill 77ms
```
Neff `mb_26b_256.pt` (21.9GB) in `s3://xbill-gemma4-31b-usw2/w26b/neffs/`. **All five Gemma-4 variants
(E2B/E4B/12B/31B/26B-A4B) now run on Inferentia.**

**How (path A, hand-rolled — cleaner than wrapping NxD's ExpertMLPs):** reuse the 31B ModelBuilder
recipe wholesale (attention shard/replicate, ScatterKV, layer_scalar, chat-template, device
prefill/decode) and swap ONLY `Gemma4TextExperts` → `DenseExperts`:
- **all-experts-DENSE** compute (all 128 experts on every token, weighted by the top-8 router weight,
  0 for non-selected → EXACT match to HF's sparse top-8; static-shape + traceable). The math + TP
  sharding logic were unit-tested vs HF on CPU (MAXDIFF ~2e-6, cos 1.0) before any device run.
- expert weights reshaped onto TWO standard parallel linears ModelBuilder auto-shards:
  `gate_up [E*2I,H]` ColumnParallelLinear (rank r → experts 16r..16r+15) + `down [H,E*I]`
  RowParallelLinear (input-sharded → all-reduce). Reshape in `checkpoint_loader`.
- HF's dual-path decoder layer forward + `Gemma4TextRouter` kept UNCHANGED (router replicated; its
  top_k_weights already carry renorm + per_expert_scale).

**THE key bug (cost 2 compiles):** `scatter_to_tensor_model_parallel_region` uses a Python-int
`get_tensor_model_parallel_rank()` → in ModelBuilder's SINGLE-RANK trace it BAKES rank 0's slice
(`Wd[:,0:16]`), so at runtime every rank weights experts 0-15 while `gate_up` computes its OWN experts
→ misaligned garbage (empty output). FIX: register `SPMDRank(TP)` in DenseExperts (checkpoint
`experts.spmd_rank.rank = torch.arange(TP, int32)`, sharded dim0 → each rank gets its number) and use
`scatter_to_process_group_spmd(Wd, 1, spmd_rank.get_rank())` — a RUNTIME rank-aware scatter. This is
the mechanism NxD's own ExpertMLPs uses (`enable_spmd_rank`). Symptom of the bug: device output empty
(immediate EOS) while CPU ref correct and the math/sharding unit-tests pass → it's a runtime-rank issue.

**Remaining (optional):** server wrap + build production 512/128 + publish (reuse the 31B server/Docker).

---

# Gemma-4 26B-A4B → Inferentia2 (inf2.24xlarge, TP=8) — port scaffold

HF: `google/gemma-4-26B-A4B-it`. **A4B = 26B total params, ~3.2B active/token (MoE).**
Seeded from the E4B `tp_alias` recipe, but the MoE block is **new capability** on Neuron —
this is a research spike, NOT a port. Config saved as `config.reference.json`.
**Recommendation on record: land 31B (dense) first, then tackle this.**

## Confirmed architecture (from config.json)

| field | value | implication |
|---|---|---|
| `enable_moe_block` | **True** | MoE — dense-MLP shard path is WRONG here (see gap below) |
| `num_experts` | **128** | all 128 resident in HBM |
| `top_k_experts` | **8** | 8 experts routed per token |
| `moe_intermediate_size` | 704 | per-expert FFN (tiny) |
| `intermediate_size` | 2112 | dense/shared FFN size (confirm if a shared expert exists) |
| `num_hidden_layers` | 30 | |
| `hidden_size` | 2816 | |
| `num_attention_heads` / `kv` | 16 / 8 | TP=8 → 2 heads, 1 kv/rank ✓ |
| `head_dim` | 256 | |
| `layer_types` | 25 sliding + 5 full | sliding_window=1024 |
| `hidden_size_per_layer_input` | **0** | NO PLE (like 31B) — "closer to 4B" is about *active* size, not PLE |
| `final_logit_softcapping` | 30.0 | keep ✓ |
| `tie_word_embeddings` | True | |
| `vocab_size` | 262144 | |
| `vision_config` | present | text path only |

## ⚠️ Memory: A4B saves COMPUTE, not MEMORY

- Total ~24.6B params → **bf16 RESIDENT weights ~49 GB**. **ALL 128 experts must sit in HBM**;
  top-8 routing only reduces FLOPs, not footprint. Budget for the full 26B, not 4B.
- At TP=8: ~6.1 GB/core weights (+ unsharded lm_head 2.82 GB/core). Fits 16 GB/core.
- Expert-parallel option: 128 experts / TP=8 = **16 experts/rank**.

## ✅ Architecture DECODED (2026-07-16, live meta-device inspection of transformers 5.13)

Every layer is a **dual-path FFN** (shared dense MLP running in parallel with a 128-expert MoE), NOT
"MLP replaced by MoE". Layer children: `self_attn, mlp, router, experts` + FOUR feed-forward norms
(`pre_feedforward_layernorm`, `post_feedforward_layernorm`, `_1`, `_2`, `pre_feedforward_layernorm_2`).

**Decoder layer forward (all 30 layers, `enable_moe_block=True`):**
```
residual = h (post-attention)
dense = post_feedforward_layernorm_1( mlp( pre_feedforward_layernorm(residual) ) )   # shared MLP, intermediate 2112
_,w,idx = router(residual_flat)                                                       # top-8
moe   = post_feedforward_layernorm_2( experts( pre_feedforward_layernorm_2(residual_flat), idx, w ) )
h = post_feedforward_layernorm( dense + moe ) + residual
h *= layer_scalar                                                                     # buffer, load manually (like 31B)
```
- **Router** `Gemma4TextRouter`: params `proj (128,2816)`, `scale (2816,)`, `per_expert_scale (128,)` (+ own RMSNorm).
  Forward: `x=norm(x)*scale*√d; p=softmax(proj(x),fp32,dim=128); w,idx=topk(p,8); w/=w.sum; w*=per_expert_scale[idx]`.
- **Experts** `Gemma4TextExperts`: FUSED `gate_up_proj (128,1408,2816)` + `down_proj (128,2816,704)`, GELUtanh.
  HF forward is a SPARSE gather/scatter loop (`torch.where`, `index_add_`) → **will NOT trace statically**.
- **Shared MLP** `Gemma4TextMLP`: standard `down(act(gate(x))*up(x))`, intermediate 2112.
- Attention: 25 sliding (8 kv, head_dim 256) + 5 global (2 kv, head_dim 512, `attention_k_eq_v`) — SAME
  shard/replicate class as the 31B port (sliding shard, global replicate; 8%8==0 shard, 2%8!=0 replicate).

## ⚠️ The crux: expert weights (45.6 GB) MUST shard — can't replicate

128 experts ≈ 45.6 GB bf16 (30 layers). Replicating on TP=8 = 45.6 GB/core ≫ 16 GB. Must shard
expert-parallel (16 experts/rank) or tensor-parallel within expert (shard the 704) → ~5.7 GB/rank.
ModelBuilder does NOT auto-shard 3D expert Parameters (only ColumnParallel/RowParallel Linears).

## ➡️ Plan: NxD `modules.moe` (path B) — it has the exact primitives

`neuronx_distributed.modules.moe` provides **`ExpertMLPs`** (sharded expert compute, tensor+expert
parallel groups, blockwise matmul, `glu_mlp` for fused gate+up), **`RouterTopK`**, and — crucially —
**`SharedExperts`** + a **`MoE`** wrapper (router + expert_mlps + shared_experts + rmsnorm) that models
Gemma4's dual-path natively. Approach:
1. Reuse the 31B ModelBuilder recipe wholesale (attention shard/replicate, ScatterKV, layer_scalar
   buffers, chat-template prompt, device prefill/decode).
2. Replace `Gemma4TextExperts` with NxD `ExpertMLPs` (map fused gate_up_proj/down_proj → its weight
   layout; glu_mlp=True); keep Gemma4's dual-path layer forward + shared `mlp` (shard TP) + the 4 norms.
3. Router: either NxD `RouterTopK` or keep Gemma4's router patched traceable (must replicate the extra
   `scale`/`per_expert_scale` + renorm — RouterTopK may not model these exactly → correctness risk).
4. First-light fallback if ExpertMLPs mapping fights: hand-rolled all-experts-dense + manual 3D-param
   expert-parallel shard + all-reduce (static-shape, exact match, slow).

Env: `aws_neuronx_venv_pytorch_2_8_nxd_inference` + `transformers==5.13.0` + fx shim +
`PATH=$VENV/bin:/opt/aws/neuron/bin`. Weights banked: `s3://xbill-gemma4-31b-usw2/w26b/weights/` (51.6GB).

## Bring-up sequence

1. Inspect the loaded `Gemma4` module to see the exact MoE block structure (router + experts + any
   shared FFN); decide expert-compute strategy (start all-experts-dense).
2. Download weights → `/workspace/real-gemma4-26B-A4B-it`; set `MP` (currently `real-gemma4-26B-it`).
3. Rewrite `_shard()` + `DecWrap.forward` for MoE; strip PLE (as in 31B).
4. `KV_MAX=256 KV_BUCKET=64 TP_DEGREE=8 python tp_alias_trace.py` → validate SEQ_MATCH.
5. Server + Dockerfile + publish.

Note: `MP` in the scaffold reads `/workspace/real-gemma4-26B-it` — update to `-26B-A4B-it`.
