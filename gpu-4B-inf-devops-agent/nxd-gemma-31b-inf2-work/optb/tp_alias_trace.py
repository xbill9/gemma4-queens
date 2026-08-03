"""TP=8 + on-device KV aliasing, DECODE-ONLY trace for Gemma4-31B (DENSE).
Adapted from the E4B tp_alias recipe. Deltas for 31B (config-confirmed):
  * NO PLE: hidden_size_per_layer_input=0 -> the per_layer_inputs / get_per_layer_inputs path
    is removed (E4B/E2B are E-family with PLE; 31B is a plain dense transformer).
  * TP=8 (heads 32 / kv 16 both divisible; TP=12 FAILS).
  * Variable head_dim: sliding layers use head_dim=256, full/global layers use global_head_dim=512.
    LINFO keys KV width off each layer's a.head_dim, so buffers size per-layer automatically.
  * Optional lm_head column-parallel shard (env SHARD_LMHEAD=1) to reclaim ~2.5 GB/core of the
    replicated 2.82 GB vocab head. Default OFF (replicated head is the known-good E4B pattern).
Correct GQA sharding: q/o/k/v/gate/up/down across cores; k/v -> nkv//TP heads/rank when divisible.
KV buffers are device-resident, aliased via input_output_aliases so the cache never leaves the core.
Prefill runs on CPU at run time (one-time seed) so only ONE neff is on-device. Saves via parallel_model_save."""
import sys, os, types, time
m = types.ModuleType("transformers.utils.fx"); m.HFTracer=object; m.symbolic_trace=None
sys.modules["transformers.utils.fx"] = m
import multiprocessing
MP = "/workspace/real-gemma4-31B-it"
TP = int(os.environ.get("TP_DEGREE", "8"))          # 31B: heads 32 / kv 16 -> TP=8 OK, TP=12 fails
MAX = int(os.environ.get("KV_MAX", "2048"))
BUCKET = int(os.environ.get("KV_BUCKET", "512"))
SHARD_LMHEAD = os.environ.get("SHARD_LMHEAD", "0") == "1"   # memory lever (see docstring)
CARGS = ["--model-type","transformer","--auto-cast","all","--auto-cast-type","bf16"]

def _kv_rank_width(nkv):
    return nkv // TP if nkv % TP == 0 else nkv

def _build_shared():
    import torch
    from transformers import Gemma4ForConditionalGeneration
    mm = Gemma4ForConditionalGeneration.from_pretrained(MP, torch_dtype=torch.float32, attn_implementation="eager"); mm.eval()
    lang = mm.model.language_model; cfg = lang.config
    class GeluTanh(torch.nn.Module):
        def forward(s, x): return 0.5*x*(1.0+torch.tanh(0.7978845608028654*(x+0.044715*x*x*x)))
    for mod in lang.modules():
        if hasattr(mod, "act_fn"): mod.act_fn = GeluTanh()
    NONSHARED, LINFO = [], {}
    for i, lyr in enumerate(lang.layers[:cfg.num_hidden_layers]):
        a = lyr.self_attn
        # Dense 31B may not carry the E-family is_kv_shared_layer attr -> default False (all non-shared).
        if not getattr(a, "is_kv_shared_layer", False):
            hd=a.head_dim; NONSHARED.append(i); LINFO[i]=(a.k_proj.out_features//hd, hd)
    softcap = getattr(mm.config.text_config, "final_logit_softcapping", None)
    return mm, lang, mm.lm_head, softcap, NONSHARED, LINFO, cfg.sliding_window

def _shard(lang, nlayers):
    import torch
    from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
    from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_rank
    rank = get_tensor_model_parallel_rank()
    def col(o):
        out,inf=o.weight.shape; l=ColumnParallelLinear(inf,out,bias=False,gather_output=False,dtype=o.weight.dtype); c=out//TP
        l.weight.data.copy_(o.weight.data[rank*c:(rank+1)*c,:]); return l
    def row(o):
        out,inf=o.weight.shape; l=RowParallelLinear(inf,out,bias=False,input_is_parallel=True,dtype=o.weight.dtype); c=inf//TP
        l.weight.data.copy_(o.weight.data[:,rank*c:(rank+1)*c]); return l
    for lyr in lang.layers[:nlayers]:
        a=lyr.self_attn; hd=a.head_dim
        k_out=a.k_proj.out_features if getattr(a,"k_proj",None) is not None else None
        nkv=(k_out//hd) if k_out is not None else None
        # Attention: shard q/k/v/o ONLY when kv heads divide TP (clean contiguous GQA sharding:
        # rank r's q-head block maps to rank r's kv-head block, groups preserved).
        # 31B global (full_attention) layers have 4 kv heads (< TP=8) and v_proj=None (k_eq_v) ->
        # can't split; REPLICATE the whole attention (leave q/k/v/o full) and keep 4 kv heads on
        # every rank. Contiguous q-sharding would misalign q-heads against the replicated KV.
        if nkv is not None and nkv % TP == 0:
            a.q_proj=col(a.q_proj); a.o_proj=row(a.o_proj); a.k_proj=col(a.k_proj)
            if getattr(a,"v_proj",None) is not None: a.v_proj=col(a.v_proj)
        # else: replicated attention (q/k/v/o unchanged); num_key_value_groups stays model default.
        # MLP: always sharded.
        mp=lyr.mlp; mp.gate_proj=col(mp.gate_proj); mp.up_proj=col(mp.up_proj); mp.down_proj=row(mp.down_proj)

def _shard_lmhead(lm_head):
    """Column-parallel the vocab head, gather_output=True so every rank still sees full logits
    (argmax on host is unchanged). Weight drops from full vocab to vocab//TP per rank."""
    import torch
    from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear
    from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_rank
    rank = get_tensor_model_parallel_rank()
    out,inf = lm_head.weight.shape           # [vocab, hidden]
    assert out % TP == 0, f"vocab {out} not divisible by TP {TP}"
    c = out // TP
    l = ColumnParallelLinear(inf, out, bias=False, gather_output=True, dtype=lm_head.weight.dtype)
    l.weight.data.copy_(lm_head.weight.data[rank*c:(rank+1)*c, :]); return l

def _sc(softcap, lg):
    import torch
    return softcap*torch.tanh(lg/softcap) if softcap else lg

def get_dec():
    import torch
    mm, lang, lm_head, softcap, NONSHARED, LINFO, SW = _build_shared()
    _shard(lang, lang.config.num_hidden_layers); NK=len(NONSHARED)
    head = _shard_lmhead(lm_head) if SHARD_LMHEAD else lm_head
    W = {i: _kv_rank_width(LINFO[i][0]) for i in NONSHARED}   # per-rank KV width
    class StaticKV:
        is_compileable=False
        def __init__(s,kb,vb,oh): s.key={i:kb[j] for j,i in enumerate(NONSHARED)}; s.val={i:vb[j] for j,i in enumerate(NONSHARED)}; s.oh=oh
        def update(s,k,v,idx,*a,**kw): s.key[idx]=s.key[idx]*(1.0-s.oh)+k*s.oh; s.val[idx]=s.val[idx]*(1.0-s.oh)+v*s.oh; return s.key[idx],s.val[idx]
        def get_seq_length(s,*a,**k): return 0
        def export(s): return [s.key[i] for i in NONSHARED],[s.val[i] for i in NONSHARED]
    class DecWrap(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.lang=lang; s.head=head
            s.kbuf=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1,W[i],MAX,LINFO[i][1]),requires_grad=False) for i in NONSHARED])
            s.vbuf=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1,W[i],MAX,LINFO[i][1]),requires_grad=False) for i in NONSHARED])
        def forward(s, ie, position_ids, onehot, full_mask, slide_mask):   # NO PLE arg (31B)
            cache=StaticKV(list(s.kbuf),list(s.vbuf),onehot)
            out=s.lang(inputs_embeds=ie,position_ids=position_ids,attention_mask={"full_attention":full_mask,"sliding_attention":slide_mask},use_cache=True,past_key_values=cache)
            lg=_sc(softcap,s.head(out.last_hidden_state)); ks,vs=cache.export(); return (lg,ks,vs)
    dec=DecWrap().eval()
    aliases={}
    for j in range(NK): aliases[dec.kbuf[j]]=1+j
    for j in range(NK): aliases[dec.vbuf[j]]=1+NK+j
    return dec, aliases

def main():
    import torch
    from transformers import AutoTokenizer
    import neuronx_distributed
    from neuronx_distributed.trace import parallel_model_save
    NEG=torch.finfo(torch.float32).min
    tok=AutoTokenizer.from_pretrained(MP)
    _mm, rlang, _h, _sc2, NONSHARED, LINFO, SW = _build_shared()
    hds=sorted({LINFO[i][1] for i in NONSHARED})
    from collections import Counter
    dist=Counter((LINFO[i][1], LINFO[i][0], "shard" if LINFO[i][0]%TP==0 else "replicate") for i in NONSHARED)
    print(f"31B: {len(NONSHARED)} non-shared layers, distinct head_dims={hds}, TP={TP}, SHARD_LMHEAD={SHARD_LMHEAD}",flush=True)
    print(f"  (head_dim, nkv, mode) distribution: {dict(dist)}",flush=True)
    def embed_ids(idl):                       # NO PLE (31B)
        ids=torch.tensor([idl])
        with torch.no_grad(): ie=rlang.embed_tokens(ids)
        return ie
    enc=tok.apply_chat_template([{"role":"user","content":"What is the capital of France?"}],add_generation_prompt=True,return_tensors="pt",return_dict=True)
    prompt=enc["input_ids"][0].tolist(); n0=len(prompt); assert n0<=BUCKET
    ie1=embed_ids([prompt[-1]])
    ar=torch.arange(MAX); oh=(ar==n0).view(1,1,MAX,1).to(torch.float32); valid=ar<=n0
    fm=torch.where(valid,0.0,NEG).view(1,1,1,MAX); sm=torch.where(valid&(ar>n0-SW),0.0,NEG).view(1,1,1,MAX); pid=torch.tensor([[n0]],dtype=torch.long)
    # max_parallel_compilations=1 serializes per-rank build+compile (rendezvous barrier) so peak
    # host RAM is ~1-2 ranks, not TP=8 simultaneously — the documented OOM remedy for large models.
    MPC=int(os.environ.get("MAX_PAR_COMPILE","1"))
    print(f"tracing TP+alias DECODE (only), max_parallel_compilations={MPC} ...",flush=True)
    dec=neuronx_distributed.trace.parallel_model_trace(get_dec,(ie1,pid,oh,fm,sm),tp_degree=TP,compiler_args=CARGS,max_parallel_compilations=MPC)
    parallel_model_save(dec,"/workspace/tpa_dec"); print("DECODE_SAVED",flush=True)
    print("TPA_TRACE_OK",flush=True)

if __name__ == "__main__":
    if os.environ.get("_TP_CHILD") != "1":
        os.environ["_TP_CHILD"]="1"; multiprocessing.set_start_method("spawn", force=True); main()
