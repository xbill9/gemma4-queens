"""Gemma-4 31B (dense, model_type gemma4) TP=8 prefill+decode via NxD ModelBuilder — ONE bf16 weight
set shared across both buckets + on-device aliased KV. Adapted from the 12B tp_mb.py.

Why ModelBuilder (not the tp_alias/E4B recipe): at 31B the manual parallel_model_trace OOMs
(compiles all 8 ranks in-process) or deadlocks at weight-collection (MPC=1). ModelBuilder compiles
ONE rank and loads weights per-rank via checkpoint_loader — the designed large-model path.

31B deltas vs 12B:
  * Model class Gemma4ForConditionalGeneration (not ...Unified); text path = mm.model.language_model.
  * Two attention layouts: 50 sliding layers (head_dim 256, 16 kv heads) shard k/v; 10 global
    (full_attention) layers (head_dim 512, 4 kv heads, v_proj=None / attention_k_eq_v) REPLICATE the
    WHOLE attention (q/k/v/o) — 4 kv < TP=8 and contiguous q-sharding would misalign q-heads to the
    replicated KV (12B could shard q because its global nkv=1; 31B nkv=4 cannot).
  * No PLE (hidden_size_per_layer_input=0), no kv-shared layers (all 60 non-shared), softcap 30.
Set MODEL_DIR + MB_WDTYPE=bf16 (mandatory to fit ~15GB/rank), TP_DEGREE=8."""
import sys, os, types, time
m=types.ModuleType("transformers.utils.fx"); m.HFTracer=object; m.symbolic_trace=None; sys.modules["transformers.utils.fx"]=m
import multiprocessing
MP=os.environ.get("MODEL_DIR","/data/real-gemma4-31B-it"); TP=int(os.environ.get("TP_DEGREE","8"))
MAX=int(os.environ.get("KV_MAX","256")); BUCKET=int(os.environ.get("KV_BUCKET","64"))
CARGS="--model-type transformer --auto-cast all --auto-cast-type bf16"

def _kv_rank_width(nkv): return nkv//TP if nkv%TP==0 else nkv

def _discover():
    """Load ref model on host once (real weights) for NONSHARED/LINFO/softcap/SW + CPU reference."""
    import torch
    from transformers import Gemma4ForConditionalGeneration
    mm=Gemma4ForConditionalGeneration.from_pretrained(MP,torch_dtype=torch.float32,attn_implementation="eager"); mm.eval()
    lang=mm.model.language_model; cfg=lang.config
    NONSHARED,LINFO=[],{}
    for i,lyr in enumerate(lang.layers[:cfg.num_hidden_layers]):
        a=lyr.self_attn
        if not getattr(a,"is_kv_shared_layer",False):
            hd=a.head_dim; NONSHARED.append(i); LINFO[i]=(a.k_proj.out_features//hd,hd)
    softcap=getattr(mm.config.text_config,"final_logit_softcapping",None)
    return mm,lang,mm.lm_head,softcap,NONSHARED,LINFO,cfg.sliding_window

def _structure():
    """Build model STRUCTURE from config only (no weight load) — safe under ModelBuilder meta context."""
    import torch
    from transformers import Gemma4ForConditionalGeneration, AutoConfig
    cfg=AutoConfig.from_pretrained(MP)
    cfg._attn_implementation="eager"
    if hasattr(cfg,"text_config"): cfg.text_config._attn_implementation="eager"  # sliding_window fused-attn SBUF overflow
    mm=Gemma4ForConditionalGeneration(cfg); mm.eval()
    lang=mm.model.language_model; lc=lang.config
    NONSHARED,LINFO=[],{}
    for i,lyr in enumerate(lang.layers[:lc.num_hidden_layers]):
        a=lyr.self_attn
        if not getattr(a,"is_kv_shared_layer",False):
            hd=a.head_dim; NONSHARED.append(i); LINFO[i]=(a.k_proj.out_features//hd,hd)
    softcap=getattr(mm.config.text_config,"final_logit_softcapping",None)
    return mm,lang,mm.lm_head,softcap,NONSHARED,LINFO,lc.sliding_window

def build_module():
    """Sharded Gemma4 Wrap: parallel-layer STRUCTURE only (weights come from checkpoint_loader)."""
    import torch
    from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
    mm,lang,lm_head,softcap,NONSHARED,LINFO,SW=_structure()
    class GeluTanh(torch.nn.Module):
        def forward(s,x): return 0.5*x*(1.0+torch.tanh(0.7978845608028654*(x+0.044715*x*x*x)))
    for mod in lang.modules():
        if hasattr(mod,"act_fn"): mod.act_fn=GeluTanh()
    WDT=torch.bfloat16 if os.environ.get("MB_WDTYPE","bf16")=="bf16" else torch.float32
    def col(o): return ColumnParallelLinear(o.in_features,o.out_features,bias=False,gather_output=False,dtype=WDT)
    def row(o): return RowParallelLinear(o.in_features,o.out_features,bias=False,input_is_parallel=True,dtype=WDT)
    nshard=nrepl=0
    for lyr in lang.layers[:lang.config.num_hidden_layers]:
        a=lyr.self_attn; hd=a.head_dim
        k_out=a.k_proj.out_features if getattr(a,"k_proj",None) is not None else None
        nkv=(k_out//hd) if k_out is not None else None
        if nkv is not None and nkv%TP==0:
            # sliding layers (nkv=16): shard q/k/v/o; rank r q-block maps to rank r kv-block, groups kept.
            a.q_proj=col(a.q_proj); a.o_proj=row(a.o_proj); a.k_proj=col(a.k_proj)
            if getattr(a,"v_proj",None) is not None: a.v_proj=col(a.v_proj)
            nshard+=1
        else:
            # global layers (nkv=4 < TP=8, v_proj=None): REPLICATE whole attention (q/k/v/o stay plain
            # Linear -> loaded full on every rank). Each rank computes full 32q x 4kv, default groups=8.
            # (Cannot shard q: 4 consecutive q-heads map to ONE kv head, so col-sharded q + groups=1 would
            #  misalign — unlike 12B nkv=1 where sharding q + groups=nq//TP is valid.)
            nrepl+=1
        mp=lyr.mlp; mp.gate_proj=col(mp.gate_proj); mp.up_proj=col(mp.up_proj); mp.down_proj=row(mp.down_proj)  # MLP always sharded
    print(f"build_module: {nshard} sharded-attn layers, {nrepl} replicated-attn (global) layers, TP={TP}",flush=True)
    # CRITICAL: load per-layer `layer_scalar` BUFFERS from the checkpoint. 31B (like 12B) scales each
    # layer by self.layer_scalar; it's a BUFFER (not a Parameter), and ModelBuilder's
    # get_sharded_checkpoint loads PARAMS only — so without this every layer uses the config default
    # 1.0 instead of the learned value, compounding over all 60 layers into garbage (cos~0).
    import glob
    from safetensors import safe_open
    lsv={}
    for fp in sorted(glob.glob(MP+"/*.safetensors")):
        with safe_open(fp,framework="pt") as sf:
            for k in sf.keys():
                if k.startswith("model.language_model.layers.") and k.endswith(".layer_scalar"):
                    lsv[int(k.split("layers.")[1].split(".")[0])]=sf.get_tensor(k)
    with torch.no_grad():
        for i,lyr in enumerate(lang.layers[:lang.config.num_hidden_layers]):
            if hasattr(lyr,"layer_scalar") and i in lsv:
                lyr.layer_scalar.copy_(lsv[i].to(lyr.layer_scalar.dtype))
    print(f"  loaded {len(lsv)} layer_scalar buffers",flush=True)
    lang.embed_tokens=torch.nn.Embedding(2, lang.config.hidden_size)   # dummy: embeddings computed on host
    NK=len(NONSHARED); W={i:_kv_rank_width(LINFO[i][0]) for i in NONSHARED}
    # softcap NOT applied on-device (monotonic -> argmax unchanged; server applies host-side for sampling).
    class ScatterKV:
        is_compileable=False
        def __init__(s,kb,vb,oh): s.key={i:kb[j] for j,i in enumerate(NONSHARED)}; s.val={i:vb[j] for j,i in enumerate(NONSHARED)}; s.oh=oh
        def _wr(s,buf,x):
            oh=s.oh[0,0]; scat=torch.einsum('bhsd,ms->bhmd',x,oh); r=oh.sum(-1).view(1,1,MAX,1); return buf*(1.0-r)+scat
        def update(s,k,v,idx,*a,**kw): s.key[idx]=s._wr(s.key[idx],k); s.val[idx]=s._wr(s.val[idx],v); return s.key[idx],s.val[idx]
        def get_seq_length(s,*a,**k): return 0
        def export(s): return [s.key[i] for i in NONSHARED],[s.val[i] for i in NONSHARED]
    class Wrap(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.lang=lang; s.head=lm_head
            s.kbuf=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1,W[i],MAX,LINFO[i][1]),requires_grad=False) for i in NONSHARED])
            s.vbuf=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1,W[i],MAX,LINFO[i][1]),requires_grad=False) for i in NONSHARED])
        def forward(s, ie, position_ids, onehot, full_mask, slide_mask):
            cache=ScatterKV(list(s.kbuf),list(s.vbuf),onehot)
            out=s.lang(inputs_embeds=ie,position_ids=position_ids,
                       attention_mask={"full_attention":full_mask,"sliding_attention":slide_mask},use_cache=True,past_key_values=cache)
            lg=s.head(out.last_hidden_state); ks,vs=cache.export(); return (lg,)+tuple(ks)+tuple(vs)  # RAW logits
    w=Wrap().eval()
    aliases={}
    for j in range(NK): aliases[w.kbuf[j]]=1+j
    for j in range(NK): aliases[w.vbuf[j]]=1+NK+j
    return w, aliases

def checkpoint_loader():
    """Full fp32 state dict keyed to the Wrap (lang.*/head.*), loaded straight from safetensors."""
    import torch, glob
    from safetensors.torch import load_file
    raw={}
    for fp in sorted(glob.glob(MP+"/*.safetensors")): raw.update(load_file(fp))
    sd={}
    for k,v in raw.items():
        if k=="model.language_model.embed_tokens.weight":
            sd["head.weight"]=v.to(torch.float32)                 # tied lm_head
        elif k.startswith("model.language_model."):
            sd["lang."+k[len("model.language_model."):]]=v.to(torch.float32)
        elif k.startswith("lm_head."):
            sd["head."+k[len("lm_head."):]]=v.to(torch.float32)
    return sd

def _inputs(lang, ids, SW, NEG, positions):
    import torch
    ids_t=torch.tensor([ids])
    with torch.no_grad(): ie=lang.embed_tokens(ids_t)   # scaled word-embedding on host; no PLE
    seq=len(positions); ar=torch.arange(MAX); pos=torch.tensor([positions],dtype=torch.long)
    oh=(ar.view(MAX,1)==pos.view(1,seq)).view(1,1,MAX,seq).to(torch.float32)
    q=pos.view(seq,1); mm2=ar.view(1,MAX)
    full=torch.where(mm2<=q,0.0,NEG).view(1,1,seq,MAX)
    slide=torch.where((mm2<=q)&(mm2>q-SW),0.0,NEG).view(1,1,seq,MAX)
    return ie,pos,oh,full,slide

def main():
    import torch
    from transformers import AutoTokenizer, DynamicCache
    from neuronx_distributed.trace.model_builder import ModelBuilder, BaseModelInstance
    NEG=torch.finfo(torch.float32).min
    tok=AutoTokenizer.from_pretrained(MP)
    _mm,rlang,_h,softcap,NONSHARED,LINFO,SW=_discover()
    print(f"31B discover: {len(NONSHARED)} non-shared layers, head_dims={sorted({LINFO[i][1] for i in NONSHARED})}, softcap={softcap}",flush=True)
    # Gemma-4's chat tokens are <|turn>=105 / <turn|>=106 (NOT <start_of_turn>), and the weight
    # snapshot ships the chat template as a separate chat_template.jinja not embedded in
    # tokenizer_config — so a manual string prompt tokenizes the turn markers into literal chars and
    # the model emits garbage. Fetch+set the canonical template, then apply_chat_template.
    if not getattr(tok,"chat_template",None):
        try:
            import subprocess,json
            from huggingface_hub import hf_hub_download
            raw=subprocess.check_output(["aws","secretsmanager","get-secret-value","--region","us-east-1","--secret-id","hf_token","--query","SecretString","--output","text"]).decode().strip()
            try: tv=json.loads(raw); tv=tv.get("hf_token") or list(tv.values())[0]
            except Exception: tv=raw
            tok.chat_template=open(hf_hub_download(os.environ.get("HF_REPO","google/gemma-4-31B-it"),"chat_template.jinja",token=tv)).read()
        except Exception as e:
            print("chat_template fetch failed, using hardcoded ids:",e,flush=True)
    if getattr(tok,"chat_template",None):
        d=tok.apply_chat_template([{"role":"user","content":"What is the capital of France?"}],add_generation_prompt=True)
        prompt=d if isinstance(d,list) else d["input_ids"]   # BatchEncoding (not a plain dict) or list
    else:
        prompt=[2,105,2364,107,3689,563,506,5279,529,7001,236881,106,107,105,4368,107,100,45518,107,101]
    n0=len(prompt); assert n0<=BUCKET
    pre_in=_inputs(rlang, prompt+[0]*(BUCKET-n0), SW, NEG, list(range(BUCKET)))
    dec_in=_inputs(rlang, [prompt[-1]], SW, NEG, [n0])
    load_from=os.environ.get("MB_LOAD")
    if load_from:
        # Reload previously-compiled neffs instead of recompiling (~39min saved) — lets validation run
        # in a short spot window once a compile-only pass has banked mb_31b_256.pt to S3.
        print("loading saved model from",load_from,flush=True)
        model=torch.jit.load(load_from)
        # torch.jit.load restores the graph + serialized weights but does NOT push them onto the
        # NeuronCores — a forward before this raises "This model is not initialized". The 108GB save
        # embeds the weights (initialize_model_weights=True at trace time), so re-init from saved.
        model.nxd_model.initialize_with_saved_weights(torch.tensor([0],dtype=torch.int32)); print("MB_LOADED",flush=True)
    else:
        class GemmaInstance(BaseModelInstance):
            def __init__(self): self.module=None; self.input_output_aliases=[{}]
            def load_module(self):
                self.module, aliases = build_module()
                self.input_output_aliases=[aliases]
            def get(self, bucket_rank, **kwargs): return self.module, self.input_output_aliases[0]
        inst=GemmaInstance()
        print("building ModelBuilder ...",flush=True)
        mb=ModelBuilder(router=None, tp_degree=TP, checkpoint_loader=checkpoint_loader, compiler_workdir="/data/mb_wd")
        mb.add("prefill", inst, [pre_in], compiler_args=CARGS)
        mb.add("decode",  inst, [dec_in], compiler_args=CARGS)
        print("tracing (ModelBuilder, single-rank compile) ...",flush=True)
        model=mb.trace(initialize_model_weights=True)
        print("MB_TRACED",flush=True)
        # Save neffs immediately after a successful trace (the expensive, reclaim-vulnerable step) so a
        # short spot window that gets us this far is banked — validation can reload the neffs on any box.
        save_to=os.environ.get("MB_SAVE")
        if save_to:
            print("saving model to",save_to,flush=True); torch.jit.save(model, save_to); print("MB_SAVED",save_to,flush=True)
        if os.environ.get("SKIP_VALIDATE")=="1":
            print("SKIP_VALIDATE=1 -> compile-only run complete",flush=True); print("TPMB_OK",flush=True); return
    # === in-process validation vs CPU fp32 reference ===
    head=_mm.lm_head
    ec=_mm.generation_config.eos_token_id; EOS=set(ec) if isinstance(ec,(list,tuple)) else {ec}
    class _G(torch.nn.Module):
        def forward(s,x): return 0.5*x*(1.0+torch.tanh(0.7978845608028654*(x+0.044715*x*x*x)))
    for mod in rlang.modules():
        if hasattr(mod,"act_fn"): mod.act_fn=_G()
    def _sc2(lg): return softcap*torch.tanh(lg/softcap) if softcap else lg
    def embR(ids):
        idt=torch.tensor([ids])
        with torch.no_grad(): ie=rlang.embed_tokens(idt)
        return ie
    def cpuref():
        ie=embR(prompt); c=DynamicCache()
        with torch.no_grad():
            out=rlang(inputs_embeds=ie,attention_mask=torch.ones(1,n0,dtype=torch.long),use_cache=True,past_key_values=c)
            first=int(_sc2(head(out.last_hidden_state))[0,n0-1].argmax())
        KS=[c.layers[i].keys for i in NONSHARED]; VS=[c.layers[i].values for i in NONSHARED]
        class SKV:
            is_compileable=False
            def __init__(s,kb,vb,oh): s.key={i:kb[j] for j,i in enumerate(NONSHARED)}; s.val={i:vb[j] for j,i in enumerate(NONSHARED)}; s.oh=oh
            def update(s,k,v,idx,*a,**kw): s.key[idx]=s.key[idx]*(1.0-s.oh)+k*s.oh; s.val[idx]=s.val[idx]*(1.0-s.oh)+v*s.oh; return s.key[idx],s.val[idx]
            def get_seq_length(s,*a,**k): return 0
        def hp(pos):
            ar=torch.arange(MAX); oh=(ar==pos).view(1,1,MAX,1).float(); valid=ar<=pos
            fm=torch.where(valid,0.0,NEG).view(1,1,1,MAX); sm=torch.where(valid&(ar>pos-SW),0.0,NEG).view(1,1,1,MAX)
            return torch.tensor([[pos]]),oh,fm,sm
        kb=[torch.zeros(1,LINFO[i][0],MAX,LINFO[i][1]) for i in NONSHARED]; vb=[torch.zeros(1,LINFO[i][0],MAX,LINFO[i][1]) for i in NONSHARED]
        for j in range(len(NONSHARED)): kb[j][:,:,:n0,:]=KS[j][:,:,:n0,:]; vb[j][:,:,:n0,:]=VS[j][:,:,:n0,:]
        seq=[first]; cur=n0
        for _ in range(30):
            if seq[-1] in EOS: break
            e1=embR([seq[-1]]); pi,oh,fm,sm=hp(cur); cache=SKV(kb,vb,oh)
            with torch.no_grad():
                out=rlang(inputs_embeds=e1,position_ids=pi,attention_mask={"full_attention":fm,"sliding_attention":sm},use_cache=True,past_key_values=cache)
                lg=_sc2(head(out.last_hidden_state))
            kb=[cache.key[i] for i in NONSHARED]; vb=[cache.val[i] for i in NONSHARED]
            seq.append(int(lg[0,0].argmax())); cur+=1
        return seq
    DEVICE_ONLY=os.environ.get("DEVICE_ONLY")=="1"   # skip the 121GB-fp32 CPU generation under short spot windows
    if DEVICE_ONLY:
        cpu_seq=None; print("DEVICE_ONLY=1 -> skipping CPU reference",flush=True)
    else:
        print("=== CPU ref ===",flush=True); cpu_seq=cpuref()
    def dcall(a): r=model(*a); return r[0] if isinstance(r,(tuple,list)) else r
    pad=prompt+[0]*(BUCKET-n0)
    print("=== DEVICE (bucketed) ===",flush=True)
    dcall(_inputs(rlang,pad,SW,NEG,list(range(BUCKET))))                # warmup
    t0=time.time(); lg=dcall(_inputs(rlang,pad,SW,NEG,list(range(BUCKET)))); pf=time.time()-t0
    first=int(lg[0,n0-1].argmax()); seq=[first]; cur=n0
    for _ in range(30):
        if seq[-1] in EOS: break
        l1=dcall(_inputs(rlang,[seq[-1]],SW,NEG,[cur])); seq.append(int(l1[0,0].argmax())); cur+=1
    dev_txt=tok.decode([x for x in seq if x not in EOS],skip_special_tokens=True)
    if cpu_seq is not None:
        print("CPU GEN:",repr(tok.decode([x for x in cpu_seq if x not in EOS],skip_special_tokens=True)),flush=True)
        print("SEQ_MATCH", cpu_seq==seq, flush=True)
    else:
        print("DEVICE_PARIS", "paris" in dev_txt.lower(), flush=True)   # coherence check w/o CPU ref
    print("DEV GEN:",repr(dev_txt),flush=True)
    print(f"DEVICE PREFILL first-token: {pf*1000:.0f} ms",flush=True)
    print("TPMB_OK",flush=True)

if __name__=="__main__":
    if os.environ.get("_TP_CHILD")!="1":
        os.environ["_TP_CHILD"]="1"; multiprocessing.set_start_method("spawn",force=True); main()
