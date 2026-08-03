"""Load TP+alias DECODE neff only. Seed device-resident KV from a CPU prefill (host, fp32),
slicing head-r to rank r. Run device greedy decode; compare to a CPU reference greedy that uses
the same seed. Prints SEQ_MATCH + tok/s. Only the decode neff is on-device (no co-residency)."""
import sys, os, types, time
m = types.ModuleType("transformers.utils.fx"); m.HFTracer=object; m.symbolic_trace=None
sys.modules["transformers.utils.fx"] = m
import multiprocessing
MP="/workspace/real-gemma4-E4B-it"; TP=2
MAX=int(os.environ.get("KV_MAX","2048")); BUCKET=int(os.environ.get("KV_BUCKET","512")); MAXNEW=int(os.environ.get("MAXNEW","30"))

def _kv_rank_width(nkv): return nkv//TP if nkv%TP==0 else nkv

def main():
    import torch
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration, DynamicCache
    from neuronx_distributed.trace import parallel_model_load
    NEG=torch.finfo(torch.float32).min
    tok=AutoTokenizer.from_pretrained(MP)
    mm=Gemma4ForConditionalGeneration.from_pretrained(MP,torch_dtype=torch.float32,attn_implementation="eager"); mm.eval()
    lang=mm.model.language_model; cfg=lang.config; SW=cfg.sliding_window; head=mm.lm_head
    softcap=getattr(mm.config.text_config,"final_logit_softcapping",None)
    class GeluTanh(torch.nn.Module):
        def forward(s,x): return 0.5*x*(1.0+torch.tanh(0.7978845608028654*(x+0.044715*x*x*x)))
    for mod in lang.modules():
        if hasattr(mod,"act_fn"): mod.act_fn=GeluTanh()
    def _sc(lg): return softcap*torch.tanh(lg/softcap) if softcap else lg
    NONSHARED,LINFO=[],{}
    for i,lyr in enumerate(lang.layers[:cfg.num_hidden_layers]):
        a=lyr.self_attn
        if not a.is_kv_shared_layer:
            hd=a.head_dim; NONSHARED.append(i); LINFO[i]=(a.k_proj.out_features//hd,hd)
    NK=len(NONSHARED)
    ec=mm.generation_config.eos_token_id; EOS=set(ec) if isinstance(ec,(list,tuple)) else {ec}
    def embed_ids(idl):
        ids=torch.tensor([idl])
        with torch.no_grad(): ie=lang.embed_tokens(ids); ple=lang.get_per_layer_inputs(ids,ie)
        return ie,ple
    def host_pos(pos):
        ar=torch.arange(MAX); oh=(ar==pos).view(1,1,MAX,1).to(torch.float32); valid=ar<=pos
        fm=torch.where(valid,0.0,NEG).view(1,1,1,MAX); sm=torch.where(valid&(ar>pos-SW),0.0,NEG).view(1,1,1,MAX)
        return torch.tensor([[pos]],dtype=torch.long),oh,fm,sm
    enc=tok.apply_chat_template([{"role":"user","content":"What is the capital of France?"}],add_generation_prompt=True,return_tensors="pt",return_dict=True)
    prompt=enc["input_ids"][0].tolist(); n0=len(prompt); print("prompt",n0,"MAX",MAX,flush=True)

    # ---- CPU prefill (host, fp32) -> full KV + first token ----
    def cpu_prefill():
        ie,ple=embed_ids(prompt); c=DynamicCache()
        with torch.no_grad():
            out=lang(inputs_embeds=ie,per_layer_inputs=ple,attention_mask=torch.ones(1,n0,dtype=torch.long),use_cache=True,past_key_values=c)
            lg=_sc(head(out.last_hidden_state))
        ks=[c.layers[i].keys for i in NONSHARED]; vs=[c.layers[i].values for i in NONSHARED]  # each [1,nkv,n0,hd]
        return int(lg[0,n0-1].argmax()), ks, vs
    first, KS, VS = cpu_prefill()
    print("CPU prefill first token:", first, repr(tok.decode([first])), "KV heads:", [KS[j].shape[1] for j in range(3)], flush=True)

    # ---- CPU reference greedy (full KV static-cache decode) ----
    class SKV:
        is_compileable=False
        def __init__(s,kb,vb,oh): s.key={i:kb[j] for j,i in enumerate(NONSHARED)}; s.val={i:vb[j] for j,i in enumerate(NONSHARED)}; s.oh=oh
        def update(s,k,v,idx,*a,**kw): s.key[idx]=s.key[idx]*(1.0-s.oh)+k*s.oh; s.val[idx]=s.val[idx]*(1.0-s.oh)+v*s.oh; return s.key[idx],s.val[idx]
        def get_seq_length(s,*a,**k): return 0
    def cpu_ref():
        kb=[torch.zeros(1,LINFO[i][0],MAX,LINFO[i][1]) for i in NONSHARED]; vb=[torch.zeros(1,LINFO[i][0],MAX,LINFO[i][1]) for i in NONSHARED]
        for j in range(NK): kb[j][:,:,:n0,:]=KS[j][:,:,:n0,:]; vb[j][:,:,:n0,:]=VS[j][:,:,:n0,:]
        seq=[first]; cur=n0
        for _ in range(MAXNEW):
            if seq[-1] in EOS: break
            e1,p1=embed_ids([seq[-1]]); pi,oh,fm,sm=host_pos(cur); cache=SKV(kb,vb,oh)
            with torch.no_grad():
                out=lang(inputs_embeds=e1,per_layer_inputs=p1,position_ids=pi,attention_mask={"full_attention":fm,"sliding_attention":sm},use_cache=True,past_key_values=cache)
                lg=_sc(head(out.last_hidden_state))
            kb=[cache.key[i] for i in NONSHARED]; vb=[cache.val[i] for i in NONSHARED]
            seq.append(int(lg[0,0].argmax())); cur+=1
        return seq
    print("=== CPU ref greedy ===",flush=True); cpu_seq=cpu_ref()

    # ---- device decode ----
    t=time.time(); dec=parallel_model_load("/workspace/tpa_dec"); print("dec loaded",round(time.time()-t,1),flush=True)
    def state_params(r):
        sp=[(int(''.join(c for c in n if c.isdigit())),p) for n,p in dec.models[r].named_parameters() if n.startswith("states.")]
        return [p for _,p in sorted(sp,key=lambda x:x[0])]
    SP=[state_params(r) for r in range(TP)]
    print("per-rank states:",[len(s) for s in SP],"(expect",2*NK,")",flush=True)
    # seed: rank r gets kv head-slice [r*w:(r+1)*w]
    for r in range(TP):
        for j in range(NK):
            nkv,hd=LINFO[NONSHARED[j]]; w=_kv_rank_width(nkv)
            sl=slice(r*w,(r+1)*w) if nkv%TP==0 else slice(0,nkv)
            sk=torch.zeros(1,w,MAX,hd); sk[:,:,:n0,:]=KS[j][:,sl,:n0,:]
            sv=torch.zeros(1,w,MAX,hd); sv[:,:,:n0,:]=VS[j][:,sl,:n0,:]
            SP[r][j].data.copy_(sk); SP[r][NK+j].data.copy_(sv)
    def dev_greedy():
        seq=[first]; cur=n0; td=0.0; steps=0
        for _ in range(MAXNEW):
            if seq[-1] in EOS: break
            e1,p1=embed_ids([seq[-1]]); pi,oh,fm,sm=host_pos(cur); a0=time.time()
            r=dec(e1,p1,pi,oh,fm,sm); lg1=r[0] if isinstance(r,(tuple,list)) else r
            td+=time.time()-a0; seq.append(int(lg1[0,0].argmax())); cur+=1; steps+=1
        return seq,td,steps
    print("=== warmup (excluded from timing) ===",flush=True); _=dev_greedy()
    # re-seed after warmup consumed the KV buffers
    for r in range(TP):
        for j in range(NK):
            nkv,hd=LINFO[NONSHARED[j]]; w=_kv_rank_width(nkv)
            sl=slice(r*w,(r+1)*w) if nkv%TP==0 else slice(0,nkv)
            sk=torch.zeros(1,w,MAX,hd); sk[:,:,:n0,:]=KS[j][:,sl,:n0,:]
            sv=torch.zeros(1,w,MAX,hd); sv[:,:,:n0,:]=VS[j][:,sl,:n0,:]
            SP[r][j].data.copy_(sk); SP[r][NK+j].data.copy_(sv)
    print("=== DEVICE greedy (timed) ===",flush=True); dev_seq,secs,steps=dev_greedy()
    print("CPU GEN:",repr(tok.decode([x for x in cpu_seq if x not in EOS],skip_special_tokens=True)),flush=True)
    print("DEV GEN:",repr(tok.decode([x for x in dev_seq if x not in EOS],skip_special_tokens=True)),flush=True)
    print("SEQ_MATCH",cpu_seq==dev_seq,flush=True)
    print("TP+ALIAS DECODE tok/s:",round(steps/secs,1),"|",round(secs/max(steps,1)*1000),"ms/tok |",steps,"steps",flush=True)
    print("TPA_RUN_OK",flush=True)

if __name__ == "__main__":
    if os.environ.get("_TP_CHILD") != "1":
        os.environ["_TP_CHILD"]="1"; multiprocessing.set_start_method("spawn", force=True); main()
