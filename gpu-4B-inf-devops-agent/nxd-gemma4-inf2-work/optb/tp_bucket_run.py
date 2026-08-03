"""Load the bucketed model (parallel_model_load) and run ON-DEVICE prefill (prefill bucket) +
device decode (decode bucket) sharing device-resident KV. Compare to CPU reference (SEQ_MATCH),
measure first-token (device-prefill) latency vs the ~1.5s CPU-prefill floor, and decode tok/s."""
import sys, os, types, time
m=types.ModuleType("transformers.utils.fx"); m.HFTracer=object; m.symbolic_trace=None; sys.modules["transformers.utils.fx"]=m
import multiprocessing
MP="/workspace/real-gemma4-E4B-it"; TP=2
MAX=int(os.environ.get("KV_MAX","256")); BUCKET=int(os.environ.get("KV_BUCKET","64")); MAXNEW=int(os.environ.get("MAXNEW","30"))
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
    class G(torch.nn.Module):
        def forward(s,x): return 0.5*x*(1.0+torch.tanh(0.7978845608028654*(x+0.044715*x*x*x)))
    for mod in lang.modules():
        if hasattr(mod,"act_fn"): mod.act_fn=G()
    def _sc(lg): return softcap*torch.tanh(lg/softcap) if softcap else lg
    NONSHARED,LINFO=[],{}
    for i,lyr in enumerate(lang.layers[:cfg.num_hidden_layers]):
        a=lyr.self_attn
        if not a.is_kv_shared_layer:
            hd=a.head_dim; NONSHARED.append(i); LINFO[i]=(a.k_proj.out_features//hd,hd)
    NK=len(NONSHARED); ec=mm.generation_config.eos_token_id; EOS=set(ec) if isinstance(ec,(list,tuple)) else {ec}
    def embed(ids):
        idt=torch.tensor([ids])
        with torch.no_grad(): ie=lang.embed_tokens(idt); ple=lang.get_per_layer_inputs(idt,ie)
        return ie,ple
    def dev_inputs(ids, positions):
        ie,ple=embed(ids); ie=ie.to(torch.bfloat16); ple=ple.to(torch.bfloat16); seq=len(positions); ar=torch.arange(MAX)
        pos=torch.tensor([positions],dtype=torch.long)
        oh=(ar.view(MAX,1)==pos.view(1,seq)).view(1,1,MAX,seq).to(torch.bfloat16)
        q=pos.view(seq,1); mm2=ar.view(1,MAX)
        full=torch.where(mm2<=q,0.0,NEG).view(1,1,seq,MAX).to(torch.bfloat16)
        slide=torch.where((mm2<=q)&(mm2>q-SW),0.0,NEG).view(1,1,seq,MAX).to(torch.bfloat16)
        return ie,ple,pos,oh,full,slide
    enc=tok.apply_chat_template([{"role":"user","content":"What is the capital of France?"}],add_generation_prompt=True,return_tensors="pt",return_dict=True)
    prompt=enc["input_ids"][0].tolist(); n0=len(prompt); print("prompt",n0,"MAX",MAX,"BUCKET",BUCKET,flush=True)

    # ---- CPU reference (fp32, DynamicCache prefill + static decode) ----
    def cpu_ref():
        ie,ple=embed(prompt); c=DynamicCache()
        with torch.no_grad():
            out=lang(inputs_embeds=ie,per_layer_inputs=ple,attention_mask=torch.ones(1,n0,dtype=torch.long),use_cache=True,past_key_values=c)
            lg=_sc(head(out.last_hidden_state))
        first=int(lg[0,n0-1].argmax()); KS=[c.layers[i].keys for i in NONSHARED]; VS=[c.layers[i].values for i in NONSHARED]
        class SKV:
            is_compileable=False
            def __init__(s,kb,vb,oh): s.key={i:kb[j] for j,i in enumerate(NONSHARED)}; s.val={i:vb[j] for j,i in enumerate(NONSHARED)}; s.oh=oh
            def update(s,k,v,idx,*a,**kw): s.key[idx]=s.key[idx]*(1.0-s.oh)+k*s.oh; s.val[idx]=s.val[idx]*(1.0-s.oh)+v*s.oh; return s.key[idx],s.val[idx]
            def get_seq_length(s,*a,**k): return 0
        def hp(pos):
            ar=torch.arange(MAX); oh=(ar==pos).view(1,1,MAX,1).to(torch.float32); valid=ar<=pos
            fm=torch.where(valid,0.0,NEG).view(1,1,1,MAX); sm=torch.where(valid&(ar>pos-SW),0.0,NEG).view(1,1,1,MAX)
            return torch.tensor([[pos]],dtype=torch.long),oh,fm,sm
        kb=[torch.zeros(1,LINFO[i][0],MAX,LINFO[i][1]) for i in NONSHARED]; vb=[torch.zeros(1,LINFO[i][0],MAX,LINFO[i][1]) for i in NONSHARED]
        for j in range(NK): kb[j][:,:,:n0,:]=KS[j][:,:,:n0,:]; vb[j][:,:,:n0,:]=VS[j][:,:,:n0,:]
        seq=[first]; cur=n0
        for _ in range(MAXNEW):
            if seq[-1] in EOS: break
            e1,p1=embed([seq[-1]]); pi,oh,fm,sm=hp(cur); cache=SKV(kb,vb,oh)
            with torch.no_grad():
                out=lang(inputs_embeds=e1,per_layer_inputs=p1,position_ids=pi,attention_mask={"full_attention":fm,"sliding_attention":sm},use_cache=True,past_key_values=cache)
                lg=_sc(head(out.last_hidden_state))
            kb=[cache.key[i] for i in NONSHARED]; vb=[cache.val[i] for i in NONSHARED]
            seq.append(int(lg[0,0].argmax())); cur+=1
        return seq
    print("=== CPU ref ===",flush=True); cpu_seq=cpu_ref()

    # ---- device: prefill bucket (on-device) then decode bucket ----
    t=time.time(); model=parallel_model_load("/workspace/tpb_dec"); print("loaded",round(time.time()-t,1),flush=True)
    def call(ie,ple,pos,oh,fm,sm):
        r=model(ie,ple,pos,oh,fm,sm); return r[0] if isinstance(r,(tuple,list)) else r
    pad=prompt+[0]*(BUCKET-n0)
    # warmup (loads neffs onto cores) then timed
    call(*dev_inputs(pad,list(range(BUCKET))))
    def gen():
        tpf=time.time(); lg=call(*dev_inputs(pad,list(range(BUCKET)))); pf=time.time()-tpf   # DEVICE prefill
        print("DEV prefill logits shape:", tuple(lg.shape), "dtype", lg.dtype, flush=True)
        first=int(lg[0,n0-1].argmax())
        top=torch.topk(lg[0,n0-1].float(),5); print("DEV first tok:",first,repr(tok.decode([first])),"top5:",top.indices.tolist(),flush=True)
        print("CPU first tok: 818 'The' (expected)",flush=True)
        seq=[first]; cur=n0; td=0.0; steps=0
        for _ in range(MAXNEW):
            if seq[-1] in EOS: break
            a0=time.time(); l1=call(*dev_inputs([seq[-1]],[cur])); td+=time.time()-a0
            seq.append(int(l1[0,0].argmax())); cur+=1; steps+=1
        return seq,pf,td,steps
    print("=== DEVICE (bucketed) ===",flush=True); dev_seq,pf,td,steps=gen()
    print("CPU GEN:",repr(tok.decode([x for x in cpu_seq if x not in EOS],skip_special_tokens=True)),flush=True)
    print("DEV GEN:",repr(tok.decode([x for x in dev_seq if x not in EOS],skip_special_tokens=True)),flush=True)
    print("SEQ_MATCH",cpu_seq==dev_seq,flush=True)
    print(f"DEVICE PREFILL (first token): {pf*1000:.0f} ms   [vs ~1400-1600ms CPU prefill]",flush=True)
    print(f"DECODE tok/s: {steps/td:.1f} | {td/max(steps,1)*1000:.0f} ms/tok | {steps} steps",flush=True)
    print("TPB_RUN_OK",flush=True)

if __name__=="__main__":
    if os.environ.get("_TP_CHILD")!="1":
        os.environ["_TP_CHILD"]="1"; multiprocessing.set_start_method("spawn",force=True); main()
