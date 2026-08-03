"""Isolation test: seed the bucketed model's device KV from a CPU fp32 prefill, then call the
DECODE bucket alone. If device 2nd token == CPU 2nd token -> decode bucket + bf16 weights are FINE
(so the prefill bucket/scatter is the bug). If garbage -> bf16 weights are the culprit."""
import sys, os, types, time
m=types.ModuleType("transformers.utils.fx"); m.HFTracer=object; m.symbolic_trace=None; sys.modules["transformers.utils.fx"]=m
import multiprocessing
MP="/workspace/real-gemma4-E4B-it"; TP=2
MAX=int(os.environ.get("KV_MAX","256")); BUCKET=int(os.environ.get("KV_BUCKET","64"))
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
    NK=len(NONSHARED)
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
    prompt=enc["input_ids"][0].tolist(); n0=len(prompt)

    # ---- CPU prefill fp32 -> first token + full KV, and CPU 2nd token (oracle) ----
    ie,ple=embed(prompt); c=DynamicCache()
    with torch.no_grad():
        out=lang(inputs_embeds=ie,per_layer_inputs=ple,attention_mask=torch.ones(1,n0,dtype=torch.long),use_cache=True,past_key_values=c)
        first=int(_sc(head(out.last_hidden_state))[0,n0-1].argmax())
    KS=[c.layers[i].keys for i in NONSHARED]; VS=[c.layers[i].values for i in NONSHARED]
    # CPU 2nd token
    class SKV:
        is_compileable=False
        def __init__(s,kb,vb,oh): s.key={i:kb[j] for j,i in enumerate(NONSHARED)}; s.val={i:vb[j] for j,i in enumerate(NONSHARED)}; s.oh=oh
        def update(s,k,v,idx,*a,**kw): s.key[idx]=s.key[idx]*(1.0-s.oh)+k*s.oh; s.val[idx]=s.val[idx]*(1.0-s.oh)+v*s.oh; return s.key[idx],s.val[idx]
        def get_seq_length(s,*a,**k): return 0
    ar=torch.arange(MAX); oh=(ar==n0).view(1,1,MAX,1).float(); valid=ar<=n0
    fm=torch.where(valid,0.0,NEG).view(1,1,1,MAX); sm=torch.where(valid&(ar>n0-SW),0.0,NEG).view(1,1,1,MAX)
    kb=[torch.zeros(1,LINFO[i][0],MAX,LINFO[i][1]) for i in NONSHARED]; vb=[torch.zeros(1,LINFO[i][0],MAX,LINFO[i][1]) for i in NONSHARED]
    for j in range(NK): kb[j][:,:,:n0,:]=KS[j][:,:,:n0,:]; vb[j][:,:,:n0,:]=VS[j][:,:,:n0,:]
    e1,p1=embed([first]); cache=SKV(kb,vb,oh)
    with torch.no_grad():
        out=lang(inputs_embeds=e1,per_layer_inputs=p1,position_ids=torch.tensor([[n0]]),attention_mask={"full_attention":fm,"sliding_attention":sm},use_cache=True,past_key_values=cache)
        cpu2=int(_sc(head(out.last_hidden_state))[0,0].argmax())
    print(f"CPU: first={first} {tok.decode([first])!r}  second={cpu2} {tok.decode([cpu2])!r}",flush=True)

    # ---- load bucketed model, warmup, then seed device KV (sliced per rank) + call DECODE bucket ----
    model=parallel_model_load("/workspace/tpb_dec")
    def call(a): return model(*a)[0]
    call(dev_inputs([first],[n0]))   # warmup: triggers _load + moves states to device
    def state_params(r):
        sp=[(int(''.join(ch for ch in n if ch.isdigit())),p) for n,p in model.models[r].named_parameters() if n.startswith("states.")]
        return [p for _,p in sorted(sp,key=lambda x:x[0])]
    SP=[state_params(r) for r in range(TP)]
    print("per-rank device states:",[len(s) for s in SP],"(expect",2*NK,")",flush=True)
    for r in range(TP):
        for j in range(NK):
            nkv,hd=LINFO[NONSHARED[j]]; w=_kv_rank_width(nkv); sl=slice(r*w,(r+1)*w) if nkv%TP==0 else slice(0,nkv)
            sk=torch.zeros(1,w,MAX,hd,dtype=torch.bfloat16); sk[:,:,:n0,:]=KS[j][:,sl,:n0,:].to(torch.bfloat16)
            sv=torch.zeros(1,w,MAX,hd,dtype=torch.bfloat16); sv[:,:,:n0,:]=VS[j][:,sl,:n0,:].to(torch.bfloat16)
            SP[r][j].data.copy_(sk); SP[r][NK+j].data.copy_(sv)
    lg=call(dev_inputs([first],[n0]))     # DECODE bucket, seeded KV
    dev2=int(lg[0,0].float().argmax())
    top=torch.topk(lg[0,0].float(),5)
    print(f"DEVICE decode 2nd token: {dev2} {tok.decode([dev2])!r}  top5:{top.indices.tolist()}",flush=True)
    print("DECODE_BUCKET_MATCH", dev2==cpu2, flush=True)
    print("ISO_DONE",flush=True)

if __name__=="__main__":
    if os.environ.get("_TP_CHILD")!="1":
        os.environ["_TP_CHILD"]="1"; multiprocessing.set_start_method("spawn",force=True); main()
