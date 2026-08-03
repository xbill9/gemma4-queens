"""TP=2 + bucketing: ONE weight-sharing model with prefill (seq=BUCKET) + decode (seq=1) buckets,
sharing device-resident aliased KV. Prefill runs ON DEVICE (no host CPU prefill). Unified forward:
write K/V into the MAX buffer via a one-hot scatter, then attention is [seq x MAX]; prefill = seq=BUCKET,
decode = seq=1, same graph two shapes. Saves the bucketed model via parallel_model_save."""
import sys, os, types, time
m = types.ModuleType("transformers.utils.fx"); m.HFTracer=object; m.symbolic_trace=None
sys.modules["transformers.utils.fx"] = m
import multiprocessing
MP="/workspace/real-gemma4-E4B-it"
TP=2
MAX=int(os.environ.get("KV_MAX","256"))
BUCKET=int(os.environ.get("KV_BUCKET","64"))
CARGS=["--model-type","transformer","--auto-cast","all","--auto-cast-type","bf16"]

def _kv_rank_width(nkv): return nkv//TP if nkv%TP==0 else nkv

def _build_shared():
    import torch
    from transformers import Gemma4ForConditionalGeneration
    mm=Gemma4ForConditionalGeneration.from_pretrained(MP,torch_dtype=torch.bfloat16,attn_implementation="eager"); mm.eval()
    lang=mm.model.language_model; cfg=lang.config
    class GeluTanh(torch.nn.Module):
        def forward(s,x): return 0.5*x*(1.0+torch.tanh(0.7978845608028654*(x+0.044715*x*x*x)))
    for mod in lang.modules():
        if hasattr(mod,"act_fn"): mod.act_fn=GeluTanh()
    NONSHARED,LINFO=[],{}
    for i,lyr in enumerate(lang.layers[:cfg.num_hidden_layers]):
        a=lyr.self_attn
        if not a.is_kv_shared_layer:
            hd=a.head_dim; NONSHARED.append(i); LINFO[i]=(a.k_proj.out_features//hd,hd)
    softcap=getattr(mm.config.text_config,"final_logit_softcapping",None)
    return mm,lang,mm.lm_head,softcap,NONSHARED,LINFO,cfg.sliding_window

def _shard(lang,nlayers):
    import torch
    from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
    from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_rank
    rank=get_tensor_model_parallel_rank()
    def col(o):
        out,inf=o.weight.shape; l=ColumnParallelLinear(inf,out,bias=False,gather_output=False); c=out//TP
        l.weight.data.copy_(o.weight.data[rank*c:(rank+1)*c,:]); l.weight.data=l.weight.data.to(torch.bfloat16); return l
    def row(o):
        out,inf=o.weight.shape; l=RowParallelLinear(inf,out,bias=False,input_is_parallel=True); c=inf//TP
        l.weight.data.copy_(o.weight.data[:,rank*c:(rank+1)*c]); l.weight.data=l.weight.data.to(torch.bfloat16); return l
    for lyr in lang.layers[:nlayers]:
        a=lyr.self_attn; hd=a.head_dim
        a.q_proj=col(a.q_proj); a.o_proj=row(a.o_proj)
        if getattr(a,"k_proj",None) is not None:
            nkv=a.k_proj.out_features//hd
            if nkv%TP==0:
                a.k_proj=col(a.k_proj)
                if getattr(a,"v_proj",None) is not None: a.v_proj=col(a.v_proj)
            else:
                a.num_key_value_groups=(a.q_proj.out_features//hd)//nkv
        mp=lyr.mlp; mp.gate_proj=col(mp.gate_proj); mp.up_proj=col(mp.up_proj); mp.down_proj=row(mp.down_proj)

def _sc(softcap,lg):
    import torch
    return softcap*torch.tanh(lg/softcap) if softcap else lg

def get_model():
    """Unified prefill/decode wrapper. KV = device-resident aliased Parameters."""
    import torch
    mm,lang,lm_head,softcap,NONSHARED,LINFO,SW=_build_shared()
    _shard(lang,lang.config.num_hidden_layers); NK=len(NONSHARED)
    W={i:_kv_rank_width(LINFO[i][0]) for i in NONSHARED}
    class ScatterKV:
        """Write k/v into a fixed MAX buffer at positions given by onehot[1,1,MAX,seq]."""
        is_compileable=False
        def __init__(s,kb,vb,oh): s.key={i:kb[j] for j,i in enumerate(NONSHARED)}; s.val={i:vb[j] for j,i in enumerate(NONSHARED)}; s.oh=oh
        def _wr(s,buf,x):
            # buf:[1,H,MAX,hd]  x:[1,H,seq,hd]  oh:[1,1,MAX,seq] -> scatter x rows into buf by onehot
            oh=s.oh[0,0]                                        # [MAX,seq]
            scat=torch.einsum('bhsd,ms->bhmd', x, oh)          # [1,H,MAX,hd]
            row=oh.sum(-1).view(1,1,MAX,1)                      # [1,1,MAX,1] rows written
            return buf*(1.0-row)+scat
        def update(s,k,v,idx,*a,**kw):
            s.key[idx]=s._wr(s.key[idx],k); s.val[idx]=s._wr(s.val[idx],v); return s.key[idx],s.val[idx]
        def get_seq_length(s,*a,**k): return 0
        def export(s): return [s.key[i] for i in NONSHARED],[s.val[i] for i in NONSHARED]
    class Wrap(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.lang=lang; s.head=lm_head
            s.kbuf=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1,W[i],MAX,LINFO[i][1],dtype=torch.bfloat16),requires_grad=False) for i in NONSHARED])
            s.vbuf=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1,W[i],MAX,LINFO[i][1],dtype=torch.bfloat16),requires_grad=False) for i in NONSHARED])
        def forward(s, ie, ple, position_ids, onehot, full_mask, slide_mask):
            cache=ScatterKV(list(s.kbuf),list(s.vbuf),onehot)
            out=s.lang(inputs_embeds=ie,per_layer_inputs=ple,position_ids=position_ids,
                       attention_mask={"full_attention":full_mask,"sliding_attention":slide_mask},
                       use_cache=True,past_key_values=cache)
            lg=_sc(softcap,s.head(out.last_hidden_state)); ks,vs=cache.export(); return (lg,ks,vs)
    w=Wrap().eval()
    aliases={}
    for j in range(NK): aliases[w.kbuf[j]]=1+j
    for j in range(NK): aliases[w.vbuf[j]]=1+NK+j
    return w, aliases

def kernel_factory():
    import torch
    def kernel(inputs: list[torch.Tensor]):
        # route by query seq length (ie is inputs[0], shape [1,seq,H])
        idx=torch.zeros(1,dtype=torch.long) if inputs[0].size(1)>1 else torch.ones(1,dtype=torch.long)
        return inputs, idx
    return torch.jit.script(kernel)

def _inputs(lang, ids, SW, NEG, seq_positions):
    """Build (ie,ple,position_ids,onehot,full_mask,slide_mask) for a set of query positions."""
    import torch
    ids_t=torch.tensor([ids])
    with torch.no_grad(): ie=lang.embed_tokens(ids_t); ple=lang.get_per_layer_inputs(ids_t,ie)
    seq=len(seq_positions); ar=torch.arange(MAX)
    pos=torch.tensor([seq_positions],dtype=torch.long)                       # [1,seq]
    onehot=(ar.view(MAX,1)==pos.view(1,seq)).view(1,1,MAX,seq).to(torch.bfloat16)   # [1,1,MAX,seq]
    q=pos.view(seq,1); m=ar.view(1,MAX)
    full=torch.where(m<=q,0.0,NEG).view(1,1,seq,MAX).to(torch.bfloat16)                          # causal query->buffer
    slide=torch.where((m<=q)&(m>q-SW),0.0,NEG).view(1,1,seq,MAX).to(torch.bfloat16)
    return ie,ple,pos,onehot,full,slide

def main():
    import torch, neuronx_distributed
    from transformers import AutoTokenizer
    from torch_neuronx import BucketModelConfig
    from neuronx_distributed.trace import parallel_model_save
    NEG=torch.finfo(torch.bfloat16).min
    tok=AutoTokenizer.from_pretrained(MP)
    _mm,rlang,_h,_s,NONSHARED,LINFO,SW=_build_shared(); NK=len(NONSHARED)
    enc=tok.apply_chat_template([{"role":"user","content":"What is the capital of France?"}],add_generation_prompt=True,return_tensors="pt",return_dict=True)
    prompt=enc["input_ids"][0].tolist(); n0=len(prompt); assert n0<=BUCKET
    # bucket 0 = prefill (BUCKET query positions 0..BUCKET-1); bucket 1 = decode (1 query at pos n0)
    pre_ids=prompt+[0]*(BUCKET-n0)
    pre_in=_inputs(rlang, pre_ids, SW, NEG, list(range(BUCKET)))
    dec_in=_inputs(rlang, [prompt[-1]], SW, NEG, [n0])
    state0=([torch.zeros(1,_kv_rank_width(LINFO[i][0]),MAX,LINFO[i][1],dtype=torch.bfloat16) for i in NONSHARED]
           +[torch.zeros(1,_kv_rank_width(LINFO[i][0]),MAX,LINFO[i][1],dtype=torch.bfloat16) for i in NONSHARED])  # kbuf..+vbuf.. (distinct)
    bc=BucketModelConfig(kernel_factory, shared_state_buffer=state0)
    print("tracing bucketed prefill+decode ...",flush=True)
    model=neuronx_distributed.trace.parallel_model_trace(
        get_model, [pre_in, dec_in], bucket_config=bc, tp_degree=TP,
        inline_weights_to_neff=False, compiler_args=CARGS)
    parallel_model_save(model,"/workspace/tpb_dec"); print("BUCKET_SAVED",flush=True)
    print("TPB_TRACE_OK",flush=True)

if __name__=="__main__":
    if os.environ.get("_TP_CHILD")!="1":
        os.environ["_TP_CHILD"]="1"; multiprocessing.set_start_method("spawn",force=True); main()
