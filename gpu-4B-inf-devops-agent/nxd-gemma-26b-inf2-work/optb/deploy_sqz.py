"""Slim 2-core (inf2.xlarge, 16GB host) deploy of the int8-squeeze 26B-A4B neff.

Loads ONLY the compiled neff (MB_LOAD) + the host embedding table — never the full model.
Structure (which layers are kv-shared, head dims) comes from a META-device model instantiation
(~0 host RAM). Host token embeddings come from a 1.48GB embed_tokens.pt (extracted separately).
This is what lets the 26B run on a 16GB-host box: compilation needs ~180GB, but deploying the
saved neff needs only the neff-load peak (covered by host swap) + the embedding table.

Env: NEFF=/data/sqz_neff.pt  MODEL_DIR=/data/cfg  EMBED=/data/embed_tokens.pt
     KV_MAX=256 KV_BUCKET=64  PROMPT="..."  (NEURON_RT_NUM_CORES=2)"""
import sys, os, types, time, json
_m=types.ModuleType("transformers.utils.fx"); _m.HFTracer=object; _m.symbolic_trace=None; sys.modules["transformers.utils.fx"]=_m
import torch
MP=os.environ.get("MODEL_DIR","/data/cfg")
NEFF=os.environ["NEFF"]; EMBED=os.environ.get("EMBED","/data/embed_tokens.pt")
MAX=int(os.environ.get("KV_MAX","256")); BUCKET=int(os.environ.get("KV_BUCKET","64"))
NEG=torch.finfo(torch.float32).min

from transformers import AutoConfig, AutoTokenizer, Gemma4ForConditionalGeneration
from accelerate import init_empty_weights
# register neuron.SPMDModel TorchScript class BEFORE torch.jit.load (else "Unknown type name neuron.SPMDModel")
from neuronx_distributed.trace.model_builder import ModelBuilder  # noqa: F401

cfg=AutoConfig.from_pretrained(MP)
cfg._attn_implementation="eager"
if hasattr(cfg,"text_config"): cfg.text_config._attn_implementation="eager"
print("instantiating meta model for structure ...",flush=True)
with init_empty_weights():
    mm=Gemma4ForConditionalGeneration(cfg)
lang=mm.model.language_model; lc=lang.config
NONSHARED=[]; LINFO={}
for i,lyr in enumerate(lang.layers[:lc.num_hidden_layers]):
    a=lyr.self_attn
    if not getattr(a,"is_kv_shared_layer",False):
        hd=a.head_dim; NONSHARED.append(i); LINFO[i]=(a.k_proj.out_features//hd, hd)
SW=lc.sliding_window; H=lc.hidden_size
print(f"structure: {len(NONSHARED)} non-shared, head_dims={sorted({LINFO[i][1] for i in NONSHARED})}, SW={SW}",flush=True)

# host embedding table (fp32 to match the trace's fp32 inputs_embeds; monotone softcap omitted -> greedy argmax unaffected)
# Gemma4 uses Gemma4TextScaledWordEmbedding: embeds are multiplied by embed_scale = hidden_size**0.5
# INSIDE the embedding (modeling_gemma4.py ~L1476), and the model forward does NOT re-normalize. The neff
# was traced on these SCALED embeds, so the host lookup MUST apply the same scale (a plain nn.Embedding
# without it makes embeds ~sqrt(H)x too small -> degenerate/repeated-token output).
EMBED_SCALE=float(H)**0.5
emb=torch.nn.Embedding(lc.vocab_size, H)
emb.weight.data=torch.load(EMBED).to(torch.float32)
del mm, lang  # drop the meta model
def _inputs(ids, positions):
    ids_t=torch.tensor([ids])
    with torch.no_grad(): ie=emb(ids_t)*EMBED_SCALE
    seq=len(positions); ar=torch.arange(MAX); pos=torch.tensor([positions],dtype=torch.long)
    oh=(ar.view(MAX,1)==pos.view(1,seq)).view(1,1,MAX,seq).to(torch.float32)
    q=pos.view(seq,1); mm2=ar.view(1,MAX)
    full=torch.where(mm2<=q,0.0,NEG).view(1,1,seq,MAX)
    slide=torch.where((mm2<=q)&(mm2>q-SW),0.0,NEG).view(1,1,seq,MAX)
    return ie,pos,oh,full,slide

gc=json.load(open(os.path.join(MP,"generation_config.json")))
ec=gc.get("eos_token_id"); EOS=set(ec) if isinstance(ec,(list,tuple)) else {ec}
tok=AutoTokenizer.from_pretrained(MP)
ct=os.path.join(MP,"chat_template.jinja")
if os.path.exists(ct): tok.chat_template=open(ct).read()
d=tok.apply_chat_template([{"role":"user","content":os.environ.get("PROMPT","What is the capital of France?")}],add_generation_prompt=True)
prompt=d if isinstance(d,list) else d["input_ids"]
n0=len(prompt); assert n0<=BUCKET, f"prompt {n0} > BUCKET {BUCKET}"

print("loading neff",NEFF,flush=True); t0=time.time()
model=torch.jit.load(NEFF)
model.nxd_model.initialize_with_saved_weights(torch.tensor([0],dtype=torch.int32))
print(f"NEFF_LOADED in {time.time()-t0:.0f}s",flush=True)

def dcall(a): r=model(*a); return r[0] if isinstance(r,(tuple,list)) else r
pad=prompt+[0]*(BUCKET-n0)
dcall(_inputs(pad,list(range(BUCKET))))  # warmup
t0=time.time(); lg=dcall(_inputs(pad,list(range(BUCKET)))); pf=time.time()-t0
first=int(lg[0,n0-1].argmax()); seq=[first]; cur=n0
td=time.time()
for _ in range(60):
    if seq[-1] in EOS: break
    l1=dcall(_inputs([seq[-1]],[cur])); seq.append(int(l1[0,0].argmax())); cur+=1
dt=time.time()-td; ntok=len(seq)
txt=tok.decode([x for x in seq if x not in EOS],skip_special_tokens=True)
print("DEV GEN:",repr(txt),flush=True)
print("DEVICE_PARIS","paris" in txt.lower(),flush=True)
print(f"PREFILL {pf*1000:.0f} ms | DECODE {ntok} tok @ {ntok/max(dt,1e-6):.1f} tok/s",flush=True)
print("DEPLOY_OK",flush=True)
