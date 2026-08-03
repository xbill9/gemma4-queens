"""Gemma-4 26B-A4B (MoE int8) TP=8 HTTP server — NxD ModelBuilder edition.

Loads the single ModelBuilder-traced model (torch.jit) that holds BOTH prefill and decode buckets
plus on-device aliased KV, and serves it. Prefill and decode both run ON THE DEVICE (the ModelBuilder
graph dispatches by input shape); the host only computes word embeddings (no PLE on 31B) and the
attention masks. This replaces the earlier tp_alias design (host-CPU prefill + per-rank aliased-KV
seeding) — see git history.

Load path (the two non-obvious steps):
  model = torch.jit.load(MB_LOAD)
  model.nxd_model.initialize_with_saved_weights(torch.tensor([0], dtype=torch.int32))  # push weights to cores
Chat prompt: relies on chat_template.jinja living in MODEL_DIR (Gemma-4's turn tokens are <|turn>=105 /
<turn|>=106, and the template ships as a separate file — bundle it with the weights so AutoTokenizer
auto-loads it). Falls back to fetching it from HF, then to a hardcoded template.

Full serving layer kept from the E4B/tp_alias server: sampling (temperature/top_k/top_p), SSE
streaming, /metrics, spot-drain + bounded queue + per-request timeout + graceful SIGTERM.
Endpoints: /generate /v1/chat/completions /v1/completions /v1/models /health /metrics.
Env: MODEL_DIR, MB_LOAD, TP_DEGREE (8), KV_MAX (256), KV_BUCKET (64), PORT (8080), MODEL_NAME."""
import sys, os, types, time, json, threading, signal
import urllib.request as _urlreq
m = types.ModuleType("transformers.utils.fx"); m.HFTracer=object; m.symbolic_trace=None
sys.modules["transformers.utils.fx"] = m
import multiprocessing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
MP=os.environ.get("MODEL_DIR","/data/real-gemma4-26B-A4B-it"); TP=int(os.environ.get("TP_DEGREE","8"))
MAX=int(os.environ.get("KV_MAX","256")); BUCKET=int(os.environ.get("KV_BUCKET","64"))
PORT=int(os.environ.get("PORT","8080")); MODEL_NAME=os.environ.get("MODEL_NAME","gemma-4-26B-A4B-it-int8")
MB_LOAD=os.environ.get("MB_LOAD","/data/mb_26b_int8_tp8.pt")
HF_REPO=os.environ.get("HF_REPO","google/gemma-4-26B-A4B-it-int8")
NEG_INF=float("-inf")
MAX_QUEUE=int(os.environ.get("MAX_QUEUE","8"))          # max concurrent+queued requests -> 429
GEN_TIMEOUT=float(os.environ.get("GEN_TIMEOUT","120"))  # per-request wall-clock cap (s)
GRACE_SECONDS=float(os.environ.get("GRACE_SECONDS","25"))
LOCK=threading.Lock(); READY=threading.Event()

# ---- metrics (Prometheus text, no deps) ----
_START=time.time(); _MLOCK=threading.Lock(); _counter=[0]
_METRICS={"requests_total":0,"errors_total":0,"timeouts_total":0,"prompt_tokens_total":0,
          "completion_tokens_total":0,"generation_seconds_total":0.0,"last_tok_per_s":0.0}
def _bump(reqs=0,ptoks=0,ctoks=0,secs=0.0,err=0,to=0,tps=None):
    with _MLOCK:
        _METRICS["requests_total"]+=reqs; _METRICS["prompt_tokens_total"]+=ptoks
        _METRICS["completion_tokens_total"]+=ctoks; _METRICS["generation_seconds_total"]+=secs
        _METRICS["errors_total"]+=err; _METRICS["timeouts_total"]+=to
        if tps is not None: _METRICS["last_tok_per_s"]=tps
def _cur_rss_bytes():
    try:
        with open("/proc/self/statm") as f: return int(f.read().split()[1])*os.sysconf("SC_PAGE_SIZE")
    except Exception: return 0
def _render_metrics():
    with _MLOCK: mm=dict(_METRICS)
    up=time.time()-_START
    avg=(mm["completion_tokens_total"]/mm["generation_seconds_total"]) if mm["generation_seconds_total"]>0 else 0.0
    def g(n,h,t,v): return f"# HELP {n} {h}\n# TYPE {n} {t}\n{n} {v}"
    return "\n".join([
        g("gemma_up","1 if ready","gauge",1 if READY.is_set() else 0),
        g("gemma_uptime_seconds","seconds since start","gauge",f"{up:.1f}"),
        g("gemma_requests_total","total requests","counter",mm["requests_total"]),
        g("gemma_errors_total","total errors","counter",mm["errors_total"]),
        g("gemma_timeouts_total","generations cut by wall-clock cap","counter",mm["timeouts_total"]),
        g("gemma_gen_timeout_seconds","per-request wall-clock cap","gauge",GEN_TIMEOUT),
        g("gemma_prompt_tokens_total","total prompt tokens","counter",mm["prompt_tokens_total"]),
        g("gemma_completion_tokens_total","total generated tokens","counter",mm["completion_tokens_total"]),
        g("gemma_generation_seconds_total","total generation wall seconds","counter",f"{mm['generation_seconds_total']:.3f}"),
        g("gemma_tokens_per_second_last","tok/s of most recent request","gauge",f"{mm['last_tok_per_s']:.2f}"),
        g("gemma_tokens_per_second_avg","avg tok/s over all requests","gauge",f"{avg:.2f}"),
        g("gemma_max_total_tokens","configured KV_MAX","gauge",MAX),
        g("gemma_max_prompt_tokens","configured KV_BUCKET","gauge",BUCKET),
        g("gemma_draining","1 if draining on spot interruption","gauge",1 if _DRAINING.is_set() else 0),
        g("gemma_inflight_requests","current in-flight+queued requests","gauge",_INFLIGHT[0]),
        g("gemma_max_queue","max concurrent+queued before 429","gauge",MAX_QUEUE),
        g("process_resident_memory_bytes","resident set size","gauge",_cur_rss_bytes()),
    ])+"\n"

# ---- spot-interruption drain + bounded queue ----
_DRAINING=threading.Event(); _INFLIGHT=[0]; _QLOCK=threading.Lock()
class _Full(Exception): pass
def _slot_acquire():
    with _QLOCK:
        if _INFLIGHT[0]>=MAX_QUEUE: raise _Full()
        _INFLIGHT[0]+=1
def _slot_release():
    with _QLOCK: _INFLIGHT[0]=max(0,_INFLIGHT[0]-1)
def _watch_spot():
    base="http://169.254.169.254"
    while not _DRAINING.is_set():
        tokn=None
        try:
            tokn=_urlreq.urlopen(_urlreq.Request(base+"/latest/api/token",method="PUT",
                 headers={"X-aws-ec2-metadata-token-ttl-seconds":"60"}),timeout=2).read().decode()
        except Exception: pass
        try:
            req=_urlreq.Request(base+"/latest/meta-data/spot/instance-action")
            if tokn: req.add_header("X-aws-ec2-metadata-token",tokn)
            if _urlreq.urlopen(req,timeout=2).status==200:
                _DRAINING.set(); print("SPOT INTERRUPTION NOTICE — draining",flush=True); return
        except Exception: pass
        time.sleep(5)

def _ensure_chat_template(tok):
    """Gemma-4 ships its chat template as a separate chat_template.jinja. Prefer the one bundled in
    MODEL_DIR (AutoTokenizer auto-loads it); else fetch from HF; else a minimal hardcoded template."""
    if getattr(tok,"chat_template",None): return
    p=os.path.join(MP,"chat_template.jinja")
    if os.path.exists(p): tok.chat_template=open(p).read(); return
    try:
        import subprocess,json as _j
        from huggingface_hub import hf_hub_download
        raw=subprocess.check_output(["aws","secretsmanager","get-secret-value","--region",
            os.environ.get("HF_SECRET_REGION","us-east-1"),"--secret-id",
            os.environ.get("HF_SECRET_ID","hf_token"),"--query","SecretString","--output","text"]).decode().strip()
        try: tv=_j.loads(raw); tv=tv.get("hf_token") or list(tv.values())[0]
        except Exception: tv=raw
        tok.chat_template=open(hf_hub_download(HF_REPO,"chat_template.jinja",token=tv)).read()
    except Exception as e:
        print("chat_template fetch failed, using minimal fallback:",e,flush=True)
        tok.chat_template=("{{ bos_token }}{% for m in messages %}{{ '<|turn>' + m['role'] + '\n' + "
                           "m['content'] + '<turn|>\n' }}{% endfor %}{% if add_generation_prompt %}"
                           "{{ '<|turn>model\n' }}{% endif %}")

def boot():
    global torch, tok, rlang, head, softcap, NONSHARED, LINFO, SW, EOS, model, NEG, _mm, tp_mb
    import torch, tp_mb_moe_int8 as tp_mb
    # importing model_builder registers the custom TorchScript classes (torch.classes.neuron.SPMDModel);
    # without it torch.jit.load raises "Unknown type name '__torch__.torch.classes.neuron.SPMDModel'".
    from neuronx_distributed.trace.model_builder import ModelBuilder  # noqa: F401
    from transformers import AutoTokenizer
    NEG=torch.finfo(torch.float32).min
    torch.set_num_threads(int(os.environ.get("PREFILL_THREADS", os.cpu_count() or 16)))
    tok=AutoTokenizer.from_pretrained(MP); _ensure_chat_template(tok)
    # _discover loads the host fp32 reference model — we use only rlang.embed_tokens (word embeddings,
    # no PLE) + LINFO/SW. The neffs carry the transformer weights on-device.
    _mm,rlang,head,softcap,NONSHARED,LINFO,SW=tp_mb._discover()
    ec=_mm.generation_config.eos_token_id; EOS=set(ec) if isinstance(ec,(list,tuple)) else {ec}
    t=time.time()
    model=torch.jit.load(MB_LOAD)
    model.nxd_model.initialize_with_saved_weights(torch.tensor([0],dtype=torch.int32))
    try:
        warm,_=_prep_chat([{"role":"user","content":"Hi"}],None)
        generate_ids(warm,3,0.0,0,1.0,set())
    except Exception as e: print("warmup:",e,flush=True)
    print(f"READY in {round(time.time()-t,1)}s — ModelBuilder TP={TP}, MAX={MAX} BUCKET={BUCKET}",flush=True); READY.set()

def _sc(lg): return (softcap*torch.tanh(lg/softcap)) if softcap else lg
def _dcall(a):
    r=model(*a); return r[0] if isinstance(r,(tuple,list)) else r

def pick(logits, temperature, top_k, top_p):
    if temperature is None or temperature<=0.0: return int(torch.argmax(logits))
    logits=logits.float()/float(temperature)
    if top_k and top_k>0:
        k=min(int(top_k),logits.numel()); kth=torch.topk(logits,k).values[-1]
        logits=torch.where(logits<kth,torch.full_like(logits,NEG_INF),logits)
    probs=torch.softmax(logits,dim=-1)
    if top_p and 0.0<top_p<1.0:
        sp,si=torch.sort(probs,descending=True); cum=torch.cumsum(sp,dim=-1)
        keep=cum-sp<=top_p; sp=torch.where(keep,sp,torch.zeros_like(sp))
        probs=torch.zeros_like(probs).scatter(0,si,sp)
    total=probs.sum()
    if total<=0: return int(torch.argmax(logits))
    return int(torch.multinomial(probs/total,1))

LAST_PREFILL_S=0.0
def _generate(prompt_ids, max_new, temperature, top_k, top_p, stop_ids, timeout_s=None):
    """Device prefill (bucketed) + device decode loop. On-device aliased KV persists between calls;
    each request re-prefills positions 0..BUCKET so stale KV from a prior request is overwritten/masked."""
    global LAST_PREFILL_S
    prompt=prompt_ids[:BUCKET]; n0=len(prompt)
    if n0>BUCKET: raise ValueError(f"prompt {n0} tokens > BUCKET {BUCKET}")
    pad=prompt+[0]*(BUCKET-n0); t0=time.time()
    lg=_sc(_dcall(tp_mb._inputs(rlang,pad,SW,NEG,list(range(BUCKET)))))   # device prefill
    LAST_PREFILL_S=time.time()-t0
    nxt=pick(lg[0,n0-1],temperature,top_k,top_p)
    cur=n0; cap=min(max_new,MAX-n0-1); deadline=time.time()+(timeout_s or GEN_TIMEOUT); steps=0
    while True:
        if nxt in EOS or nxt in stop_ids: return n0,"stop"
        yield nxt
        if steps>=cap: return n0,"length"
        if time.time()>deadline: return n0,"timeout"
        l1=_sc(_dcall(tp_mb._inputs(rlang,[nxt],SW,NEG,[cur])))          # device decode (1 token)
        nxt=pick(l1[0,0],temperature,top_k,top_p); cur+=1; steps+=1

def generate_ids(prompt_ids, max_new, temperature, top_k, top_p, stop_ids, timeout_s=None):
    g=_generate(prompt_ids,max_new,temperature,top_k,top_p,stop_ids,timeout_s)
    ids,n0,finish=[],len(prompt_ids),"length"
    try:
        while True: ids.append(next(g))
    except StopIteration as e: n0,finish=e.value
    return ids,n0,finish

def _prep_chat(messages, stop):
    d=tok.apply_chat_template(messages,add_generation_prompt=True)
    prompt_ids=d if isinstance(d,list) else d["input_ids"]
    stop_ids=set()
    if isinstance(stop,str): stop=[stop]
    for s in (stop or []):
        try: stop_ids.update(tok.encode(s,add_special_tokens=False))
        except Exception: pass
    return prompt_ids,stop_ids

def run_chat(messages, max_new, temperature, top_k, top_p, stop, timeout_s=None):
    prompt_ids,stop_ids=_prep_chat(messages,stop); t_req=time.time()
    with LOCK: gen,n0,finish=generate_ids(prompt_ids,max_new,temperature,top_k,top_p,stop_ids,timeout_s)
    text=tok.decode(gen,skip_special_tokens=True); dt=time.time()-t_req; ct=len(gen)
    _bump(reqs=1,ptoks=n0,ctoks=ct,secs=dt,to=(1 if finish=="timeout" else 0),tps=(ct/dt if dt>0 else 0.0))
    dec_s=max(dt-LAST_PREFILL_S,1e-6)
    print(f"[req] pt={n0} ct={ct} prefill={round(LAST_PREFILL_S,2)}s decode={round(ct/dec_s,1)}tok/s e2e={round(ct/dt,1) if dt>0 else 0}tok/s {round(dt,2)}s finish={finish}",flush=True)
    return text,n0,ct,finish

class H(BaseHTTPRequestHandler):
    def log_message(s,*a): pass
    def _send(s,c,o): b=json.dumps(o).encode(); s.send_response(c); s.send_header("Content-Type","application/json"); s.send_header("Content-Length",str(len(b))); s.end_headers(); s.wfile.write(b)
    def _send_text(s,c,t): b=t.encode(); s.send_response(c); s.send_header("Content-Type","text/plain; version=0.0.4"); s.send_header("Content-Length",str(len(b))); s.end_headers(); s.wfile.write(b)
    def _body(s): n=int(s.headers.get("Content-Length",0)); return json.loads(s.rfile.read(n) or b"{}")
    def do_GET(s):
        p=s.path.rstrip("/")
        if p=="/v1/models": s._send(200,{"object":"list","data":[{"id":MODEL_NAME,"object":"model","owned_by":"local-inferentia2"}]})
        elif p in ("/health","/ping"):
            if _DRAINING.is_set(): s._send(503,{"status":"draining"})
            elif not READY.is_set(): s._send(503,{"status":"loading"})
            else: s._send(200,{"status":"ok"})
        elif p=="/metrics": s._send_text(200,_render_metrics())
        else: s._send(200,{"status":"ok","model":f"{MODEL_NAME} (ModelBuilder TP={TP})","device":"Inferentia2","max_total_tokens":MAX,"max_prompt_tokens":BUCKET,"routes":["/generate","/v1/chat/completions","/v1/completions","/v1/models","/health","/metrics"]})
    def do_POST(s):
        path=s.path.rstrip("/")
        if path not in ("/v1/chat/completions","/v1/completions","/generate"): return s._send(404,{"error":{"message":f"unknown route {path}"}})
        if not READY.is_set(): return s._send(503,{"error":{"message":"server loading (warming up); retry shortly"}})
        if _DRAINING.is_set(): return s._send(503,{"error":{"message":"server draining (spot interruption); retry"}})
        try: _slot_acquire()
        except _Full: return s._send(429,{"error":{"message":f"server busy (> {MAX_QUEUE} queued); retry later"}})
        try:
            body=s._body()
            if path=="/v1/chat/completions":
                msgs=body.get("messages")
                if not msgs: return s._send(400,{"error":{"message":"missing 'messages'"}})
                otype="chat"
            elif path=="/v1/completions":
                prompt=body.get("prompt","")
                if isinstance(prompt,list): prompt=prompt[0] if prompt else ""
                if not prompt: return s._send(400,{"error":{"message":"missing 'prompt'"}})
                msgs=[{"role":"user","content":prompt}]; otype="text"
            else:
                prompt=body.get("prompt","")
                if not prompt: return s._send(400,{"error":{"message":"missing 'prompt'"}})
                msgs=[{"role":"user","content":prompt}]; otype="generate"
            gd=otype=="generate"
            max_new=int(body.get("max_tokens",body.get("max_completion_tokens",body.get("max_new_tokens",110 if gd else 192))))
            temp=float(body.get("temperature",0.0 if gd else 0.7))
            top_k=int(body.get("top_k",0)); top_p=float(body.get("top_p",1.0 if gd else 0.95))
            max_new=max(1,min(max_new,MAX-1)); temp=max(0.0,min(temp,5.0)); top_p=max(0.0,min(top_p,1.0)); top_k=max(0,top_k)
            stop=body.get("stop"); model=body.get("model",MODEL_NAME)
            timeout_s=float(body["timeout"]) if body.get("timeout") else None
            if bool(body.get("stream")) and otype in ("chat","text"):
                s._stream(msgs,max_new,temp,top_k,top_p,stop,otype,model,timeout_s)
            else:
                text,pt,ct,finish=run_chat(msgs,max_new,temp,top_k,top_p,stop,timeout_s)
                s._completion(otype,text,pt,ct,finish,model,body.get("prompt",""))
        except ValueError as e:
            _bump(err=1)
            try: s._send(400,{"error":{"message":str(e)}})
            except Exception: pass
        except Exception as e:
            _bump(err=1)
            try: s._send(500,{"error":{"message":repr(e)}})
            except Exception: pass
        finally: _slot_release()
    def _completion(s,otype,text,pt,ct,finish,model,prompt):
        _counter[0]+=1; now=int(time.time())
        if otype=="chat":
            s._send(200,{"id":f"chatcmpl-{now}-{_counter[0]}","object":"chat.completion","created":now,"model":model,
                "choices":[{"index":0,"message":{"role":"assistant","content":text},"finish_reason":finish}],
                "usage":{"prompt_tokens":pt,"completion_tokens":ct,"total_tokens":pt+ct}})
        elif otype=="text":
            s._send(200,{"id":f"cmpl-{now}-{_counter[0]}","object":"text_completion","created":now,"model":model,
                "choices":[{"index":0,"text":text,"logprobs":None,"finish_reason":finish}],
                "usage":{"prompt_tokens":pt,"completion_tokens":ct,"total_tokens":pt+ct}})
        else:
            s._send(200,{"prompt":prompt,"response":text,"prompt_tokens":pt,"gen_tokens":ct,"finish_reason":finish})
    def _stream(s,msgs,max_new,temp,top_k,top_p,stop,otype,model,timeout_s=None):
        prompt_ids,stop_ids=_prep_chat(msgs,stop)
        if len(prompt_ids)>BUCKET: return s._send(400,{"error":{"message":f"prompt {len(prompt_ids)} tokens > BUCKET {BUCKET}"}})
        _counter[0]+=1; now=int(time.time()); cid=f"chatcmpl-{now}-{_counter[0]}"
        s.send_response(200); s.send_header("Content-Type","text/event-stream"); s.send_header("Cache-Control","no-cache"); s.send_header("Connection","close"); s.end_headers()
        def chunk(delta=None,finish=None):
            if otype=="text": return {"id":cid,"object":"text_completion","created":now,"model":model,"choices":[{"index":0,"text":delta or "","finish_reason":finish}]}
            d={} if delta is None else {"content":delta}
            return {"id":cid,"object":"chat.completion.chunk","created":now,"model":model,"choices":[{"index":0,"delta":d,"finish_reason":finish}]}
        def sse(obj): s.wfile.write(b"data: "+json.dumps(obj).encode()+b"\n\n"); s.wfile.flush()
        t_req=time.time(); ids=[]; prev=""; n0=len(prompt_ids); finish="length"
        try:
            with LOCK:
                gen=_generate(prompt_ids,max_new,temp,top_k,top_p,stop_ids,timeout_s)
                try:
                    while True:
                        ids.append(next(gen)); text=tok.decode(ids,skip_special_tokens=True)
                        delta=text[len(prev):]; prev=text
                        if delta: sse(chunk(delta=delta))
                except StopIteration as e: n0,finish=e.value
            sse(chunk(finish=finish)); s.wfile.write(b"data: [DONE]\n\n"); s.wfile.flush()
            dt=time.time()-t_req; ct=len(ids)
            _bump(reqs=1,ptoks=n0,ctoks=ct,secs=dt,to=(1 if finish=="timeout" else 0),tps=(ct/dt if dt>0 else 0.0))
            print(f"[req] stream pt={n0} ct={ct} {round(ct/dt,1) if dt>0 else 0}tok/s {round(dt,2)}s finish={finish}",flush=True)
        except (BrokenPipeError,ConnectionResetError):
            _bump(reqs=1,ptoks=n0,ctoks=len(ids),secs=time.time()-t_req)

def main():
    threading.Thread(target=_watch_spot,daemon=True).start()
    httpd=ThreadingHTTPServer(("0.0.0.0",PORT),H)
    def _graceful_term(signum,frame):
        print(f"signal {signum} — draining (up to {GRACE_SECONDS}s), then shutting down",flush=True)
        _DRAINING.set()
        def _drain_then_stop():
            end=time.time()+GRACE_SECONDS
            while _INFLIGHT[0]>0 and time.time()<end: time.sleep(0.2)
            httpd.shutdown()
        threading.Thread(target=_drain_then_stop,daemon=True).start()
    signal.signal(signal.SIGTERM,_graceful_term)
    threading.Thread(target=boot,daemon=True).start()
    print(f"HTTP listening on :{PORT} (loading neffs...)",flush=True); httpd.serve_forever()

if __name__ == "__main__":
    if os.environ.get("_TP_CHILD") != "1":
        os.environ["_TP_CHILD"]="1"; multiprocessing.set_start_method("spawn", force=True); main()
