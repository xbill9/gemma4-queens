# %% [markdown]
# # Run Gemma-4 (E2B) text generation on AWS Inferentia2 with `torch-neuronx`
#
# This notebook compiles and runs [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it)
# on a **single NeuronCore** of an Inferentia2 instance (`inf2.xlarge`), producing greedy output that is
# **token-for-token identical to the CPU reference**.
#
# Gemma-4's smaller models use **Per-Layer Embeddings (PLE)** and a **per-layer KV-sharing** attention
# pattern that don't trace through the vendor stack directly. This sample uses a small, explicit recipe:
#
# * **Two graphs** — a *prefill* graph (whole prompt) and a *single-token decode* graph — traced with
#   `torch_neuronx.trace`.
# * **KV cache as graph I/O** — a fixed `MAX`-length KV buffer is passed in and out of the decode graph and
#   updated in place with a one-hot masked write, so the cache stays on-device across steps with no host
#   round-trip of the graph itself.
# * **Host-side embeddings** — token embeddings and the per-layer inputs are computed on the host and passed
#   in as `inputs_embeds` / `per_layer_inputs`, keeping the traced graph free of the PLE lookup.
# * **Eager attention + `tanh` GELU + logit softcap** — matched to the reference so device == CPU.
#
# **Requirements:** an `inf2.xlarge` (or larger) with the Neuron SDK runtime, and access to the gated Gemma-4
# weights on Hugging Face.

# %% [markdown]
# ## 1. Install dependencies
#
# Pinned to the versions this sample was validated against (Neuron SDK 2.23).

# %%
# fmt: off
%pip install --quiet --extra-index-url https://pip.repos.neuron.amazonaws.com \
    "torch-neuronx==2.8.*" "neuronx-cc==2.*" "transformers==5.13.0" "huggingface_hub>=0.34"
# fmt: on

# %% [markdown]
# ## 2. Authenticate to Hugging Face
#
# Gemma-4 is a **gated** model — accept the license on the
# [model page](https://huggingface.co/google/gemma-4-E2B-it) first, then log in with a token that has access.

# %%
from huggingface_hub import notebook_login
notebook_login()

# %% [markdown]
# ## 3. Configuration

# %%
import os, time, torch
torch.manual_seed(0)

MODEL_ID = "google/gemma-4-E2B-it"
MAX      = 128   # max total sequence length (KV buffer length)
BUCKET   = 32    # prefill bucket: prompts are right-padded to this length
NEG      = torch.finfo(torch.float32).min

# %% [markdown]
# ## 4. Load the model
#
# We load in **bf16** so the on-device weight constants fit a single 16 GB core. Gemma-4 uses a `tanh`
# approximation of GELU and a final-logit softcap; we set the activation explicitly and apply the softcap
# in the graph so the device output matches the reference exactly.

# %%
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration, DynamicCache

tok  = AutoTokenizer.from_pretrained(MODEL_ID)
m    = Gemma4ForConditionalGeneration.from_pretrained(
           MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="eager").eval()
lang = m.model.language_model
lm_head = m.lm_head
cfg  = lang.config
SW   = cfg.sliding_window
WDT  = m.dtype                       # bf16 — host KV buffers / one-hot must match this
softcap = getattr(m.config.text_config, "final_logit_softcapping", None)

class GeluTanh(torch.nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
for mod in lang.modules():
    if hasattr(mod, "act_fn"):
        mod.act_fn = GeluTanh()

# Discover the layers that own their K/V (non KV-shared) and their (n_kv_heads, head_dim).
NONSHARED, LINFO = [], {}
for i, lyr in enumerate(lang.layers[:cfg.num_hidden_layers]):
    a = lyr.self_attn
    if not a.is_kv_shared_layer:
        hd = a.head_dim
        NONSHARED.append(i)
        LINFO[i] = (a.k_proj.out_features // hd, hd)
print("non-shared KV layers:", NONSHARED)

def softcap_logits(lg):
    return softcap * torch.tanh(lg / softcap) if softcap else lg

# %% [markdown]
# ## 5. Define the prefill and decode graphs
#
# The **prefill** graph runs the whole (padded) prompt and returns the logits plus the K/V for every
# non-shared layer. The **decode** graph runs a single new token against a fixed `MAX`-length KV buffer,
# writing the new K/V into the buffer with a one-hot mask and returning the updated buffers.

# %%
class PreWrap(torch.nn.Module):
    def __init__(s): super().__init__(); s.lang = lang; s.head = lm_head
    def forward(s, ie, am, ple):
        cache = DynamicCache()
        out = s.lang(inputs_embeds=ie, per_layer_inputs=ple, attention_mask=am,
                     use_cache=True, past_key_values=cache)
        lg = softcap_logits(s.head(out.last_hidden_state))
        ks = [cache.layers[i].keys   for i in NONSHARED]
        vs = [cache.layers[i].values for i in NONSHARED]
        return (lg, ks, vs)

class StaticKV:
    """A KV cache backed by fixed MAX-length buffers; update() writes the new token via a one-hot mask."""
    is_compileable = False
    def __init__(s, key_bufs, val_bufs, onehot):
        s.key = {i: key_bufs[j] for j, i in enumerate(NONSHARED)}
        s.val = {i: val_bufs[j] for j, i in enumerate(NONSHARED)}
        s.oh  = onehot                                    # [1,1,MAX,1] float
    def update(s, k, v, idx, *a, **kw):                   # k,v: [1,nkv,1,hd]
        s.key[idx] = s.key[idx] * (1.0 - s.oh) + k * s.oh
        s.val[idx] = s.val[idx] * (1.0 - s.oh) + v * s.oh
        return s.key[idx], s.val[idx]
    def get_seq_length(s, *a, **k): return 0              # position_ids are supplied explicitly
    def export(s):
        return [s.key[i] for i in NONSHARED], [s.val[i] for i in NONSHARED]

class DecWrap(torch.nn.Module):
    def __init__(s): super().__init__(); s.lang = lang; s.head = lm_head
    def forward(s, ie, ple, position_ids, onehot, full_mask, slide_mask, key_bufs, val_bufs):
        cache = StaticKV(key_bufs, val_bufs, onehot)
        masks = {"full_attention": full_mask, "sliding_attention": slide_mask}
        out = s.lang(inputs_embeds=ie, per_layer_inputs=ple, position_ids=position_ids,
                     attention_mask=masks, use_cache=True, past_key_values=cache)
        lg = softcap_logits(s.head(out.last_hidden_state))
        ks, vs = cache.export()
        return (lg, ks, vs)

pre = PreWrap().eval()
dec = DecWrap().eval()

# Host helpers: embeddings (incl. per-layer inputs) and all position-dependent tensors.
def embed_ids(id_list):
    ids = torch.tensor([id_list])
    with torch.no_grad():
        ie  = lang.embed_tokens(ids)
        ple = lang.get_per_layer_inputs(ids, ie)
    return ie, ple

def host_pos_tensors(pos):
    ar     = torch.arange(MAX)
    onehot = (ar == pos).view(1, 1, MAX, 1).to(WDT)
    valid  = ar <= pos
    full   = torch.where(valid, 0.0, NEG).view(1, 1, 1, MAX)
    slide  = torch.where(valid & (ar > pos - SW), 0.0, NEG).view(1, 1, 1, MAX)
    return torch.tensor([[pos]], dtype=torch.long), onehot, full, slide

ec  = m.generation_config.eos_token_id
EOS = set(ec) if isinstance(ec, (list, tuple)) else {ec}

# %% [markdown]
# ## 6. Greedy generation loop (works with either the eager modules or the traced graphs)

# %%
def run_greedy(pre_fn, dec_fn, prompt, maxnew=30):
    n0  = len(prompt)
    pad = prompt + [0] * (BUCKET - n0)
    ie, ple = embed_ids(pad)
    am = torch.tensor([[1] * n0 + [0] * (BUCKET - n0)])
    with torch.no_grad():
        lg, ks, vs = pre_fn(ie, am, ple)
    first = int(lg[0, n0 - 1].argmax())

    key_bufs = [torch.zeros(1, LINFO[i][0], MAX, LINFO[i][1], dtype=WDT) for i in NONSHARED]
    val_bufs = [torch.zeros(1, LINFO[i][0], MAX, LINFO[i][1], dtype=WDT) for i in NONSHARED]
    for j in range(len(NONSHARED)):
        key_bufs[j][:, :, :n0, :] = ks[j][:, :, :n0, :]
        val_bufs[j][:, :, :n0, :] = vs[j][:, :, :n0, :]

    seq, cur = [first], n0
    for _ in range(maxnew):
        if seq[-1] in EOS:
            break
        ie1, ple1 = embed_ids([seq[-1]])
        position_ids, onehot, full_mask, slide_mask = host_pos_tensors(cur)
        with torch.no_grad():
            lg1, key_bufs, val_bufs = dec_fn(ie1, ple1, position_ids, onehot,
                                             full_mask, slide_mask, key_bufs, val_bufs)
        seq.append(int(lg1[0, 0].argmax()))
        cur += 1
    return seq

msgs   = [{"role": "user", "content": "What is the capital of France?"}]
enc    = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
prompt = enc["input_ids"][0].tolist()
assert len(prompt) <= BUCKET, "prompt longer than BUCKET; increase BUCKET (and re-trace)"

# %% [markdown]
# ## 7. Compile the two graphs for Inferentia
#
# `torch_neuronx.trace` compiles each graph to a NEFF. Compilation takes a few minutes the first time.

# %%
import torch_neuronx

ie, ple = embed_ids(prompt + [0] * (BUCKET - len(prompt)))
am = torch.tensor([[1] * len(prompt) + [0] * (BUCKET - len(prompt))])
CARGS = ["--model-type", "transformer", "--auto-cast", "all", "--auto-cast-type", "bf16"]

t = time.time()
pre_neff = torch_neuronx.trace(pre, (ie, am, ple), compiler_args=CARGS)
print(f"prefill compiled in {time.time()-t:.0f}s")

key_bufs = [torch.zeros(1, LINFO[i][0], MAX, LINFO[i][1], dtype=WDT) for i in NONSHARED]
val_bufs = [torch.zeros(1, LINFO[i][0], MAX, LINFO[i][1], dtype=WDT) for i in NONSHARED]
ie1, ple1 = embed_ids([prompt[-1]])
position_ids, onehot, full_mask, slide_mask = host_pos_tensors(len(prompt))
t = time.time()
dec_neff = torch_neuronx.trace(
    dec, (ie1, ple1, position_ids, onehot, full_mask, slide_mask, key_bufs, val_bufs),
    compiler_args=CARGS)
print(f"decode compiled in {time.time()-t:.0f}s")

# Optionally persist the NEFFs so you can reload without recompiling:
# torch.jit.save(pre_neff, "gemma4_e2b_prefill.pt")
# torch.jit.save(dec_neff, "gemma4_e2b_decode.pt")

# %% [markdown]
# ## 8. Validate: device output == CPU reference

# %%
cpu_seq = run_greedy(pre,      dec,      prompt)
dev_seq = run_greedy(pre_neff, dec_neff, prompt)

print("CPU:   ", repr(tok.decode([s for s in cpu_seq if s not in EOS], skip_special_tokens=True)))
print("Device:", repr(tok.decode([s for s in dev_seq if s not in EOS], skip_special_tokens=True)))
print("SEQ_MATCH:", cpu_seq == dev_seq)
assert cpu_seq == dev_seq, "device output diverged from CPU reference"

# %% [markdown]
# ## 9. Generate and measure throughput

# %%
def generate(question, maxnew=80):
    enc = tok.apply_chat_template([{"role": "user", "content": question}],
                                  add_generation_prompt=True, return_tensors="pt", return_dict=True)
    p = enc["input_ids"][0].tolist()
    t = time.time()
    seq = run_greedy(pre_neff, dec_neff, p, maxnew=maxnew)
    dt = time.time() - t
    print(tok.decode([s for s in seq if s not in EOS], skip_special_tokens=True))
    print(f"\n[{len(seq)} tokens in {dt:.1f}s = {len(seq)/dt:.1f} tok/s]")

generate("In one sentence, what is AWS Inferentia?")

# %% [markdown]
# ## Notes / troubleshooting (Gemma-4 specifics)
#
# * **Per-Layer Embeddings (PLE).** The smaller Gemma-4 models add per-layer inputs. We compute
#   `embed_tokens` + `get_per_layer_inputs` on the host and pass them in, so the traced graph never does the
#   PLE lookup. (The larger encoder-free `gemma4_unified` models have no PLE.)
# * **KV-sharing as graph I/O.** Passing the fixed `MAX`-length KV buffers in and out of the decode graph is
#   what lets the per-layer KV-sharing pattern trace cleanly — the sharing shows up as ordinary graph data
#   dependencies. The one-hot masked write (`buf*(1-oh) + kv*oh`) avoids any dynamic index op in the graph.
# * **bf16 is required for a single core.** In fp32 the on-device weight constants (~15 GB for E2B's larger
#   sibling) exceed a 16 GB core; bf16 halves them. `--auto-cast bf16` matches the host bf16 embeddings.
# * **Eager attention + `tanh` GELU + softcap.** Set `attn_implementation="eager"`, replace `act_fn` with the
#   `tanh` GELU, and apply the final-logit softcap in the graph — all three are needed for exact CPU parity.
# * **Bigger models need tensor parallelism.** E2B fits one core. The 4B/12B variants exceed 16 GB even in
#   bf16 and require a TP=2 build across both NeuronCores of an `inf2.8xlarge`.
