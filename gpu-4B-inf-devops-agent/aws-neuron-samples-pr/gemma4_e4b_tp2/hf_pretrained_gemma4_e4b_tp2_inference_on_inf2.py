# %% [markdown]
# # Run Gemma-4 (E4B) on AWS Inferentia2 across both NeuronCores (TP=2)
#
# This sample compiles and runs [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it)
# **tensor-parallel across the two NeuronCores of an `inf2.8xlarge`**, with greedy output
# **token-for-token identical to the CPU reference** and the prompt **prefilled on-device**.
#
# It's the tensor-parallel follow-up to the single-core
# [E2B sample](../hf_pretrained_gemma4_e2b_inference_on_inf2.ipynb): E4B is ~2× larger and does **not**
# fit one 16 GB core even in bf16, so it must be sharded across both cores.
#
# **How it works** — one weight-sharing `neuronx-distributed` **`ModelBuilder`** trace holds two buckets
# (a *prefill* graph and a single-token *decode* graph) that share **one bf16 sharded weight set** and a
# **device-resident KV cache** aliased as graph I/O. Attention is GQA-sharded across the two ranks; token
# embeddings and per-layer inputs are computed on the host.
#
# > **Why a helper script for the compile:** the tensor-parallel trace spawns worker processes
# > (`multiprocessing`, needs an `if __name__ == "__main__"` guard), which doesn't run inside a notebook
# > cell — so the **compile** step calls the bundled `tp2_build.py`. Everything after (load + generate)
# > runs in-process, right here in the notebook.
#
# **Requires** an `inf2.8xlarge` (2 NeuronCores) with the Neuron runtime, and access to the gated weights.

# %% [markdown]
# ## 1. Install dependencies

# %%
# fmt: off
%pip install --quiet --extra-index-url https://pip.repos.neuron.amazonaws.com \
    "torch-neuronx==2.8.*" "neuronx-cc==2.*" "neuronx-distributed>=0.17" \
    "transformers==5.13.0" "huggingface_hub>=0.34"
# fmt: on

# %% [markdown]
# ## 2. Authenticate + download the (gated) model
#
# Accept the license on the [model page](https://huggingface.co/google/gemma-4-E4B-it), then log in.
# The build script reads the weights from a local directory, so we download them there.

# %%
from huggingface_hub import notebook_login, snapshot_download
notebook_login()

MODEL_ID  = "google/gemma-4-E4B-it"
MODEL_DIR = "./gemma-4-E4B-it"
snapshot_download(MODEL_ID, local_dir=MODEL_DIR)

# %% [markdown]
# ## 3. Add swap (the 2-rank compile peaks past host RAM)
#
# `neuronx-cc` compiles both tensor-parallel ranks concurrently and briefly exceeds a 128 GB host; a
# swapfile prevents an OOM during compilation. The resulting model runs fine without swap.

# %%
import subprocess
subprocess.run("fallocate -l 55G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile",
               shell=True)  # sudo may be required depending on your setup
subprocess.run("swapon --show", shell=True)

# %% [markdown]
# ## 4. Compile the tensor-parallel model
#
# `tp2_build.py` shards the model across `TP_DEGREE=2` cores, traces the prefill + decode buckets, runs an
# **in-process `SEQ_MATCH` check** (device greedy vs. CPU reference), and saves the model with
# `torch.jit.save`. First-time compilation takes ~15–25 minutes.

# %%
import os
env = dict(os.environ,
           MODEL_DIR=MODEL_DIR, TP_DEGREE="2",
           KV_MAX="256", KV_BUCKET="64", MB_WDTYPE="bf16",
           MB_SAVE="./gemma4_e4b_tp2.pt")
subprocess.run(["python", "tp2_build.py"], env=env, check=True)
# -> prints "SEQ_MATCH True", "DEV GEN: 'The capital of France is Paris.'", "MB_SAVED ..."

# %% [markdown]
# ## 5. Load the saved model and generate (in-process — no spawn needed)
#
# The saved `NxDModel` loads with `torch.jit.load` and runs across both cores from a single process. The
# host provides only the token embeddings + per-layer inputs; the transformer, head, and softcap run on
# the cores. Each request prefills the prompt **on-device**, then decodes token-by-token against the
# device-resident aliased KV cache.

# %%
import torch, time, torch_neuronx
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

KV_MAX, KV_BUCKET = 256, 64
NEG = torch.finfo(torch.float32).min

tok  = AutoTokenizer.from_pretrained(MODEL_DIR)
mm   = Gemma4ForConditionalGeneration.from_pretrained(
           MODEL_DIR, torch_dtype=torch.float32, attn_implementation="eager").eval()
lang = mm.model.language_model
SW   = lang.config.sliding_window
ec   = mm.generation_config.eos_token_id
EOS  = set(ec) if isinstance(ec, (list, tuple)) else {ec}

dec = torch.jit.load("./gemma4_e4b_tp2.pt")
dec.nxd_model.initialize_with_saved_weights(torch.tensor(0))   # load sharded weights onto both cores

def _inputs(ids, positions):
    """Host-side: token embeddings + per-layer inputs, a one-hot KV-write map, and [seq x MAX] masks."""
    ids_t = torch.tensor([ids])
    with torch.no_grad():
        ie  = lang.embed_tokens(ids_t)
        ple = lang.get_per_layer_inputs(ids_t, ie)
    seq = len(positions); ar = torch.arange(KV_MAX); pos = torch.tensor([positions], dtype=torch.long)
    oh    = (ar.view(KV_MAX, 1) == pos.view(1, seq)).view(1, 1, KV_MAX, seq).float()
    q     = pos.view(seq, 1); m2 = ar.view(1, KV_MAX)
    full  = torch.where(m2 <= q, 0.0, NEG).view(1, 1, seq, KV_MAX)
    slide = torch.where((m2 <= q) & (m2 > q - SW), 0.0, NEG).view(1, 1, seq, KV_MAX)
    return ie, ple, pos, oh, full, slide

def _logits(out):
    return out[0] if isinstance(out, (tuple, list)) else out

def generate(question, maxnew=80):
    p  = tok.apply_chat_template([{"role": "user", "content": question}],
                                 add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"][0].tolist()
    n0 = len(p); assert n0 <= KV_BUCKET
    t0 = time.time()
    lg = _logits(dec(*_inputs(p + [0] * (KV_BUCKET - n0), list(range(KV_BUCKET)))))   # on-device prefill
    ft = time.time() - t0
    nxt = int(lg[0, n0 - 1].argmax()); seq = [nxt]; cur = n0
    for _ in range(maxnew):
        if nxt in EOS: break
        lg = _logits(dec(*_inputs([nxt], [cur])))                                     # on-device decode
        nxt = int(lg[0, 0].argmax()); seq.append(nxt); cur += 1
    dt = time.time() - t0
    print(tok.decode([s for s in seq if s not in EOS], skip_special_tokens=True))
    print(f"\n[first token {ft*1000:.0f} ms | {len(seq)} tokens, {len(seq)/dt:.1f} tok/s]")

generate("In one sentence, what is AWS Inferentia?")

# %% [markdown]
# ## Notes / troubleshooting (Gemma-4 TP=2 specifics)
#
# * **Weight-sharing buckets.** `ModelBuilder` shards **one** bf16 weight set once and shares it across the
#   prefill and decode buckets, so weights are resident a single time (~7–8 GB/core) and the KV cache never
#   leaves the cores.
# * **GQA head-sharding.** Shard q/o/gate/up/down across ranks; shard k/v to `n_kv // TP` heads per rank
#   when divisible (and keep `num_key_value_groups`), otherwise replicate k/v — this preserves the query→KV
#   head mapping. Global (full-attention) layers with a single KV head are replicated.
# * **`layer_scalar` is a buffer.** Gemma-4 scales each layer's output by a per-layer `layer_scalar`
#   *buffer*; `ModelBuilder`'s weight-sharding loads *parameters* only, so it must be copied from the
#   checkpoint by hand (see `tp2_build.py`) — otherwise every layer over-scales and the output is garbage.
# * **Host embeddings.** Token embeddings + per-layer inputs (PLE) are computed on the host and passed as
#   `inputs_embeds` / `per_layer_inputs`.
# * **Launcher.** The tensor-parallel trace uses `multiprocessing` spawn with an `if __name__=="__main__"`
#   guard and a small `transformers.utils.fx` shim (transformers 5.13 dropped that module) — see the top of
#   `tp2_build.py`. That's why the compile is a script, not a notebook cell.
# * **Scaling further.** The 12B/31B dense variants use the same recipe at higher TP degrees
#   (`TP_DEGREE=8` on an `inf2.24xlarge`); bf16 is mandatory as fp32 constants overflow a 16 GB core.
