# Porting Gemma‑4 (E2B / E4B / 12B) to AWS Inferentia2

*A field report on running Google's Gemma‑4 family on AWS Inferentia2 (`inf2`), covering the
three architectural obstacles that break the vendor stack — **mixed attention heads**, the
**vLLM / `optimum‑neuron` / NxD** dead‑ends, and the **Neuron compiler** (`neuronx‑cc`) limits —
and the recipe that got all three model sizes serving coherently.*

| | |
|---|---|
| Models | `google/gemma-4-E2B-it`, `google/gemma-4-E4B-it`, `google/gemma-4-12B-it` |
| Hardware | AWS Inferentia2 — `inf2.xlarge` (1 chip / 2 cores / 32 GB HBM) and `inf2.8xlarge` (same accelerator, 128 GB host RAM) |
| Software | Neuron SDK 2.23 · `torch-neuronx` 2.8.0 · `neuronx-cc` 2.23.6484 · `neuronx-distributed` 0.17 · `transformers` 5.13.0 |
| Result | E2B ~44 tok/s (1 core), E4B ~33–39 tok/s (TP=2), 12B ~15 tok/s (TP=2) — greedy decode **token‑for‑token identical to the CPU reference** on all three |
| Artifacts | HF: `xbill9/gemma-4-{E2B,E4B,12B}-it-inferentia2` · Docker Hub: `xbill9/gemma4-optb{,-e4b,-12b}` |

---

## 1. Background: why this is hard

Gemma‑4 is not a vanilla decoder. Across the family it combines several features that each map
cleanly onto TPU/XLA (where the model was designed) but individually break the AWS inference path:

- **Per‑Layer Embeddings (PLE)** and **MatFormer** nesting on E2B/E4B (the "effective 2B/4B" trick).
- **Cross‑layer KV‑sharing**: on E2B/E4B many layers do not compute their own Key/Value — they
  reuse a neighbour's KV projection.
- **Grouped‑Query Attention (GQA)** with small KV‑head counts.
- **Mixed attention types**: interleaved *sliding‑window* and *global* attention layers, with
  **different KV‑head counts per type** on the 12B.
- **Logit soft‑capping** (`tanh` cap at 30) over a **262 144‑token vocabulary**.

The AWS vendor stack (`optimum-neuron` + `neuronx-distributed` + the Neuron vLLM backend) has **no
Gemma‑4 model class at all**, and its graph builder cannot express KV‑sharing. The public Neuron
vLLM endpoint we started from loaded *something* and served **fluent‑looking gibberish**. Everything
below is about getting from there to correct, fast, cheap inference.

### 1.1 The three models at a glance

| | **E2B** | **E4B** | **12B** |
|---|---|---|---|
| HF class | `Gemma4ForConditionalGeneration` (text) | same | `Gemma4UnifiedForConditionalGeneration` |
| model_type | `gemma4_text` | `gemma4_text` | `gemma4_unified` (encoder‑free multimodal) |
| Params (effective) | ~5B (2B) | ~8B (4B) | 12B |
| Per‑Layer Embeddings | **yes** | **yes** | **no** (`hidden_size_per_layer_input=0`) |
| Cross‑layer KV‑sharing | **yes** (15 non‑shared layers own KV) | **yes** | **no** (`num_kv_shared_layers=0`, every layer owns KV) |
| Query / KV heads | GQA | **8 q / 2 kv** | **16 q / 8 kv**, global layers **nkv=1** |
| Attention | sliding + global, `sliding_window=512` | sliding + global, `sliding_window=512` | sliding + global, `sliding_window=1024`, `attention_k_eq_v=true` |
| head_dim | — | 256 | 256 (global 512) |
| Logit softcap | 30 | 30 | 30 |
| Vocab | 262 144 | 262 144 | 262 144 (tied embeddings) |
| Fits one 16 GB core? | **yes** (bf16) | **no** → TP=2 | **no** → TP=2 |

Three model sizes, three *different* reasons the vendor path fails, and — as it turns out — three
distinct meanings of "mixed heads."

---

## 2. The three faces of "mixed heads"

"Mixed heads" was the single most expensive class of bug in this port. It shows up in three
completely different forms as you move up the size ladder.

### 2.1 Mixed KV‑*sharing* across layers (E2B, E4B)

On E2B/E4B a layer's attention is flagged `self_attn.is_kv_shared_layer`. Shared layers do **not**
run their own `k_proj`/`v_proj`; they attend against a KV tensor produced by an earlier "owner"
layer. On TPU this is a free graph view. In AWS's `neuronx-distributed` (NxD) `ModelBuilder`, which
wants a static, per‑layer weight→buffer mapping, it **cannot be represented** — there is no place to
say "this layer's K/V is that layer's K/V, computed once, live."

The fix (**"Option B"**, see §3) is to stop fighting the framework and trace the Hugging Face
forward pass directly. KV‑sharing then falls out as an ordinary live data dependency in the traced
graph. The port only has to enumerate which layers actually own a cache:

```python
# discover the layers that write KV; shared layers never touch the cache
NONSHARED = []
for i, lyr in enumerate(lang.layers[:cfg.num_hidden_layers]):
    a = lyr.self_attn
    if not a.is_kv_shared_layer:                 # <- the crux
        NONSHARED.append(i)
        LINFO[i] = (a.k_proj.out_features // a.head_dim, a.head_dim)   # (n_kv, head_dim)
```

Only the `NONSHARED` layers (15 of them on E2B) allocate a KV buffer; the shared layers read
through. This is why the KV cache is far smaller than a naïve "one buffer per layer" implementation
would allocate — and why it fits a single core in bf16.

### 2.2 Mixed *query vs KV* head counts — GQA under tensor parallelism (E4B)

E4B does not fit one 16 GB core (§4.3), so it must be sharded across both NeuronCores (TP=2). GQA
now bites: E4B has **8 query heads and 2 KV heads**. Column‑sharding the query projection across 2
ranks is trivial (4 q‑heads/rank). The KV projection has only **2** heads, and the subtle failure is
that `repeat_kv` inside attention broadcasts each KV head to a *group* of query heads — so you cannot
independently halve the KV heads and the query heads without scrambling **which query attends which
KV**.

The rule that makes it correct: shard KV to `nkv // TP` heads per rank **only when it divides**, and
crucially **keep `num_key_value_groups` unchanged** so `repeat_kv` still maps each rank's 4 query
heads onto its 1 KV head:

```python
def _kv_rank_width(nkv):
    return nkv // TP if nkv % TP == 0 else nkv     # 2 // 2 = 1 KV head / rank on E4B

# ... per layer:
nkv = a.k_proj.out_features // a.head_dim
if nkv % TP == 0:                                  # divisible: shard k/v, keep groups
    a.k_proj = col(a.k_proj)
    a.v_proj = col(a.v_proj)
# else: see 2.3
```

Naïvely also dividing `num_key_value_groups` produces the classic "4‑vs‑8 `repeat_kv` mismatch":
plausible‑looking but wrong output. The KV cache is then kept **device‑resident and sliced per rank**
(head *r* → rank *r*), because NxD's `forward()` only returns rank 0's KV (§5.2).

### 2.3 Mixed *attention types* with different KV‑head counts (12B)

The 12B is where "mixed heads" becomes structural. `gemma4_unified` interleaves two attention types
with **different KV‑head counts**:

- **sliding‑window layers**: `nkv = 8` → divisible by TP=2 → shard to 4 heads/rank (the §2.2 path).
- **global layers**: `num_global_key_value_heads = 1` → **indivisible by TP=2**.

You cannot column‑shard a single KV head across two ranks. The working answer is to **leave the
global layers' K/V replicated** (a plain `nn.Linear`, which NxD loads identically on every rank) and
**shrink the group count** so `repeat_kv` matches the *sharded* query‑head count per rank:

```python
a = lyr.self_attn
hd = a.head_dim
nq = a.q_proj.out_features // hd            # capture BEFORE replacing q_proj (see gotcha)
a.q_proj = col(a.q_proj); a.o_proj = row(a.o_proj)
nkv = a.k_proj.out_features // hd
if nkv % TP == 0:                            # sliding layers: nkv=8 -> 4/rank, keep groups
    a.k_proj = col(a.k_proj); a.v_proj = col(a.v_proj)
else:                                        # global layers: nkv=1, indivisible
    # leave k/v REPLICATED; shrink groups to the per-rank sharded q-head count
    a.num_key_value_groups = (nq // TP) // nkv
```

Two gotchas made this a multi‑hour fight (commit `d70dc94`):

1. **Ordering:** you must read `nq = q_proj.out_features // head_dim` **before** replacing `q_proj`
   with a `ColumnParallelLinear`, whose `.out_features` reports the *sharded* width — otherwise you
   compute the group count from already‑halved numbers and get an `AttributeError`/wrong groups.
2. The global‑layer `else` branch had literally never been exercised by E4B (all E4B layers are
   `nkv=2`, divisible), so it was dead, untested code until the 12B walked into it.

**Takeaway:** "mixed heads" is not one problem. It is (a) mixed KV *sharing* across layers, (b) mixed
query/KV *counts* within a layer (GQA), and (c) mixed attention *types* with different KV counts per
type. Each needs a different sharding rule, and the naïve "just divide everything by TP" is wrong in
all three.

---

## 3. Why vLLM / `optimum‑neuron` / NxD couldn't do it

The port deliberately does **not** use the AWS vendor inference path. Here is why each vendor layer
failed, and what replaced it.

**`optimum-neuron` has no Gemma‑4.** There is simply no `gemma4_text` / `gemma4_unified` model class
in `optimum-neuron`. The Neuron vLLM backend dispatches through `optimum-neuron`, so vLLM has nothing
to instantiate.

**NxD `ModelBuilder` cannot represent KV‑sharing.** NxD wants a static graph where each layer maps
its own weights to its own KV buffers. Gemma‑4's cross‑layer KV‑sharing (§2.1) has no expression in
that model, so the graph builder either refuses the model or silently drops the sharing.

**The vendor endpoint served gibberish.** The public Neuron vLLM/NxD deployment we began with came up
"healthy" and produced fluent‑looking but semantically empty text — the worst failure mode, because
it *looks* like it works.

**An automated port falsely reported success.** A framework auto‑port pass claimed **"100% PASS"**.
On inspection its golden reference had been built from a **PLE‑stripped** checkpoint, so both sides of
the comparison were wrong in the same way. A green check on a broken oracle is worse than a red one.

**The replacement — "Option B."** Instead of teaching NxD about Gemma‑4, `torch_neuronx.trace()` (for
single‑core E2B) and `neuronx_distributed.trace.parallel_model_trace` / `ModelBuilder` (for TP E4B/12B)
are pointed **straight at the Hugging Face `transformers` 5.13 text forward pass**. KV‑sharing,
PLE, GQA, and the sliding/global mix all trace as ordinary live graph dependencies — exactly as they
do on TPU/XLA. The vendor abstraction is bypassed; the reference implementation *is* the graph.

There is still a TP‑launch wrinkle: `parallel_model_trace` spawns rank workers, and
`transformers` 5.13's FX utilities plus the spawn need a `transformers.utils.fx` shim and a
`_TP_CHILD` environment sentinel to guard against infinite re‑spawn.

---

## 4. The Neuron compiler (`neuronx‑cc`) wall

Once the graph is traceable, `neuronx-cc` imposes its own hard limits. The recurring one is **SBUF**,
the on‑chip state buffer (SRAM), capped at **196 608 B per partition**. Several Gemma‑4 ops blow
straight through it, each with a distinct fix.

### 4.1 The PLE table cannot live on device (E2B/E4B)

The Per‑Layer Embedding table is `262144 × hidden`. Materialised on device it trips the compiler and
by itself over‑commits the 16 GB core. **Fix:** keep the PLE (and word) embedding **on the host**,
gather on CPU, and feed the result in **as activations**. The device graph never sees the table.

### 4.2 Logit soft‑capping overflows SBUF (12B)

Gemma‑4 applies `softcap * tanh(logits / softcap)` over the full 262 144‑token vocabulary. In fp32
that `tanh` is a custom‑call whose working set is `128 × 524288` bytes — **524 288 > 196 608 B/partition**:

```
[NCC_INLA001] Allocated memory out of bound (128x524288) 524288 vs 196608 B/partition
```

**Fix (commit `328ca88`):** *don't apply softcap on device.* Soft‑capping is **monotonic**, so
`argmax(softcap(x)) == argmax(x)` — greedy decode is completely unaffected. The device returns **raw
logits**; the host server re‑applies the cap only when sampling (`temperature > 0`). Correct and free.
(On E2B/E4B, at the smaller hidden size, the explicit `tanh` cap does fit and is kept in‑graph.)

### 4.3 Fused/SDPA attention overflows SBUF at `sliding_window=1024` (12B)

Dropping softcap was necessary but not sufficient — the real overflow on the 12B was the **fused
attention custom‑call**, because the 12B's `sliding_window = 1024` is **2× E4B's 512**, doubling the
attention tile. **Fix (commit `a0a2a75`):** force **eager attention** on the device model
(`cfg._attn_implementation = "eager"` on both the config and its `text_config`). Eager tiles attention
as explicit matmuls that fit SBUF. (E2B/E4B use eager anyway; at `sliding_window=512` the fused path
also would have fit, but eager is the portable choice.)

### 4.4 fp32 constants exceed the 16 GB core → bf16 **and** tensor parallelism

A single Option‑B neff for E4B materialises **~15.4 GB of fp32 model constants**; a second neff
(prefill + decode) on the same core tips past the 16 GB NeuronCore budget →
`NRT_RESOURCE / status=4 Allocation Failure`. Two things are needed:

- **bf16 weights** (`MB_WDTYPE=bf16`): NxD's `shard_children` casts the checkpoint to the layer dtype,
  halving the neff. **But bf16 shrinks the *on‑disk* neff, not the *on‑device* constant count** — the
  15.4 GB of resident constants is still there.
- **TP=2**: the only way to actually fit E4B/12B is to **shard the model across both cores**, so each
  rank holds ~half the weights (~7.7 GB/core on E4B; ~12 GB/core on 12B at 256 context).

E2B is the exception: in bf16 it fits one core, which is why it stays single‑core at ~44 tok/s.

### 4.5 Keep the graph in elementary ops (all sizes)

`neuronx-cc` is happiest with plain arithmetic. Several things are written out by hand rather than
left to library helpers or dynamic control flow:

- **tanh‑GELU** spelled out as `0.5*x*(1+tanh(0.7978845608*(x + 0.044715*x^3)))`.
- **`DynamicCache` at trace time**, but a **fixed static KV buffer at decode**; the per‑step write is a
  **one‑hot masked scatter** — `buf*(1-oh) + k*oh` — pure arithmetic, trace‑safe, no scatter op or
  data‑dependent indexing.
- language model + LM head registered as **real submodules** so they compile into the graph instead of
  being called opaquely.

### 4.6 A silent, non‑compiler gotcha: `layer_scalar` is a *buffer*

Each Gemma‑4 layer does `hidden_states *= self.layer_scalar`. `layer_scalar` is a registered
**buffer** (default `1.0`, real value ≈ 0.06). NxD's weight‑sharding (`shard_children` /
`get_sharded_checkpoint`) loads **parameters only, never buffers** — so without intervention every
layer over‑scales ~16×, compounding across all layers into `cos ≈ 0` garbage. **Fix (commit
`e785f6d`):** read the per‑layer `layer_scalar` from the checkpoint and copy it into the module by
hand before tracing. This one‑line‑of‑physics bug cost real time on both E4B and 12B and is the
single most important "gotcha" in the whole port.

### 4.7 Compilation ergonomics

`neuronx-cc` runs on **CPU** — no NeuronCore needed to compile, so you can build neffs on any box.
But a TP=2 compile runs `neuronx-cc` for **both ranks concurrently** and peaks **past 128 GB host RAM**;
add ≥55 GB of swap before compiling (the resulting neff runs fine without swap).

---

## 5. Device‑resident KV cache and the prefill problem

Correctness and SBUF are only half the story; throughput comes from keeping the KV cache **on the
device** and never round‑tripping it through the host. Three designs evolved:

### 5.1 Two‑graph static KV (E2B, single core)

Two neffs share one static KV buffer: **prefill** (padded prompt ≤ `KV_BUCKET`, returns the 15
non‑shared layers' K/V) and **decode** (single‑token forward against a fixed `KV_MAX` buffer, one‑hot
masked write). KV tensors are graph inputs/outputs. ~44 tok/s, ~0.06 s prefill after a one‑time
~100 s neff load.

### 5.2 TP + on‑device aliased KV (E4B/12B)

Under TP a neff spans **both** cores, so you cannot park prefill on core 0 and decode on core 1. Two
sub‑designs handle prefill:

- **`tp_alias`**: only the **decode** neff is resident; **prefill runs once on the host CPU** to seed
  the cache. The KV buffers are **device‑resident `nn.Parameter`s aliased as graph I/O**
  (`input_output_aliases`), updated in place each step, so the cache never leaves the cores. Each
  rank's KV is seeded with head *r*. First token ~1.4–1.6 s (host prefill), decode ~33 tok/s.
- **device‑prefill (`ModelBuilder`)**: prefill and decode are **two buckets of one weight‑sharing
  model**, both on device, sharing the aliased KV. First token drops to **~0.1–0.16 s**; decode
  ~39 tok/s (E4B) / ~15 tok/s (12B). This is the recommended build.

```python
# KV parameters aliased as graph I/O so the cache is never round-tripped through the host
aliases = {}
for j in range(NK): aliases[w.kbuf[j]] = 1 + j
for j in range(NK): aliases[w.vbuf[j]] = 1 + NK + j
```

Per‑token cost is essentially **flat across context length** because the cache is device‑resident.

### 5.3 Save / reload

The traced executor is saved with `torch.jit.save(model, path)` and reloaded with
`torch.jit.load` + `nxd_model.initialize_with_saved_weights(torch.tensor(start_rank))`. (The executor's
own `.save` is TorchScript's, not NxD's — a subtle trap.)

---

## 6. Results

All three ports match the CPU float reference **token‑for‑token** on greedy decode
(`SEQ_MATCH True`), e.g. *"The capital of France is **Paris**."*

| Model | Build | Hardware | TP | Context | First token | Decode | Host RAM |
|---|---|---|---|---|---|---|---|
| **E2B** | two‑graph static KV | `inf2.8xlarge` / `inf2.xlarge` (slim) | 1 core | 512 / 128 | ~0.06 s | **~44 tok/s** | ~6 GB (slim) |
| **E4B** | `tp_alias` | `inf2.8xlarge` | 2 | 512 / 128 · 2048 / 512 | ~1.4–1.6 s | ~33 tok/s | — |
| **E4B** | device‑prefill (bf16) | `inf2.8xlarge` / `inf2.xlarge` (slim) | 2 | 512 / 128 | **~0.11–0.16 s** | **~36–39 tok/s** | ~8 GB (slim) |
| **12B** | device‑prefill (bf16) | `inf2.8xlarge` / `inf2.xlarge` (slim) | 2 | 256 / 64 | **~0.1 s** | **~15 tok/s** | ~8 GB (slim) |

The 12B's lower decode rate is inherent (dense 12B, all weights active per token), not a port defect;
per‑token cost matches the 8xlarge full server. Notably, **all three run on a single `inf2.xlarge`**
(¼ the price of the 8xlarge, same 2‑core accelerator) via "slim" servers that keep only the
host‑side embedding on the CPU.

---

## 7. Operational gotchas worth their own paragraph

- **A missing `tokenizer.json` masquerades as a device bug.** A hung `hf download` once left
  `tokenizer.json` absent; `GemmaTokenizer` loaded with no vocab and mapped *every* prompt to a single
  `<unk>`. The model then faithfully emitted garbage (unused high‑id tokens), which looked exactly like
  a device/reload/precision failure and consumed hours. The decisive diagnostic was a **CPU‑reference
  run on the same box**: the ground‑truth CPU model produced the *same* garbage, exonerating the
  accelerator and pointing at the input. **Always sanity‑check `tok("hello world").input_ids` before
  suspecting the compiler.**
- **`inf2.xlarge` (16 GB) needs swap.** Serving fits in ~3.6–8 GB, but the one‑time neff load peaks
  ~14.5 GB; a stock DLAMI with no swap OOM‑kills the container *and* the SSM agent. Add a swapfile
  first.
- **Docker images are 70+ GB and cannot be *built* on the `inf2.xlarge`.** The base image plus the
  model files plus swap overflow the root disk mid‑extract. Build the slim tags on an `inf2.8xlarge`
  (700 GB root); the push only uploads the tiny server layer since the base is already on Hub.

---

## 8. Cross‑cutting findings

The three obstacles above were the entry point. The port also produced a set of findings that
generalize beyond Gemma‑4 and were, in aggregate, worth more than any single fix.

### 8.1 "Garbage out" almost never means "the accelerator is broken"

Every time this port produced garbage, the NeuronCore was **innocent**. The causes, in order of how
often they bit: a broken/missing **tokenizer** (§7), a mis‑restored **weight reload** (§8.2), a wrong
**buffer** (`layer_scalar`, §4.6), and a mismatched **driver/SDK** version (a precompiled neff can
*mis‑execute into garbage rather than error* on the wrong runtime). All four are **cheaper to rule out
than a device bug**, and all four *look* like a precision/hardware failure.

The fastest oracle turned out to be a **CPU‑reference run on the same box**: load the model in bf16
with `from_pretrained` (≈5 s off the page cache once the weights are downloaded) and run one forward.
If the CPU reference produces the *same* garbage as the device, the accelerator is exonerated and the
bug is upstream (tokenizer, input, weights). This single technique collapsed a multi‑hour "device
bug" into a two‑minute tokenizer fix. **Reach for the CPU oracle before profiling the neff.**

### 8.2 Validate the *serving path*, not the *trace*

The most dangerous illusion in the whole project was a green `SEQ_MATCH`. Correctness was validated
**in‑process** on the freshly traced model (`ModelBuilder.trace(initialize_model_weights=True)`) — but
the server loads a **saved** model in a **fresh process** and calls
`initialize_with_saved_weights()`. Those are different code paths, and the in‑process pass never
exercised the one the server actually uses. (`torch.jit.load` *without* the init call fails outright —
"This model is not initialized" — so the init step is load‑bearing, not cosmetic.)

Combined with the auto‑port's **false "100% PASS"** against a PLE‑stripped golden (§3), the lesson is
blunt: **a passing test against the wrong oracle, or on the wrong execution path, is worse than a
failing one.** Validate (a) the exact artifact you will ship, (b) in the exact way you will load it,
(c) against an **independent** float reference — not the trace, not a golden derived from the same
broken assumption.

### 8.3 "Effective params" is a capacity lie, and bf16 doesn't rescue the core

E2B is marketed as "2B" and E4B as "4B" — the MatFormer/PLE *effective* parameter counts. But the
**device footprint is the full parameter count** (~5B and ~8B), and that is what has to fit 16 GB. This
is exactly why E4B, the "4B" model, does **not** fit one core while E2B, the "2B" model, does.
**Plan capacity from real parameters, never the effective headline.**

The second half of the trap: **bf16 halves the on‑disk neff but not the on‑device constant count.**
The intuition "just use lower precision to fit" is wrong here — E4B's ~15.4 GB of resident fp32
constants stay ~15.4 GB of *slots* regardless of dtype tricks at the boundaries, and a second neff
still tips past 16 GB. **Tensor parallelism, not precision, is the lever that actually fits the model
on the core** (§4.4).

### 8.4 KV‑sharing is also a *gift*, not only an obstacle

The same cross‑layer KV‑sharing that NxD can't represent (§2.1, §3) is what makes E2B fit one core in
the first place. Only the **15 non‑shared layers allocate a KV buffer**; the rest read through. The
device‑resident cache is therefore far smaller than a naïve one‑buffer‑per‑layer implementation, which
is a large part of why the memory budget closes. An architectural feature that breaks the vendor
abstraction can still be a **net win** once you stop fighting it.

### 8.5 Exploit numerical invariances to fit SBUF

Two of the compiler wins came from *math*, not from the compiler:

- **Monotonic ⇒ argmax‑invariant.** Logit soft‑capping is monotonic, so it cannot change
  `argmax` — greedy decode is identical whether or not it runs. Moving it host‑side (only needed for
  sampling) freed the SBUF it was overflowing (§4.2). Generalize: **any monotonic, per‑element output
  transform can leave the device graph for greedy decode.**
- **Parameters vs buffers.** Frameworks that "load the checkpoint" often mean *parameters*, silently
  skipping **buffers** like `layer_scalar` (§4.6). When a model multiplies by a learned scalar that
  lives in a buffer, that scalar will be wrong after any param‑only load — a bug with no error message,
  only degraded output.

### 8.6 The cheap box is the *same accelerator*

`inf2.xlarge` and `inf2.8xlarge` carry the **identical 2‑NeuronCore / 32 GB‑HBM accelerator**; they
differ only in **host** vCPU (4 vs 32) and RAM (16 vs 128 GB). Since Gemma‑4's transformer runs
entirely on the cores, the only thing standing between the ~4× cheaper box and full performance is
**host memory** — solved by "slim" servers that keep just the embedding table on the CPU and drop the
transformer layers (they live in the neff). Result: **all three models serve on a single
`inf2.xlarge`**, and because throughput barely drops, the cheap box is **cheaper per token**, not just
per hour. (The one non‑obvious requirement is swap: the neff *load* briefly peaks ~14.5 GB on a 16 GB
host — §7.)

### 8.7 Prefill is a workload decision, not a fixed design

Two prefill strategies coexist and trade off cleanly, so the "right" one depends on the workload:

| | host‑seed (`tp_alias`) | device‑prefill (`ModelBuilder`) |
|---|---|---|
| First token | ~1.4–1.6 s (CPU prefill) | **~0.1–0.16 s** |
| Decode | up to ~60 tok/s (E4B slim) | ~36–39 tok/s (E4B) |
| Best for | long generations | chat / first‑token‑bound |

Because the KV cache is **device‑resident** in both, per‑token latency is essentially **flat across
context length** — there is no host round‑trip that grows with the sequence. The knob to turn is
first‑token latency vs sustained decode, chosen per use case, not per model.

## 9. Artifacts

**Hugging Face** (recipe + compiled neffs + Dockerfiles + model cards):
- `xbill9/gemma-4-E2B-it-inferentia2`
- `xbill9/gemma-4-E4B-it-inferentia2`
- `xbill9/gemma-4-12B-it-inferentia2`

**Docker Hub** (prebuilt, run with `--device /dev/neuron0 --ipc=host`):
- `xbill9/gemma4-optb` — E2B: `latest`/`512-128`, `slim`, `tp2-slim`, `tp2-2048`
- `xbill9/gemma4-optb-e4b` — E4B: `latest`/`512-128`, `tp2-devprefill-512`, `slim-devprefill`
- `xbill9/gemma4-optb-12b` — 12B: `tp2-devprefill-256`, `slim-devprefill`

**Key source** (branch `gemma4-inf2-nxd-kvshare`): `optb_kv.py` (E2B two‑graph), `tp_alias_trace.py`
(E4B TP+alias), `tp_mb.py` (E4B/12B device‑prefill `ModelBuilder`), `optb_server_*.py` (full / slim /
device‑prefill HTTP servers with OpenAI routes). Fix history: `e785f6d` (layer_scalar), `d70dc94`
(12B global‑layer sharding), `328ca88` (drop on‑device softcap), `a0a2a75` (force eager attention),
`78967ac` (bf16 weights).

---

## 10. Limitations & future work

- **Batch size 1**, single‑stream greedy/sampled decode. No continuous batching / paged attention —
  the largest gap between "works" and "serves production traffic."
- **Context is baked into the neff.** E2B/E4B ship at 512/128 and 2048/512; 12B at 256/64. Larger
  windows need a recompile (and, on 12B, may not fit the ~4 GB/core headroom).
- **Text only.** The 12B `gemma4_unified` audio (640 samples/40 ms) and image (48×48 patch) projection
  paths exist but are not yet wired to the device graph.
- Compiled specifically for Inferentia2 and the pinned Neuron SDK versions.

Natural next steps: static batching, a 12B 512‑context recompile, a `tp_alias`‑style 12B decode path
(the E4B data suggests ~2× throughput headroom), and wiring the 12B multimodal projections.

---

*Base models © Google, Apache‑2.0. The compiled neffs embed the base weights in bf16 and are
redistributed as Apache‑2.0 derivatives with attribution. The Option‑B recipe, TP sharding rules,
device‑resident KV design, and servers are Apache‑2.0. Not affiliated with or endorsed by Google or
AWS; provided for research/testing.*
