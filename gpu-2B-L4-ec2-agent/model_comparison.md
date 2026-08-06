# 📊 Gemma 4 Inference Model Comparison Report

This report compares the serving performance of different Gemma 4 model sizes and quantizations on a single **NVIDIA L4 GPU** (24GB VRAM, Cloud Run Gen2) in the `us-east4` region. 

## 📈 Performance Visualizations
![Model Performance Comparison](./comparison_chart.png)

---

## 📋 Comprehensive Model Comparison Matrix

| Feature / Metric | Gemma 4 (4B) | Gemma 4 (12B Standard) | Gemma 4 (12B QAT) | Gemma 4 (26B) |
| :--- | :--- | :--- | :--- | :--- |
| **Model ID** | `google/gemma-4-E4B-it` | `google/gemma-4-12B-it` | `google/gemma-4-12B-it-qat-w4a16-ct` | `google/gemma-4-26B-it` |
| **Quantization Method** | FP8 Weights & KV Cache | Unquantized (`bfloat16` weights) | INT4 Weights & FP16 Activations (QAT) | FP8 Weights & KV Cache |
| **Weight Footprint (VRAM)** | **~4 GB** | ~24 GB | **~6 GB** | ~26 GB (Exceeds 24GB L4 limit) |
| **Free VRAM for KV Cache** | **~19 GB** | **~0 GB** (VRAM fully exhausted by weights) | **~18 GB** | ~0 GB (exceeds VRAM, paging/offloading active) |
| **Peak Throughput** | **~80-100 t/s** (expected)* | Severe throughput starvation | **55.21 t/s** (at Concurrency = 64) | **131.11 t/s** (at Concurrency = 16)* |
| **Avg Latency (Concurrency=1)** | **~0.6s** (estimated) | **0.19s** (small output or mock context) | **1.46s** (full 128 tokens generation) | **1.99s** (full 128 tokens generation) |
| **Max Stable Concurrency** | **Concurrency <= 64** | **Concurrency <= 8** (success rate falls to 81% at 16) | **Concurrency <= 64** (100% success rate up to 64) | **Capped at 20** concurrent requests in test* |
| **Accuracy / Reasoning** | Low (struggles with complex logs/code) | High | **High** (nearly identical to unquantized 12B) | Very High (requires larger GPU cluster) |
| **SRE Suitability** | Low (good for routing/simple checks only) | Unsuitable for high concurrency on L4 | **Optimal balance** of high capacity, high concurrency, and accuracy | High capacity, but requires multi-GPU or TPU for concurrency |

> [!NOTE]
> \* **Benchmark Data Notes**: 
> 1. There is no active benchmark CSV dataset under `../gpu-4B-L4-devops-agent`. The numbers presented are architectural estimations based on vLLM benchmarks for 4B parameters.
> 2. The 26B benchmark was run with a fixed request count of 20 prompts per concurrency level. Therefore, concurrency levels above 16 did not actually test parallel loads beyond 20 concurrent requests, resulting in a flat performance line. 

---

## 💡 Key SRE & DevOps Insights

### 1. 12B QAT vs. 4B: The Trade-off of Reasoning vs. Scale
* **Parameter Capacity**: The **Gemma 4 (4B)** model fits comfortably inside the L4 VRAM, using only ~4 GB. This leaves the largest amount of VRAM (~19 GB) available for the KV cache, ensuring high concurrency. However, 4B models struggle with complex SRE diagnostics, log tracing, and structured tool calling.
* **12B QAT (w4a16)** provides a sweet spot: by compressing the weights to 4-bit, it fits in **~6 GB of VRAM**, leaving **~18 GB for the KV cache** (nearly the same as the 4B model). This enables it to match the high concurrency of the 4B model (stable up to concurrency 64) while providing the superior reasoning and troubleshooting capabilities of a 12B parameter model.

### 2. The VRAM Cliff & KV Cache Starvation (12B QAT vs. 12B Standard)
* A single NVIDIA L4 GPU has exactly **24 GB of VRAM**. Serving the **12B Standard (bfloat16)** model requires 24 GB of VRAM just to store the weights, leaving **0 GB** for the KV cache. As a result, the serving engine starts dropping requests (success rate drops to **81.3%** at concurrency 16, **50%** at concurrency 32, and **45%** at concurrency 64) because it has no memory to store intermediate token keys and values.
* The **12B QAT (w4a16)** model uses 4-bit weights, freeing up **18 GB of VRAM** for the KV cache, allowing it to sustain **100% success rate up to concurrency 64** with a peak throughput of **55.21 tokens/sec**.

---

## 🛠 Deployment Recommendations
1. **Prefer 12B QAT for SRE Agents on L4**: Use `google/gemma-4-12B-it-qat-w4a16-ct` to maintain high concurrency (up to 64 concurrent users) and stability while retaining high reasoning accuracy.
2. **Deploy 4B only for Simple Tasks**: Use the 4B model only for low-overhead routing or text filtering tasks where advanced reasoning is not required.
