# 📊 Gemma 4 serving Performance Benchmark

This report details the throughput and latency characteristics of the Gemma 4 (E4B-it) model deployed on a Cloud Run instance configured with an NVIDIA L4 GPU in the `us-east4` region.

## 📈 Performance Visualizations

![Gemma 4 Performance Chart](/home/xbill/.gemini/antigravity-cli/brain/cfff7322-2fa5-4792-8342-48ec261708ab/benchmark_chart.png)

---

## 🔍 Key Performance Insights

### 1. Throughput Scaling (Left Chart)
* **High Efficiency at Low Concurrency**: With single-user requests (concurrency = 1), throughput is highly consistent across smaller context sizes.
* **Concurrency Scaling**: As concurrency climbs from 1 to 16, throughput scales up significantly, peaking around **80 to 90 tokens/second** at smaller context windows. This proves the efficacy of vLLM's continuous batching implementation.
* **Large Context Handling**: At the maximum context window size of **16,384 tokens**, throughput remains extremely stable around **25 to 27 tokens/second**, demonstrating that the GPU is fully capable of serving long-context requests under high loads without degradation.

### 2. Latency Characteristics (Right Chart)
* **Sub-Second Response Times**: Across all tested context windows (from 8 to 16,384 tokens), average latency remains well below 1 second for standard workloads (concurrency levels 1 to 32).
* **Predictable Scaling**: Even at maximum concurrency (128 concurrent users) and largest context sizes, latency stays under **1 second** on average, making this stack highly suitable for real-time applications and SRE automated diagnostic workflows.
