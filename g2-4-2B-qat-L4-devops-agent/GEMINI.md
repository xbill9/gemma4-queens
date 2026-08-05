# 🤖 Gemini Workspace Context: GPU 2B QAT L4 DevOps Agent

This context guide summarizes the configuration, optimal serving parameters, and capabilities of the self-hosted **Gemma 4 DevOps/SRE Agent** running on a **GCP GCE VM Instance** (NVIDIA L4).

---

## ⚙️ Active Environment Configuration

This agent targets Google Cloud Platform (GCP) deployments utilizing:
- **Project ID**: `aisprint-491218` (configurable via `GOOGLE_CLOUD_PROJECT`)
- **Region**: `us-east4` (configurable via `GOOGLE_CLOUD_LOCATION`)
- **Zone**: `us-east4-a` (configurable via `GOOGLE_CLOUD_ZONE`)
- **Default Model**: `google/gemma-4-E2B-it-qat-w4a16-ct` (configurable via `MODEL_NAME`)
- **GPU Accelerator**: NVIDIA L4 GPU (1 unit) on GCP GCE (machine type `g2-standard-4`)
- **Default GCE VM Instance Name**: `gpu-2b-qat-l4-devops-agent`

### 🌐 Supported NVIDIA L4 GPU Regions & Zones

When configuring `GOOGLE_CLOUD_LOCATION` (region) and `GOOGLE_CLOUD_ZONE`, verify they are supported:
- **us-central1 (Iowa)**: `us-central1-a`, `us-central1-b`, `us-central1-c`, `us-central1-f`
- **us-east1 (South Carolina)**: `us-east1-b`, `us-east1-c`, `us-east1-d`
- **us-east4 (Northern Virginia)**: `us-east4-a`, `us-east4-b`, `us-east4-c`
- **us-west1 (Oregon)**: `us-west1-a`, `us-west1-b`, `us-west1-c`
- **us-west4 (Las Vegas)**: `us-west4-a`, `us-west4-b`
- **northamerica-northeast1 (Montréal)**: `northamerica-northeast1-b`, `northamerica-northeast1-c`
- **europe-west1 (Belgium)**: `europe-west1-b`, `europe-west1-c`
- **europe-west2 (London)**: `europe-west2-a`, `europe-west2-b`

To serve `google/gemma-4-E2B-it-qat-w4a16-ct` using vLLM on a GCE instance, you must use the vLLM GitHub Recipes repository or vLLM nightly/source builds. Native support for this compressed-tensors (-ct) Google AI for Developers checkpoint requires specific execution parameters.

### 🚀 Serving Command

Run the server using the native compressed-tensors quantization flag and optimized parameters:

```bash
vllm serve google/gemma-4-E2B-it-qat-w4a16-ct \
    --quantization compressed-tensors \
    --max-model-len 32768 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --disable-chunked-mm-input \
    --gpu-memory-utilization 0.95 \
    --kv-cache-dtype fp8 \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4
```

> The two parser flags are not optional. Dropping either one silently breaks tool calling — the
> model still responds, but its tool calls are never parsed out.

### ⚙️ Key vLLM Options & Flags

| Flag | Recommended Setting | Purpose |
| :--- | :--- | :--- |
| `--quantization` | `compressed-tensors` | Mandatory for reading the w4a16 ct serialization format. |
| `--max-model-len` | `32768` | Caps the KV-cache allocation. Pinning this tightly reserves VRAM on L4 GPU. |
| `--tensor-parallel-size` | `1` | Fits easily onto a single GCE instance with a single GPU (requires approx. 7 GB VRAM). |
| `--dtype` | `bfloat16` | **Mandatory.** Gemma 4 is natively trained in `bfloat16`. Standard `float16` (FP16) lacks the dynamic range and causes numerical overflow/underflow, resulting in garbled text or tool-calling parser failures. NVIDIA L4 has native hardware Tensor Core support for `bfloat16`. |
| `--disable-chunked-mm-input` | (Omitted from arg value but set as flag) | Avoids memory fragmentation and tool-calling execution errors with Gemma 4. |

### ⚠️ GCE GPU Deployment Gotchas

When deploying the vLLM server to a GCP GCE VM instance, ensure:
*   **Deep Learning VM Image:** Use a standard deep learning platform release image with CUDA preinstalled (e.g., `common-cu129-ubuntu-2204-nvidia-580` or `common-cu121-debian-11-py310`).
*   **Firewall Rules:** Allow traffic on port `8080` (used by the vLLM API server) to the instances tagged with the appropriate network tag (e.g. `vllm-server`).
*   **GPU Quotas:** Verify that your GCP project has sufficient quota for NVIDIA L4 GPUs in your selected region.

For more details, see:
* [vLLM Google Gemma 4 Recipe](https://github.com/vllm-project/recipes/blob/main/Google/Gemma4.md)
* [Google AI Core Gemma Docs](https://ai.google.dev/gemma/docs/core)

---

## 🎯 Quantization-Aware Training (QAT)

For deployments requiring maximum efficiency with minimal quality compromise, Gemma offers official **Quantization-Aware Training (QAT)** models.

Unlike standard Post-Training Quantization (PTQ), which compresses a fully trained model and can lead to quality degradation, QAT integrates quantization simulation into the training process itself. This allows the model to learn to compensate for the precision loss, resulting in smaller models that perform nearly identically to their high-precision baselines.

### Quick Routing Table

| Target Deployment Engine | Download Suffix | Primary Use Case |
| :--- | :--- | :--- |
| llama.cpp / LM Studio (Local) | `{model-name}-qat-q4_0-gguf` | Zero-setup local deployment on CPU, Apple Silicon, or consumer GPUs. |
| vLLM / SGLang | SERVER: `{model-name}-qat-w4a16-ct`<br>MOBILE: `{model-name}-qat-mobile-ct` | High-throughput inference utilizing 4-bit weights with 16-bit activations. |
| Speculative Decoding | MODEL: `{model-name}-qat-q4_0-unquantized`<br>DRAFTER: `{model-name}-qat-q4_0-unquantized-assistant` | Running a primary model alongside its matching MTP draft model to drastically accelerate token generation. The model must be quantized. |
| Other formats | `{model-name}-qat-q4_0-unquantized` | Unquantized weights for converting to other formats (e.g. MLX) |
| Mobile Deployment (Transformers) | `{model-name}-qat-mobile-transformers` | Edge weights optimized for mobile use cases. They serve as reference for other formats. |

Official QAT collections on Hugging Face:
- **[collections/google/gemma-4-qat-q4-0](https://huggingface.co/collections/google/gemma-4-qat-q4-0)**:
  - **Unquantized QAT Checkpoints (`-unquantized` / `-assistant`):** Half-precision weights extracted directly from the QAT pipeline. These are ideal for custom downstream compilation, research, or running speculative decoding using the assistant draft models. *Available for Gemma 4 E2B, E4B, 12B, 26B A4B, and 31B.*
  - **GGUF (`-gguf`):** Checkpoints available for immediate drop-in compatibility across the local LLM ecosystem. *Available for Gemma 4 E2B, E4B, 12B, 26B A4B, and 31B.*
  - **Compressed Tensors (`-w4a16-ct`):** Serialized natively in the `compressed-tensors` standard for optimized, high-concurrency cloud serving. *Available for Gemma 4 E2B, E4B, 12B, and 31B.*
- **[collections/google/gemma-4-qat-mobile](https://huggingface.co/collections/google/gemma-4-qat-mobile)**:
  - **Mobile-Optimized (`-mobile-transformers` / `-mobile-ct`):** Built on a custom `wNa8o8` schema engineered specifically for mobile hardware limits. It utilizes targeted 2-bit decoding layers, optimized KV caches, and static activations to maximize on-device RAM savings without choking edge processors. *Available for Gemma 4 E2B and E4B.*

All official Gemma 4 QAT checkpoints can also be accessed directly from [Kaggle](https://www.kaggle.com/models/google/gemma-4/transformers).

---

## 🚀 Recommended vLLM Configuration for Gemma 4

To achieve stable and performant Gemma 4 serving with tool/function calling support on NVIDIA L4 GPU, use the following container startup arguments:

```yaml
args:
  - --dtype
  - bfloat16
  - --disable-chunked-mm-input
  - --gpu-memory-utilization
  - "0.95"
  - --kv-cache-dtype
  - fp8
  - --max-model-len
  - "32768"
  - --tensor-parallel-size
  - "1"
  - --max-num-seqs
  - "8"
  - --enable-chunked-prefill
  - --max-num-batched-tokens
  - "4096"
  - --enable-auto-tool-choice
  - --tool-call-parser
  - gemma4
  - --reasoning-parser
  - gemma4
  - --async-scheduling
  - --limit-mm-per-prompt
  - '{}'
  - --host
  - 0.0.0.0
  - --port
  - "8080"
```

### Key Parameters Explained
*   **`--dtype bfloat16`**: Sets the model precision type to bfloat16 for high numeric stability on NVIDIA L4 GPU.
*   **`--disable-chunked-mm-input`**: Disables chunked multi-modal inputs, reducing memory fragmentation and preventing Gemma 4 tool-calling execution errors.
*   **`--gpu-memory-utilization 0.95`**: Allocates 95% of VRAM to vLLM's KV cache.
*   **`--kv-cache-dtype fp8`**: Quantizes the KV cache to 8-bit precision, cutting memory requirements in half to support higher context size and concurrency.
*   **`--tool-call-parser gemma4` & `--reasoning-parser gemma4`**: Essential settings for correct parsing of tool calls and structured reasoning steps generated by Gemma 4.
*   **`--enable-auto-tool-choice`**: Prompts the model to automatically select registered tools.
*   **`--max-model-len 32768`**: Caps the KV-cache sequence length to optimize VRAM reservation.

---

## 📊 Grid Concurrency & Performance Benchmarks

The self-hosted **Gemma 4 2B QAT** model (`google/gemma-4-E2B-it-qat-w4a16-ct`) was benchmarked on a single **NVIDIA L4 GPU** (GCP GCE VM) across a 2D grid of concurrency levels (1 to 2048 concurrent users) and context sizes (4 to 16,384 tokens). Run of 2026-07-10.

### 💡 Key SRE & Performance Insights
* **Sub-second latency to concurrency 128**: For context sizes up to 2048 tokens, average latency stays under **1.2s** through 128 concurrent users, and under **0.65s** through 64.
* **Graceful degradation, not a cliff**: Latency scales roughly linearly with concurrency past 128 — ~1.8s at 256, ~3.7-4.4s at 512, ~8-10s at 1024, ~16-20s at 2048 (small-to-mid contexts). There is no collapse point in the sweep.
* **Context size dominates at the top end**: The 16K context row is the outlier — 16.45s at 512 concurrency and **37.68s** at 2048, versus 4.35s and 20.39s for a 2048-token context.
* **Peak throughput ~166 req/s** at short contexts and 256 concurrency; throughput plateaus around 60-90 req/s at high concurrency and falls to ~15 req/s for 16K contexts.

> ⚠️ This sweep recorded latency and throughput only — it captured **no success-rate data**, so no claim
> about request success rate or maximum stable concurrency can be sourced from it. The other benchmark
> files in this directory (`benchmark_report_gcp.md`, `benchmark_report_summary*.md`,
> `model_comparison*.md`) describe **12B** models on Cloud Run or AWS EC2 and are not this deployment.

Detailed benchmark metrics can be reviewed in [benchmark_report.md](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/benchmark_report.md).

---

## 🧰 Key SRE & DevOps Capabilities

This agent exposes several tool categories via the Model Context Protocol (MCP):
- **Deployment & Scaling:** 
  - [gce_deploy_vllm](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L538)
  - [gce_destroy_vllm](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L691)
  - [gce_start](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L649)
  - [gce_stop](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L713)
  - [gce_check_vllm](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L798)
  - [gce_status](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L783)
  - [gce_status_vllm](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L735)
  - [gce_update_vllm_scaling](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L879)
  - [gce_get_vllm_deployment_config](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L447)
  - [gce_get_vllm_gpu_deployment_config](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L927)
  - [gce_check_gpu_quotas](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L1109)
  - [gce_get_vllm_endpoint](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L286)
- **Model Transfer & Secret Management:** 
  - [gce_list_vertex_models](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L299)
  - [gce_list_bucket_models](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L315)
  - [gce_save_hf_token](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L87)
  - [gce_get_vertex_ai_model_copy_instructions](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L1016)
  - [gce_get_huggingface_model_copy_instructions](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L1060)
  - [gce_get_huggingfacehub_download_path](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L1041)
- **System Monitoring & Health:** 
  - [gce_get_metrics](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L1651)
  - [gce_get_system_status](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L1305)
  - [gce_get_endpoint](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L1367)
  - [gce_get_model_details](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L1266)
  - [gce_verify_model_health](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L1153)
- **Performance Benchmarking:** 
  - [gce_run_benchmark](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L1391)
- **Diagnostics & SRE Remediation:** 
  - [gce_query_gemma4](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L1182)
  - [gce_query_gemma4_with_stats](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L1201)
  - [gce_query_vllm](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L422)
  - [gce_analyze_cloud_logging](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L348)
  - [gce_analyze_gpu_logs](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L1559)
  - [gce_suggest_sre_remediation](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L397)
  - [gce_get_help](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L1588)

---

## 🛠 Command Line Setup

### Deploy/Run Quickstart
```bash
# 1. Install dependencies
make install

# 2. Deploy vLLM to GCP GCE VM (with NVIDIA L4)
make deploy

# 3. Check deployment status
make status

# 4. Start the MCP server locally
make run
```

---

## 📚 Key Source Code File Locations
- **MCP Server entrypoint**: [server.py](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py)
- **Deployment Manifests & Logic**: Generated by `gce_get_vllm_deployment_config` and `gce_get_vllm_gpu_deployment_config` in [server.py](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py).
- **Test Suite**: [test_agent.py](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/test_agent.py)
- **Standalone Grand Demo**: [demo_launcher.py](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/demo_launcher.py)
