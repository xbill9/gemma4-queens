# Self-Hosted vLLM DevOps Agent (MCP Server)

This project provides an automated DevOps/SRE assistant that leverages **Gemma models self-hosted via vLLM on GCP Compute Engine (GCE) VM instances**. It bridges GCP infrastructure logs with a private inference endpoint to analyze issues and suggest remediations.

To deploy and run this project, you need to address two main components: the **Inference Stack** (vLLM on GCP GCE) and the **MCP Server** itself.

### 1. Infrastructure Requirements (The Inference Stack)
The MCP server expects a running vLLM instance. Your GCE VM deployment for the model needs:
*   **Hardware:** NVIDIA L4 GPU (1 unit).
*   **Compute:** `g2-standard-4` machine type (4 vCPUs, 16GiB RAM).
*   **Storage:** A GCS Bucket containing the Gemma model weights (e.g., `gs://PROJECT_ID-bucket/gemma-4-E2B-it-qat-w4a16-ct/`) or direct Hugging Face access.
*   **Networking:** Firewall rule allowing traffic on port `8080`.

### 2. Agent Requirements (The MCP Server)
*   **Runtime:** Python 3.13.
*   **Libraries:** `mcp`, `fastmcp`, `google-cloud-logging`, `google-cloud-aiplatform`, `google-cloud-storage`, `google-adk`, `huggingface_hub`, `openai`, `httpx`, and `python-dotenv`.
*   **Permissions:** The service account running the agent needs:
    *   `logging.logEntries.list` (to read logs).
    *   `aiplatform.models.list` (to list Vertex AI models).
    *   `compute.instances.get` and `compute.instances.list` (to discover vLLM endpoints).
    *   Access to the GCE instance (via external IP or VPC).

### 3. Environment Variables
You can configure the following variables for the MCP server:
*   `GOOGLE_CLOUD_PROJECT`: Your GCP Project ID (defaults to `aisprint-491218`).
*   `GOOGLE_CLOUD_LOCATION`: The region for Vertex AI (defaults to `us-east4`).
*   `GOOGLE_CLOUD_ZONE`: The zone for GCE deployment (defaults to `us-east4-a`).
*   `VLLM_BASE_URL`: The URL of your GCE vLLM service. **If omitted, the agent will attempt to auto-discover it using the GCE external IP.**
*   `MODEL_NAME`: The model identifier used by vLLM (defaults to `google/gemma-4-E2B-it-qat-w4a16-ct`).

## 🛠 Usage & Setup

### Step 1: Prepare Model Weights
Use the built-in tool `gce_get_vertex_ai_model_copy_instructions` or `gce_get_huggingface_model_copy_instructions` to move Gemma weights to your GCS bucket.

### Step 2: Deploy vLLM to GCP GCE
Run the `gce_deploy_vllm` tool within the MCP server to provision a `g2-standard-4` instance and start the vLLM container, or use the provided [Makefile](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/Makefile):
```bash
make deploy
```

> [!IMPORTANT]
> **Critical Deployment Configurations:**
> *   **GCE Machine Type:** Uses `g2-standard-4` which is optimized for NVIDIA L4 GPUs.
> *   **Startup Script:** The deployment uses a startup script to install Docker and launch the `vllm/vllm-openai:nightly` container with optimized Gemma 4 parameters.
> *   **Numeric Stability (`--dtype=bfloat16`):** Gemma 4 models are natively trained and optimized in `bfloat16`. Running them with standard `float16` (FP16) can lead to numerical overflow/underflow, resulting in garbled outputs or tool-calling errors. The NVIDIA L4 GPU natively accelerates `bfloat16` operations via its Tensor Cores.
> *   **Quantization Support:** For `-ct` (compressed-tensors) models, the `--quantization compressed-tensors` flag is mandatory.
> *   **Tool Calling (`--tool-call-parser gemma4` and `--reasoning-parser gemma4`):** Both are required. Dropping either one silently breaks tool use — the model answers, but its tool calls are never parsed out.

### Step 3: Run the MCP Server
Install dependencies and run the server:
```bash
make install
# Optional: export VLLM_BASE_URL="your-vllm-url"
make run
```

## 🛠 Available Tools

The following tools are available via the MCP server:

### 🐳 Infrastructure & Deployment
*   **[gce_start](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L649)**: Starts the GCE instance and ensures vLLM is running.
*   **[gce_status_vllm](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L735)**: Reports the GCE VM's type, state, public IP, zone and launch time.
*   **[gce_status](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L783)**: Alias for `gce_status_vllm`.
*   **[gce_stop](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L713)**: Safely stops the GCE VM to save costs.
*   **[gce_check_vllm](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L798)**: Validates that the vLLM engine is responsive on the GCE instance.
*   **[gce_deploy_vllm](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L538)**: Provisions a new GCP GCE VM instance with NVIDIA L4.
*   **[gce_destroy_vllm](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L691)**: Deletes the GCE VM instance and associated resources.
*   **[gce_get_vllm_deployment_config](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L447)**: Generates the `gcloud` command and startup script for manual GCE deployment.
*   **[gce_get_vllm_gpu_deployment_config](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L927)**: Generates a GKE manifest and node-pool command for vLLM on NVIDIA L4.
*   **[gce_update_vllm_scaling](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L879)**: Scales the GCE instance vertically to a different machine type and restarts it.
*   **[gce_check_gpu_quotas](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L1109)**: Checks availability for L4 GPUs in the target region.
*   **[gce_get_vllm_endpoint](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L286)**: Resolves the external IP of the GCE instance.

### 📦 Model Management
*   **[gce_list_vertex_models](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L299)**: Lists models in the Vertex AI Registry.
*   **[gce_list_bucket_models](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L315)**: Lists model weights in GCS bucket.
*   **[gce_save_hf_token](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L87)**: Securely saves a Hugging Face API token to Secret Manager.
*   **[gce_get_vertex_ai_model_copy_instructions](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L1016)**: Guide to transfer Gemma models from Vertex AI Model Garden to GCS.
*   **[gce_get_huggingface_model_copy_instructions](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L1060)**: Guide to transfer Gemma models from Hugging Face and upload to GCS.
*   **[gce_get_huggingfacehub_download_path](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L1041)**: Resolves local cache path using huggingface_hub.

### 📊 Monitoring & Status
*   **[gce_get_metrics](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L1651)**: Fetches raw Prometheus metrics from the running vLLM service.
*   **[gce_get_system_status](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L1305)**: Provides a high-level status dashboard of the GCE instance and container health.
*   **[gce_get_endpoint](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L1367)**: Verifies connectivity and returns the active service URL.
*   **[gce_get_model_details](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L1266)**: Retrieves detailed model metadata and engine state from `/v1/models`.
*   **[gce_verify_model_health](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L1153)**: Deep health check by querying the model with a simple prompt and measuring latency.

### 📈 Performance & Benchmarking
*   **[gce_run_benchmark](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L1391)**: Runs performance/concurrency benchmark sweeps against the vLLM GPU endpoint.

### 💬 Interaction & Diagnostics
*   **[gce_query_gemma4](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L1182)**: Primary tool to query the self-hosted model with standard chat message format.
*   **[gce_query_gemma4_with_stats](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L1201)**: Queries the model and returns streaming performance statistics (TTFT, throughput).
*   **[gce_query_vllm](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L422)**: Direct text completions querying tool.
*   **[gce_analyze_cloud_logging](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L348)**: Fetches logs from GCP Logging and analyzes them using the model.
*   **[gce_analyze_gpu_logs](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L1559)**: Fetches GCE service logs and uses Gemma 4 to analyze them for SRE/DevOps errors.
*   **[gce_suggest_sre_remediation](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L397)**: Suggests remediation plans for SRE errors using the model.
*   **[gce_get_help](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py#L1588)**: Provides help text and summarizes the configuration options and all available SRE/DevOps tools.

## 📦 Resources
The server exposes the following MCP resources:
*   **`config://vllm-deployment-template`**: The baseline machine type, image and `docker run` command for a GCE L4 GPU vLLM deployment.

## 📊 Performance Benchmarks

The self-hosted **Gemma 4 2B QAT** model (`google/gemma-4-E2B-it-qat-w4a16-ct`) was swept on a single **NVIDIA L4 GPU** (GCP GCE VM) across concurrency 1–2048 and context sizes 4–16,384 tokens. Run of 2026-07-10:

* **Sub-second latency to concurrency 128**: for contexts up to 2048 tokens, average latency stays under **1.2s** at 128 concurrent users and under **0.65s** at 64.
* **Linear degradation past that**: ~1.8s at 256, ~3.7–4.4s at 512, ~8–10s at 1024, ~16–20s at 2048. No collapse point appears within the sweep.
* **Context size dominates at the top end**: the 16K row reaches **16.45s** at 512 concurrency and **37.68s** at 2048, against 4.35s and 20.39s for a 2048-token context.
* **Peak throughput ~166 req/s** at short contexts and 256 concurrency, settling to 60–90 req/s under heavy concurrency and ~15 req/s at 16K contexts.

> [!NOTE]
> This sweep captured latency and throughput only — **no success-rate data** — so it supports no claim about request success rate or maximum stable concurrency. The other benchmark files in this directory (`benchmark_report_gcp.md`, `benchmark_report_summary*.md`, `model_comparison*.md`, and the charts) measure **12B** models on Cloud Run or AWS EC2 and do not describe this deployment.

Full latency and throughput matrices: [benchmark_report.md](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/benchmark_report.md).

## 🌟 Grand Demo
A standalone demo script is included to showcase the agent's capabilities:
```bash
python demo_launcher.py
```
This script simulates log analysis, remediation suggestions, and infrastructure configuration generation.

## 🛠 Makefile Helpers
The included [Makefile](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/Makefile) provides several shortcuts:
*   `make install`: Installs Python dependencies listed in [requirements.txt](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/requirements.txt).
*   `make run`: Starts the MCP server via [server.py](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/server.py).
*   `make deploy`: Deploys vLLM to GCP GCE with GPU.
*   `make destroy`: Removes the vLLM GCE VM instance.
*   `make status`: Checks the status of the vLLM GCE service.
*   `make query PROMPT="your prompt"`: Queries the vLLM model directly via `curl`.
*   `make test`: Runs the test suite in [test_agent.py](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/test_agent.py).

## 🧪 Testing
Run the included test suite in [test_agent.py](file:///home/xbill/gemma4-queens/g2-4-2B-qat-L4-devops-agent/test_agent.py) to verify the tool registration and basic functionality:
```bash
make test
```
