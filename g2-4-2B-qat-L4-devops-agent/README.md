# Self-Hosted vLLM DevOps Agent (MCP Server)

This project provides an automated DevOps/SRE assistant that leverages **Gemma models self-hosted via vLLM on GCP Compute Engine (GCE) VM instances**. It bridges GCP infrastructure logs with a private inference endpoint to analyze issues and suggest remediations.

To deploy and run this project, you need to address two main components: the **Inference Stack** (vLLM on GCP GCE) and the **MCP Server** itself.

### 1. Infrastructure Requirements (The Inference Stack)
The MCP server expects a running vLLM instance. Your GCE VM deployment for the model needs:
*   **Hardware:** NVIDIA L4 GPU (1 unit).
*   **Compute:** `g2-standard-4` machine type (4 vCPUs, 16GiB RAM).
*   **Storage:** A GCS Bucket containing the Gemma model weights (e.g., `gs://PROJECT_ID-bucket/gemma-4-E2B-it-qat-w4a16-ct/`) or direct Hugging Face access.
*   **Networking:** Firewall rule allowing traffic on port `8080`.

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
Use the built-in tool `get_vertex_ai_model_copy_instructions` or `get_huggingface_model_copy_instructions` to move Gemma weights to your GCS bucket.

### Step 2: Deploy vLLM to GCP GCE
Run the `deploy_vllm` tool within the MCP server to provision a `g2-standard-4` instance and start the vLLM container, or use the provided [Makefile](file:///home/xbill/gemma4-tips/g2-4-2B-qat-L4-devops-agent/Makefile):
```bash
make deploy
```

> [!IMPORTANT]
> **Critical Deployment Configurations:**
> *   **GCE Machine Type:** Uses `g2-standard-4` which is optimized for NVIDIA L4 GPUs.
> *   **Startup Script:** The deployment uses a startup script to install Docker and launch the `vllm/vllm-openai:nightly` container with optimized Gemma 4 parameters.
> *   **Numeric Stability (`--dtype=bfloat16`):** Gemma 4 models are natively trained and optimized in `bfloat16`. Running them with standard `float16` (FP16) can lead to numerical overflow/underflow, resulting in garbled outputs or tool-calling errors. The NVIDIA L4 GPU natively accelerates `bfloat16` operations via its Tensor Cores.
> *   **Quantization Support:** For `-ct` (compressed-tensors) models, the `--quantization compressed_tensors` flag is mandatory.

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
*   **[start_gce](file:///home/xbill/gemma4-tips/g2-4-2B-qat-L4-devops-agent/server.py#L1567)**: Starts the GCE instance and ensures vLLM is running.
*   **[status_gce](file:///home/xbill/gemma4-tips/g2-4-2B-qat-L4-devops-agent/server.py#L1568)**: Returns the current status of the GCE VM.
*   **[stop_gce](file:///home/xbill/gemma4-tips/g2-4-2B-qat-L4-devops-agent/server.py#L1569)**: Safely stops the GCE VM to save costs.
*   **[check_vllm](file:///home/xbill/gemma4-tips/g2-4-2B-qat-L4-devops-agent/server.py#L1570)**: Validates that the vLLM engine is responsive on the GCE instance.
*   **[deploy_vllm](file:///home/xbill/gemma4-tips/g2-4-2B-qat-L4-devops-agent/server.py#L509)**: Provisions a new GCP GCE VM instance with NVIDIA L4.
*   **[destroy_vllm](file:///home/xbill/gemma4-tips/g2-4-2B-qat-L4-devops-agent/server.py#L1572)**: Deletes the GCE VM instance and associated resources.
*   **[get_vllm_deployment_config](file:///home/xbill/gemma4-tips/g2-4-2B-qat-L4-devops-agent/server.py#L418)**: Generates the `gcloud` command and startup script for manual GCE deployment.
*   **[check_gpu_quotas](file:///home/xbill/gemma4-tips/g2-4-2B-qat-L4-devops-agent/server.py#L783)**: Checks availability for L4 GPUs in the target region.
*   **[get_vllm_endpoint](file:///home/xbill/gemma4-tips/g2-4-2B-qat-L4-devops-agent/server.py#L237)**: Resolves the external IP of the GCE instance.

### 📦 Model Management
*   **[list_vertex_models](file:///home/xbill/gemma4-tips/g2-4-2B-qat-L4-devops-agent/server.py#L251)**: Lists models in the Vertex AI Registry.
*   **[list_bucket_models](file:///home/xbill/gemma4-tips/g2-4-2B-qat-L4-devops-agent/server.py#L267)**: Lists model weights in GCS bucket.
*   **[save_hf_token](file:///home/xbill/gemma4-tips/g2-4-2B-qat-L4-devops-agent/server.py#L49)**: Securely saves a Hugging Face API token to Secret Manager.
*   **[get_vertex_ai_model_copy_instructions](file:///home/xbill/gemma4-tips/g2-4-2B-qat-L4-devops-agent/server.py#L693)**: Guide to transfer Gemma models from Vertex AI Model Garden to GCS.
*   **[get_huggingface_model_copy_instructions](file:///home/xbill/gemma4-tips/g2-4-2B-qat-L4-devops-agent/server.py#L737)**: Guide to transfer Gemma models from Hugging Face and upload to GCS.
*   **[get_huggingfacehub_download_path](file:///home/xbill/gemma4-tips/g2-4-2B-qat-L4-devops-agent/server.py#L718)**: Resolves local cache path using huggingface_hub.

### 📊 Monitoring & Status
*   **[get_metrics](file:///home/xbill/gemma4-tips/g2-4-2B-qat-L4-devops-agent/server.py#L1605)**: Fetches raw Prometheus metrics from the running vLLM service.
*   **[get_system_status](file:///home/xbill/gemma4-tips/g2-4-2B-qat-L4-devops-agent/server.py#L1260)**: Provides a high-level status dashboard of the GCE instance and container health.
*   **[get_endpoint](file:///home/xbill/gemma4-tips/g2-4-2B-qat-L4-devops-agent/server.py#L1322)**: Verifies connectivity and returns the active service URL.
*   **[get_model_details](file:///home/xbill/gemma4-tips/g2-4-2B-qat-L4-devops-agent/server.py#L1221)**: Retrieves detailed model metadata and engine state from `/v1/models`.
*   **[verify_model_health](file:///home/xbill/gemma4-tips/g2-4-2B-qat-L4-devops-agent/server.py#L824)**: Deep health check by querying the model with a simple prompt and measuring latency.

### 📈 Performance & Benchmarking
*   **[run_benchmark](file:///home/xbill/gemma4-tips/g2-4-2B-qat-L4-devops-agent/server.py#L1066)**: Runs performance/concurrency benchmark sweeps against the Cloud Run vLLM GPU endpoint.

### 💬 Interaction & Diagnostics
*   **[query_gemma4](file:///home/xbill/gemma4-tips/g2-4-2B-qat-L4-devops-agent/server.py#L853)**: Primary tool to query the self-hosted model with standard chat message format.
*   **[query_gemma4_with_stats](file:///home/xbill/gemma4-tips/g2-4-2B-qat-L4-devops-agent/server.py#L872)**: Queries the model and returns streaming performance statistics (TTFT, throughput).
*   **[query_vllm](file:///home/xbill/gemma4-tips/g2-4-2B-qat-L4-devops-agent/server.py#L368)**: Direct text completions querying tool.
*   **[analyze_cloud_logging](file:///home/xbill/gemma4-tips/g2-4-2B-qat-L4-devops-agent/server.py#L297)**: Fetches logs from GCP Logging and analyzes them using the model.
*   **[analyze_gpu_logs](file:///home/xbill/gemma4-tips/g2-4-2B-qat-L4-devops-agent/server.py#L1514)**: Fetches GCE service logs and uses Gemma 4 to analyze them for SRE/DevOps errors.
*   **[suggest_sre_remediation](file:///home/xbill/gemma4-tips/g2-4-2B-qat-L4-devops-agent/server.py#L343)**: Suggests remediation plans for SRE errors using the model.
*   **[get_help](file:///home/xbill/gemma4-tips/g2-4-2B-qat-L4-devops-agent/server.py#L1543)**: Provides help text and summarizes the configuration options and all available SRE/DevOps tools.

## 📦 Resources
The server exposes the following MCP resources:
*   **`config://vllm-deployment-template`**: A YAML template for Cloud Run GPU deployment.

## 📊 Performance Benchmarks (Standard vs. QAT)

The self-hosted **Gemma 4 2B QAT** model has been benchmarked on a single **NVIDIA L4 GPU** (GCP GCE VM) to measure concurrency limits:
* **High Concurrency Stability**: The QAT INT4 model maintains a **100% request success rate** up to **512 concurrent users** (with context windows up to 2048 tokens).
* **The QAT Advantage**: The standard 2B model (bfloat16) leaves 0 GB of free VRAM for the KV cache on a single L4 GPU, failing at concurrencies above 8. The QAT model (w4a16) frees up **~18 GB of VRAM** for the KV cache, representing a **~64x improvement in concurrency capacity**.
* Detailed matrix results and SRE insights are available in [benchmark_report_summary_gcp.md](file:///home/xbill/gemma4-tips/g2-4-2B-qat-L4-devops-agent/benchmark_report_summary_gcp.md).

## 🌟 Grand Demo
A standalone demo script is included to showcase the agent's capabilities:
```bash
python demo_launcher.py
```
This script simulates log analysis, remediation suggestions, and infrastructure configuration generation.

## 🛠 Makefile Helpers
The included [Makefile](file:///home/xbill/gemma4-tips/g2-4-2B-qat-L4-devops-agent/Makefile) provides several shortcuts:
*   `make install`: Installs Python dependencies listed in [requirements.txt](file:///home/xbill/gemma4-tips/g2-4-2B-qat-L4-devops-agent/requirements.txt).
*   `make run`: Starts the MCP server via [server.py](file:///home/xbill/gemma4-tips/g2-4-2B-qat-L4-devops-agent/server.py).
*   `make deploy`: Deploys vLLM to GCP GCE with GPU.
*   `make destroy`: Removes the vLLM GCE VM instance.
*   `make status`: Checks the status of the vLLM GCE service.
*   `make query PROMPT="your prompt"`: Queries the vLLM model directly via `curl`.
*   `make test`: Runs the test suite in [test_agent.py](file:///home/xbill/gemma4-tips/g2-4-2B-qat-L4-devops-agent/test_agent.py).

## 🧪 Testing
Run the included test suite in [test_agent.py](file:///home/xbill/gemma4-tips/g2-4-2B-qat-L4-devops-agent/test_agent.py) to verify the tool registration and basic functionality:
```bash
make test
```
