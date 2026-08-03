# Self-Hosted vLLM DevOps Agent (MCP Server)

This project provides an automated DevOps/SRE assistant that leverages **Gemma models self-hosted via vLLM on Cloud Run GPU**. It bridges Google Cloud Logging with a private inference endpoint to analyze infrastructure issues and suggest remediations.

## 🚀 Deployment Requirements

To deploy and run this project, you need to address two main components: the **Inference Stack** (vLLM on Cloud Run) and the **MCP Server** itself.

### 1. Infrastructure Requirements (The Inference Stack)
The MCP server expects a running vLLM instance. Your Cloud Run deployment for the model needs:
*   **Hardware:** NVIDIA L4 GPU (1 unit).
*   **Compute:** Minimum 4 vCPUs and 16GiB RAM.
*   **Execution Environment:** `gen2` (required for GPU and GCS FUSE).
*   **Storage:** A GCS Bucket containing the Gemma model weights (e.g., `gs://PROJECT_ID-bucket/gemma-4-E4B-it/`).
*   **Networking:** Private Google Access must be enabled on the VPC subnet if using GCS FUSE.

### 2. Software & API Dependencies
The agent relies on several Google Cloud services and Python libraries:
*   **Libraries:** `mcp`, `fastmcp`, `google-cloud-logging`, `google-cloud-aiplatform`, `google-cloud-storage`, `google-adk`, `huggingface_hub`, and `requests`.
*   **Permissions:** The service account running the agent needs:
    *   `logging.logEntries.list` (to read logs).
    *   `aiplatform.models.list` (to list Vertex AI models).
    *   Access to the vLLM endpoint (either public with auth or via VPC).

### 3. Environment Variables
You can configure the following variables for the MCP server:
*   `GOOGLE_CLOUD_PROJECT`: Your GCP Project ID (defaults to `aisprint-491218`).
*   `GOOGLE_CLOUD_LOCATION`: The region for Vertex AI (defaults to `us-east4`).
*   `VLLM_BASE_URL`: The URL of your Cloud Run vLLM service. **If omitted, the agent will attempt to auto-discover it using `gcloud`.**
*   `MODEL_NAME`: The model identifier used by vLLM (defaults to `google/gemma-4-E4B-it`).

## 🛠 Usage & Setup

### Step 1: Prepare Model Weights
Use the built-in tool `get_vertex_ai_model_copy_instructions` or `get_huggingface_model_copy_instructions` to move Gemma weights to your GCS bucket.

### Step 2: Deploy vLLM to Cloud Run
Run the `get_vllm_deployment_config` tool within the MCP server to generate the exact `gcloud` command for deployment, or use the provided `Makefile`:
```bash
make deploy
```

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
*   **`deploy_vllm`**: Deploys vLLM to Cloud Run GPU (NVIDIA L4 in us-east4).
*   **`destroy_vllm`**: Deletes the Cloud Run vLLM service.
*   **`status_vllm`**: Checks the status of the Cloud Run vLLM service.
*   **`update_vllm_scaling`**: Updates min/max instances for scaling.
*   **`get_vllm_deployment_config`**: Generates the `gcloud` deployment command.
*   **`get_vllm_gpu_deployment_config`**: Generates a GKE manifest for GPU (NVIDIA L4).
*   **`check_gpu_quotas`**: Checks L4 and other GPU quotas for a region.
*   **`get_vllm_endpoint`**: Returns the current active vLLM endpoint URL.

### 📦 Model Management
*   **`list_vertex_models`**: Lists models in the Vertex AI Registry.
*   **`list_bucket_models`**: Lists model weights in GCS bucket.
*   **`save_hf_token`**: Securely saves a Hugging Face API token to Secret Manager.
*   **`get_vertex_ai_model_copy_instructions`**: Guide to transfer Gemma models from Vertex AI Model Garden to GCS.
*   **`get_huggingface_model_copy_instructions`**: Guide to transfer Gemma models from Hugging Face and upload to GCS.
*   **`get_huggingfacehub_download_path`**: Resolves local cache path using huggingface_hub.

### 📊 Monitoring & Status
*   **`get_system_status`**: Provides a high-level status dashboard of the Cloud Run service and health.
*   **`get_endpoint`**: Verifies connectivity and returns the active service URL.
*   **`get_model_details`**: Retrieves detailed model metadata and engine state from `/v1/models`.
*   **`verify_model_health`**: Deep health check by querying the model with a simple prompt and measuring latency.

### 📈 Performance & Benchmarking
*   **`run_benchmark`**: Runs performance/concurrency benchmark sweeps against the Cloud Run vLLM GPU endpoint.

### 💬 Interaction & Diagnostics
*   **`query_gemma4`**: Primary tool to query the self-hosted model with standard chat message format.
*   **`query_gemma4_with_stats`**: Queries the model and returns streaming performance statistics (TTFT, throughput).
*   **`query_vllm`**: Direct text completions querying tool.
*   **`analyze_cloud_logging`**: Fetches logs from GCP Logging and analyzes them using the model.
*   **`analyze_gpu_logs`**: Fetches Cloud Run logs and uses Gemma 4 to analyze them for SRE/DevOps errors.
*   **`suggest_sre_remediation`**: Suggests remediation plans for SRE errors using the model.
*   **`get_help`**: Provides help text and summarizes the configuration options and all available SRE/DevOps tools.

## 📦 Resources
The server exposes the following MCP resources:
*   **`config://vllm-deployment-template`**: A YAML template for Cloud Run GPU deployment.

## 🌟 Grand Demo
A standalone demo script is included to showcase the agent's capabilities:
```bash
python demo_launcher.py
```
This script simulates log analysis, remediation suggestions, and infrastructure configuration generation.

## 🛠 Makefile Helpers
The included `Makefile` provides several shortcuts:
*   `make install`: Installs Python dependencies.
*   `make run`: Starts the MCP server.
*   `make deploy`: Deploys vLLM to Cloud Run with GPU.
*   `make destroy`: Removes the vLLM Cloud Run service.
*   `make status`: Checks the status of the vLLM service.
*   `make query PROMPT="your prompt"`: Queries the vLLM model directly via `curl`.
*   `make test`: Runs the test suite.

## 🧪 Testing
Run the included test suite to verify the tool registration and basic functionality:
```bash
make test
```
