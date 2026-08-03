import asyncio
import csv
import json
import logging
import os
import statistics
import subprocess
import sys
import time
from typing import Optional

import httpx
from dotenv import load_dotenv
load_dotenv(override=True)

from google.cloud import aiplatform, secretmanager, storage
from google.cloud import logging as cloud_logging
from mcp.server.fastmcp import FastMCP
from openai import AsyncOpenAI

# Setup logging to stderr ONLY to avoid interfering with MCP stdio communication
logging.basicConfig(
    stream=sys.stderr, level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("vllm-devops-agent")
logger.info("Initializing DevOps Agent MCP Server...")

# Initialize FastMCP server
mcp = FastMCP("Self-Hosted vLLM DevOps Agent")

# Load AWS credentials if .aws_creds exists
aws_creds_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".aws_creds")
if os.path.exists(aws_creds_path):
    with open(aws_creds_path, "r") as f:
        for line in f:
            if "=" in line:
                key, val = line.strip().split("=", 1)
                os.environ[key] = val

# Configuration
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "aisprint-491218")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-east4")
ZONE = os.getenv("GOOGLE_CLOUD_ZONE", "us-east4-a")
BUCKET_NAME = f"{PROJECT_ID}-bucket"

# The URL of the self-hosted vLLM service on Cloud Run or GCP GCE
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-4-E2B-it-qat-w4a16-ct")
HF_SECRET_ID = "hf-token"


async def run_gcloud(cmd: list[str]) -> tuple[int, str, str]:
    """Helper to run a gcloud command asynchronously and return (returncode, stdout, stderr)."""
    try:
        process = await asyncio.create_subprocess_exec(
            "gcloud", *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        rc = process.returncode if process.returncode is not None else -1
        return rc, stdout.decode("utf-8").strip(), stderr.decode("utf-8").strip()
    except Exception as e:
        return -1, "", str(e)


async def get_secret(secret_id: str = HF_SECRET_ID) -> Optional[str]:
    """Retrieves a secret from GCP Secret Manager or environment variables."""
    # 1. Check environment variable
    val = os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY")
    if val:
        return val

    # 2. Check GCP Secret Manager
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest"
        response = await asyncio.to_thread(client.access_secret_version, request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        logger.debug(f"GCP Secret Manager failed: {e}")

    return None


@mcp.tool()
async def save_hf_token(token: str) -> str:
    """Securely saves a Hugging Face API token to GCP Secret Manager."""
    saved_gcp = False

    try:
        client = secretmanager.SecretManagerServiceClient()
        secret_parent = f"projects/{PROJECT_ID}/secrets/{HF_SECRET_ID}"
        try:
            await asyncio.to_thread(client.get_secret, request={"name": secret_parent})
        except Exception:
            await asyncio.to_thread(
                client.create_secret,
                request={
                    "parent": f"projects/{PROJECT_ID}",
                    "secret_id": HF_SECRET_ID,
                    "secret": {"replication": {"automatic": {}}},
                },
            )
        await asyncio.to_thread(
            client.add_secret_version,
            request={"parent": secret_parent, "payload": {"data": token.encode("UTF-8")}},
        )
        saved_gcp = True
    except Exception as e:
        logger.warning(f"GCP Secret Manager failed: {e}")

    if saved_gcp:
        return "✅ Token saved to GCP Secret Manager."
    else:
        return "❌ Failed to save token to Secret Manager (GCP failed)."


DEFAULT_SERVICE_NAME = "gpu-2b-qat-l4-devops-agent"


def discover_vllm_url(service_name: str = DEFAULT_SERVICE_NAME) -> Optional[str]:
    """Attempts to automatically discover the GCP GCE instance external IP."""
    if VLLM_BASE_URL:
        logger.info(f"Using provided VLLM_BASE_URL: {VLLM_BASE_URL}")
        return VLLM_BASE_URL

    logger.info(f"Attempting to discover GCP GCE external IP for instance: {service_name}")
    try:
        cmd = [
            "compute",
            "instances",
            "describe",
            service_name,
            f"--project={PROJECT_ID}",
            f"--zone={ZONE}",
            "--format=value(networkInterfaces[0].accessConfigs[0].natIP)",
        ]
        process = subprocess.run(["gcloud"] + cmd, capture_output=True, text=True, timeout=15)
        if process.returncode == 0:
            ip = process.stdout.strip()
            if ip:
                url = f"http://{ip}:8080"
                logger.info(f"📡 Automatically discovered GCP GCE vLLM at: {url}")
                return url
            else:
                logger.warning("⚠️ gcloud returned empty IP for GCE instance.")
        else:
            logger.warning(f"⚠️ gcloud failed to discover GCE IP (code {process.returncode}): {process.stderr.strip()}")
    except subprocess.TimeoutExpired:
        logger.warning("⚠️ Discovery timed out after 15 seconds.")
    except Exception as e:
        logger.warning(f"⚠️ Error during GCE vLLM discovery: {str(e)}")

    logger.error("❌ Failed to discover service URL.")
    return None


# Resolve base URL at runtime
_ACTIVE_VLLM_URL = None


def get_vllm_url() -> str:
    """Returns the cached vLLM URL or discovers it if needed."""
    global _ACTIVE_VLLM_URL
    if not _ACTIVE_VLLM_URL:
        _ACTIVE_VLLM_URL = discover_vllm_url()

    if not _ACTIVE_VLLM_URL:
        raise Exception(
            "Could not determine vLLM service URL. Ensure you are authenticated and the service/instance exists."
        )

    return _ACTIVE_VLLM_URL


def get_auth_token() -> str:
    """For GCE VM, returning empty token since access is direct via firewall-opened port 8080."""
    return ""


async def get_vllm_client() -> AsyncOpenAI:
    """Initializes and returns an AsyncOpenAI client for the Cloud Run vLLM service."""
    vllm_url = get_vllm_url()
    token = get_auth_token()
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return AsyncOpenAI(
        base_url=f"{vllm_url}/v1",
        api_key=token or "not-needed",
        default_headers=headers,
    )


async def get_active_model_name(client: AsyncOpenAI) -> str:
    """Queries the vLLM endpoint to find the active model name, or falls back to configuration."""
    try:
        models_response = await client.models.list()
        if models_response.data:
            return models_response.data[0].id
    except Exception as e:
        logger.warning(f"⚠️ Failed to dynamically query active model from vLLM: {e}")

    # Fallback
    if "/" not in MODEL_NAME:
        return f"/mnt/models/{MODEL_NAME}"
    return MODEL_NAME


# Initialize Vertex AI SDK
aiplatform.init(project=PROJECT_ID, location=LOCATION)


@mcp.resource("config://vllm-deployment-template")
def get_deployment_template() -> str:
    """Returns a base template for AWS EC2 L4 GPU vLLM deployment."""
    return """
# AWS EC2 vLLM Deployment Template
# Required Instance: g6.2xlarge (1x NVIDIA L4 GPU, 24GB VRAM)
# Recommended AMI: Deep Learning OSS Nvidia Driver AMI GPU PyTorch (Ubuntu 22.04)

InstanceType: g6.2xlarge
ImageId: ami-012ba162b9cd2729c (us-east-1)
Ports:
  - Container Port: 8080
  - Host Port: 8080

Docker Run Command:
docker run -d --name vllm-server \\
  --gpus all \\
  --ipc=host \\
  --restart always \\
  -p 8080:8080 \\
  -e HF_TOKEN=$HF_TOKEN \\
  vllm/vllm-openai:nightly \\
  --model google/gemma-4-E2B-it-qat-w4a16-ct \\
  --quantization compressed-tensors \\
  --dtype bfloat16 \\
  --max-model-len 32768 \\
  --disable-chunked-mm-input \\
  --gpu-memory-utilization 0.95 \\
  --kv-cache-dtype fp8 \\
  --tensor-parallel-size 1 \\
  --max-num-seqs 8 \\
  --enable-chunked-prefill \\
  --max-num-batched-tokens 4096 \\
  --enable-auto-tool-choice \\
  --tool-call-parser gemma4 \\
  --reasoning-parser gemma4 \\
  --async-scheduling \\
  --limit-mm-per-prompt '{}' \\
  --host 0.0.0.0 \\
  --port 8080
"""


@mcp.tool()
def get_vllm_endpoint(service_name: str = DEFAULT_SERVICE_NAME) -> Optional[str]:
    """
    Returns the current active vLLM endpoint URL.

    Args:
        service_name: The service name or instance Name tag to describe (defaults to 'gpu-2b-qat-l4-devops-agent').
    """
    if service_name == DEFAULT_SERVICE_NAME:
        return get_vllm_url()
    return discover_vllm_url(service_name)


@mcp.tool()
def list_vertex_models() -> str:
    """
    Uses the Vertex AI SDK (part of ADK ecosystem) to list models in the project registry.
    """
    try:
        models = aiplatform.Model.list()
        if not models:
            return "No models found in Vertex AI Model Registry."

        model_list = [f"- {m.display_name} (ID: {m.name})" for m in models]
        return "### Vertex AI Model Registry\n" + "\n".join(model_list)
    except Exception as e:
        return f"Error listing models from Vertex AI: {str(e)}"


@mcp.tool()
def list_bucket_models(bucket_name: Optional[str] = None) -> str:
    """
    Lists the contents of a GCS bucket to check for uploaded model files.

    Args:
        bucket_name: The GCS bucket name. Defaults to BUCKET_NAME.
    """
    if not bucket_name:
        bucket_name = BUCKET_NAME

    clean_bucket = bucket_name.replace("gs://", "")

    try:
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(clean_bucket)
        blobs = list(bucket.list_blobs(max_results=100))

        if not blobs:
            return f"The GCS bucket '{clean_bucket}' is empty."

        file_list = [f"- gs://{clean_bucket}/{b.name} ({b.size / 1024 / 1024:.2f} MB)" for b in blobs[:50]]
        summary = f"### Contents of GCS Bucket: {clean_bucket}\n"
        summary += "\n".join(file_list)

        if len(blobs) > 50:
            summary += f"\n\n(Showing 50 of {len(blobs)} items)"

        return summary
    except Exception as e:
        return f"Error listing objects in bucket '{clean_bucket}': {str(e)}"


@mcp.tool()
async def analyze_cloud_logging(filter_query: str, limit: int = 5) -> str:
    """
    Fetches and summarizes error logs from Google Cloud Logging.

    Args:
        filter_query: Query filter. A standard Google Cloud log query.
        limit: Number of recent logs to fetch.
    """
    combined_logs = ""

    try:
        logging_client = cloud_logging.Client(project=PROJECT_ID)
        entries = list(
            logging_client.list_entries(filter_=filter_query, order_by=cloud_logging.DESCENDING, page_size=limit)
        )
        if entries:
            log_texts = [
                f"Timestamp: {e.timestamp} | Severity: {e.severity} | Message: {str(e.payload)[:1000] if isinstance(e.payload, str) else json.dumps(e.payload)[:1000]}"
                for e in entries
            ]
            combined_logs = "\n---\n".join(log_texts)
    except Exception as e:
        logger.warning(f"Failed to fetch Google Cloud logs: {e}")

    if not combined_logs:
        return "No matching logs found."

    try:
        if len(combined_logs) > 12000:
            combined_logs = combined_logs[:12000] + "... (truncated)"

        prompt = f"Analyze the following DevOps logs and provide a high-level summary of the critical issues and potential root causes:\n\n{combined_logs}\n\nSummary:"

        client = await get_vllm_client()
        model_name = await get_active_model_name(client)
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            max_tokens=512,
            temperature=0.2,
        )
        response_text = chat_completion.choices[0].message.content or ""
        return f"### Log Analysis (Self-Hosted vLLM)\n\n{response_text}"

    except Exception as e:
        return f"Error analyzing logs via self-hosted vLLM: {str(e)}"


@mcp.tool()
async def suggest_sre_remediation(error_message: str) -> str:
    """
    Proposes remediation steps for a specific SRE incident using self-hosted vLLM.

    Args:
        error_message: The error or incident description to remediate.
    """
    prompt = f"As an expert SRE, suggest a 3-step remediation plan for the following error:\n\nError: {error_message}\n\nRemediation Plan:"

    try:
        client = await get_vllm_client()
        model_name = await get_active_model_name(client)
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            max_tokens=512,
            temperature=0.2,
        )
        response_text = chat_completion.choices[0].message.content or ""
        return f"### Remediation Plan\n\n{response_text}"
    except Exception as e:
        return f"Error fetching remediation plan: {str(e)}"


@mcp.tool()
async def query_vllm(prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
    """
    Directly queries the self-hosted vLLM model with a custom prompt.

    Args:
        prompt: The text prompt to send to the model.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (0.0 for deterministic).
    """
    try:
        client = await get_vllm_client()
        model_name = await get_active_model_name(client)
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        response_text = chat_completion.choices[0].message.content or ""
        return f"### vLLM Response\n\n{response_text}"
    except Exception as e:
        return f"Error querying vLLM: {str(e)}"


@mcp.tool()
def get_vllm_deployment_config(
    service_name: str = DEFAULT_SERVICE_NAME,
    model_path: str = "google/gemma-4-E2B-it-qat-w4a16-ct",
    zone: str = ZONE,
    gpu_memory_utilization: float = 0.95,
) -> str:
    """
    Generates the gcloud compute command and startup script to deploy vLLM to a GCP GCE g2-standard-4 instance (NVIDIA L4).

    Args:
        service_name: The name of the GCE VM instance.
        model_path: Hugging Face repo ID or GCS URI of the model.
        zone: GCP zone for the deployment.
        gpu_memory_utilization: The fraction of GPU VRAM to use for KV cache (default: 0.95).
    """
    quant_arg = (
        "--quantization compressed-tensors" if any(q in model_path.lower() for q in ["qat", "w4a16", "ct"]) else ""
    )

    startup_script = f"""#!/bin/bash
if ! command -v docker &> /dev/null; then
    apt-get update -y
    apt-get install -y docker.io
    systemctl start docker
    systemctl enable docker
fi
docker run -d --name vllm-server \\
  --gpus all \\
  --ipc=host \\
  --restart always \\
  -p 8080:8080 \\
  -e HF_TOKEN="$(gcloud secrets versions access latest --secret=hf-token 2>/dev/null || echo '')" \\
  vllm/vllm-openai:nightly \\
  --model {model_path} \\
  {quant_arg} \\
  --dtype bfloat16 \\
  --max-model-len 32768 \\
  --disable-chunked-mm-input \\
  --gpu-memory-utilization {gpu_memory_utilization} \\
  --kv-cache-dtype fp8 \\
  --tensor-parallel-size 1 \\
  --max-num-seqs 8 \\
  --enable-chunked-prefill \\
  --max-num-batched-tokens 4096 \\
  --enable-auto-tool-choice \\
  --tool-call-parser gemma4 \\
  --reasoning-parser gemma4 \\
  --async-scheduling \\
  --limit-mm-per-prompt '{{}}' \\
  --host 0.0.0.0 \\
  --port 8080
"""

    gcloud_cmd = (
        f"gcloud compute instances create {service_name} \\\n"
        f"  --project={PROJECT_ID} \\\n"
        f"  --zone={zone} \\\n"
        f"  --machine-type=g2-standard-4 \\\n"
        f"  --accelerator=type=nvidia-l4,count=1 \\\n"
        f"  --maintenance-policy=TERMINATE \\\n"
        f"  --image-family=common-cu121-debian-11-py310 \\\n"
        f"  --image-project=deeplearning-platform-release \\\n"
        f"  --boot-disk-size=150GB \\\n"
        f"  --boot-disk-type=pd-balanced \\\n"
        f"  --metadata-from-file=startup-script=startup_script.sh \\\n"
        f"  --tags=vllm-server"
    )

    fw_cmd = (
        f"gcloud compute firewall-rules create allow-vllm-8080 \\\n"
        f"  --project={PROJECT_ID} \\\n"
        f"  --allow=tcp:8080 \\\n"
        f"  --target-tags=vllm-server \\\n"
        f"  --description='Allow port 8080 for vLLM'"
    )

    return (
        f"### 🚀 GCP GCE g2-standard-4 (NVIDIA L4) Instance vLLM Deployment Config\n\n"
        f"#### 1. Startup Script (`startup_script.sh`):\n"
        f"```bash\n{startup_script}\n```\n\n"
        f"#### 2. Create GCE Instance Command:\n"
        f"```bash\n{gcloud_cmd}\n```\n\n"
        f"#### 3. Create Firewall Rule Command:\n"
        f"```bash\n{fw_cmd}\n```\n\n"
        f"#### 4. Prerequisites:\n"
        f"- Save your HF Token in GCP Secret Manager: `gcloud secrets create hf-token --data-file=-` (and paste your token)\n"
        f"- Ensure the service account assigned to the GCE instance has permission to access the secret."
    )


@mcp.tool()
async def deploy_vllm(
    service_name: str = DEFAULT_SERVICE_NAME,
    model_path: str = "google/gemma-4-E2B-it-qat-w4a16-ct",
    zone: str = ZONE,
) -> str:
    """
    Deploys vLLM to GCP GCE g2-standard-4 (NVIDIA L4) VM instance.

    Args:
        service_name: Name of the GCE VM instance.
        model_path: Hugging Face repo ID or GCS URI.
        zone: GCP zone to launch the VM in.
    """
    hf_token = await get_secret() or ""

    quant_arg = (
        "--quantization compressed-tensors" if any(q in model_path.lower() for q in ["qat", "w4a16", "ct"]) else ""
    )

    script_content = f"""#!/bin/bash
if ! command -v docker &> /dev/null; then
    apt-get update -y
    apt-get install -y docker.io
    systemctl start docker
    systemctl enable docker
fi
docker run -d --name vllm-server \\
  --gpus all \\
  --ipc=host \\
  --restart always \\
  -p 8080:8080 \\
  -e HF_TOKEN="{hf_token}" \\
  vllm/vllm-openai:nightly \\
  --model {model_path} \\
  {quant_arg} \\
  --dtype bfloat16 \\
  --max-model-len 32768 \\
  --disable-chunked-mm-input \\
  --gpu-memory-utilization 0.95 \\
  --kv-cache-dtype fp8 \\
  --tensor-parallel-size 1 \\
  --max-num-seqs 8 \\
  --enable-chunked-prefill \\
  --max-num-batched-tokens 4096 \\
  --enable-auto-tool-choice \\
  --tool-call-parser gemma4 \\
  --reasoning-parser gemma4 \\
  --async-scheduling \\
  --limit-mm-per-prompt '{{}}' \\
  --host 0.0.0.0 \\
  --port 8080
"""
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".sh") as tmp:
        tmp.write(script_content)
        script_path = tmp.name

    try:
        # Create GCE instance
        code, stdout, stderr = await run_gcloud(
            [
                "compute",
                "instances",
                "create",
                service_name,
                f"--project={PROJECT_ID}",
                f"--zone={zone}",
                "--machine-type=g2-standard-4",
                "--accelerator=type=nvidia-l4,count=1",
                "--maintenance-policy=TERMINATE",
                "--image-family=common-cu129-ubuntu-2204-nvidia-580",
                "--image-project=deeplearning-platform-release",
                "--boot-disk-size=150GB",
                "--boot-disk-type=pd-balanced",
                f"--metadata-from-file=startup-script={script_path}",
                "--tags=vllm-server",
            ]
        )

        if code != 0:
            return f"Failed to deploy GCP GCE VM instance:\nError: {stderr}"

        # Create firewall rule (ignore if exists)
        await run_gcloud(
            [
                "compute",
                "firewall-rules",
                "create",
                "allow-vllm-8080",
                f"--project={PROJECT_ID}",
                "--allow=tcp:8080",
                "--target-tags=vllm-server",
                "--description=Allow port 8080 for vLLM",
            ]
        )

        return (
            f"🚀 Successfully requested GCP GCE g2-standard-4 Instance deployment for service '{service_name}' in zone '{zone}'.\n"
            f"Machine Type: `g2-standard-4`\n"
            f"Accelerator: `1x NVIDIA L4 (24GB VRAM)`\n"
            f"Please wait a few minutes for the instance to initialize, install drivers/Docker, and start the container."
        )
    finally:
        try:
            os.unlink(script_path)
        except Exception:
            pass


@mcp.tool()
async def start_gce(
    service_name: str = DEFAULT_SERVICE_NAME,
    model_path: str = "google/gemma-4-E2B-it-qat-w4a16-ct",
    zone: str = ZONE,
) -> str:
    """
    Starts an existing GCE instance, or provisions a new one if none exists.

    Args:
        service_name: Name of the GCE VM instance.
        model_path: Model ID (used if provisioning a new instance).
        zone: GCP zone for the VM.
    """
    code, stdout, stderr = await run_gcloud(
        [
            "compute",
            "instances",
            "describe",
            service_name,
            f"--project={PROJECT_ID}",
            f"--zone={zone}",
            "--format=value(status)",
        ]
    )

    if code == 0:
        status = stdout.strip()
        if status in ["TERMINATED", "STOPPED"]:
            code_start, stdout_start, stderr_start = await run_gcloud(
                ["compute", "instances", "start", service_name, f"--project={PROJECT_ID}", f"--zone={zone}", "--quiet"]
            )
            if code_start == 0:
                return f"🚀 Successfully started existing GCE instance '{service_name}' in zone '{zone}'."
            else:
                return f"Failed to start GCE instance '{service_name}':\nError: {stderr_start}"
        else:
            return f"GCE instance '{service_name}' is already in status: '{status}'."
    else:
        return await deploy_vllm(service_name=service_name, model_path=model_path, zone=zone)


@mcp.tool()
async def destroy_vllm(
    service_name: str = DEFAULT_SERVICE_NAME,
    zone: str = ZONE,
) -> str:
    """
    Deletes the GCP GCE vLLM VM instance.

    Args:
        service_name: Name of the GCE VM instance.
        zone: Zone where VM is located.
    """
    code, stdout, stderr = await run_gcloud(
        ["compute", "instances", "delete", service_name, f"--project={PROJECT_ID}", f"--zone={zone}", "--quiet"]
    )

    if code == 0:
        return f"🗑️ Successfully deleted GCP GCE instance: {service_name} in zone {zone}."
    else:
        return f"Failed to delete GCE instance {service_name}:\nError: {stderr}"


@mcp.tool()
async def stop_gce(
    service_name: str = DEFAULT_SERVICE_NAME,
    zone: str = ZONE,
) -> str:
    """
    Stops GCP GCE instance.

    Args:
        service_name: Name of the GCE VM instance to stop.
        zone: Zone of the instance.
    """
    code, stdout, stderr = await run_gcloud(
        ["compute", "instances", "stop", service_name, f"--project={PROJECT_ID}", f"--zone={zone}", "--quiet"]
    )

    if code == 0:
        return f"🛑 Successfully requested stopping for GCE Instance: {service_name}"
    else:
        return f"Failed to stop GCE instance {service_name}:\nError: {stderr}"


@mcp.tool()
async def status_vllm(service_name: str = DEFAULT_SERVICE_NAME, zone: str = ZONE) -> str:
    """
    Checks the status of the GCP GCE instance(s) matching the specified service name.

    Args:
        service_name: Name of the instance to check.
        zone: Zone of the instance.
    """
    code, stdout, stderr = await run_gcloud(
        ["compute", "instances", "describe", service_name, f"--project={PROJECT_ID}", f"--zone={zone}", "--format=json"]
    )

    if code != 0:
        return f"Failed to get status for GCE instance '{service_name}':\nError: {stderr}"

    try:
        data = json.loads(stdout)
        status = data.get("status")
        machine_type = data.get("machineType", "").split("/")[-1]
        nat_ip = "None"
        network_interfaces = data.get("networkInterfaces", [])
        if network_interfaces:
            access_configs = network_interfaces[0].get("accessConfigs", [])
            if access_configs:
                nat_ip = access_configs[0].get("natIP", "None")

        info = (
            f"- **Instance ID**: `{data.get('id')}`\n"
            f"  - **Type**: `{machine_type}`\n"
            f"  - **State**: `{status}`\n"
            f"  - **Public IP**: `{nat_ip}`\n"
            f"  - **Zone**: `{zone}`\n"
            f"  - **Launch Time**: `{data.get('creationTimestamp')}`\n"
        )
        return f"### GCP GCE Status for '{service_name}':\n\n{info}"
    except Exception as e:
        return f"Failed to parse instance info:\nError: {str(e)}"


@mcp.tool()
async def status_gce(
    service_name: str = DEFAULT_SERVICE_NAME,
    zone: str = ZONE,
) -> str:
    """
    Checks status of GCP GCE instance.

    Args:
        service_name: Name of the GCE VM instance.
        zone: Zone of the instance.
    """
    return await status_vllm(service_name=service_name, zone=zone)


@mcp.tool()
async def check_vllm(
    service_name: str = DEFAULT_SERVICE_NAME,
    zone: str = ZONE,
) -> str:
    """
    Checks the status of the vLLM container and engine running on the GCE instance.

    Args:
        service_name: Name of the GCE VM instance to check.
        zone: Zone of the instance.
    """
    # 1. Check GCE VM Status
    code, stdout, stderr = await run_gcloud(
        ["compute", "instances", "describe", service_name, f"--project={PROJECT_ID}", f"--zone={zone}", "--format=json"]
    )

    if code != 0:
        return f"Failed to describe GCE instance '{service_name}':\nError: {stderr}"

    try:
        data = json.loads(stdout)
        status = data.get("status")
        nat_ip = None
        network_interfaces = data.get("networkInterfaces", [])
        if network_interfaces:
            access_configs = network_interfaces[0].get("accessConfigs", [])
            if access_configs:
                nat_ip = access_configs[0].get("natIP")

        report = f"### 🖥️ Instance: `{service_name}` ({status})\n"
        if status != "RUNNING":
            report += f"❌ Instance is not running (Current State: `{status}`). Skipping container checks.\n"
            return report

        if not nat_ip:
            report += "⚠️ No Public IP associated with this running instance.\n"
            return report

        # 2. Check Docker Container status via gcloud compute ssh
        code_ssh, stdout_ssh, stderr_ssh = await run_gcloud(
            [
                "compute",
                "ssh",
                service_name,
                f"--project={PROJECT_ID}",
                f"--zone={zone}",
                "--command=docker inspect -f '{{.State.Status}}' vllm-server 2>&1",
                "--quiet",
            ]
        )

        docker_status = stdout_ssh.strip() if code_ssh == 0 else f"Failed to query (Error: {stderr_ssh})"
        report += f"- **Docker Container (`vllm-server`)**: `{docker_status}`\n"

        # 3. Check vLLM HTTP health endpoint
        http_status = "Unreachable"
        try:
            async with httpx.AsyncClient(timeout=3) as http_client:
                res = await http_client.get(f"http://{nat_ip}:8080/health")
                if res.status_code == 200:
                    http_status = "Healthy ✅"
                else:
                    http_status = f"Unhealthy (HTTP Code: {res.status_code}) ❌"
        except Exception as e:
            http_status = f"Unreachable (Error: {e}) ❌"

        report += f"- **vLLM API Endpoint (`http://{nat_ip}:8080/health`)**: `{http_status}`\n"
        return report
    except Exception as e:
        return f"Failed to parse instance info:\nError: {str(e)}"


@mcp.tool()
async def update_vllm_scaling(
    instance_type: str,
    service_name: str = DEFAULT_SERVICE_NAME,
    zone: str = ZONE,
) -> str:
    """
    Updates the GCP GCE instance type (scaling vertically) for the vLLM service instance.
    Note: The instance must be stopped to change its machine type.

    Args:
        instance_type: The new GCP machine type (e.g. 'g2-standard-8', 'g2-standard-16').
        service_name: The name of the GCE VM instance to scale.
        zone: Zone of the instance.
    """
    # 1. Stop instance
    code, stdout, stderr = await run_gcloud(
        ["compute", "instances", "stop", service_name, f"--project={PROJECT_ID}", f"--zone={zone}", "--quiet"]
    )
    if code != 0:
        return f"Failed to stop instance for scaling:\nError: {stderr}"

    # 2. Set machine type
    code, stdout, stderr = await run_gcloud(
        [
            "compute",
            "instances",
            "set-machine-type",
            service_name,
            f"--project={PROJECT_ID}",
            f"--zone={zone}",
            f"--machine-type={instance_type}",
            "--quiet",
        ]
    )
    if code != 0:
        return f"Failed to change machine type to {instance_type}:\nError: {stderr}"

    # 3. Start instance
    code, stdout, stderr = await run_gcloud(
        ["compute", "instances", "start", service_name, f"--project={PROJECT_ID}", f"--zone={zone}", "--quiet"]
    )
    if code != 0:
        return f"Failed to restart instance after scaling:\nError: {stderr}"

    return f"🚀 Successfully scaled GCE instance `{service_name}` to `{instance_type}` and restarted it."


@mcp.tool()
def get_vllm_gpu_deployment_config(
    cluster_name: str = "gpu-cluster", model_name: str = "google/gemma-4-E2B-it-qat-w4a16-ct"
) -> str:
    """
    Generates a GKE manifest and setup instructions for deploying vLLM on GPU (NVIDIA L4).

    Args:
        cluster_name: The name of the GKE cluster.
        model_name: The model identifier (e.g., 'google/gemma-4-E2B-it-qat-w4a16-ct').
    """
    manifest = f"""
### 🌀 vLLM on GPU (GKE Deployment)

To deploy vLLM on GPUs, use the following GKE manifest. This configuration targets a single **NVIDIA L4 GPU** using a **g2-standard-4** node.

#### 1. Create a GPU Node Pool (if not exists)
```bash
gcloud container node-pools create gpu-l4 \\
    --cluster={cluster_name} \\
    --location={LOCATION} \\
    --machine-type=g2-standard-4 \\
    --accelerator=type=nvidia-l4,count=1 \\
    --num-nodes=1
```

#### 2. Kubernetes Manifest (vllm-gpu.yaml)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-gpu
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-gpu
  template:
    metadata:
      labels:
        app: vllm-gpu
    spec:
      nodeSelector:
        cloud.google.com/gke-gpu: "true"
      containers:
      - name: vllm-gpu
        image: vllm/vllm-openai:nightly
        resources:
          limits:
            nvidia.com/gpu: "1"
          requests:
            nvidia.com/gpu: "1"
        command: ["python3", "-m", "vllm.entrypoints.openai.api_server"]
        args:
        - "--model={model_name}"
        - "--gpu-memory-utilization=0.95"
        - "--max-model-len=32768"
        ports:
        - containerPort: 8080
        volumeMounts:
        - name: dshm
          mountPath: /dev/shm
      volumes:
      - name: dshm
        emptyDir:
          medium: Memory
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-gpu-service
spec:
  selector:
    app: vllm-gpu
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: ClusterIP
```

#### 3. Deployment Steps
1. Install NVIDIA Device Plugin for Kubernetes on GKE.
2. Save the YAML above to `vllm-gpu.yaml`.
3. Apply it: `kubectl apply -f vllm-gpu.yaml`.
"""
    return manifest


@mcp.tool()
def get_vertex_ai_model_copy_instructions(model_name: str = "gemma-4-E2B-it-qat-w4a16-ct") -> str:
    """
    Provides instructions and commands to transfer Gemma model artifacts from Vertex AI Model Garden to your GCS bucket.
    """
    instructions = f"""
### 🚀 Transferring {model_name} from Vertex AI Model Garden

To use vLLM with Cloud Storage FUSE without Hugging Face, follow these steps:

1. **Accept Terms:** Go to the Vertex AI Model Garden page for Gemma (https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/335) and click 'Accept' on the license agreement.
2. **Download via Signed URL:** After accepting, the console provides a 'Download' button or a signed URL.
3. **Transfer to GCS:**
   If you have the artifacts locally after downloading from the console, use:
   `gcloud storage cp -r ./model_artifacts/* gs://{BUCKET_NAME}/{model_name}/`

4. **Alternative (Direct GCS Copy):**
   Google occasionally provides a managed GCS path for verified projects. If accessible, you can use:
   `gcloud storage cp -r gs://vertex-ai-models/gemma/{model_name}/* gs://{BUCKET_NAME}/{model_name}/`

Once the artifacts are in your bucket, use `get_vllm_deployment_config` to generate your Cloud Run deployment command.
"""
    return instructions


@mcp.tool()
async def get_huggingfacehub_download_path(
    repo_id: str = "google/gemma-4-E2B-it-qat-w4a16-ct",
) -> str:
    """
    Returns the local cache path for a Hugging Face model using huggingface_hub.
    Note: This may trigger a download if the model is not already in the cache.
    """
    try:
        from huggingface_hub import snapshot_download

        token = await get_secret() or os.getenv("HF_TOKEN")
        # Run synchronous snapshot_download in a separate thread to avoid blocking the async event loop
        path = await asyncio.to_thread(snapshot_download, repo_id=repo_id, token=token)
        return f"Model '{repo_id}' is available at: {path}"
    except Exception as e:
        return f"Error resolving huggingface_hub path: {str(e)}"


@mcp.tool()
def get_huggingface_model_copy_instructions(
    repo_id: str = "google/gemma-4-E2B-it-qat-w4a16-ct",
    bucket_name: Optional[str] = None,
) -> str:
    """
    Provides instructions and commands to transfer Gemma model weights from Hugging Face to your GCS bucket.

    Args:
        repo_id: The Hugging Face repo ID (e.g., 'google/gemma-4-E2B-it-qat-w4a16-ct').
        bucket_name: The target bucket name (defaults to BUCKET_NAME).
    """
    if not bucket_name:
        bucket_name = BUCKET_NAME

    model_name = repo_id.split("/")[-1]

    instructions = f"""
### 📦 Transferring {model_name} from Hugging Face to GCS

To use Hugging Face weights with vLLM on Cloud Run via GCS FUSE, follow these steps:

#### Option A: Using `huggingface_hub` Python Library (Recommended)
`huggingface_hub` simplifies the download process and can be run directly from python:

1. **Download Model:**
   `python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('{repo_id}'))"`

2. **Upload to GCS:**
   The command above outputs the local path. Use it to copy the artifacts:
   `gcloud storage cp -r /path/to/downloaded/model/* gs://{bucket_name}/{model_name}/`

#### Option B: Using `huggingface-cli`
1. **Setup Hugging Face CLI:**
   `pip install huggingface_hub`
   `huggingface-cli login`

2. **Download Model Artifacts:**
   `huggingface-cli download {repo_id} --local-dir ./{model_name}`

3. **Upload to GCS Bucket:**
   `gcloud storage cp -r ./{model_name}/* gs://{bucket_name}/{model_name}/`

Once uploaded, you can deploy using:
`get_vllm_deployment_config(model_path="{model_name}")`
"""
    return instructions


@mcp.tool()
def check_gpu_quotas(region: Optional[str] = None) -> str:
    """
    Checks GPU quotas for a specific Google Cloud region.

    Args:
        region: The GCP region (defaults to LOCATION).
    """
    if not region:
        region = LOCATION

    cmd = [
        "gcloud",
        "compute",
        "regions",
        "describe",
        region,
        f"--project={PROJECT_ID}",
        "--format=json(quotas)",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        quotas = data.get("quotas", [])

        # Filter for GPU quotas
        gpu_quotas = []
        for q in quotas:
            metric = q.get("metric", "")
            if "GPU" in metric or "ACCELERATOR" in metric:
                gpu_quotas.append(f"- **{metric}**:\n  - Limit: `{q.get('limit')}`\n  - Usage: `{q.get('usage')}`")

        if not gpu_quotas:
            return f"No GPU/Accelerator quotas found in GCP region `{region}`."

        return f"### 📊 GCP GPU Quotas for region `{region}`\n\n" + "\n".join(gpu_quotas)

    except subprocess.CalledProcessError as e:
        return f"Failed to retrieve GPU quotas for region `{region}`:\nError: {e.stderr}\nOutput: {e.stdout}"
    except Exception as e:
        return f"Error checking GPU quotas: {str(e)}"


@mcp.tool()
async def verify_model_health() -> str:
    """Runs a deep health check with latency reporting on the Cloud Run GPU-hosted model."""
    try:
        client = await get_vllm_client()
        model_name = await get_active_model_name(client)
        start_time = time.monotonic()
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello, is the model working?"}],
            model=model_name,
            max_tokens=200,
        )
        end_time = time.monotonic()
        latency = end_time - start_time
        response_content = chat_completion.choices[0].message.content

        if response_content:
            return (
                f"✅ Model health check PASSED.\n"
                f"Model: {model_name}\n"
                f"Response: '{response_content[:50]}...'\n"
                f"Latency: {latency:.2f} seconds."
            )
        else:
            return "❌ Model health check FAILED: Empty response."
    except Exception as e:
        return f"❌ Model health check FAILED: {e}"


@mcp.tool()
async def query_gemma4(prompt: str) -> str:
    """Queries the self-hosted Gemma 4 model on Cloud Run."""
    logger.info(f"Querying Cloud Run model with prompt: '{prompt[:50]}...'")
    try:
        client = await get_vllm_client()
        model_name = await get_active_model_name(client)
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
        )
        response = chat_completion.choices[0].message.content or "No response from model."
        logger.info(f"Model response: '{response[:100]}...'")
        return response
    except Exception as e:
        logger.error(f"Error querying model: {e}")
        return f"❌ An error occurred while querying the model: {e}"


@mcp.tool()
async def query_gemma4_with_stats(prompt: str) -> str:
    """
    Queries the self-hosted Gemma 4 model on Cloud Run and returns detailed performance statistics.

    This tool provides:
    - The full model response.
    - Time to First Token (TTFT).
    - Total generation time.
    - Tokens per second.
    """
    logger.info(f"Querying model with stats with prompt: '{prompt[:50]}...'")
    try:
        client = await get_vllm_client()
        model_name = await get_active_model_name(client)

        start_time = time.monotonic()
        ttft = None
        response_content = ""
        total_tokens = 0

        stream = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            stream=True,
        )

        async for chunk in stream:
            if ttft is None:
                ttft = time.monotonic() - start_time

            delta = chunk.choices[0].delta if chunk.choices else None
            if delta:
                content = delta.content or getattr(delta, "reasoning", None)
                if content:
                    response_content += content
                    total_tokens += 1  # Rough token count

        end_time = time.monotonic()
        total_time = end_time - start_time

        if not response_content:
            return "❌ Model returned an empty response."

        tokens_per_second = total_tokens / (total_time - ttft) if ttft and total_time > ttft else 0

        stats_report = (
            f"### 📊 Performance Stats\n"
            f"- **Model:** `{model_name}`\n"
            f"- **Time to First Token (TTFT):** `{ttft:.3f}s`\n"
            f"- **Total Generation Time:** `{total_time:.3f}s`\n"
            f"- **Tokens per Second:** `{tokens_per_second:.2f} tokens/s`\n"
            f"- **Total Tokens (approx.):** `{total_tokens}`\n"
            f"\n### 💬 Model Response\n"
            f"{response_content}"
        )

        logger.info(f"Model response with stats: TTFT={ttft:.3f}s, TotalTime={total_time:.3f}s")
        return stats_report

    except Exception as e:
        logger.error(f"Error querying model with stats: {e}")
        return f"❌ An error occurred while querying the model with stats: {e}"


@mcp.tool()
async def get_model_details() -> str:
    """Retrieves detailed information about the running Cloud Run model, engine, and versions."""
    report = ""
    try:
        vllm_url = get_vllm_url()
        report += f"### 🧩 Model Details ({vllm_url})\n\n"
        client = await get_vllm_client()

        # 1. Get Model Details from /v1/models
        try:
            models_res = await client.models.list()
            report += "**Model Information (`/v1/models`):**\n"
            models_list = [{"id": m.id, "object": m.object, "owned_by": m.owned_by} for m in models_res.data]
            report += f"```json\n{json.dumps(models_list, indent=2)}\n```\n"
        except Exception as e:
            report += f"❌ Error fetching model details via client: {e}\n\n"

        # 2. Get Health Status
        token = get_auth_token()
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        async with httpx.AsyncClient(timeout=10) as http_client:
            try:
                res = await http_client.get(f"{vllm_url}/health", headers=headers)
                if res.status_code == 200:
                    report += "**Health Status (`/health`):**\n- Status: `Healthy` ✅\n\n"
                else:
                    report += f"**Health Status (`/health`):**\n- Status: `Unhealthy` (Code: {res.status_code}) ❌\n\n"
            except Exception as e:
                report += f"**Health Status (`/health`):**\n- Status: `Unreachable` (Error: {e}) ❌\n\n"
    except Exception as e:
        report += f"❌ Error retrieving system URL or auth token: {e}"

    return report


@mcp.tool()
async def get_system_status(service_name: str = DEFAULT_SERVICE_NAME) -> str:
    """
    Provides a high-level dashboard of GCP GCE VM system status and vLLM health.

    Args:
        service_name: The name of the GCE VM instance.
    """
    health = "🔴 Offline"
    url = None
    try:
        url = get_vllm_url()
        token = get_auth_token()
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        async with httpx.AsyncClient(timeout=5) as client:
            res = await client.get(f"{url}/health", headers=headers)
            if res.status_code == 200:
                health = f"🟢 Online ({url})"
            else:
                health = f"🔴 Offline (Status {res.status_code}) ({url})"
    except Exception as e:
        logger.warning(f"Health check failed: {e}")

    gce_status = "🔴 Unknown"
    try:
        code, stdout, stderr = await run_gcloud(
            [
                "compute",
                "instances",
                "describe",
                service_name,
                f"--project={PROJECT_ID}",
                f"--zone={ZONE}",
                "--format=value(status)",
            ]
        )
        if code == 0:
            status = stdout.strip()
            if status == "RUNNING":
                gce_status = f"🟢 Running ({service_name})"
            else:
                gce_status = f"🔴 {status.capitalize()} ({service_name})"
        else:
            gce_status = f"🔴 GCE Error ({stderr})"
    except Exception as e:
        gce_status = f"🔴 GCE Error: {str(e)}"

    if "🟢" in health:
        next_step = "Use `query_gemma4` to interact with the model."
    else:
        next_step = f"Call `deploy_vllm` to provision/start the GCE instance `{service_name}`."

    return (
        f"### 🌀 GPU vLLM System Status\n"
        f"- **vLLM Health:** {health}\n"
        f"- **Hosting Status:** {gce_status}\n"
        f"**👉 Next Step:** {next_step}"
    )


@mcp.tool()
async def get_endpoint(service_name: str = DEFAULT_SERVICE_NAME) -> str:
    """
    Returns the active vLLM service URL if available.

    Args:
        service_name: The name of the service or instance Name tag to query.
    """
    try:
        url = get_vllm_url()
        token = get_auth_token()
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        async with httpx.AsyncClient(timeout=5) as client:
            res = await client.get(f"{url}/health", headers=headers)
            if res.status_code == 200:
                return f"🟢 vLLM is Online at: {url}"
            else:
                return f"🔴 vLLM is configured at {url} but returned status {res.status_code}."
    except Exception as e:
        return f"🔴 vLLM endpoint check failed: {e}. Try deploying/starting it with `deploy_vllm`."


@mcp.tool()
async def run_benchmark(
    model: Optional[str] = None,
    num_prompts: int = 20,
    random_output_len: int = 128,
    max_concurrency: int = 8,
) -> str:
    """
    Runs a performance/concurrency benchmark sweep against the Cloud Run vLLM GPU endpoint.

    Args:
        model: Model name to request (defaults to the active model from /v1/models).
        num_prompts: Number of requests to send per concurrency level.
        random_output_len: Max tokens to generate per request.
        max_concurrency: Maximum concurrency level to sweep up to (powers of 2, e.g. 1, 2, 4, 8).
    """
    from datetime import datetime

    try:
        url = get_vllm_url()
        token = get_auth_token()
    except Exception as e:
        return f"❌ Cannot run benchmark: {e}"

    # Get active model name if not provided
    client = await get_vllm_client()
    if not model:
        model = await get_active_model_name(client)

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    base_url = f"{url.rstrip('/')}/v1/completions"
    prompt = "Explain the importance of Site Reliability Engineering for large scale AI deployments."

    concurrencies = []
    c = 1
    while c <= max_concurrency:
        concurrencies.append(c)
        c *= 2
    if max_concurrency not in concurrencies:
        concurrencies.append(max_concurrency)

    results = []

    async def send_request(http_client: httpx.AsyncClient, sem: asyncio.Semaphore) -> dict:
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": random_output_len,
            "temperature": 0.0,
            "stream": False,
        }
        async with sem:
            start_time = time.perf_counter()
            try:
                response = await http_client.post(base_url, json=payload, headers=headers, timeout=120)
                end_time = time.perf_counter()
                if response.status_code == 200:
                    latency = end_time - start_time
                    data = response.json()
                    tokens = data.get("usage", {}).get("completion_tokens", random_output_len)
                    return {"success": True, "latency": latency, "tokens": tokens}
                else:
                    return {"success": False, "error": f"Status {response.status_code}"}
            except Exception as e:
                return {"success": False, "error": str(e)}

    # Warmup
    logger.info("Warming up model for benchmark...")
    async with httpx.AsyncClient() as http_client:
        await send_request(http_client, asyncio.Semaphore(1))

    logger.info(f"Starting GPU benchmark sweep against {url} with model {model}...")
    for concurrency in concurrencies:
        logger.info(f"Running sweep with concurrency={concurrency}...")
        sem = asyncio.Semaphore(concurrency)

        async with httpx.AsyncClient() as http_client:
            start_batch = time.perf_counter()
            tasks = [send_request(http_client, sem) for _ in range(num_prompts)]
            batch_results = await asyncio.gather(*tasks)
            total_time = time.perf_counter() - start_batch

        successes = [r for r in batch_results if r["success"]]
        latencies = [r["latency"] for r in successes]

        if not latencies:
            results.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "concurrency": concurrency,
                    "total_requests": num_prompts,
                    "success_rate": 0.0,
                    "avg_latency": 0.0,
                    "p95_latency": 0.0,
                    "req_per_sec": 0.0,
                    "tokens_per_sec": 0.0,
                }
            )
            continue

        avg_lat = statistics.mean(latencies)
        p95_lat = sorted(latencies)[int(len(latencies) * 0.95)]
        throughput = len(successes) / total_time
        tokens_per_sec = sum(r["tokens"] for r in successes) / total_time

        results.append(
            {
                "timestamp": datetime.now().isoformat(),
                "concurrency": concurrency,
                "total_requests": num_prompts,
                "success_rate": len(successes) / num_prompts,
                "avg_latency": avg_lat,
                "p95_latency": p95_lat,
                "req_per_sec": throughput,
                "tokens_per_sec": tokens_per_sec,
            }
        )

    # Save to CSV
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.csv")
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "concurrency",
                "total_requests",
                "success_rate",
                "avg_latency",
                "p95_latency",
                "req_per_sec",
                "tokens_per_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    summary_str = f"### 📊 GPU Benchmark Results (Model: `{model}`)\n\n"
    summary_str += "| Concurrency | Success Rate | Req/s | Tokens/s | Avg Latency | P95 Latency |\n"
    summary_str += "|---:|---:|---:|---:|---:|---:|\n"
    for r in results:
        summary_str += f"| {r['concurrency']} | {r['success_rate']:.1%} | {r['req_per_sec']:.2f} | {r['tokens_per_sec']:.2f} | {r['avg_latency']:.2f}s | {r['p95_latency']:.2f}s |\n"
    summary_str += f"\n\nResults saved to `{output_file}`"
    return summary_str


async def fetch_gce_logs(service_name: str, limit: int = 50) -> str:
    """Fetches docker logs from the running GCE instance via gcloud compute ssh."""
    code, stdout, stderr = await run_gcloud(
        [
            "compute",
            "ssh",
            service_name,
            f"--project={PROJECT_ID}",
            f"--zone={ZONE}",
            f"--command=docker logs --tail {limit} vllm-server 2>&1",
            "--quiet",
        ]
    )
    if code == 0:
        return stdout
    else:
        return f"Failed to fetch logs: {stderr}"


@mcp.tool()
async def analyze_gpu_logs(limit: int = 15, service_name: str = DEFAULT_SERVICE_NAME) -> str:
    """
    Fetches vLLM logs for the specified service and uses Gemma 4 to analyze them for errors.

    Args:
        limit: Number of log entries to fetch.
        service_name: Name of the GCP GCE VM instance.
    """
    logger.info(f"Fetching GCE logs for instance {service_name}...")
    raw_logs = await fetch_gce_logs(service_name, limit)

    # Prepare prompt for Gemma
    prompt = f"Analyze the following vLLM docker container logs and provide a high-level summary of critical issues:\n\n{raw_logs}\n\nSummary:"
    try:
        client = await get_vllm_client()
        model_name = await get_active_model_name(client)
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            max_tokens=512,
            temperature=0.2,
        )
        response_text = chat_completion.choices[0].message.content or ""
        return f"### GCP GCE Log Analysis (Self-Hosted vLLM)\n\n{response_text}"
    except Exception as e:
        return f"Failed to analyze logs: {str(e)}"


@mcp.tool()
async def get_help() -> str:
    """Provides help text and summarizes the configuration options and all available SRE/DevOps tools for this GCP GCE MCP server."""
    return (
        "### 🛠️ GCP Gemma 4 SRE Agent Help & Configuration\n\n"
        "You can configure this MCP server using the following environment variables:\n\n"
        "**GCP Configuration:**\n"
        f"- **`GOOGLE_CLOUD_PROJECT`**: Your GCP Project ID.\n"
        f"  - *Current Value:* `{PROJECT_ID}`\n"
        f"- **`GOOGLE_CLOUD_LOCATION`**: The GCP Region/Location.\n"
        f"  - *Current Value:* `{LOCATION}`\n"
        f"- **`GOOGLE_CLOUD_ZONE`**: The GCP Zone for GCE VM deployment.\n"
        f"  - *Current Value:* `{ZONE}`\n"
        f"- **`BUCKET_NAME`**: GCS Bucket used to store model weights.\n"
        f"  - *Current Value:* `{BUCKET_NAME}`\n\n"
        "**General serving:**\n"
        f"- **`MODEL_NAME`**: Default Hugging Face repository or GCS path.\n"
        f"  - *Current Value:* `{MODEL_NAME}`\n"
        f"- **`VLLM_BASE_URL`**: The explicit URL of your vLLM GCE service. (If not set, it is auto-discovered via GCE VM external IP)\n"
        f"  - *Current Value:* `{VLLM_BASE_URL or 'Not set (auto-discovering)'}`\n\n"
        "### ℹ️ Active Mode Summary\n"
        "The server is running in **GCP GCE VM** mode targeting a `g2-standard-4` host VM with NVIDIA L4 GPU.\n\n"
        "### 🧰 Available MCP Tools\n\n"
        "Below is a summary of the tools exposed by this SRE/DevOps agent:\n\n"
        "#### 🐳 Infrastructure & Deployment\n"
        "- **`start_gce`**: Starts an existing GCE instance, or provisions a new one if none exists.\n"
        "- **`status_gce`**: Checks GCE instance status.\n"
        "- **`stop_gce`**: Stops GCE instance.\n"
        "- **`check_vllm`**: Checks the status of the vLLM container and engine running on the GCE instance.\n"
        "- **`deploy_vllm`**: Deploys vLLM to GCP GCE g2-standard-4 (NVIDIA L4) VM instance.\n"
        "- **`destroy_vllm`**: Deletes the GCP GCE vLLM VM instance.\n"
        "- **`status_vllm`**: Checks GCE instance status.\n"
        "- **`update_vllm_scaling`**: Scales GCE instance type vertically.\n"
        "- **`get_vllm_deployment_config`**: Generates the gcloud compute command and startup script.\n"
        "- **`get_vllm_gpu_deployment_config`**: Generates a GKE manifest for GPU (NVIDIA L4).\n"
        "- **`check_gpu_quotas`**: Checks GPU/Accelerator quotas for a region.\n"
        "- **`get_vllm_endpoint`**: Returns the current active vLLM endpoint URL.\n\n"
        "#### 📊 Model Management\n"
        "- **`list_vertex_models`**: Lists models in the Vertex AI Registry.\n"
        "- **`list_bucket_models`**: Lists model weights in GCS bucket.\n"
        "- **`save_hf_token`**: Securely saves a Hugging Face API token to Secret Manager.\n"
        "- **`get_vertex_ai_model_copy_instructions`**: Instructions to copy model from Vertex AI Model Garden to GCS.\n"
        "- **`get_huggingface_model_copy_instructions`**: Instructions to download model from Hugging Face and upload to GCS.\n"
        "- **`get_huggingfacehub_download_path`**: Resolves local cache path using huggingface_hub.\n\n"
        "#### 📊 Monitoring & Status\n"
        "- **`get_metrics`**: Fetches raw Prometheus metrics from the running vLLM service's /metrics endpoint.\n"
        "- **`get_system_status`**: Provides a high-level status dashboard of the service and health.\n"
        "- **`get_endpoint`**: Verifies connectivity and returns the active service URL.\n"
        "- **`get_model_details`**: Retrieves detailed model metadata and engine state from `/v1/models`.\n"
        "- **`verify_model_health`**: Deep health check by querying the model with a simple prompt and measuring latency.\n\n"
        "#### 📈 Performance & Benchmarking\n"
        "- **`run_benchmark`**: Runs performance/concurrency benchmark sweeps against the vLLM GPU endpoint.\n\n"
        "#### 💬 Interaction & Diagnostics\n"
        "- **`query_gemma4`**: Primary tool to query the self-hosted model with standard chat message format.\n"
        "- **`query_gemma4_with_stats`**: Queries the model and returns streaming performance statistics (TTFT, throughput).\n"
        "- **`query_vllm`**: Direct text completions querying tool.\n"
        "- **`analyze_cloud_logging`**: Fetches logs from GCP Logging and analyzes them using the model.\n"
        "- **`analyze_gpu_logs`**: Fetches service logs and uses Gemma 4 to analyze them for SRE/DevOps errors.\n"
        "- **`suggest_sre_remediation`**: Suggests remediation plans for SRE errors using the model.\n"
    )


@mcp.tool()
async def get_metrics() -> str:
    """
    Fetches the Prometheus metrics from the active vLLM service.
    """
    try:
        url = get_vllm_url()
        token = get_auth_token()
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        async with httpx.AsyncClient(timeout=10) as client:
            res = await client.get(f"{url}/metrics", headers=headers)
            if res.status_code == 200:
                return res.text
            else:
                return f"🔴 Failed to retrieve metrics. Status code: {res.status_code}\n\nResponse:\n{res.text[:1000]}"
    except Exception as e:
        return f"🔴 Error fetching metrics: {e}"


if __name__ == "__main__":
    mcp.run()
