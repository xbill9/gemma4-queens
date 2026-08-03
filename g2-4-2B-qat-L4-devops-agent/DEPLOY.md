# Deployment Guide: Self-Hosted vLLM on AWS EC2 (Gemma 4 E2B-it QAT)

This document summarizes the deployment state, configuration, and architecture for the self-hosted vLLM inference server running on AWS EC2.

---

## 📦 Model Artifacts
The model is served using the official quantized w4a16 compressed-tensors (QAT) checkpoint:
*   **Source:** Hugging Face (`google/gemma-4-E2B-it-qat-w4a16-ct`)
*   **Alternative Storage:** AWS S3 (`s3://vllm-models-bucket/gemma-4-E2B-it-qat-w4a16-ct/`)
*   **Format:** Hugging Face Transformers (Safetensors with compressed-tensors)

---

## 🚀 AWS Inference Stack (EC2 g6.2xlarge Spot Instance)
The inference server is hosted on an AWS Spot EC2 instance for cost-effective machine learning workloads.

*   **Instance Type:** `g6.2xlarge`
*   **Market Type:** Spot (One-time request)
*   **GPU Accelerator:** 1x NVIDIA L4 (24 GiB VRAM)
*   **vCPUs / RAM:** 8 vCPUs / 32 GiB RAM
*   **Operating System / AMI:** Ubuntu 22.04 Deep Learning AMI (DLAMI)
*   **Container Port:** `8080` (mapped to Host Port `8080`)
*   **Security Group:** `vllm-devops-sg` (allows inbound TCP on `8080` and `22`)

### AWS CLI Launch Command (Spot)
```bash
aws ec2 run-instances \
  --image-id ami-012ba162b9cd2729c \
  --instance-type g6.2xlarge \
  --key-name alinux \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=gpu-2b-qat-l4-devops-agent}]' \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}' \
  --user-data file://user_data.sh
```

---

## 🛠 Deployment & Startup Script

The instance is deployed with a `UserData` script that automatically provisions Docker and launches the vLLM engine inside a container with optimized parameters.

### vLLM Run Arguments
```bash
docker run -d --name vllm-server \
  --gpus all \
  --ipc=host \
  --restart always \
  -p 8080:8080 \
  -e HF_TOKEN="<your-hf-token>" \
  vllm/vllm-openai:nightly \
  --model google/gemma-4-E2B-it-qat-w4a16-ct \
  --quantization compressed-tensors \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --disable-chunked-mm-input \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 1 \
  --max-num-seqs 8 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 4096 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --async-scheduling \
  --limit-mm-per-prompt '{}' \
  --host 0.0.0.0 \
  --port 8080
```

### Key Parameters Explained
*   `--dtype bfloat16`: Prevents numerical overflow/underflow and matches Gemma 4's native precision.
*   `--quantization compressed-tensors`: Required to deserialize the QAT INT4 model weights.
*   `--gpu-memory-utilization 0.95`: Allocates 95% of GPU memory to vLLM's cache.
*   `--kv-cache-dtype fp8`: Cuts KV cache footprint in half, drastically improving concurrency.
*   `--tool-call-parser gemma4` & `--reasoning-parser gemma4`: Critical for correct parsing of tool calls.

---

## 🔗 Integration with SRE Agent
To connect the MCP SRE Agent to the newly deployed AWS endpoint:

1. Discover the public IP of your EC2 instance (e.g. `54.1.2.3`).
2. Export the endpoint URL in your terminal environment:
   ```bash
   export VLLM_BASE_URL="http://54.1.2.3:8080"
   export MODEL_NAME="google/gemma-4-E2B-it-qat-w4a16-ct"
   ```
3. Start the agent:
   ```bash
   make run
   ```
