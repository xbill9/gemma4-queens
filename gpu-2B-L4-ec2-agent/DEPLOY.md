# Deployment Guide: Self-Hosted vLLM on AWS EC2 (Gemma 4 E2B-it QAT)

This document summarizes the deployment state, configuration, and architecture for the self-hosted vLLM inference server running on AWS EC2.

---

## 📦 Model Artifacts
The model is served using the official quantized w4a16 compressed-tensors (QAT) checkpoint:
*   **Source:** Hugging Face (`google/gemma-4-E2B-it-qat-w4a16-ct`)
*   **Alternative Storage:** AWS S3 (`s3://vllm-models-bucket/gemma-4-E2B-it-qat-w4a16-ct/`)
*   **Format:** Hugging Face Transformers (Safetensors with compressed-tensors)

---

## 🚀 AWS Inference Stack (EC2 g6.xlarge)

*   **Instance Type:** `g6.xlarge`
*   **GPU Accelerator:** 1x NVIDIA L4 (24 GiB VRAM)
*   **vCPUs / RAM:** 4 vCPUs / 16 GiB RAM
*   **Operating System / AMI:** Ubuntu 22.04 Deep Learning AMI (DLAMI), resolved per region — see below
*   **Root Volume:** 150 GiB gp3
*   **Container Port:** `8080` (mapped to Host Port `8080`)
*   **Security Group:** `vllm-devops-sg` (inbound TCP on `8080` and `22`)
*   **Market Type:** on-demand by default; spot is opt-in

### AMI selection
AMI IDs are region-specific, so nothing here pins one. Both the Makefile and `server.py` resolve a
DLAMI at deploy time, in this order:

1. `$VLLM_AMI_ID` if set
2. the SSM public parameter `/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id`
3. newest `describe-images` match on the DLAMI name pattern
4. a pinned us-east-1 fallback (`ami-012ba162b9cd2729c`), last resort only

In us-east-1 this currently resolves to **`ami-09f0204c6e7c717ab`** (Deep Learning Base OSS Nvidia
Driver GPU AMI, Ubuntu 22.04, 2026-07-31). Note this is the *Base* DLAMI, not the PyTorch one the
old hardcoded `ami-012ba162b9cd2729c` pointed at — that pin was the PyTorch 2.7 DLAMI, still
available at `.../oss-nvidia-driver-gpu-pytorch-2.7-ubuntu-22.04/latest/ami-id`. Base is the right
fit here: vLLM runs in a container, so the host only needs the driver, Docker, and the NVIDIA
container toolkit, all of which Base ships. Set `VLLM_AMI_ID` to go back to the PyTorch flavor.

Check what your region resolves to with `make ami`.

### Spot vs on-demand
Spot is roughly 60–70% cheaper but comes with two hard constraints:

*   It can be reclaimed at any time, and a `one-time` request is **not** automatically replaced.
*   EC2 **cannot stop or resize a one-time spot instance** — only terminate it. So `make stop`,
    `make start`, and the `update_vllm_scaling` tool do not work against spot instances. They
    detect this and say so rather than surfacing a raw `UnsupportedOperation`.

Deploy on-demand if you want stop/start or vertical scaling. Use `MARKET_TYPE=spot` for
throwaway benchmark runs.

---

## 🛠 Deployment

### Prerequisites
```bash
# 1. Store the HF token in both places the instance looks
make save-token HF_TOKEN=hf_xxx

# 2. Confirm the instance profile can actually read it back
make check-token-access
```

The instance resolves its token itself, using its instance profile, so the token never enters
UserData (which stays readable via IMDS for the life of the instance). It tries, in order:

1. SSM Parameter Store `/vllm/HF_TOKEN`
2. Secrets Manager secret `hf_token`
3. abort the boot with a `FATAL:` line in `/var/log/cloud-init-output.log`

Both are attempted because reading the SSM copy is a **SecureString under `alias/aws/ssm`**, which
needs `kms:Decrypt` — and `AmazonSSMManagedInstanceCore` does not grant it. In this account the
default profile `aws-elasticbeanstalk-ec2-role` evaluates to:

| Action | Decision |
| --- | --- |
| `ssm:GetParameter` on `/vllm/HF_TOKEN` | allowed |
| `kms:Decrypt` | implicitDeny |
| `secretsmanager:GetSecretValue` on `hf_token` | allowed |
| `secretsmanager:GetSecretValue` on `hf-token` | implicitDeny |

So the Secrets Manager fallback is the path that actually works here. Note the two similarly named
secrets: the instance role can read `hf_token` (underscore) only, while the agent's own
`get_secret()` reads `hf-token` (hyphen) with *your* credentials. Both rotation paths now write
every store the deployment reads — `make save-token` covers the instance-side pair, and the
`save_hf_token` MCP tool writes `hf-token`, `hf_token`, and the SSM parameter in one call, reporting
which ones it reached. If it warns that the instance-side stores were not updated, a fresh launch
will still boot on the old token.

Re-run `make check-token-access` after any IAM change — it simulates the role rather than waiting
for a boot to fail.

### Deploy
```bash
make deploy IAM_PROFILE=vllm-ec2-profile          # on-demand, default
make deploy IAM_PROFILE=vllm-ec2-profile MARKET_TYPE=spot
make wait                                          # block until /health responds
make query PROMPT='"What is SRE?"'
```

`make deploy` resolves the AMI, finds or creates `vllm-devops-sg`, picks a public subnet **in the
same VPC as that security group**, generates the UserData script, and launches the instance.

### ⚠️ Security group state in this account
Two groups named `vllm-devops-sg` already exist, in different VPCs, and **both allow ports 22 and
8080 from `0.0.0.0/0`**:

| Group | VPC | Ingress |
| --- | --- | --- |
| `sg-065c6975cbd84dad3` | `vpc-0bfdd15d906e0c008` (default) | 22, 8080 from `0.0.0.0/0` |
| `sg-0c33f618ffdf23a2d` | `vpc-0b52b7e6ebc50ba38` | 22, 8080 from `0.0.0.0/0` |

vLLM serves **without authentication**, so deploying into either group publishes an open inference
endpoint. New groups are created restricted to your current public IP, but an existing group is
reused as-is — `make deploy` warns when it finds one that is open. To fix it:
```bash
make lock-down                        # replace 0.0.0.0/0 with your current IP on both ports
make deploy INGRESS_CIDR=203.0.113.7/32
```
Because the name is ambiguous, both the Makefile and the agent now select the group **in the
default VPC**; pin it explicitly with `VPC_ID=` (Makefile) or `VLLM_VPC_ID=` (agent).

### Other targets
| Target | Effect |
| --- | --- |
| `make status` | Instances by tag, with type/state/lifecycle/IP |
| `make endpoint` | Prints `http://<public-ip>:8080` |
| `make logs` | Last 100 lines of the vLLM container, over SSM |
| `make clean-container` | Removes the container, leaves the instance up |
| `make stop` / `make start` | On-demand instances only |
| `make destroy` | Terminates all instances with the service tag |
| `make quotas` | On-demand and spot G-instance vCPU quotas |
| `make check-token-access` | Simulates whether the instance profile can read the token |
| `make lock-down` | Replaces `0.0.0.0/0` ingress with `INGRESS_CIDR` |

### Verified account state (us-east-1, account 106059658660)
* `g6.xlarge` offered in all five AZs (`us-east-1a`–`f`)
* Quotas: 16 vCPUs on-demand G, 16 vCPUs spot G — 4 concurrent `g6.xlarge` either way
* Key pair `alinux` exists; default VPC `vpc-0bfdd15d906e0c008` has public subnets in all 5 AZs
* No instances currently carry the `gpu-2b-qat-l4-ec2-agent` tag

---

## 📋 vLLM Run Arguments
```bash
docker run -d --name vllm-server \
  --gpus all \
  --ipc=host \
  --restart always \
  -p 8080:8080 \
  -e HF_TOKEN="$(aws ssm get-parameter --name /vllm/HF_TOKEN --with-decryption --query Parameter.Value --output text)" \
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
*   `--quantization compressed-tensors`: Required to deserialize the QAT INT4 model weights. Applied
    automatically when the model path carries a real quantization marker (`qat`, `w4a16`, `w8a8`,
    `compressed-tensors`, or a trailing `-ct`).
*   `--gpu-memory-utilization 0.95`: Allocates 95% of GPU memory to vLLM's cache.
*   `--kv-cache-dtype fp8`: Cuts KV cache footprint in half, drastically improving concurrency.
*   `--tool-call-parser gemma4` & `--reasoning-parser gemma4`: Critical for correct parsing of tool calls.

The DLAMI ships Docker and the NVIDIA container toolkit. The `apt-get install docker.io` branch in
UserData is only a safety net — on a plain Ubuntu AMI `--gpus all` would still fail without the
container toolkit.

---

## 🔗 Integration with SRE Agent
The agent auto-discovers the endpoint from the EC2 `Name` tag when AWS credentials are present. To
point it somewhere explicitly:
```bash
export VLLM_BASE_URL="http://54.1.2.3:8080"
export MODEL_NAME="google/gemma-4-E2B-it-qat-w4a16-ct"
make run
```

Relevant environment variables:

| Variable | Purpose |
| --- | --- |
| `AWS_REGION` / `AWS_DEFAULT_REGION` | Target region (default `us-east-1`) |
| `VLLM_SERVICE_NAME` | EC2 `Name` tag the agent manages (default `gpu-2b-qat-l4-ec2-agent`) |
| `VLLM_BASE_URL` | Explicit endpoint; skips discovery |
| `VLLM_AMI_ID` | Pin a specific AMI instead of resolving one |
| `VLLM_INSTANCE_PROFILE` | Default instance profile for `deploy_vllm` |
| `VLLM_INGRESS_CIDR` | Ingress CIDR for the security group |
| `VLLM_VPC_ID` | Pins which VPC's `vllm-devops-sg` to use |
| `HF_SSM_PARAMETER` / `HF_SECRET_NAME` | Where the instance looks for its token |
| `MODEL_NAME` | Model id reported to the API |

Note on credentials: `server.py` reads `.aws_creds` **only** when no other credential source
resolves. It used to export those values unconditionally, which shadowed a working profile or role
with whatever was in the file — usually an expired session token, producing `InvalidClientTokenId`
even though `aws` worked in the same shell.
