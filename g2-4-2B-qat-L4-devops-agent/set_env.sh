#!/bin/bash

# Environment configuration for the Gemma 4 SRE Agent on GCE (g2-standard-4 / NVIDIA L4).
#
# These values must match the Makefile and server.py defaults. server.py calls
# load_dotenv(override=True), so .env wins over anything exported by the MCP client
# (.mcp.json) or your shell -- a wrong value here silently overrides everything else.
#
# Must be SOURCED to export into your current shell:  source ./set_env.sh

# Canonical values, pinned to match the Makefile and server.py. Deliberately NOT
# inherited from the ambient shell: a stray GOOGLE_CLOUD_PROJECT would otherwise be
# written into .env and then override every other config source. Edit .env to change.
PROJECT_ID="aisprint-491218"
REGION="us-east4"
ZONE="${REGION}-a"
MODEL_NAME="google/gemma-4-E2B-it-qat-w4a16-ct"

if [ -n "$GOOGLE_CLOUD_PROJECT" ] && [ "$GOOGLE_CLOUD_PROJECT" != "$PROJECT_ID" ]; then
    echo "Warning: ignoring ambient GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT; pinning $PROJECT_ID."
fi

# Never clobber an existing .env silently -- it may hold HF_TOKEN or a pinned VLLM_BASE_URL.
if [ -f .env ]; then
    cp .env .env.bak
    echo "Existing .env backed up to .env.bak"
fi

cat <<EOF > .env
GOOGLE_CLOUD_PROJECT=$PROJECT_ID
GOOGLE_CLOUD_LOCATION=$REGION
GOOGLE_CLOUD_ZONE=$ZONE
MODEL_NAME=$MODEL_NAME
EOF

# Carry over HF_TOKEN if it was already set; otherwise server.py falls back to the
# Secret Manager secret `hf-token`.
if [ -n "$HF_TOKEN" ]; then
    echo "HF_TOKEN=$HF_TOKEN" >> .env
fi

# VLLM_BASE_URL is intentionally left unset: discover_vllm_url() resolves the VM's
# external IP via gcloud. Uncomment to pin it to a fixed endpoint.
# echo "VLLM_BASE_URL=http://<vm-external-ip>:8080" >> .env

set -a
# shellcheck disable=SC1091
source .env
set +a

echo "Current Environment:"
cat .env
