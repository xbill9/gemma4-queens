#!/bin/bash
# Provision a v5e-1 (v5litepod-1) Flex-start Queued Resource running the PyTorch
# TPU startup script instead of the vLLM one.
#
# Why a Queued Resource and not a GCE flex-start instance: the GCE path in the
# tpu-pytorch-v5e1 MCP server only knows ct6e/ct5p machine types, and this rig's
# capacity for the single-chip v5e shape has only ever been granted through the
# queued-resources API in us-west4-a. The create call below is the same one
# server.py's _create_queued_resource issues, with the vLLM startup script
# swapped for startup_script_pytorch_template.sh.
#
# Deployment parameters come from tpu.env (the rig's single source of truth);
# the PyTorch backend specifics are TPU_BACKEND_* and stay out of git — pass
# them in the environment or accept the defaults resolved below.
#
#   ./build_pytorch_v5e1.sh [resource_id]
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
[ -f "$HERE/tpu.env" ] && source "$HERE/tpu.env"

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:?set GOOGLE_CLOUD_PROJECT or provide tpu.env}"
ZONE="${GOOGLE_CLOUD_ZONE:-us-west4-a}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-v5litepod-1}"
# v5e is spelled v5litepod to gcloud, and its Flex-start runtime is the tpuv5-lite
# alpha image (NOT v2-alpha-tpuv6e, which the tpu-pytorch-v5e1 server hardcodes).
TPU_RUNTIME_VERSION="${TPU_RUNTIME_VERSION:-v2-alpha-tpuv5-lite}"
RESOURCE_ID="${1:-${RESOURCE_ID:-torchtpu-v5e1-qr}}"
MAX_RUN_DURATION="${MAX_RUN_DURATION:-4h}"
VALID_UNTIL_DURATION="${VALID_UNTIL_DURATION:-4h}"

# The PyTorch startup template lives with the pytorch project, not this rig.
TEMPLATE="${TPU_PYTORCH_TEMPLATE:-$HOME/tpu-pytorch-v5e1/.claude/skills/tpu-management/mcp/startup_script_pytorch_template.sh}"

# Backend install config. No public defaults exist for the package index, so
# these mirror what the tpu-pytorch-v5e1 server expects in its environment.
TPU_BACKEND_PIP_SPEC="${TPU_BACKEND_PIP_SPEC:-torch_tpu torch==2.13.0+cpu}"
TPU_BACKEND_INDEX="${TPU_BACKEND_INDEX:-}"
TPU_BACKEND_WHEELS_GCS="${TPU_BACKEND_WHEELS_GCS:-gs://${PROJECT_ID}-torchtpu-wheels}"
TPU_BACKEND_PIP_EXTRAS="${TPU_BACKEND_PIP_EXTRAS:-transformers compressed-tensors 'jinja2>=3.1' 'setuptools<81'}"
TPU_BACKEND_ENV_EXPORTS="${TPU_BACKEND_ENV_EXPORTS:-}"

if [ ! -f "$TEMPLATE" ]; then
    echo "❌ PyTorch startup template not found: $TEMPLATE" >&2
    echo "   Set TPU_PYTORCH_TEMPLATE to its path." >&2
    exit 1
fi
if [ -z "$TPU_BACKEND_INDEX" ] && [ -z "$TPU_BACKEND_WHEELS_GCS" ]; then
    echo "❌ Set TPU_BACKEND_INDEX (authenticated pip index) or TPU_BACKEND_WHEELS_GCS" >&2
    echo "   (project-owned wheel mirror). The backend wheels are not on public PyPI." >&2
    exit 1
fi

# The template is consumed by str.format() — render it with Python so the {{ }}
# escapes in its bash body survive. sed would corrupt them.
SCRIPT_FILE=$(mktemp -t tpu-pytorch-startup-XXXXXX.sh)
trap 'rm -f "$SCRIPT_FILE"' EXIT
TEMPLATE="$TEMPLATE" OUT="$SCRIPT_FILE" \
PROJECT_ID="$PROJECT_ID" ZONE="$ZONE" \
PIP_SPEC="$TPU_BACKEND_PIP_SPEC" INDEX="$TPU_BACKEND_INDEX" \
WHEELS_GCS="$TPU_BACKEND_WHEELS_GCS" EXTRAS="$TPU_BACKEND_PIP_EXTRAS" \
ENV_EXPORTS="$TPU_BACKEND_ENV_EXPORTS" \
python3 - <<'PYEOF'
import os

with open(os.environ["TEMPLATE"]) as f:
    template = f.read()
rendered = template.format(
    project_id=os.environ["PROJECT_ID"],
    zone=os.environ["ZONE"],
    torch_pip_spec=os.environ["PIP_SPEC"],
    torch_index=os.environ["INDEX"],
    wheels_gcs=os.environ["WHEELS_GCS"],
    torch_pip_extras=os.environ["EXTRAS"],
    env_exports=os.environ["ENV_EXPORTS"],
)
with open(os.environ["OUT"], "w") as f:
    f.write(rendered)
PYEOF

if [ "${DRY_RUN:-0}" != "0" ]; then
    echo "✅ Startup script rendered ($(wc -l < "$SCRIPT_FILE") lines). DRY_RUN set — not creating anything."
    cp "$SCRIPT_FILE" "${DRY_RUN_OUT:-./rendered-pytorch-startup.sh}"
    echo "   Wrote ${DRY_RUN_OUT:-./rendered-pytorch-startup.sh}"
    exit 0
fi

echo "🚀 Creating Queued Resource $RESOURCE_ID"
echo "   zone=$ZONE accelerator=$ACCELERATOR_TYPE runtime=$TPU_RUNTIME_VERSION"
echo "   wheels=$TPU_BACKEND_WHEELS_GCS spec=$TPU_BACKEND_PIP_SPEC"

gcloud alpha compute tpus queued-resources create "$RESOURCE_ID" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --node-id="${RESOURCE_ID}-node" \
    --accelerator-type="$ACCELERATOR_TYPE" \
    --runtime-version="$TPU_RUNTIME_VERSION" \
    --provisioning-model=flex-start \
    --max-run-duration="$MAX_RUN_DURATION" \
    --valid-until-duration="$VALID_UNTIL_DURATION" \
    --labels=purpose=flex-start,workload=pytorch \
    --metadata-from-file=startup-script="$SCRIPT_FILE"

cat <<EOF

Watch it come up:
  gcloud alpha compute tpus queued-resources describe $RESOURCE_ID --zone=$ZONE --project=$PROJECT_ID --format='value(state.state)'

Once ACTIVE, the startup script's progress is in the node's serial console and
/var/log/tpu-startup.log. Success marker: 'TPU environment ready.'
  gcloud compute tpus tpu-vm ssh ${RESOURCE_ID}-node --zone=$ZONE --project=$PROJECT_ID --command 'sudo tail -40 /var/log/tpu-startup.log'

Re-run the compile smoke test any time:
  gcloud compute tpus tpu-vm ssh ${RESOURCE_ID}-node --zone=$ZONE --project=$PROJECT_ID --command 'python3.12 /opt/tpu_smoke.py'
EOF
