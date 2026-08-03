#!/bin/bash
# Pull the compiled model + weights from HF into /data on first start, then serve.
# The 31B artifacts are ~170 GB; the first start downloads them (subsequent starts reuse the volume).
set -e
DATA=/data
mkdir -p "$DATA"
NEED=0
[ -f "$MB_LOAD" ] || NEED=1
[ -f "$MODEL_DIR/model.safetensors.index.json" ] || NEED=1
if [ "$NEED" = "1" ]; then
  echo "[entrypoint] fetching artifacts from HF repo $HF_REPO_INF into $DATA ..."
  python - <<PY
import os
from huggingface_hub import snapshot_download
tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
p = snapshot_download(os.environ["HF_REPO_INF"], local_dir="$DATA", token=tok,
                      allow_patterns=["mb_26b_int8_tp8.pt", "real-gemma4-26B-A4B-it/*"])
print("[entrypoint] downloaded to", p)
PY
else
  echo "[entrypoint] artifacts already present in $DATA"
fi
exec python /app/optb_server_int8.py
