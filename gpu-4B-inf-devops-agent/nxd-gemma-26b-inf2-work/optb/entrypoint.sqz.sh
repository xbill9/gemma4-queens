#!/bin/bash
# Slim int8-squeeze 26B-A4B on a single inf2.xlarge. Pull the compiled neff + host embedding table +
# config bundle from the HF inference repo into /data on first start, then serve. Artifacts are ~28 GB
# (neff 26.7 GB + embed_tokens 1.48 GB + cfg) — a fraction of the 24xlarge int8 build (which needs the
# full 51 GB weights). Subsequent starts reuse the volume.
set -e
DATA=/data
mkdir -p "$DATA"
NEED=0
[ -f "$MB_LOAD" ] || NEED=1
[ -f "$EMBED" ] || NEED=1
[ -f "$MODEL_DIR/config.json" ] || NEED=1
if [ "$NEED" = "1" ]; then
  echo "[entrypoint] fetching slim artifacts from HF repo $HF_REPO_INF into $DATA ..."
  python - <<PY
import os
from huggingface_hub import snapshot_download
tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
p = snapshot_download(os.environ["HF_REPO_INF"], local_dir="$DATA", token=tok,
                      allow_patterns=["sqz_neff.pt", "embed_tokens.pt", "cfg/*"])
print("[entrypoint] downloaded to", p)
PY
else
  echo "[entrypoint] artifacts already present in $DATA"
fi
exec python /app/optb_server_sqz.py
