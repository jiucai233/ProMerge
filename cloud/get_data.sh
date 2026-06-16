#!/usr/bin/env bash
# Download LIBERO-Spatial (5.8 GB) into data/libero_spatial/ from the official HF source.
set -euo pipefail
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || true
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="yifengzhu-hf/LIBERO-datasets", repo_type="dataset",
                      local_dir="data", allow_patterns="libero_spatial/*")
print("LIBERO-Spatial ->", p)
PY
echo "✅ data/libero_spatial ready"
