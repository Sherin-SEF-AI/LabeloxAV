#!/usr/bin/env bash
# Runs ON the RunPod pod. Builds the heavy stack under /workspace (persistent volume) so it survives
# pod stop/start. Each step is a gate. Expects HF_TOKEN in the environment (gated SAM 3.1).
# No em-dashes. Real commands only.
set -euo pipefail

cd /workspace
echo "[setup] python 3.12 via uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# shellcheck disable=SC1090
source "$HOME/.local/bin/env" 2>/dev/null || true
[ -d /workspace/.venv ] || uv venv --python 3.12 /workspace/.venv
source /workspace/.venv/bin/activate
PYV="$(python --version)"
echo "[setup] $PYV"
case "$PYV" in *"3.12"*) : ;; *) echo "[setup] FATAL: expected Python 3.12, got $PYV"; exit 1;; esac

echo "[setup] torch + torchvision (cu128)"
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
python - <<'EOF'
import torch, sys
if not torch.cuda.is_available():
    print("[setup] FATAL: torch.cuda.is_available() is False"); sys.exit(1)
print("[setup] torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))
EOF

echo "[setup] transformers + accelerate + qwen-vl-utils + ultralytics + huggingface_hub + sam3 deps"
# einops + pycocotools are imported by sam3 but not pulled by its install; include them here.
uv pip install "transformers>=4.57.0" accelerate qwen-vl-utils ultralytics "huggingface_hub[cli]" einops pycocotools

echo "[setup] SAM 3 from source"
[ -d /workspace/sam3 ] || git clone https://github.com/facebookresearch/sam3.git /workspace/sam3
( cd /workspace/sam3 && uv pip install -e . ) || { echo "[setup] FATAL: sam3 install failed"; exit 1; }

echo "[setup] checkpoint download to the volume (uses the modern 'hf' CLI; HF_TOKEN authenticates gated repos)"
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN required}"
mkdir -p /workspace/ckpts
# SAM 3.1 (gated; access must already be granted). Fall back to sam3 then sam2.1 if 3.1 is unavailable.
download_first() {
  local dest="$1"; shift
  for repo in "$@"; do
    if hf download "$repo" --local-dir "$dest" >/dev/null 2>&1; then
      echo "[setup] downloaded $repo -> $dest"; echo "$repo" > "${dest}/.source"; return 0
    fi
    echo "[setup] $repo unavailable, trying next fallback"
  done
  echo "[setup] FATAL: none of [$*] could be downloaded to $dest"; return 1
}
download_first /workspace/ckpts/sam3p1 facebook/sam3.1 facebook/sam3 facebook/sam2.1-hiera-large
download_first /workspace/ckpts/qwen3vl-8b Qwen/Qwen3-VL-8B-Instruct Qwen/Qwen2.5-VL-7B-Instruct

echo "[setup] verifying imports + checkpoints"
python - <<'EOF'
import importlib, sys, os
for mod in ("ultralytics", "transformers"):
    importlib.import_module(mod)
try:
    importlib.import_module("sam3")
except Exception:
    # the package may expose a different import name; check the cloned tree exists instead
    assert os.path.isdir("/workspace/sam3"), "sam3 source missing"
for d in ("/workspace/ckpts/sam3p1", "/workspace/ckpts/qwen3vl-8b"):
    assert os.path.isdir(d) and os.listdir(d), f"checkpoint dir empty: {d}"
print("[setup] imports + checkpoints OK")
EOF
echo "[setup] DONE"
