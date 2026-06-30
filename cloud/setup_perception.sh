#!/usr/bin/env bash
# Pod-side lane stack (CLRerNet) in an ISOLATED venv so its older mmdet/mmcv pins never disturb the
# smoke-verified SAM 3.1 / torch venv at /workspace/.venv that drivable depends on. Best-effort: if it
# fails, drivable (SAM 3.1 PCS) is unaffected and lanes are reported empty with the error.
set -u
LANE_VENV=/workspace/.venv-lanes
CLR=/workspace/CLRerNet

if [ ! -d "$CLR" ]; then
  git clone --depth 1 https://github.com/hirotomusiker/CLRerNet.git "$CLR" || { echo "[lanes] clone failed"; exit 1; }
fi
[ -d "$LANE_VENV" ] || uv venv --python 3.11 "$LANE_VENV"
# CLRerNet (recent) targets mmdet 3.x / mmcv 2.x, which have torch 2.x wheels; keep it off the drivable venv.
"$LANE_VENV/bin/python" -m pip install --quiet --upgrade pip
"$LANE_VENV/bin/pip" install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu128 || true
"$LANE_VENV/bin/pip" install --quiet -U openmim && "$LANE_VENV/bin/mim" install "mmcv>=2.0.0" "mmdet>=3.0.0" || { echo "[lanes] mmcv/mmdet install failed"; exit 1; }
( cd "$CLR" && "$LANE_VENV/bin/pip" install --quiet -e . ) || { echo "[lanes] CLRerNet editable install failed"; exit 1; }

# CULane DLA-34 checkpoint (CLRerNet release). Override CLRERNET_CKPT/CONFIG if a different one is staged.
mkdir -p /workspace/ckpts
CK=/workspace/ckpts/clrernet_culane_dla34_ema.pth
[ -f "$CK" ] || wget -q -O "$CK" "https://github.com/hirotomusiker/CLRerNet/releases/download/v0.1.0/clrernet_culane_dla34_ema.pth" || echo "[lanes] checkpoint download failed; set CLRERNET_CKPT to a staged file"
echo "[lanes] CLRerNet setup complete (venv=$LANE_VENV)"
