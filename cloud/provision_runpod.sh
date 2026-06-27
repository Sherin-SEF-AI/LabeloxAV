#!/usr/bin/env bash
# LabeloxAV cloud provisioning runbook (local orchestration).
# Provisions a RunPod A100 80GB on-demand pod with a persistent network volume, installs and verifies
# the heavy stack (SAM 3.1 + Qwen3-VL + ultralytics) on the pod, runs a smoke test, then STOPS the pod
# to cap billing. Each step is a gate: on failure the pod is stopped and we exit non zero.
# Constraints: parse JSON (never tables), no em-dashes, keep persistent artifacts under /workspace.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${ROOT}/provision.log"
PY="${ROOT}/.venv/bin/python"
API="${ROOT}/cloud/runpod_api.py"
POD_ID=""
VOL_ID="${LBX_RUNPOD_VOLUME_ID:-}"
START_TS="$(date +%s)"

log()  { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
die()  {
  log "FATAL: $*"
  if [ -n "$POD_ID" ]; then
    log "stopping pod ${POD_ID} to protect the budget"
    "$PY" "$API" stop-pod "$POD_ID" >>"$LOG" 2>&1 || log "WARN: stop-pod failed, stop it manually"
  fi
  exit 1
}

log "=== LabeloxAV cloud provision started ==="

# Step 0: preconditions (hard gate)
[ -n "${RUNPOD_API_KEY:-}" ] || die "RUNPOD_API_KEY is not set. export it and retry."
[ -n "${HF_TOKEN:-}" ]       || die "HF_TOKEN is not set (needed for gated SAM 3.1). export it and retry."
[ -x "$PY" ]                 || die "project venv python not found at ${PY} (run make install)."

# Step 1: install + configure runpodctl (used for stop/exec; the API client does the rest)
if ! command -v runpodctl >/dev/null 2>&1; then
  arch="$(uname -m)"; case "$arch" in x86_64) a=amd64;; aarch64|arm64) a=arm64;; *) die "unsupported arch ${arch}";; esac
  log "installing runpodctl (${a})"
  wget -q "https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-${a}" -O /tmp/runpodctl \
    && chmod +x /tmp/runpodctl && sudo mv /tmp/runpodctl /usr/local/bin/runpodctl || die "runpodctl install failed"
fi
runpodctl config --apiKey="$RUNPOD_API_KEY" >>"$LOG" 2>&1 || die "runpodctl config failed"

# Report any pod already billing (do not auto-create blindly on top)
log "existing pods (JSON):"; runpodctl get pod 2>>"$LOG" | tee -a "$LOG" || log "WARN: could not list pods"

# Step 2: pick a GPU (A100 80GB, else H100), parse JSON
log "querying GPU types"
GPU_JSON="$("$PY" "$API" gpu-types)" || die "gpu-types query failed: ${GPU_JSON}"
GPU_ID="$(printf '%s' "$GPU_JSON" | "$PY" - <<'EOF'
import json,sys
types=json.load(sys.stdin)
def find(sub):
    for t in types:
        if sub.lower() in (t.get("displayName") or "").lower() and (t.get("memoryInGb") or 0) >= 80:
            return t["id"]
    return ""
gid=find("A100") or find("H100")
print(gid)
EOF
<<<"$GPU_JSON")" || true
[ -n "$GPU_ID" ] || die "no A100 80GB or H100 available. GPU types: ${GPU_JSON}"
log "selected GPU id: ${GPU_ID}"

# Image: a CUDA 12.8 devel base; we layer Python 3.12 via uv on the pod (Step 5). Override with
# LBX_CLOUD_IMAGE. The default tag is logged; verify it exists in the RunPod catalog on first run.
IMAGE="${LBX_CLOUD_IMAGE:-runpod/pytorch:2.8.0-py3.11-cuda12.8.0-devel-ubuntu22.04}"
log "using image: ${IMAGE} (override with LBX_CLOUD_IMAGE; verify the tag exists)"

# Step 3: network volume (persistent). runpodctl cannot create one; use the REST API, or reuse an id.
if [ -z "$VOL_ID" ]; then
  if [ -n "${LBX_RUNPOD_DC:-}" ]; then
    log "creating network volume labeloxav-vol (100GB) in DC ${LBX_RUNPOD_DC}"
    VOL_JSON="$("$PY" "$API" create-volume labeloxav-vol 100 "$LBX_RUNPOD_DC")" || die "volume create failed: ${VOL_JSON}"
    VOL_ID="$(printf '%s' "$VOL_JSON" | "$PY" -c 'import json,sys;print(json.load(sys.stdin).get("id",""))')"
  fi
fi
[ -n "$VOL_ID" ] || die "no network volume. Create 'labeloxav-vol' in the RunPod console (pick a DC), then re-run with: export LBX_RUNPOD_VOLUME_ID=<id>"
log "network volume id: ${VOL_ID}"

# Step 4: create the pod, poll to RUNNING (5 min), extract ssh
log "deploying on-demand pod"
POD_JSON="$("$PY" "$API" create-pod "$GPU_ID" "$IMAGE" "$VOL_ID")" || die "pod create failed: ${POD_JSON}"
POD_ID="$(printf '%s' "$POD_JSON" | "$PY" -c 'import json,sys;print(json.load(sys.stdin).get("id",""))')"
[ -n "$POD_ID" ] || die "pod id empty: ${POD_JSON}"
log "pod id: ${POD_ID}; waiting for RUNNING + ssh"
SSH_IP=""; SSH_PORT=""
for i in $(seq 1 30); do
  S="$("$PY" "$API" pod-status "$POD_ID")" || true
  read -r SSH_IP SSH_PORT < <(printf '%s' "$S" | "$PY" -c 'import json,sys;d=json.load(sys.stdin);s=d.get("ssh") or {};print(s.get("ip","") , s.get("port",""))')
  if [ -n "$SSH_IP" ] && [ -n "$SSH_PORT" ]; then break; fi
  sleep 10
done
[ -n "$SSH_IP" ] && [ -n "$SSH_PORT" ] || die "pod did not become ready with ssh in 5 min"
log "pod ready: ssh root@${SSH_IP} -p ${SSH_PORT}"

# Step 5 + 6: set up the stack on the pod and smoke test (SSH). Keys must be on your RunPod account.
SSHO="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
log "uploading setup + smoke scripts to the pod"
scp $SSHO -P "$SSH_PORT" "${ROOT}/cloud/setup_pod.sh" "${ROOT}/cloud/smoke_test.py" "root@${SSH_IP}:/workspace/" >>"$LOG" 2>&1 \
  || die "scp to pod failed (is your SSH key on your RunPod account?)"
log "running setup_pod.sh on the pod (this installs the stack + downloads checkpoints)"
ssh $SSHO -p "$SSH_PORT" "root@${SSH_IP}" "HF_TOKEN='${HF_TOKEN}' bash /workspace/setup_pod.sh" 2>&1 | tee -a "$LOG" \
  || die "pod setup failed"
log "running smoke test on the pod"
ssh $SSHO -p "$SSH_PORT" "root@${SSH_IP}" "cd /workspace && .venv/bin/python smoke_test.py" 2>&1 | tee -a "$LOG" || true
scp $SSHO -P "$SSH_PORT" "root@${SSH_IP}:/workspace/smoke_test_result.json" "${ROOT}/cloud/smoke_test_result.json" >>"$LOG" 2>&1 || log "WARN: could not fetch smoke result"
SMOKE="FAIL"
if [ -f "${ROOT}/cloud/smoke_test_result.json" ]; then
  SMOKE="$("$PY" -c 'import json;print(json.load(open("'"${ROOT}"'/cloud/smoke_test_result.json")).get("verdict","FAIL"))' 2>/dev/null || echo FAIL)"
fi
log "smoke test verdict: ${SMOKE}"

# Step 7: stop the pod (never delete), report
ELAPSED=$(( ($(date +%s) - START_TS) / 60 ))
log "elapsed ${ELAPSED} min. The pod bills on wall-clock while running."
"$PY" "$API" stop-pod "$POD_ID" >>"$LOG" 2>&1 || log "WARN: stop-pod failed, stop ${POD_ID} manually"
log "=== SUMMARY ==="
log "pod_id=${POD_ID} volume_id=${VOL_ID} image=${IMAGE} gpu_id=${GPU_ID} smoke=${SMOKE}"
log "restart later with: runpodctl start pod ${POD_ID}   (volume + venv + checkpoints persist)"
[ "$SMOKE" = "PASS" ] || die "smoke test did not PASS (pod stopped). See provision.log."
log "DONE: stack verified and pod stopped."
