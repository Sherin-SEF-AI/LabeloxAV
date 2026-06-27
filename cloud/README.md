# LabeloxAV cloud (hybrid GPU): RunPod A100 provisioning

The local RTX 5080 (16 GB) handles interactive review and small/quick training. Heavy real-model work
(SAM 3.1 PCS + Qwen3-VL labeling sweeps, YOLO26 training) runs on an on-demand RunPod A100 80 GB. This
runbook provisions the pod, installs and verifies the stack, runs a smoke test, then STOPS the pod so it
stops billing. The persistent network volume keeps the venv and checkpoints so the next run resumes fast.

## Preconditions (hard gates)

- `export RUNPOD_API_KEY=...`
- `export HF_TOKEN=...` and you must already have access to the gated `facebook/sam3` checkpoints on
  Hugging Face (request it on the model page first).
- Your SSH public key must be registered on your RunPod account (Settings, SSH Public Keys), so the
  runbook can `ssh`/`scp` into the pod.
- A network volume: either pre-create `labeloxav-vol` in the RunPod console and
  `export LBX_RUNPOD_VOLUME_ID=<id>`, or set `export LBX_RUNPOD_DC=<dataCenterId>` and the runbook will
  create one (100 GB). The pod is placed in the volume's data center.

## Run

```bash
make cloud-provision     # provision, verify, smoke test, then STOP the pod
# or directly:
bash cloud/provision_runpod.sh
```

Everything is logged to `provision.log`. The smoke result lands in `cloud/smoke_test_result.json`.

Restart the (stopped) pod later without reinstalling anything:

```bash
runpodctl start pod <POD_ID>
```

## What runs where

- `cloud/provision_runpod.sh` (local): preconditions, install runpodctl, pick GPU (A100 80GB, else
  H100), pick a CUDA 12.8 devel image (override with `LBX_CLOUD_IMAGE`), ensure the network volume,
  create the pod, wait for SSH, push and run the on-pod scripts, then stop the pod.
- `cloud/runpod_api.py` (local): thin RunPod GraphQL/REST client (JSON only). runpodctl cannot list GPU
  types or create network volumes, so those go through the API here.
- `cloud/setup_pod.sh` (on pod): Python 3.12 via uv, torch cu128 (asserts CUDA), transformers 4.57+,
  ultralytics, SAM 3 from source, and checkpoint downloads into `/workspace/ckpts`.
- `cloud/smoke_test.py` (on pod): one frame through YOLO26, SAM 3.1 PCS, and Qwen3-VL; writes a
  PASS/FAIL verdict.

## Model availability and fallback policy

Verified at build time (HF API): `facebook/sam3.1`, `facebook/sam3`, `Qwen/Qwen3-VL-8B-Instruct` all
resolve. The scripts still fall back to the nearest available model and record the substitution:

- SAM 3.1 -> `facebook/sam3.1`, else `facebook/sam3`, else `facebook/sam2.1-hiera-large`.
- YOLO26 -> `yolo26n.pt`, else `yolo11n.pt` (whatever the installed ultralytics ships).
- Qwen3-VL-8B -> `Qwen/Qwen3-VL-8B-Instruct`, else `Qwen/Qwen2.5-VL-7B-Instruct`.

The SAM 3.1 and Qwen3-VL invocation APIs drift between releases. `smoke_test.py` tries a couple of entry
points and captures the traceback in the result JSON, so on the first pod run you can adjust the call to
the installed package version. Treat the first provision as the verification run.

## Budget

A100 80 GB bills on wall-clock while RUNNING. The runbook stops the pod after the smoke test. Cap is
`cloud.budget_cap_usd` (50). Stopping halts compute billing; the volume retains a small storage charge.

## Boundary

This runbook does NOT run the production labeling sweep or training. Those are a `compute_target=cloud`
job (see `services/training/cloud.py` for the data-sync contract) and the next directive, run once fleet
frames are staged on `/workspace/data`.
