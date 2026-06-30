"""Local orchestrator for the pod perception sweep (drivable semantic segmentation; lanes when wired).

Exports frames from MinIO to a temp dir, starts the pod, pushes the frames + cloud/perception_pod.py, runs
it, pulls perception.jsonl, and ingests into DrivableMask + Lane, mirroring the proven scp/ssh path in
cloud/provision_runpod.sh. The pod is ALWAYS stopped after the run (finally) to cap billing, unless
--keep-pod. Sweeps default to clean real frames only (>=1280 wide, selected=true), so quarantined synthetic
noise is never sent to the GPU. Drivable uses SegFormer-Cityscapes (public); gated models read HF_TOKEN from
the pod env.

Run: python -m services.perception.cloud --session <id> --limit 50          # one session, clean frames
     python -m services.perception.cloud --corpus --limit 200 --batches 12   # whole corpus, batched
     python -m services.perception.cloud --frames <id,id,...>                # specific frames
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from uuid import UUID

from core.logging import get_logger
from core.storage import get_object_store

log = get_logger("perception_cloud")

POD_ID = os.environ.get("LBX_POD_ID", "y62vx3km9sq9e6")        # labeloxav-pod
SSH_KEY = os.path.expanduser("~/.runpod/ssh/runpodctl-ssh-key")
API = str(Path(__file__).resolve().parents[2] / "cloud" / "runpod_api.py")
PY = sys.executable
SSHO = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-i", SSH_KEY]


def _api(*args: str) -> dict:
    out = subprocess.run([PY, API, *args], capture_output=True, text=True, env=os.environ)
    if out.returncode != 0:
        raise RuntimeError(f"runpod_api {args[0]} failed: {out.stderr.strip()[:200]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def start_pod_and_wait() -> tuple[str, str]:
    """Start the pod if stopped and poll pod-status for a public ssh endpoint (ip, port)."""
    subprocess.run(["runpodctl", "pod", "start", POD_ID], capture_output=True, text=True)
    for _ in range(36):                                       # up to 6 min
        st = _api("pod-status", POD_ID)
        ssh = st.get("ssh")
        if ssh and ssh.get("ip") and ssh.get("port"):
            log.info("perception.pod_ready", ip=ssh["ip"], port=ssh["port"])
            return str(ssh["ip"]), str(ssh["port"])
        time.sleep(10)
    raise RuntimeError("pod did not expose ssh within 6 min")


def stop_pod() -> None:
    """Stop the pod to cap billing. Best-effort: never raises, so it is safe in a finally block."""
    try:
        _api("stop-pod", POD_ID)
        log.info("perception.pod_stopped", pod=POD_ID)
    except Exception as exc:  # noqa: BLE001  cost-safety must not mask the real error
        log.warning("perception.pod_stop_failed", error=str(exc)[:120])


def _scp(port: str, src: list[str], dst: str) -> None:
    subprocess.run(["scp", *SSHO, "-P", port, "-r", *src, dst], check=True)


def _ssh(ip: str, port: str, cmd: str) -> int:
    return subprocess.run(["ssh", *SSHO, "-p", port, f"root@{ip}", cmd]).returncode


async def export_frames(session_id: UUID | None, limit: int, local_dir: Path,
                        frame_ids: list[str] | None = None, clean_only: bool = True) -> list[dict]:
    """Download frame images from MinIO to local_dir and build the pod manifest. frame_ids targets specific
    frames; else sweep a session (or the whole corpus if session_id is None) in time order up to limit.
    clean_only restricts a sweep to real, selected dashcam frames (>=1280 wide, selected=true), so the pod
    never wastes GPU time on quarantined synthetic noise frames."""
    from sqlalchemy import and_, select

    from db.models import Frame
    from db.session import get_sessionmaker
    store = get_object_store()
    local_dir.mkdir(parents=True, exist_ok=True)
    real = and_(Frame.img_uri.like("s3://labeloxav%"), Frame.width >= 1280, Frame.selected.is_(True))
    async with get_sessionmaker()() as db:
        if frame_ids:
            rows = (await db.execute(select(Frame.frame_id, Frame.img_uri)
                    .where(Frame.frame_id.in_([UUID(f) for f in frame_ids])))).all()
        else:
            stmt = select(Frame.frame_id, Frame.img_uri).order_by(Frame.ts_ns).limit(limit)
            if session_id is not None:
                stmt = stmt.where(Frame.session_id == session_id)
            if clean_only:
                stmt = stmt.where(real)
            rows = (await db.execute(stmt)).all()
    manifest = []
    for fid, uri in rows:
        name = f"{fid}.jpg"
        (local_dir / name).write_bytes(store.get_bytes(uri))
        manifest.append({"frame_id": str(fid), "path": f"/workspace/percep_in/{name}"})
    (local_dir / "manifest.jsonl").write_text("\n".join(json.dumps(m) for m in manifest))
    log.info("perception.exported", frames=len(manifest), dir=str(local_dir))
    return manifest


async def ingest(result_path: Path) -> dict:
    """Write perception.jsonl into DrivableMask + Lane (proposed)."""
    from sqlalchemy import delete

    from db.models import DrivableMask, Frame, Lane
    from db.session import get_sessionmaker
    store = get_object_store()
    n_dr = n_lane = n_err = 0
    rows = [json.loads(ln) for ln in result_path.read_text().splitlines() if ln.strip()]
    async with get_sessionmaker()() as db:
        for rec in rows:
            fid = UUID(rec["frame_id"])
            frame = await db.get(Frame, fid)
            if frame is None:
                continue
            dr = rec.get("drivable")
            if dr and not rec.get("drivable_error"):
                key = f"masks/drivable/{frame.session_id}/{fid}.json"
                uri = store.put_bytes(key, json.dumps({"classes": dr["classes"], "width": dr["width"],
                                      "height": dr["height"]}).encode(), "application/json")
                model_v = dr.get("model", "segformer-cityscapes:pod")
                dm = await db.get(DrivableMask, fid)
                if dm is None:
                    db.add(DrivableMask(frame_id=fid, mask_uri=uri, coverage=dr["coverage"],
                           source="proposed", model_version=model_v))
                else:
                    dm.mask_uri, dm.coverage, dm.model_version, dm.source = uri, dr["coverage"], model_v, "proposed"
                n_dr += 1
            if rec.get("drivable_error") or rec.get("lane_error"):
                n_err += 1
            lanes = rec.get("lanes") or []
            if lanes:
                await db.execute(delete(Lane).where(Lane.frame_id == fid, Lane.source == "proposed"))
                for i, pts in enumerate(lanes):
                    db.add(Lane(frame_id=fid, session_id=frame.session_id, control_points=pts,
                           lane_type="solid", is_ego=False, source="proposed", model_version="mapillary-marking:pod"))
                    n_lane += 1
        await db.commit()
    log.info("perception.ingested", drivable=n_dr, lanes=n_lane, errors=n_err)
    return {"drivable": n_dr, "lanes": n_lane, "errors": n_err}


def _run_sweep_on_pod(root: Path, work: Path, lanes: bool) -> None:
    """Start the pod, push the runtime + frames, run perception, pull the result. Always stops the pod in a
    finally so a crash mid-sweep cannot leave a billing GPU running (override with --keep-pod)."""
    ip, port = start_pod_and_wait()
    try:
        target = f"root@{ip}"
        _ssh(ip, port, "mkdir -p /workspace/percep_in")
        _scp(port, [str(root / "cloud" / "perception_pod.py")], f"{target}:/workspace/")
        _scp(port, [str(p) for p in work.glob("*.jpg")] + [str(work / "manifest.jsonl")],
             f"{target}:/workspace/percep_in/")
        # Lanes are derived from the lane-marking class, so they need no extra install, but they require a
        # model that HAS that class (Mapillary). --lanes therefore switches the run to the Mapillary model
        # for both drivable and lanes; drivable-only stays on the default (public SegFormer-Cityscapes).
        env = "PERCEPTION_SEG_MODEL=facebook/mask2former-swin-large-mapillary-vistas-semantic " if lanes else ""
        lanes_flag = "--lanes" if lanes else ""
        rc = _ssh(ip, port, f"cd /workspace && {env}.venv/bin/python perception_pod.py "
                            f"--manifest percep_in/manifest.jsonl --out percep_out.jsonl {lanes_flag}")
        if rc != 0:
            raise RuntimeError("pod perception run failed (see ssh output)")
        _scp(port, [f"{target}:/workspace/percep_out.jsonl"], str(work / "percep_out.jsonl"))
    finally:
        if os.environ.get("LBX_KEEP_POD") != "1":
            stop_pod()


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=None, help="sweep one session (omit with --corpus for all sessions)")
    ap.add_argument("--corpus", action="store_true", help="sweep across all sessions (clean frames only)")
    ap.add_argument("--frames", default=None, help="comma-separated frame ids to target instead of a sweep")
    ap.add_argument("--limit", type=int, default=20, help="max frames per run (batch size for a corpus sweep)")
    ap.add_argument("--batches", type=int, default=1, help="number of --limit-sized batches to process")
    ap.add_argument("--all-frames", action="store_true", help="include quarantined synthetic frames (default: clean only)")
    ap.add_argument("--lanes", action="store_true")
    ap.add_argument("--keep-pod", action="store_true", help="do not stop the pod after the run")
    args = ap.parse_args()
    if args.keep_pod:
        os.environ["LBX_KEEP_POD"] = "1"

    root = Path(__file__).resolve().parents[2]
    work = root / ".perception_work"
    frame_ids = [f.strip() for f in args.frames.split(",")] if args.frames else None
    sid = UUID(args.session) if args.session else None
    clean_only = not args.all_frames

    totals = {"drivable": 0, "lanes": 0, "errors": 0, "frames": 0}
    for b in range(max(1, args.batches)):
        manifest = await export_frames(sid, args.limit, work, frame_ids, clean_only=clean_only)
        if not manifest:
            log.info("perception.no_frames", batch=b)
            break
        _run_sweep_on_pod(root, work, args.lanes)
        res = await ingest(work / "percep_out.jsonl")
        for k in ("drivable", "lanes", "errors"):
            totals[k] += res.get(k, 0)
        totals["frames"] += len(manifest)
        if frame_ids:                     # explicit frame list is a single pass
            break
    print(json.dumps({"session": args.session, "corpus": args.corpus, **totals}))


if __name__ == "__main__":
    asyncio.run(main())
