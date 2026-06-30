"""Local orchestrator for the pod perception sweep (drivable via SAM 3.1 PCS, lanes via CLRerNet).

Exports a session's frames from MinIO to a temp dir, starts the existing pod (runpodctl start), waits for
ssh, pushes the frames + cloud/perception_pod.py, runs it, pulls perception.jsonl, and ingests the results
into DrivableMask + Lane. Mirrors the proven scp/ssh path in cloud/provision_runpod.sh. The pod is stopped
by the caller (make target) to cap billing. Drivable uses the smoke-verified SAM 3.1 PCS loader; lanes are
guarded on the pod so a CLRerNet failure never blocks drivable.

Run: .venv/bin/python -m services.perception.cloud --session <id> --limit 20 [--lanes] [--keep-pod]
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
    subprocess.run(["runpodctl", "start", "pod", POD_ID], capture_output=True, text=True)
    for _ in range(36):                                       # up to 6 min
        st = _api("pod-status", POD_ID)
        ssh = st.get("ssh")
        if ssh and ssh.get("ip") and ssh.get("port"):
            log.info("perception.pod_ready", ip=ssh["ip"], port=ssh["port"])
            return str(ssh["ip"]), str(ssh["port"])
        time.sleep(10)
    raise RuntimeError("pod did not expose ssh within 6 min")


def _scp(port: str, src: list[str], dst: str) -> None:
    subprocess.run(["scp", *SSHO, "-P", port, "-r", *src, dst], check=True)


def _ssh(ip: str, port: str, cmd: str) -> int:
    return subprocess.run(["ssh", *SSHO, "-p", port, f"root@{ip}", cmd]).returncode


async def export_frames(session_id: UUID, limit: int, local_dir: Path,
                        frame_ids: list[str] | None = None) -> list[dict]:
    """Download frame images from MinIO to local_dir and build the pod manifest. frame_ids targets specific
    frames (e.g. the one a reviewer flagged); otherwise sweep the session in time order up to limit."""
    from sqlalchemy import select

    from db.models import Frame
    from db.session import get_sessionmaker
    store = get_object_store()
    local_dir.mkdir(parents=True, exist_ok=True)
    async with get_sessionmaker()() as db:
        if frame_ids:
            rows = (await db.execute(select(Frame.frame_id, Frame.img_uri)
                    .where(Frame.frame_id.in_([UUID(f) for f in frame_ids])))).all()
        else:
            rows = (await db.execute(select(Frame.frame_id, Frame.img_uri)
                    .where(Frame.session_id == session_id).order_by(Frame.ts_ns).limit(limit))).all()
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
    from sqlalchemy import delete, select

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
                           lane_type="solid", is_ego=False, source="proposed", model_version="clrernet:pod"))
                    n_lane += 1
        await db.commit()
    log.info("perception.ingested", drivable=n_dr, lanes=n_lane, errors=n_err)
    return {"drivable": n_dr, "lanes": n_lane, "errors": n_err}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=None)
    ap.add_argument("--frames", default=None, help="comma-separated frame ids to target instead of a sweep")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--lanes", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    work = root / ".perception_work"
    frame_ids = [f.strip() for f in args.frames.split(",")] if args.frames else None
    sid = UUID(args.session) if args.session else None
    await export_frames(sid, args.limit, work, frame_ids)

    ip, port = start_pod_and_wait()
    target = f"root@{ip}"
    _ssh(ip, port, "mkdir -p /workspace/percep_in")
    push = [str(root / "cloud" / "perception_pod.py")]
    if args.lanes:
        push.append(str(root / "cloud" / "setup_perception.sh"))
    _scp(port, push, f"{target}:/workspace/")
    _scp(port, [str(p) for p in work.glob("*.jpg")] + [str(work / "manifest.jsonl")],
         f"{target}:/workspace/percep_in/")
    if args.lanes:
        _ssh(ip, port, "bash /workspace/setup_perception.sh || echo '[lanes] CLRerNet setup failed (drivable unaffected)'")
    lanes_flag = "--lanes" if args.lanes else ""
    rc = _ssh(ip, port, f"cd /workspace && .venv/bin/python perception_pod.py "
                        f"--manifest percep_in/manifest.jsonl --out percep_out.jsonl {lanes_flag}")
    if rc != 0:
        raise RuntimeError("pod perception run failed (see ssh output)")
    _scp(port, [f"{target}:/workspace/percep_out.jsonl"], str(work / "percep_out.jsonl"))
    res = await ingest(work / "percep_out.jsonl")
    print(json.dumps({"session": args.session, **res}))


if __name__ == "__main__":
    asyncio.run(main())
