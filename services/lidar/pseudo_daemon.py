"""Off-hours hook: give the camera-only fleet 3D coverage, batch by batch, and an ego trajectory to hang it on.

The pseudo-LiDAR lift has existed end to end since the 3D module landed: a metric depth model, a
back-projection into the ego frame, and a cuboid lifter that snaps a 2D box to the ground plane. It has
covered 154 frames of 41,752, which is 0.4%, for one reason: it only ever ran from a manual, bounded
router call that a person had to make per session.

Nothing here is a new lifter. It is the scheduling and the batching that were missing, plus the ego
trajectory the cuboids need to be comparable across frames at all.

The GPU discipline is the strictest of any hook in this program, because the depth model is the largest
thing the engine loads. Clouds are built `PSEUDO_BATCH` frames per `gpu_slot` hold, `training_holds_gpu`
and a VRAM floor are checked between batches, and every batch commits as its own revertible `AgentRun`.
A batch that would run the host out of memory waits rather than pushing it over.

Ego pose is built first and only once per session, because it is CPU work that every later batch reads,
and because a lift placed in a frame nobody can locate is a cuboid in a coordinate system of one.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import AgentRun, EgoPose, Frame, PointCloud
from db.models import Session as DbSession

log = get_logger("pseudo_daemon")

KIND = "pseudo_lift"
BATCH_KIND = "pseudo_batch"
POSE_KIND = "ego_pose_build"
CREATED_BY = "scheduler"

# Frames per GPU slot hold. The depth model is resident for the whole batch, so this trades how long a
# training job waits for the card against how often the model is reloaded.
PSEUDO_BATCH = 64
# Sessions touched per night. Coverage is a long game and a night that tried to finish it would hold the
# card until morning.
SESSIONS_PER_NIGHT = 2
# Free VRAM the depth model needs before a batch may start. Measured on the first run and stored on the
# agent run, so the second night uses the real figure rather than this estimate.
EST_DEPTH_MB = 3500.0
MIN_FREE_VRAM_MB = EST_DEPTH_MB
# How long a batch waits for headroom before giving the night up and saying so.
MAX_WAIT_S = 900.0
# Host memory ceiling. The same guard the copy-paste generator uses: decoding a batch of 1080p frames and
# holding their clouds is real host memory, and this box runs a database and an API on the same RAM.
MEMORY_CEILING_FRAC = 0.90


async def _uncovered_sessions(db: AsyncSession, limit: int) -> list[tuple[uuid.UUID, int, int]]:
    """Real sessions with the most frames still lacking a cloud, worst first.

    Worst first because coverage is the point: a session at 0% teaches more per GPU minute than the tail
    of one already at 90%, and the ordering makes the daemon's progress legible session by session.
    """
    covered = (select(PointCloud.session_id, PointCloud.ts_ns)
               .where(PointCloud.source == "pseudo").subquery())
    stmt = (select(Frame.session_id,
                   func.count(Frame.frame_id).label("frames"),
                   func.count(covered.c.ts_ns).label("covered"))
            .join(DbSession, DbSession.session_id == Frame.session_id)
            .outerjoin(covered, (covered.c.session_id == Frame.session_id)
                       & (covered.c.ts_ns == Frame.ts_ns))
            .where(DbSession.origin == "real", Frame.origin == "real",
                   Frame.img_uri.isnot(None), Frame.selected.is_(True))
            .group_by(Frame.session_id)
            .having(func.count(Frame.frame_id) > func.count(covered.c.ts_ns))
            .order_by((func.count(Frame.frame_id) - func.count(covered.c.ts_ns)).desc())
            .limit(limit))
    return [(sid, int(n), int(c)) for sid, n, c in (await db.execute(stmt)).all()]


async def _pending_frames(db: AsyncSession, session_id: uuid.UUID, limit: int) -> list[dict]:
    """Frames of one session that have no cloud at their timestamp yet, oldest first."""
    covered = set((await db.execute(
        select(PointCloud.ts_ns).where(PointCloud.session_id == session_id,
                                       PointCloud.source == "pseudo"))).scalars().all())
    rows = (await db.execute(
        select(Frame.frame_id, Frame.ts_ns, Frame.cam_id, Frame.img_uri)
        .where(Frame.session_id == session_id, Frame.origin == "real",
               Frame.img_uri.isnot(None), Frame.selected.is_(True))
        .order_by(Frame.ts_ns))).all()
    out = []
    for fid, ts, cam, uri in rows:
        if int(ts) in covered:
            continue
        out.append({"frame_id": fid, "ts_ns": int(ts), "cam_id": cam, "img_uri": uri})
        if len(out) >= limit:
            break
    return out


async def _wait_for_gpu(db: AsyncSession, report: dict) -> str | None:
    """Block until the card has room and training is not holding it, or give up and name what stopped it."""
    import asyncio
    import time

    from services.labelops.class_precision import free_vram_mb
    from services.training.gpu_lease import training_holds_gpu

    started = time.monotonic()
    while time.monotonic() - started < MAX_WAIT_S:
        if await training_holds_gpu(db):
            report["waited_for_training"] = report.get("waited_for_training", 0) + 1
            await asyncio.sleep(30)
            continue
        free = await free_vram_mb()
        if free is not None and free < MIN_FREE_VRAM_MB:
            report["waited_for_vram"] = report.get("waited_for_vram", 0) + 1
            await asyncio.sleep(30)
            continue
        return None
    return f"waited {MAX_WAIT_S:.0f}s for the GPU and it stayed busy"


def _memory_pressure() -> float | None:
    from services.hardening.resources import host

    try:
        return float(host().get("memory_used_frac") or 0.0)
    except Exception:  # noqa: BLE001 - a reading that fails must not decide the job
        return None


async def lift_session_batch(session_id: uuid.UUID, frames: list[dict], *, parent_run_id: uuid.UUID,
                             created_by: str) -> dict:
    """Build clouds for one batch of frames and commit them as one revertible child run."""
    import cv2
    import numpy as np

    from compute.worker.jobs.pointcloud_build import CALIB_VERSION
    from core.gpu_slot import gpu_slot
    from core.storage import get_object_store
    from db.session import get_sessionmaker
    from services.lidar.detect3d.run import lift_frame
    from services.lidar.ingest.pseudo import lift_frame_group
    from services.lidar.ingest.store import store_cloud

    batch_run_id = uuid.uuid4()
    store = get_object_store()
    built: list[str] = []
    lifted: list[uuid.UUID] = []
    failed = 0

    async with gpu_slot(f"pseudo_lift:{session_id}", timeout_s=120):
        for fr in frames:
            try:
                buf = np.frombuffer(store.get_bytes(fr["img_uri"]), dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            except Exception:  # noqa: BLE001 - a missing blob skips one frame, never the batch
                img = None
            if img is None:
                failed += 1
                continue
            try:
                cloud = lift_frame_group({fr["cam_id"]: img}, ts_ns=fr["ts_ns"],
                                         calibration_version=CALIB_VERSION)
                res = await store_cloud(cloud, session_id, source="pseudo",
                                        depth_model=cloud.depth_model,
                                        calibration_version=CALIB_VERSION)
            except Exception as exc:  # noqa: BLE001 - one bad frame must not end the batch
                log.warning("pseudo.lift_failed", frame_id=str(fr["frame_id"]), error=str(exc)[:200])
                failed += 1
                continue
            built.append(str(res.get("cloud_id")))
            lifted.append(fr["frame_id"])

    # A cloud with nothing lifted from it is storage, not a label. The cuboid lift is CPU work over an
    # already-built cloud, so it runs outside the GPU slot: holding the card through it would keep a
    # training job waiting on geometry that does not need the card.
    cuboids = 0
    for fid in lifted:
        try:
            res3d = await lift_frame(fid)
            cuboids += int(res3d.get("cuboids") or 0)
        except Exception as exc:  # noqa: BLE001 - one frame's geometry must not end the batch
            log.warning("pseudo.lift_frame_failed", frame_id=str(fid), error=str(exc)[:200])

    async with get_sessionmaker()() as db:
        db.add(AgentRun(run_id=batch_run_id, kind=BATCH_KIND,
                        scope={"parent_run_id": str(parent_run_id), "session_id": str(session_id)},
                        status="committed", policy={},
                        counts={"clouds": len(built), "failed": failed, "cuboids": cuboids},
                        changes={"cloud_ids": built, "session_id": str(session_id)},
                        critic={}, created_by=created_by))
        await db.commit()
    log.info("pseudo.batch.committed", run_id=str(batch_run_id), clouds=len(built), failed=failed,
             cuboids=cuboids)
    return {"run_id": str(batch_run_id), "clouds": len(built), "failed": failed, "cuboids": cuboids}


async def revert_batch(db: AsyncSession, run: AgentRun) -> dict:
    """Delete the clouds one batch built, and the blobs behind them.

    Cuboids lifted from a cloud cascade with it through `object_3d`'s foreign key, so a revert takes the
    3D labels the cloud produced with it rather than leaving them pointing at nothing.
    """
    from sqlalchemy import delete

    from core.storage import get_object_store

    ids = [uuid.UUID(c) for c in ((run.changes or {}).get("cloud_ids") or [])]
    if not ids:
        return {"reverted": 0, "reason": "the run recorded no clouds"}
    rows = (await db.execute(select(PointCloud).where(PointCloud.cloud_id.in_(ids)))).scalars().all()
    store = get_object_store()
    removed = 0
    for pc in rows:
        try:
            store.remove(pc.cloud_uri)
        except Exception:  # noqa: BLE001 - a blob already gone is not a failure to revert
            pass
        removed += 1
    await db.execute(delete(PointCloud).where(PointCloud.cloud_id.in_(ids)))
    run.status = "reverted"
    run.reverted_at = datetime.now(UTC)
    await db.commit()
    log.info("pseudo.batch.reverted", run_id=str(run.run_id), clouds=removed)
    return {"reverted": removed, "skipped": len(ids) - removed}


async def maybe_lift_pending(db: AsyncSession) -> dict:
    """Once a night, give the least-covered sessions ego pose and pseudo-LiDAR clouds, batch by batch."""
    from services.agent.runtime.report import finish_run, launch, ran_since
    from services.govern.killswitch import get_state
    from services.training.gpu_lease import training_holds_gpu

    st = await get_state(db)
    if not st.loop_enabled:
        return {"ran": False, "reason": "loop disabled by the killswitch"}
    if await ran_since(db, KIND, datetime.now(UTC) - timedelta(days=1)):
        return {"ran": False, "reason": "a lift has already run today"}
    if await training_holds_gpu(db):
        return {"ran": False, "reason": "training holds the GPU"}

    targets = await _uncovered_sessions(db, SESSIONS_PER_NIGHT)
    if not targets:
        return {"ran": False, "reason": "every real selected frame already has a cloud"}

    async def worker(run_id):
        from db.session import get_sessionmaker
        from services.intelligence.ego_pose import build_ego_pose

        status, report = "committed", {"sessions": [], "child_runs": []}
        try:
            for sid, n_frames, n_covered in targets:
                entry = {"session_id": str(sid), "frames": n_frames, "covered_before": n_covered}
                async with get_sessionmaker()() as wdb:
                    have_pose = (await wdb.execute(select(func.count()).select_from(EgoPose)
                                                   .where(EgoPose.session_id == sid))).scalar_one()
                if not have_pose:
                    # CPU work, and every batch below reads it. A cuboid in a frame nobody can locate
                    # sits in a coordinate system of one.
                    entry["ego_pose"] = await build_ego_pose(sid, run_id=run_id)
                else:
                    entry["ego_pose"] = {"poses": int(have_pose), "reason": "already built"}

                mem = _memory_pressure()
                if mem is not None and mem >= MEMORY_CEILING_FRAC:
                    entry["stopped"] = f"host memory at {mem:.0%}, ceiling {MEMORY_CEILING_FRAC:.0%}"
                    report["sessions"].append(entry)
                    break
                stop = await _wait_for_gpu(db, report)
                if stop:
                    entry["stopped"] = stop
                    report["sessions"].append(entry)
                    break

                async with get_sessionmaker()() as wdb:
                    frames = await _pending_frames(wdb, sid, PSEUDO_BATCH)
                if not frames:
                    entry["clouds"] = 0
                    entry["stopped"] = "no frame of this session is still uncovered"
                    report["sessions"].append(entry)
                    continue
                res = await lift_session_batch(sid, frames, parent_run_id=run_id,
                                               created_by=CREATED_BY)
                entry["clouds"] = res["clouds"]
                entry["failed"] = res["failed"]
                entry["cuboids"] = res["cuboids"]
                report["child_runs"].append(res["run_id"])
                report["sessions"].append(entry)
        except Exception as exc:  # noqa: BLE001 - a failed night records why, never leaves a run running
            status = "error"
            report["error"] = str(exc)[:400]
            log.error("pseudo.lift_failed", error=str(exc))
        await finish_run(run_id, status=status, report=report,
                         changes={"child_runs": report.get("child_runs", [])})

    return {"ran": True, **(await launch(db, KIND, worker, created_by=CREATED_BY,
                                         policy={"sessions": [str(s) for s, _n, _c in targets],
                                                 "batch": PSEUDO_BATCH}))}


async def coverage(db: AsyncSession) -> dict:
    """How much of the real corpus has a cloud, a pose and a 3D box. The number this daemon exists to move."""
    from db.models import Object3D, Track3D

    frames = (await db.execute(
        select(func.count()).select_from(Frame)
        .join(DbSession, DbSession.session_id == Frame.session_id)
        .where(Frame.origin == "real", DbSession.origin == "real",
               Frame.img_uri.isnot(None), Frame.selected.is_(True)))).scalar_one()
    clouds = (await db.execute(select(func.count()).select_from(PointCloud)
                               .where(PointCloud.source == "pseudo"))).scalar_one()
    posed = (await db.execute(select(func.count()).select_from(EgoPose))).scalar_one()
    posed_measured = (await db.execute(select(func.count()).select_from(EgoPose)
                                       .where(EgoPose.measured.is_(True)))).scalar_one()
    obj3d = (await db.execute(select(func.count()).select_from(Object3D))).scalar_one()
    trk3d = (await db.execute(select(func.count()).select_from(Track3D))).scalar_one()
    return {"real_selected_frames": int(frames), "pseudo_clouds": int(clouds),
            "cloud_coverage": round(int(clouds) / int(frames), 5) if frames else None,
            "ego_poses": int(posed), "ego_poses_measured": int(posed_measured),
            "object_3d": int(obj3d), "track_3d": int(trk3d)}
