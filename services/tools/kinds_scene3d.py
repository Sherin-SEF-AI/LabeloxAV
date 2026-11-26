"""Background job kinds for the scene3d annotation tools (agent C: #3 label once in 3D, #5 point labels, #6
occupancy). Each is registered on the shared tool queue (services/tools/registry.py).

- scene3d_label_once (CPU): one cuboid drawn in the session scene frame -> a 3D track on every scan and a 2D box
  in every camera frame at the same instant, linked, all in review.
- scene3d_ptv3 (GPU, auto-label): PTv3 point segmentation of one or more clouds, stored as a 'model' version.
- scene3d_occupancy_lidar (CPU): occupancy grids with flow from the session's clouds (occupancy4d.py).
- scene3d_occupancy_camera (GPU, auto-label): for a session without LiDAR, lift camera frames to pseudo-LiDAR
  with Depth-Anything-V2 metric depth, then build the grids from those clouds.
"""

from __future__ import annotations

import asyncio
import io
import uuid

import numpy as np

from core.logging import get_logger
from services.tools.registry import JobContext, register

log = get_logger("kinds_scene3d")


@register("scene3d_label_once", label="Label once in 3D")
async def label_once(ctx: JobContext) -> dict:
    from services.lidar.aggregate.label_propagate import propagate_scene_label

    p = ctx.params
    await ctx.progress(0.02, "placing scans")
    res = await propagate_scene_label(
        uuid.UUID(str(p["session_id"])), p["center"], p["dims"], float(p.get("yaw", 0.0)), int(p["class_id"]),
        ground_snap=bool(p.get("ground_snap", True)), max_range_m=float(p.get("max_range_m", 60.0)),
        project=bool(p.get("project", True)), job_id=ctx.job_id, user_id=ctx.created_by, progress=ctx.progress)
    if res.get("error"):
        raise RuntimeError(res["error"])
    return res


async def _store_model_segmentation(cloud_id: uuid.UUID, res: dict) -> dict:
    from sqlalchemy import func, select

    from core.config import get_settings
    from core.storage import get_object_store
    from db.models import PointCloud, PointSegmentation
    from db.session import get_sessionmaker

    async with get_sessionmaker()() as db:
        pc = await db.get(PointCloud, cloud_id)
        version = int((await db.execute(select(func.coalesce(func.max(PointSegmentation.version), 0)).where(
            PointSegmentation.cloud_id == cloud_id, PointSegmentation.source == "model"))).scalar() or 0) + 1
        buf = io.BytesIO()
        np.savez_compressed(buf, semantic=res["semantic"].astype(np.int32), instance=res["instance"].astype(np.int32),
                            conf=res["conf"].astype(np.float32), raw=res["raw_labels"].astype(np.int32),
                            class_names=np.array(res["class_names"]), object_3d_ids=np.array([], dtype="U36"))
        store = get_object_store()
        store.ensure_bucket()
        uri = store.put_bytes(f"{get_settings().lidar.cloud_prefix}/{pc.session_id}/seg/{cloud_id}.ptv3.v{version}.npz",
                              buf.getvalue(), "application/octet-stream")
        row = PointSegmentation(cloud_id=cloud_id, labels_uri=uri, model_version=str(res["checkpoint"])[:64],
                                kind="semantic", method="ptv3", n_points=int(len(res["semantic"])),
                                low_conf_frac=float(res["low_conf_frac"]), source="model", version=version)
        db.add(row)
        await db.commit()
        return {"cloud_id": str(cloud_id), "seg_id": str(row.seg_id), "version": version,
                "classes": res["classes_present"], "low_conf_frac": round(float(res["low_conf_frac"]), 4)}


@register("scene3d_ptv3", gpu=True, autolabel=True, label="3D point segmentation (PTv3)")
async def ptv3(ctx: JobContext) -> dict:
    from db.models import PointCloud
    from db.session import get_sessionmaker
    from services.lidar.ingest.store import load_cloud
    from services.lidar.segment3d.semantic import SegmentationUnavailable, segment_ptv3

    ids = [uuid.UUID(str(c)) for c in (ctx.params.get("cloud_ids") or [])]
    out, failed = [], []
    try:
        for i, cid in enumerate(ids):
            await ctx.progress(i / max(1, len(ids)), f"segmenting cloud {i + 1} of {len(ids)}")
            async with get_sessionmaker()() as db:
                pc = await db.get(PointCloud, cid)
            if pc is None:
                failed.append({"cloud_id": str(cid), "reason": "cloud not found"})
                continue
            cloud = await asyncio.to_thread(load_cloud, pc.cloud_uri)
            try:
                res = await asyncio.to_thread(segment_ptv3, cloud)
            except SegmentationUnavailable as exc:
                raise RuntimeError(str(exc)) from exc
            out.append(await _store_model_segmentation(cid, res))
    finally:
        try:
            from services.lidar.segment3d import ptv3_infer
            ptv3_infer.release()
        except Exception:  # noqa: BLE001 - nothing loaded, nothing to free
            pass
    return {"segmented": len(out), "clouds": out, "failed": failed}


@register("scene3d_occupancy_lidar", label="Occupancy grids from LiDAR")
async def occupancy_lidar(ctx: JobContext) -> dict:
    from services.lidar.occupancy4d import build_occupancy_window

    p = ctx.params
    await ctx.progress(0.05, "building grids")
    return await build_occupancy_window(uuid.UUID(str(p["session_id"])), p.get("t0"), p.get("t1"),
                                        limit=int(p.get("limit", 500)))


@register("scene3d_occupancy_camera", gpu=True, autolabel=True, label="Occupancy grids from cameras")
async def occupancy_camera(ctx: JobContext) -> dict:
    """Camera-only occupancy: Depth-Anything-V2 metric depth -> pseudo-LiDAR clouds -> grids.

    Every `every`-th instant of the session (all cameras captured at that instant together) is lifted through
    the cameras' resolved calibration. Instants that already have a pseudo cloud are not lifted again, so a
    re-run only fills the gaps.
    """
    import cv2
    from sqlalchemy import select

    from core.storage import get_object_store
    from db.models import Frame, PointCloud
    from db.session import get_sessionmaker
    from services.calibration.resolve import resolve_calibration
    from services.lidar.ingest.pseudo import lift_with_calibration
    from services.lidar.ingest.store import store_cloud
    from services.lidar.occupancy4d import build_grid_for_cloud

    p = ctx.params
    sid = uuid.UUID(str(p["session_id"]))
    every = max(1, int(p.get("every", 5)))
    limit = max(1, min(int(p.get("limit", 40)), 300))
    async with get_sessionmaker()() as db:
        rows = (await db.execute(select(Frame.ts_ns, Frame.cam_id, Frame.img_uri, Frame.width, Frame.height)
                                 .where(Frame.session_id == sid, Frame.img_uri.is_not(None))
                                 .order_by(Frame.ts_ns))).all()
        have = set((await db.execute(select(PointCloud.ts_ns).where(
            PointCloud.session_id == sid, PointCloud.source == "pseudo"))).scalars().all())
    groups: dict[int, list] = {}
    for r in rows:
        groups.setdefault(int(r.ts_ns), []).append(r)
    if not groups:
        raise RuntimeError("this session has no camera frames to lift")
    wanted = sorted(groups)[::every][:limit]
    todo = [ts for ts in wanted if ts not in have]
    store = get_object_store()
    cal_cache: dict = {}
    lifted = 0
    try:
        for i, ts in enumerate(todo):
            await ctx.progress(0.05 + 0.8 * i / max(1, len(todo)), f"lifting instant {i + 1} of {len(todo)}")
            images = {}
            for r in groups[ts]:
                buf = np.frombuffer(await asyncio.to_thread(store.get_bytes, r.img_uri), dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                key = (r.cam_id, img.shape[1], img.shape[0])
                if key not in cal_cache:
                    cal_cache[key] = await resolve_calibration(sid, r.cam_id, img.shape[1], img.shape[0])
                images[r.cam_id] = (img, cal_cache[key])
            if not images:
                continue
            cloud = await asyncio.to_thread(lift_with_calibration, images, ts, "resolved",
                                            None, int(p.get("stride", 4)))
            await store_cloud(cloud, sid, source="pseudo", depth_model=cloud.depth_model,
                              calibration_version="resolved")
            lifted += 1
    finally:
        _release_depth()
    await ctx.progress(0.9, "building grids")
    # Only the pseudo clouds of the wanted instants: a LiDAR session lifted for comparison keeps its LiDAR
    # grids exactly as they were.
    built, occupied, reasons = 0, 0, {}
    async with get_sessionmaker()() as db:
        clouds = (await db.execute(select(PointCloud).where(
            PointCloud.session_id == sid, PointCloud.source == "pseudo", PointCloud.ts_ns.in_(wanted))
            .order_by(PointCloud.ts_ns))).scalars().all()
        for c in clouds:
            res = await build_grid_for_cloud(db, c)
            if res["built"]:
                built += 1
                occupied += res["occupied"]
            else:
                reasons[res["reason"]] = reasons.get(res["reason"], 0) + 1
        await db.commit()
    return {"lifted": lifted, "already_lifted": len(wanted) - len(todo), "instants": len(wanted),
            "calibration": sorted({c.source for c in cal_cache.values()}),
            "grids": built, "occupied_voxels": occupied, "skipped": reasons}


def _release_depth() -> None:
    """Free the depth model the pseudo-LiDAR lift cached, so the next GPU job has the card."""
    try:
        import services.lidar.ingest.pseudo as pseudo

        pseudo._DEPTH_MODEL = None
        pseudo._DEPTH_PROC = None
        pseudo._DEPTH_ID = None
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
