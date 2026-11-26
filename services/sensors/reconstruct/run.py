"""Build one reconstruction end to end: window, initial cloud, splats or point map, rendered views, stored.

What it stores under `recon/<session>/<recon_id>/` in the object store:
  splats.npz      the trained Gaussians (gsplat only)
  splats.ply      the same in the standard 3DGS .ply layout, for any splat viewer (gsplat only)
  preview.bin     float32 [x, y, z, r, g, b] per point: splat centres or map points, for the 3D view
  views/*.jpg     rendered views; each `views` entry on the row carries its camera (T_world_cam, K, size,
                  ortho) so the page can draw a cuboid onto the render with the same projection that made it

Rendered views per keyframe: the frame's own pose (a genuine novel view when the frame was held out), 1.5 m to
either side, and raised 3 m looking down; plus one orthographic top-down view of the whole window, which is
where a person places a cuboid. The cuboid is in the session scene frame and goes to every scan and camera
frame through services/lidar/aggregate/label_propagate.py (the label-once job), unchanged.
"""

from __future__ import annotations

import asyncio
import io
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

import cv2
import numpy as np

from core.logging import get_logger
from core.storage import get_object_store

log = get_logger("reconstruct")

WORLD_UP = np.array([0.0, 0.0, 1.0])
KEYFRAMES = 6
BEV_CEILING_M = 1.0
MAX_IMAGE_WIDTH = 1280
PREVIEW_POINTS = 300_000
DEFAULT_BG = np.array([0.6, 0.65, 0.7], dtype=np.float32)

ProgressFn = Callable[[float, str], Awaitable[None]]


def _prefix(session_id, recon_id) -> str:
    return f"recon/{session_id}/{recon_id}"


def _decode_rgb(data: bytes) -> np.ndarray | None:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _jpeg(rgb01: np.ndarray, quality: int = 88) -> bytes:
    bgr = cv2.cvtColor((np.clip(rgb01, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("could not encode a rendered view")
    return enc.tobytes()


def keyframe_indices(n: int, k: int = KEYFRAMES, held: list[int] | None = None) -> list[int]:
    """Up to `k` evenly spaced frames, always including the held-out ones that fall near them."""
    if n <= 0:
        return []
    base = sorted({int(round(x)) for x in np.linspace(0, n - 1, min(k, n))})
    for h in held or []:
        if len(base) >= k + 3:
            break
        if h not in base:
            base.append(h)
    return sorted(set(base))


def initial_cloud(window, images: list[np.ndarray], *, voxel_m: float = 0.08, max_points: int = 600_000,
                  loader=None) -> dict:
    """The window's scans in the world frame, inside the cameras' view, coloured from the frames."""
    from services.lidar.aggregate.scene_frame import accumulate_scene

    from .window import colourise, frustum_mask

    if not window.scans:
        return {"xyz": np.zeros((0, 3), np.float32), "rgb": np.zeros((0, 3), np.float32), "scans": 0, "seen": 0}
    every = max(1, len(window.scans) // 60)
    acc = accumulate_scene(window.scans, every=every, max_range_m=60.0, voxel_m=voxel_m,
                           max_points=max_points * 2, loader=loader)
    xyz = acc["xyz"].astype(np.float64)
    xyz = xyz[frustum_mask(xyz, window.frames)]
    rgb, seen = colourise(xyz, window.frames, images)
    xyz, rgb = xyz[seen], rgb[seen]
    if len(xyz) > max_points:
        idx = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    return {"xyz": xyz.astype(np.float32), "rgb": rgb.astype(np.float32), "scans": acc["scans"],
            "seen": int(seen.sum())}


def view_cameras(window, held: list[int], cloud_xyz: np.ndarray) -> list[dict]:
    """Every camera the reconstruction is rendered from, before rendering."""
    from .window import bev_view, novel_poses

    cams: list[dict] = []
    for i in keyframe_indices(len(window.frames), held=held):
        f = window.frames[i]
        base = {"frame_id": str(f.frame_id), "ts_ns": f.ts_ns, "frame_index": i, "held_out": i in held,
                "K": f.K.tolist(), "width": f.width, "height": f.height, "ortho": False}
        cams.append({**base, "name": "camera", "kind": "camera", "T_world_cam": f.T_world_cam.tolist()})
        for name, T in novel_poses(f, WORLD_UP).items():
            cams.append({**base, "name": name, "kind": "novel", "T_world_cam": T.tolist()})
    if len(cloud_xyz):
        eyes = np.array([f.T_world_cam[:3, 3] for f in window.frames])
        heading = eyes[-1, :2] - eyes[0, :2]
        if np.linalg.norm(heading) < 1.0:                     # parked: use the camera's own forward axis
            heading = window.frames[0].T_world_cam[:2, 2]
        b = bev_view(cloud_xyz.astype(np.float64), heading)
        cams.append({"name": "top_down", "kind": "bev", "frame_id": None, "ts_ns": None, "frame_index": None,
                     "held_out": False, "K": b["K"].tolist(), "width": b["width"], "height": b["height"],
                     "ortho": True, "T_world_cam": b["T_world_cam"].tolist(), "ppm": b["ppm"],
                     "heading": b["heading"],
                     # Only what is below 1 m over the cameras is drawn from above: tree canopies and the
                     # sky's floaters would otherwise cover the road a cuboid is placed on.
                     "z_max": float(eyes[:, 2].max()) + BEV_CEILING_M})
    return cams


def _background(images: list[np.ndarray]) -> np.ndarray:
    """The frames' top rows, where the sky is: what a pixel no splat covers is rendered as."""
    tops = [im[: max(1, im.shape[0] // 12)].reshape(-1, 3) for im in images[:: max(1, len(images) // 8)]]
    return (np.median(np.concatenate(tops), axis=0) / 255.0).astype(np.float32) if tops else DEFAULT_BG


async def build_reconstruction(recon_id: uuid.UUID, *, progress: ProgressFn | None = None,
                               should_stop: Callable[[], bool] | None = None, device: str = "cuda",
                               loader=None) -> dict:
    """Run the reconstruction a `reconstruction` row describes and fill the row in. Returns the result summary."""
    from sqlalchemy import update

    from db.models import Reconstruction
    from db.session import get_sessionmaker

    from .window import load_window

    maker = get_sessionmaker()
    store = get_object_store()
    store.ensure_bucket()

    async def _say(frac: float, stage: str) -> None:
        if progress is not None:
            await progress(frac, stage)

    async with maker() as db:
        rec = await db.get(Reconstruction, recon_id)
        if rec is None:
            raise LookupError("reconstruction not found")
        p = dict(rec.params or {})
        method = rec.method
        session_id = rec.session_id
        rec.status = "running"
        await db.commit()
        window = await load_window(db, session_id, cam_id=p.get("cam_id"), start_ts_ns=p.get("start_ts_ns"),
                                   n_frames=int(p.get("n_frames", 50)), stride=int(p.get("stride", 1)))
    if len(window.frames) < 3:
        raise ValueError(f"only {len(window.frames)} frames of this window have an ego pose; a reconstruction "
                         "needs at least 3 posed frames")

    await _say(0.03, f"loading {len(window.frames)} frames")
    images: list[np.ndarray] = []
    for f in window.frames:
        img = await asyncio.to_thread(lambda u=f.img_uri: _decode_rgb(store.get_bytes(u)))
        if img is None:
            raise ValueError(f"frame {f.frame_id} could not be decoded")
        if img.shape[1] != f.width or img.shape[0] != f.height:
            img = cv2.resize(img, (f.width, f.height), interpolation=cv2.INTER_AREA)
        images.append(img)
    max_w = int(p.get("max_width", MAX_IMAGE_WIDTH))
    if window.frames[0].width > max_w:
        s = max_w / window.frames[0].width
        for k, f in enumerate(window.frames):
            w, h = int(round(f.width * s)), int(round(f.height * s))
            images[k] = cv2.resize(images[k], (w, h), interpolation=cv2.INTER_AREA)
            f.K = f.K.copy()
            f.K[:2] *= s
            f.width, f.height = w, h

    await _say(0.08, f"placing {len(window.scans)} scans")
    cloud = await asyncio.to_thread(initial_cloud, window, images, loader=loader)
    if len(cloud["xyz"]) < 1000:
        raise ValueError(f"the window's LiDAR gives {len(cloud['xyz'])} points the cameras see; a reconstruction "
                         "needs the session's scans placed by ego poses (KITTI sessions have both)")
    bg = _background(images)
    prefix = _prefix(session_id, recon_id)
    artifacts: dict = {}
    metrics: dict = {"window": window.summary, "initial_points": int(len(cloud["xyz"])), "scans": cloud["scans"],
                     "calibration": window.cal_source}

    if method == "gsplat":
        from .gsplat_train import render_views, sh0_to_rgb, split_holdout, to_ply, train_splats

        loop = asyncio.get_running_loop()
        stop = {"flag": False}

        def _prog(frac: float, stage: str) -> None:
            fut = asyncio.run_coroutine_threadsafe(_say(0.12 + 0.68 * frac, stage), loop)
            try:
                fut.result(timeout=30)
            except Exception:  # noqa: BLE001 - a cancel surfaces here as JobCanceled; stop the loop cleanly
                stop["flag"] = True

        viewmats = np.stack([np.linalg.inv(f.T_world_cam) for f in window.frames])
        Ks = np.stack([f.K for f in window.frames])
        res = await asyncio.to_thread(
            train_splats, images, viewmats, Ks, cloud["xyz"], cloud["rgb"], iters=int(p.get("iters", 7000)),
            sh_degree=int(p.get("sh_degree", 2)), device=device, background=bg, progress=_prog,
            should_stop=lambda: stop["flag"] or bool(should_stop and should_stop()))
        if stop["flag"]:
            from services.tools.registry import JobCanceled

            raise JobCanceled()
        g = res["gaussians"]
        metrics.update(res["metrics"])
        held = res["metrics"]["held_out_indices"]
        await _say(0.82, "saving splats")
        buf = io.BytesIO()
        np.savez_compressed(buf, means=g["means"].astype(np.float32), scales=g["scales"].astype(np.float32),
                            quats=g["quats"].astype(np.float32), opacities=g["opacities"].astype(np.float32),
                            sh0=g["sh0"].astype(np.float32), shN=g["shN"].astype(np.float16),
                            sh_degree=np.array(g["sh_degree"]), background=bg)
        artifacts["splats"] = await asyncio.to_thread(store.put_bytes, f"{prefix}/splats.npz", buf.getvalue())
        artifacts["ply"] = await asyncio.to_thread(store.put_bytes, f"{prefix}/splats.ply", to_ply(g))
        opa = 1.0 / (1.0 + np.exp(-g["opacities"]))
        keep = opa > 0.2
        prev_xyz = g["means"][keep]
        prev_rgb = sh0_to_rgb(g["sh0"][keep].reshape(-1, 3))

        def _render(cam: dict) -> np.ndarray:
            gg = g
            if cam.get("z_max") is not None:
                big = np.exp(g["scales"]).max(axis=1) > 1.5
                keep_bev = (g["means"][:, 2] <= cam["z_max"]) & ~big
                gg = {**{k: g[k][keep_bev] for k in ("means", "scales", "quats", "opacities", "sh0", "shN")},
                      "sh_degree": g["sh_degree"]}
            return render_views(gg, np.linalg.inv(np.asarray(cam["T_world_cam"]))[None],
                                np.asarray(cam["K"])[None], cam["width"], cam["height"], device=device,
                                background=bg, ortho=cam["ortho"])[0]
    else:
        from .gsplat_train import psnr, split_holdout
        from .pointmap import render_points

        _train, held = split_holdout(len(window.frames))
        prev_xyz, prev_rgb = cloud["xyz"], cloud["rgb"]
        fits = []
        for i in range(0, len(window.frames), max(1, len(window.frames) // 8)):
            f = window.frames[i]
            r = await asyncio.to_thread(render_points, cloud["xyz"], cloud["rgb"], f.T_world_cam, f.K, f.width,
                                        f.height, background=bg)
            fits.append(psnr(r, images[i].astype(np.float32) / 255.0))
        metrics.update({"psnr_camera_views": round(float(np.mean(fits)), 2) if fits else None,
                        "points": int(len(cloud["xyz"])), "image_size": [window.frames[0].width,
                                                                         window.frames[0].height]})

        def _render(cam: dict) -> np.ndarray:
            xyz, rgb = cloud["xyz"], cloud["rgb"]
            if cam.get("z_max") is not None:
                low = xyz[:, 2] <= cam["z_max"]
                xyz, rgb = xyz[low], rgb[low]
            return render_points(xyz, rgb, np.asarray(cam["T_world_cam"]), np.asarray(cam["K"]),
                                 cam["width"], cam["height"], ortho=cam["ortho"], background=bg)

    # The preview cloud for the 3D view.
    if len(prev_xyz) > PREVIEW_POINTS:
        idx = np.random.default_rng(1).choice(len(prev_xyz), PREVIEW_POINTS, replace=False)
        prev_xyz, prev_rgb = prev_xyz[idx], prev_rgb[idx]
    packed = np.concatenate([prev_xyz.astype(np.float32), prev_rgb.astype(np.float32)], axis=1)
    artifacts["preview"] = await asyncio.to_thread(store.put_bytes, f"{prefix}/preview.bin", packed.tobytes())
    artifacts["preview_points"] = int(len(packed))

    cams = view_cameras(window, held, cloud["xyz"])
    views: list[dict] = []
    for n, cam in enumerate(cams):
        if should_stop is not None and should_stop():
            from services.tools.registry import JobCanceled

            raise JobCanceled()
        img = await asyncio.to_thread(_render, cam)
        uri = await asyncio.to_thread(store.put_bytes, f"{prefix}/views/{n:03d}_{cam['name']}.jpg", _jpeg(img),
                                      "image/jpeg")
        views.append({**cam, "index": n, "uri": uri})
        if n % 4 == 0:
            await _say(0.85 + 0.13 * (n + 1) / len(cams), f"rendering view {n + 1} of {len(cams)}")

    async with maker() as db:
        await db.execute(update(Reconstruction).where(Reconstruction.recon_id == recon_id).values(
            status="done", artifacts=artifacts, views=views, metrics=metrics, cam_id=window.cam_id,
            first_ts_ns=window.span_ns[0], last_ts_ns=window.span_ns[1], finished_at=datetime.now(UTC),
            error=None))
        await db.commit()
    log.info("reconstruction.done", recon_id=str(recon_id), method=method, views=len(views),
             psnr_held_out=metrics.get("psnr_held_out"))
    return {"recon_id": str(recon_id), "method": method, "views": len(views),
            "metrics": {k: v for k, v in metrics.items() if k not in ("window", "held_out_indices")}}


async def mark_failed(recon_id: uuid.UUID, status: str, error: str | None) -> None:
    from sqlalchemy import update

    from db.models import Reconstruction
    from db.session import get_sessionmaker

    async with get_sessionmaker()() as db:
        await db.execute(update(Reconstruction).where(Reconstruction.recon_id == recon_id).values(
            status=status, error=(error or "")[:2000] or None, finished_at=datetime.now(UTC)))
        await db.commit()
