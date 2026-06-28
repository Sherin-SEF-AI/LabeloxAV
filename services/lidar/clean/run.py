"""Run the cleaning pipeline on a stored cloud: derive the ground plane, the ground-removed cloud, and the
denoised cloud, each written as a point_cloud_derived row. Raw is never overwritten. Preview mode runs the
same passes on a decimated copy and returns counts only, for an interactive local preview; the bulk path
runs full resolution and stores, and can be driven from the pointcloud_build burst job.
"""

from __future__ import annotations

import uuid

from core.config import get_settings
from core.logging import get_logger
from db.models import PointCloud
from db.session import get_sessionmaker
from services.lidar.clean.denoise import denoise
from services.lidar.clean.ground import remove_ground
from services.lidar.ingest.store import load_cloud, store_derived

log = get_logger("lidar_clean")


async def clean_cloud(cloud_id: uuid.UUID, session_id: uuid.UUID, method: str | None = None,
                      rain_dust: bool = True, preview: bool = False,
                      preview_max: int | None = None) -> dict:
    """Ground-remove and denoise a stored cloud into derived variants. Preview skips storage."""
    async with get_sessionmaker()() as db:
        row = await db.get(PointCloud, cloud_id)
        if row is None:
            return {"error": "cloud not found", "cloud_id": str(cloud_id)}
        uri = row.cloud_uri
    raw = load_cloud(uri)
    work = raw.decimate(preview_max or get_settings().lidar.viewer_max_points) if preview else raw

    g = remove_ground(work, method=method)
    den = denoise(g["nonground"], rain_dust=rain_dust)

    if preview:
        return {"cloud_id": str(cloud_id), "preview": True, "input_points": work.n,
                "ground_points": g["ground_points"], "ground_removed_points": g["kept_points"],
                "denoised_points": den.n, "plane": g["plane"], "method": g["method"]}

    derived = {
        "ground_plane": await store_derived(cloud_id, session_id, g["ground"], "ground_plane",
                                            g["method"], {"plane": g["plane"]}),
        "ground_removed": await store_derived(cloud_id, session_id, g["nonground"], "ground_removed",
                                              g["method"], {"ground_points": g["ground_points"]}),
        "denoised": await store_derived(cloud_id, session_id, den, "denoised", "statistical+radius+raindust",
                                        {"source": "ground_removed"}),
    }
    log.info("lidar.cleaned", cloud=str(cloud_id), raw=raw.n, kept=g["kept_points"], denoised=den.n)
    return {"cloud_id": str(cloud_id), "raw_points": raw.n, "method": g["method"], "derived": derived}
