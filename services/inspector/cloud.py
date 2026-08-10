"""Serving a point cloud to the inspector's 3D panel, at the timestamp the clock is on.

The inspector already synchronises seven panels against one clock. The one Foxglove has and this did not is
3D, which is awkward because the corpus has 154 clouds across 124 sessions, 39 of them real LiDAR, and the
whole 3D tier (CALYX, ORACLYX, the cuboid tools) has never had a way to look at them.

Two things shape this module, and both are about size.

Clouds here run from 80 points to 500,000, averaging 9,573. Half a million points is roughly 6MB of raw
float32 and considerably more as JSON, which is not a panel refresh, it is a download. So the payload is
raw little-endian float32 over `application/octet-stream` and the browser maps it straight into a
Float32Array: no parse step, no per-number allocation. JSON here would spend more time in JSON.parse than
the renderer spends drawing.

And it is downsampled, with the count returned rather than implied. Stride sampling, not the first N: a cloud
is written in scan order, so the first 20,000 points of a 500,000-point sweep are one sector of one rotation,
and a viewer showing that would look like a working panel displaying a quarter of the scene. Stride keeps the
shape of the whole cloud at any budget, which is the only property that makes a downsampled view honest.

The nearest cloud is chosen by timestamp rather than requiring an exact match, because the clock lands
wherever the user scrubbed and clouds are sparse: 154 of them across sessions holding thousands of frames.
How far away it was comes back in the header, so a panel showing a cloud from four seconds ago can say so
instead of implying it is what the camera is looking at.
"""

from __future__ import annotations

import uuid

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import PointCloud

log = get_logger("inspector.cloud")

# What a browser can hold and draw without the panel becoming the reason the tab is slow. 200k points at
# three float32 is 2.4MB, which is a fast local fetch and a comfortable single draw call.
DEFAULT_MAX_POINTS = 200_000

# Beyond this the request is not a panel refresh. Bounded here rather than trusted from the query string,
# because the caller is a URL anybody can edit.
HARD_MAX_POINTS = 500_000


def stride_indices(n: int, max_points: int) -> np.ndarray:
    """Evenly spaced indices covering the whole cloud, at most `max_points` of them.

    Stride rather than head or random. Head is wrong because a cloud is stored in scan order, so the first N
    points are one sector of one rotation and the panel would show a quarter of the scene while looking
    entirely functional. Random is defensible but not reproducible, and two viewers scrubbed to the same
    timestamp showing different points is a bug report nobody can act on.
    """
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    budget = max(1, min(int(max_points), HARD_MAX_POINTS))
    if n <= budget:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, budget).astype(np.int64)


async def nearest_cloud(db: AsyncSession, session_id: str | uuid.UUID, ts_ns: int) -> PointCloud | None:
    """The cloud closest in time to `ts_ns` within the session.

    Nearest rather than exact: the clock lands wherever the user scrubbed, and with 154 clouds across
    sessions of thousands of frames an exact match is the rare case, not the normal one.
    """
    sid = uuid.UUID(str(session_id))
    rows = (await db.execute(
        select(PointCloud).where(PointCloud.session_id == sid))).scalars().all()
    if not rows:
        return None
    return min(rows, key=lambda c: abs(int(c.ts_ns) - int(ts_ns)))


def pack(xyz: np.ndarray, intensity: np.ndarray | None) -> bytes:
    """Interleave to [x, y, z, i] float32 little-endian, which is what the panel's buffer geometry wants.

    Interleaved rather than four arrays, so the browser makes one fetch and one typed-array view instead of
    stitching buffers it would then have to keep in step.
    """
    n = int(xyz.shape[0])
    out = np.zeros((n, 4), dtype="<f4")
    out[:, :3] = xyz.astype("<f4", copy=False)
    if intensity is not None and len(intensity) == n:
        out[:, 3] = np.asarray(intensity, dtype="<f4")
    return out.tobytes()


async def cloud_payload(db: AsyncSession, session_id: str | uuid.UUID, ts_ns: int, *,
                        max_points: int = DEFAULT_MAX_POINTS) -> dict | None:
    """The bytes for one cloud plus the metadata a panel needs to frame and caption it."""
    from services.lidar.ingest.store import load_cloud

    row = await nearest_cloud(db, session_id, ts_ns)
    if row is None:
        return None

    cloud = load_cloud(row.cloud_uri)
    idx = stride_indices(cloud.n, max_points)
    xyz = cloud.xyz[idx]
    intensity = cloud.intensity[idx] if cloud.intensity is not None else None

    delta_ms = abs(int(row.ts_ns) - int(ts_ns)) / 1e6
    log.info("inspector.cloud", cloud=str(row.cloud_id), points=int(xyz.shape[0]), of=cloud.n,
             delta_ms=round(delta_ms, 1))
    return {
        "bytes": pack(xyz, intensity),
        "cloud_id": str(row.cloud_id),
        "ts_ns": str(row.ts_ns),
        "source": row.source,
        "returned": int(xyz.shape[0]),
        # Both counts, always. A panel that shows 200,000 of 500,000 points and says only "200,000" is
        # describing its own budget as if it were the sensor's.
        "total": int(cloud.n),
        "truncated": bool(int(xyz.shape[0]) < cloud.n),
        # So the panel can say "nearest cloud, 4.2s away" rather than implying this is what the camera sees.
        "delta_ms": round(delta_ms, 1),
        "bounds": row.bounds or cloud.bounds(),
    }
