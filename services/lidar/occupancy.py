"""Milestone F: occupancy / voxel labeling from aggregated geometry, the modern occupancy ground-truth
target. A voxel is occupied when points fall in it, free when a ray from the sensor origin to an occupied
voxel passes through it (classic free-space carving), and unknown otherwise (occluded or unobserved). Built
from a point cloud, so it works on pseudo-LiDAR (vision-first) the same as real LiDAR, and on the aggregated
clip reconstruction for a keyframe-plus-interpolated occupancy volume.
"""

from __future__ import annotations

import numpy as np

from core.logging import get_logger

log = get_logger("occupancy")


def _bresenham3d(a: tuple, b: tuple) -> list[tuple]:
    """Integer voxel line from a to b (3D Bresenham), inclusive of both endpoints."""
    x1, y1, z1 = a
    x2, y2, z2 = b
    pts = [(x1, y1, z1)]
    dx, dy, dz = abs(x2 - x1), abs(y2 - y1), abs(z2 - z1)
    xs = 1 if x2 > x1 else -1
    ys = 1 if y2 > y1 else -1
    zs = 1 if z2 > z1 else -1
    if dx >= dy and dx >= dz:
        p1, p2 = 2 * dy - dx, 2 * dz - dx
        while x1 != x2:
            x1 += xs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dx
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            pts.append((x1, y1, z1))
    elif dy >= dx and dy >= dz:
        p1, p2 = 2 * dx - dy, 2 * dz - dy
        while y1 != y2:
            y1 += ys
            if p1 >= 0:
                x1 += xs
                p1 -= 2 * dy
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            pts.append((x1, y1, z1))
    else:
        p1, p2 = 2 * dy - dz, 2 * dx - dz
        while z1 != z2:
            z1 += zs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dz
            if p2 >= 0:
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            pts.append((x1, y1, z1))
    return pts


def voxelize_occupancy(points, origin, bounds, voxel_size: float, min_points: int = 1) -> dict:
    """points: Nx3 in the same frame as origin and bounds=[xmin,ymin,zmin,xmax,ymax,zmax]. Returns the grid
    dims and the occupied / free / unknown voxel counts, occupied carved into free along each sensor ray."""
    xmin, ymin, zmin, xmax, ymax, zmax = bounds
    dims = (max(1, int((xmax - xmin) / voxel_size)), max(1, int((ymax - ymin) / voxel_size)),
            max(1, int((zmax - zmin) / voxel_size)))

    def vox(p):
        return (int((p[0] - xmin) // voxel_size), int((p[1] - ymin) // voxel_size),
                int((p[2] - zmin) // voxel_size))

    def inb(v):
        return 0 <= v[0] < dims[0] and 0 <= v[1] < dims[1] and 0 <= v[2] < dims[2]

    counts: dict = {}
    for p in np.asarray(points, dtype=float):
        v = vox(p)
        if inb(v):
            counts[v] = counts.get(v, 0) + 1
    occupied = {v for v, c in counts.items() if c >= min_points}

    free: set = set()
    ov = vox(origin)
    for v in occupied:
        for fv in _bresenham3d(ov, v):
            if fv != v and inb(fv) and fv not in occupied:
                free.add(fv)

    total = dims[0] * dims[1] * dims[2]
    unknown = total - len(occupied) - len(free)
    return {"dims": list(dims), "voxel_size": voxel_size, "occupied": len(occupied), "free": len(free),
            "unknown": unknown, "occupied_voxels": sorted(occupied)}
