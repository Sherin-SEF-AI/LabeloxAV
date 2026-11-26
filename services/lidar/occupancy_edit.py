"""Human edits to an occupancy grid, kept as a delta layer so the generated grid is never rewritten.

An `occupancy_grid` row with source 'human' names the grid it edits (`base_grid_id`) and shares its geometry
(origin, voxel size, dims). Its blob is a delta: voxel indices with a value, 1 for "occupied" and 0 for
"free". The merged grid a consumer reads is the base's occupied set, plus the delta's occupied voxels, minus
its free ones. Rebuilding the base grid (occupancy4d.py upserts on its natural key) keeps the edit layer, since
the human row hangs off the base row's id.

Propagating an edit to the following grids is a change of frame. A grid placed by an ego pose has a world
origin (occupancy4d.py moves the window with the vehicle), so a voxel's centre is a world point; that point
falls in some voxel of the next grid, or outside it. A grid nothing placed sits in the ego frame of its own
instant, and an edit cannot follow the vehicle without a pose; that is refused rather than guessed.
"""

from __future__ import annotations

import io
import uuid

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger

log = get_logger("occupancy_edit")

OCCUPIED, FREE = 1, 0
# One box edit at most this many voxels: a 160 x 120 x 12 window is 230,400, so a whole-grid fill is allowed
# and anything larger is a mistake in the request.
MAX_BOX_VOXELS = 250_000


def pack_delta(voxels: np.ndarray, values: np.ndarray, dims) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, voxels=np.asarray(voxels, dtype=np.int32).reshape(-1, 3),
                        values=np.asarray(values, dtype=np.int8).reshape(-1),
                        dims=np.asarray(dims, dtype=np.int32), kind=np.array("delta"))
    return buf.getvalue()


def unpack_delta(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    with np.load(io.BytesIO(data)) as z:
        return z["voxels"].astype(np.int32).reshape(-1, 3), z["values"].astype(np.int8).reshape(-1)


def _keys(v: np.ndarray, dims) -> np.ndarray:
    v = np.asarray(v, dtype=np.int64).reshape(-1, 3)
    return (v[:, 0] * int(dims[1]) + v[:, 1]) * int(dims[2]) + v[:, 2]


def _unkey(k: np.ndarray, dims) -> np.ndarray:
    k = np.asarray(k, dtype=np.int64)
    z = k % int(dims[2])
    y = (k // int(dims[2])) % int(dims[1])
    x = k // (int(dims[1]) * int(dims[2]))
    return np.stack([x, y, z], axis=1).astype(np.int32)


def in_bounds(v: np.ndarray, dims) -> np.ndarray:
    v = np.asarray(v).reshape(-1, 3)
    return ((v >= 0) & (v < np.asarray(dims, dtype=np.int64))).all(axis=1)


def apply_ops(delta_vox: np.ndarray, delta_val: np.ndarray, ops: list[dict], dims) -> tuple[np.ndarray, np.ndarray, int]:
    """Apply paint/box operations to a delta. Returns the new delta and how many voxels the ops touched.

    op = {"kind": "paint", "value": "occupied"|"free"|"reset", "voxels": [[i, j, k], ...]}
       | {"kind": "box", "value": ..., "lo": [i, j, k], "hi": [i, j, k]}   (inclusive, clipped to the grid)
    "reset" drops the voxels from the delta, so they read as the generated grid says again.
    """
    cur: dict[int, int] = dict(zip(_keys(delta_vox, dims).tolist(), delta_val.astype(int).tolist(), strict=True))
    touched = 0
    for op in ops:
        value = op.get("value")
        if value not in ("occupied", "free", "reset"):
            raise ValueError(f"unknown value {value!r}: occupied, free or reset")
        if op.get("kind") == "paint":
            vox = np.asarray(op.get("voxels") or [], dtype=np.int64).reshape(-1, 3)
        elif op.get("kind") == "box":
            lo = np.maximum(np.minimum(op["lo"], op["hi"]), 0)
            hi = np.minimum(np.maximum(op["lo"], op["hi"]), np.asarray(dims) - 1)
            if (hi < lo).any():
                continue
            n = int(np.prod(hi - lo + 1))
            if n > MAX_BOX_VOXELS:
                raise ValueError(f"box covers {n} voxels; at most {MAX_BOX_VOXELS}")
            g = np.mgrid[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]
            vox = g.reshape(3, -1).T
        else:
            raise ValueError(f"unknown op kind {op.get('kind')!r}: paint or box")
        vox = vox[in_bounds(vox, dims)]
        keys = _keys(vox, dims).tolist()
        touched += len(keys)
        if value == "reset":
            for k in keys:
                cur.pop(k, None)
        else:
            val = OCCUPIED if value == "occupied" else FREE
            for k in keys:
                cur[k] = val
    if not cur:
        return np.zeros((0, 3), np.int32), np.zeros(0, np.int8), touched
    ks = np.fromiter(cur.keys(), dtype=np.int64, count=len(cur))
    vs = np.fromiter(cur.values(), dtype=np.int8, count=len(cur))
    order = np.argsort(ks)
    return _unkey(ks[order], dims), vs[order], touched


def merge(base_voxels: np.ndarray, base_flow: np.ndarray, delta_vox: np.ndarray, delta_val: np.ndarray,
          dims) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Base plus delta: (voxels, flow, edited) where `edited` marks voxels a person set occupied."""
    bk = _keys(base_voxels, dims)
    free_k = _keys(delta_vox[delta_val == FREE], dims) if len(delta_vox) else np.zeros(0, np.int64)
    occ_k = _keys(delta_vox[delta_val == OCCUPIED], dims) if len(delta_vox) else np.zeros(0, np.int64)
    keep = ~np.isin(bk, free_k) & ~np.isin(bk, occ_k)
    vox = np.concatenate([np.asarray(base_voxels, np.int32).reshape(-1, 3)[keep], _unkey(occ_k, dims)])
    flow = np.concatenate([np.asarray(base_flow, np.float32).reshape(-1, 3)[keep],
                           np.zeros((len(occ_k), 3), np.float32)])
    # a voxel the base already had and the person re-confirmed keeps its flow
    if len(occ_k):
        had = np.isin(occ_k, bk)
        if had.any():
            pos = {int(k): i for i, k in enumerate(bk.tolist())}
            base_flow_arr = np.asarray(base_flow, np.float32).reshape(-1, 3)
            start = int(keep.sum())
            for j, k in enumerate(occ_k.tolist()):
                if had[j]:
                    flow[start + j] = base_flow_arr[pos[int(k)]]
    edited = np.concatenate([np.zeros(int(keep.sum()), bool), np.ones(len(occ_k), bool)])
    return vox, flow, edited


def transfer(delta_vox: np.ndarray, delta_val: np.ndarray, src_origin, src_voxel: float,
             dst_origin, dst_voxel: float, dst_dims) -> tuple[np.ndarray, np.ndarray]:
    """A delta's voxels moved from one placed grid to another through their shared world frame."""
    if len(delta_vox) == 0:
        return delta_vox, delta_val
    centres = np.asarray(src_origin, np.float64) + (delta_vox.astype(np.float64) + 0.5) * float(src_voxel)
    vi = np.floor((centres - np.asarray(dst_origin, np.float64)) / float(dst_voxel)).astype(np.int64)
    ok = in_bounds(vi, dst_dims)
    return vi[ok].astype(np.int32), delta_val[ok]


# ---- persistence ----

async def human_layer(db: AsyncSession, grid_id: uuid.UUID):
    from db.models import OccupancyGrid

    return (await db.execute(select(OccupancyGrid).where(OccupancyGrid.base_grid_id == grid_id,
                                                         OccupancyGrid.source == "human"))).scalars().first()


def _load_base(base) -> dict:
    from core.storage import get_object_store
    from services.lidar.occupancy4d import unpack_grid

    return unpack_grid(get_object_store().get_bytes(base.grid_uri))


def _load_delta(layer) -> tuple[np.ndarray, np.ndarray]:
    from core.storage import get_object_store

    if layer is None:
        return np.zeros((0, 3), np.int32), np.zeros(0, np.int8)
    return unpack_delta(get_object_store().get_bytes(layer.grid_uri))


async def read_grid(db: AsyncSession, grid_id: uuid.UUID) -> dict:
    """The generated grid, its edit layer and the merged result."""
    from db.models import OccupancyGrid

    base = await db.get(OccupancyGrid, grid_id)
    if base is None:
        raise LookupError("grid not found")
    if base.source == "human":
        raise LookupError("that is an edit layer; read the grid it edits")
    g = _load_base(base)
    layer = await human_layer(db, grid_id)
    dv, dval = _load_delta(layer)
    dims = tuple(int(d) for d in base.dims)
    vox, flow, edited = merge(g["voxels"], g["flow"], dv, dval, dims)
    return {"base": base, "layer": layer, "dims": dims, "voxels": vox, "flow": flow, "edited": edited,
            "delta_voxels": dv, "delta_values": dval, "base_occupied": int(len(g["voxels"]))}


async def write_delta(db: AsyncSession, base, delta_vox: np.ndarray, delta_val: np.ndarray,
                      user_id: uuid.UUID | None) -> dict:
    """Store the delta as the base grid's human layer (created on first edit). Does not commit."""
    from core.storage import get_object_store
    from db.models import OccupancyGrid

    dims = tuple(int(d) for d in base.dims)
    layer = await human_layer(db, base.grid_id)
    if layer is None:
        clash = (await db.execute(select(OccupancyGrid.grid_id).where(
            OccupancyGrid.session_id == base.session_id, OccupancyGrid.ts_ns == base.ts_ns,
            OccupancyGrid.source == "human"))).scalar()
        if clash is not None:
            raise ValueError("another grid at this instant already carries the human edit layer")
    g = _load_base(base)
    merged, _, _ = merge(g["voxels"], g["flow"], delta_vox, delta_val, dims)
    store = get_object_store()
    store.ensure_bucket()
    version = uuid.uuid4().hex[:8]
    uri = store.put_bytes(f"occupancy/{base.session_id}/{base.ts_ns}.human.{version}.npz",
                          pack_delta(delta_vox, delta_val, dims), "application/octet-stream")
    if layer is None:
        layer = OccupancyGrid(session_id=base.session_id, ts_ns=base.ts_ns, frame_id=base.frame_id,
                              origin=list(base.origin), voxel_m=base.voxel_m, dims=list(base.dims), grid_uri=uri,
                              source="human", ego_pose_ts=base.ego_pose_ts, occupied=int(len(merged)),
                              flow_voxels=0, base_grid_id=base.grid_id, edited_by=user_id)
        db.add(layer)
    else:
        layer.grid_uri = uri
        layer.occupied = int(len(merged))
        layer.edited_by = user_id
    return {"delta_voxels": int(len(delta_vox)), "set_occupied": int((delta_val == OCCUPIED).sum()),
            "set_free": int((delta_val == FREE).sum()), "occupied_after": int(len(merged)),
            "occupied_before": int(len(g["voxels"]))}


async def edit_grid(db: AsyncSession, grid_id: uuid.UUID, ops: list[dict], user_id: uuid.UUID | None) -> dict:
    """Apply ops to a grid's human layer. Commits."""
    cur = await read_grid(db, grid_id)
    dv, dval, touched = apply_ops(cur["delta_voxels"], cur["delta_values"], ops, cur["dims"])
    res = await write_delta(db, cur["base"], dv, dval, user_id)
    await db.commit()
    return {"grid_id": str(grid_id), "touched": touched, **res}


async def propagate_edits(db: AsyncSession, grid_id: uuid.UUID, count: int, user_id: uuid.UUID | None,
                          only_ops: list[dict] | None = None) -> dict:
    """Carry a grid's edits (or just `only_ops`, applied to an empty delta) to the next `count` grids of the
    same session and source, through the world frame the ego poses put them in. Commits."""
    from db.models import OccupancyGrid

    cur = await read_grid(db, grid_id)
    base = cur["base"]
    if base.ego_pose_ts is None:
        raise ValueError("this grid was not placed by an ego pose, so an edit cannot follow the vehicle to the "
                         "next grid; recover the session's trajectory and rebuild the grids first")
    if only_ops:
        dv, dval, _ = apply_ops(np.zeros((0, 3), np.int32), np.zeros(0, np.int8), only_ops, cur["dims"])
    else:
        dv, dval = cur["delta_voxels"], cur["delta_values"]
    if len(dv) == 0:
        return {"grid_id": str(grid_id), "propagated_to": 0, "reason": "no edits to carry"}
    nxt = (await db.execute(select(OccupancyGrid).where(
        OccupancyGrid.session_id == base.session_id, OccupancyGrid.source == base.source,
        OccupancyGrid.ts_ns > base.ts_ns).order_by(OccupancyGrid.ts_ns).limit(max(1, min(count, 200))))).scalars().all()
    done, unplaced, outside = [], 0, 0
    for g in nxt:
        if g.ego_pose_ts is None:
            unplaced += 1
            continue
        tv, tval = transfer(dv, dval, base.origin, base.voxel_m, g.origin, g.voxel_m, tuple(int(d) for d in g.dims))
        if len(tv) == 0:
            outside += 1
            continue
        tcur = await read_grid(db, g.grid_id)
        ops = [{"kind": "paint", "value": "occupied", "voxels": tv[tval == OCCUPIED].tolist()},
               {"kind": "paint", "value": "free", "voxels": tv[tval == FREE].tolist()}]
        ndv, ndval, _ = apply_ops(tcur["delta_voxels"], tcur["delta_values"], ops, tcur["dims"])
        res = await write_delta(db, g, ndv, ndval, user_id)
        done.append({"grid_id": str(g.grid_id), "ts_ns": int(g.ts_ns), "carried": int(len(tv)), **res})
    await db.commit()
    log.info("occupancy_edit.propagated", grid=str(grid_id), to=len(done))
    return {"grid_id": str(grid_id), "propagated_to": len(done), "grids": done,
            "skipped_unplaced": unplaced, "skipped_outside": outside}
