"""2D projection of an embedding space, for the explorer's embeddings map.

The high-dimensional vectors already live in pgvector (object_embedding.dino_vec, frame_embedding.dino_vec /
siglip_vec). This fits a 2D layout over them so a human can see the corpus as structure rather than as a list:
clusters are visual, outliers sit apart, and a lasso over a region becomes a bulk action (tag, review, export).

UMAP is the default because it preserves local neighbourhood structure far better than a linear method, which
is what makes clusters meaningful. When umap-learn is unavailable or a fit fails, this falls back to a
deterministic PCA (numpy SVD) and records `method="pca"` on the row, so a map is never silently a different
algorithm than the one requested.

The fit is CPU-heavy and fully blocking, so callers run it off the event loop.
"""

from __future__ import annotations

import asyncio
import uuid as uuidlib
from uuid import UUID

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import (
    EmbeddingProjection,
    EmbeddingProjectionPoint,
    Frame,
    FrameEmbedding,
    Object,
    ObjectEmbedding,
)

log = get_logger("curation_projection")

# A fit over more than this many vectors is slow enough to be a bad interactive experience; the caller gets a
# deterministic random subsample instead, and `n` on the row records what was actually laid out.
MAX_POINTS = 50_000


def _pca_2d(X: np.ndarray) -> np.ndarray:
    """Deterministic 2D PCA via SVD. The fallback when UMAP is unavailable: linear, so it will not separate
    non-linear clusters the way UMAP does, but it is stable, dependency-free, and reproducible."""
    Xc = X - X.mean(axis=0, keepdims=True)
    # full_matrices=False keeps this tractable on (N, 768); we only need the top 2 right singular vectors.
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def fit_2d(X: np.ndarray, method: str = "umap", n_neighbors: int = 15, min_dist: float = 0.1,
           seed: int = 42) -> tuple[np.ndarray, str]:
    """Blocking 2D fit. Returns (coords (N,2), method actually used). Cosine is the right metric for these
    embeddings: DINOv3/SigLIP vectors are compared by angle everywhere else in the codebase (pgvector
    cosine_distance in core/embeddings.py), so the map must cluster on the same notion of similarity."""
    if X.shape[0] < 3:
        # Not enough points for a neighbourhood method; lay them out flat so the map still renders.
        return np.zeros((X.shape[0], 2), dtype=float), "pca"
    if method == "umap":
        try:
            import umap  # noqa: PLC0415

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(n_neighbors, max(2, X.shape[0] - 1)),
                min_dist=min_dist,
                metric="cosine",
                random_state=seed,
                verbose=False,
            )
            return np.asarray(reducer.fit_transform(X), dtype=float), "umap"
        except Exception as exc:  # noqa: BLE001
            log.warning("projection.umap_failed_falling_back_to_pca", error=str(exc))
    return _pca_2d(X), "pca"


def cluster_labels(X: np.ndarray, min_cluster_size: int = 15) -> np.ndarray:
    """HDBSCAN density labels over the source vectors (-1 = noise). Clustering the high-dimensional vectors
    rather than the 2D coordinates keeps the grouping faithful to the real embedding structure instead of to
    projection artifacts. Returns all-noise when the set is too small or the fit fails."""
    n = X.shape[0]
    if n < max(3, min_cluster_size):
        return np.full(n, -1, dtype=int)
    try:
        import hdbscan  # noqa: PLC0415

        return np.asarray(
            hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(X), dtype=int)
    except Exception as exc:  # noqa: BLE001
        log.warning("projection.hdbscan_failed", error=str(exc))
        return np.full(n, -1, dtype=int)


def _fit_2d(X: np.ndarray, method: str, n_neighbors: int, min_dist: float,
            seed: int, min_cluster_size: int) -> tuple[np.ndarray, str, np.ndarray]:
    """Blocking fit + cluster in one executor hop. Returns (coords, method used, cluster labels)."""
    coords, used = fit_2d(X, method, n_neighbors, min_dist, seed)
    return coords, used, cluster_labels(X, min_cluster_size)


async def _load_vectors(db: AsyncSession, kind: str, space: str,
                        session_id: str | None, limit: int) -> tuple[list[UUID], np.ndarray]:
    """Load (ref_ids, matrix) for the requested space, optionally confined to one session."""
    if kind == "object":
        stmt = select(ObjectEmbedding.object_id, ObjectEmbedding.dino_vec)
        if session_id:
            stmt = (stmt.join(Object, Object.object_id == ObjectEmbedding.object_id)
                        .join(Frame, Frame.frame_id == Object.frame_id)
                        .where(Frame.session_id == UUID(session_id)))
    else:
        col = FrameEmbedding.siglip_vec if space == "siglip" else FrameEmbedding.dino_vec
        stmt = select(FrameEmbedding.frame_id, col).where(col.isnot(None))
        if session_id:
            stmt = (stmt.join(Frame, Frame.frame_id == FrameEmbedding.frame_id)
                        .where(Frame.session_id == UUID(session_id)))
    rows = (await db.execute(stmt.limit(limit))).all()
    if not rows:
        return [], np.zeros((0, 2), dtype=float)
    ids = [r[0] for r in rows]
    X = np.asarray([list(r[1]) for r in rows], dtype=np.float32)
    return ids, X


async def fit_projection(db: AsyncSession, *, kind: str = "object", space: str = "dino",
                         session_id: str | None = None, method: str = "umap",
                         n_neighbors: int = 15, min_dist: float = 0.1, seed: int = 42,
                         min_cluster_size: int = 15,
                         limit: int = MAX_POINTS, notes: str | None = None) -> dict:
    """Fit and persist a 2D projection with density-cluster labels. Returns the projection row summary."""
    if kind not in ("object", "frame"):
        raise ValueError("kind must be object or frame")
    if space not in ("dino", "siglip"):
        raise ValueError("space must be dino or siglip")
    if kind == "object" and space == "siglip":
        raise ValueError("object crops are embedded with DINOv3 only; use space=dino")

    ids, X = await _load_vectors(db, kind, space, session_id, min(limit, MAX_POINTS))
    if len(ids) == 0:
        return {"error": "no embeddings found for this selection",
                "kind": kind, "space": space, "session_id": session_id, "n": 0}

    loop = asyncio.get_event_loop()
    coords, used, labels = await loop.run_in_executor(
        None, _fit_2d, X, method, n_neighbors, min_dist, seed, min_cluster_size)

    proj = EmbeddingProjection(
        kind=kind, space=space, method=used, n=len(ids),
        session_id=UUID(session_id) if session_id else None,
        params={"n_neighbors": n_neighbors, "min_dist": min_dist, "seed": seed,
                "min_cluster_size": min_cluster_size, "requested_method": method},
        notes=notes,
    )
    db.add(proj)
    await db.flush()  # need projection_id before the points reference it
    db.add_all([
        EmbeddingProjectionPoint(projection_id=proj.projection_id, ref_id=rid,
                                 x=float(c[0]), y=float(c[1]), cluster=int(lbl))
        for rid, c, lbl in zip(ids, coords, labels, strict=False)
    ])
    await db.commit()
    n_clusters = int(len({int(v) for v in labels} - {-1}))
    log.info("projection.fitted", projection_id=str(proj.projection_id), kind=kind, space=space,
             method=used, n=len(ids), clusters=n_clusters)
    return {"projection_id": str(proj.projection_id), "kind": kind, "space": space,
            "method": used, "n": len(ids), "clusters": n_clusters, "session_id": session_id}


async def list_projections(db: AsyncSession, limit: int = 50) -> list[dict]:
    rows = (await db.execute(
        select(EmbeddingProjection).order_by(EmbeddingProjection.created_at.desc()).limit(limit)
    )).scalars().all()
    return [{"projection_id": str(p.projection_id), "kind": p.kind, "space": p.space, "method": p.method,
             "n": p.n, "session_id": str(p.session_id) if p.session_id else None, "params": p.params,
             "notes": p.notes, "created_at": p.created_at.isoformat() if p.created_at else None}
            for p in rows]


async def projection_points(db: AsyncSession, projection_id: str, limit: int = MAX_POINTS) -> dict:
    """The 2D points plus the per-point attributes the explorer colours by, in one payload so the map renders
    from a single request."""
    proj = await db.get(EmbeddingProjection, UUID(projection_id))
    if proj is None:
        return {"error": "projection not found", "projection_id": projection_id}

    pts = (await db.execute(
        select(EmbeddingProjectionPoint.ref_id, EmbeddingProjectionPoint.x, EmbeddingProjectionPoint.y,
               EmbeddingProjectionPoint.cluster)
        .where(EmbeddingProjectionPoint.projection_id == proj.projection_id).limit(limit)
    )).all()
    if not pts:
        return {"projection_id": projection_id, "kind": proj.kind, "method": proj.method, "points": []}

    ref_ids = [r[0] for r in pts]
    meta: dict[uuidlib.UUID, dict] = {}
    if proj.kind == "object":
        rows = (await db.execute(
            select(Object.object_id, Object.class_id, Object.state, Object.source, Object.conf,
                   Object.tags, Object.frame_id)
            .where(Object.object_id.in_(ref_ids))
        )).all()
        for oid, class_id, state, source, conf, tags, frame_id in rows:
            meta[oid] = {"class_id": class_id, "state": state, "source": source,
                         "conf": round(float(conf), 4) if conf is not None else None,
                         "tags": tags or [], "frame_id": str(frame_id)}
    else:
        rows = (await db.execute(
            select(Frame.frame_id, Frame.session_id, Frame.scene, Frame.quality, Frame.tags)
            .where(Frame.frame_id.in_(ref_ids))
        )).all()
        for fid, sid, scene, quality, tags in rows:
            meta[fid] = {"session_id": str(sid), "scene": scene or {},
                         "quality": round(float(quality), 4) if quality is not None else None,
                         "tags": tags or []}

    return {
        "projection_id": projection_id, "kind": proj.kind, "space": proj.space, "method": proj.method,
        "n": proj.n,
        "points": [{"id": str(rid), "x": round(float(x), 5), "y": round(float(y), 5),
                    "cluster": int(cl) if cl is not None else -1, **meta.get(rid, {})}
                   for rid, x, y, cl in pts],
    }


async def delete_projection(db: AsyncSession, projection_id: str) -> dict:
    """Drop a fitted projection (points cascade)."""
    proj = await db.get(EmbeddingProjection, UUID(projection_id))
    if proj is None:
        return {"deleted": False, "reason": "not found"}
    await db.execute(delete(EmbeddingProjectionPoint).where(
        EmbeddingProjectionPoint.projection_id == proj.projection_id))
    await db.delete(proj)
    await db.commit()
    return {"deleted": True, "projection_id": projection_id}
