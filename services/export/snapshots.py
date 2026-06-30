"""Milestone I: dataset snapshot lineage and diff. DatasetCommit already seals an immutable, content-hashed
snapshot of a slice (commit_id is the hash, export_uris point at the frozen artifacts), but parent_id was
always None, so successive snapshots of the same slice were not chained and could not be compared. This adds
the lineage link (a new snapshot's parent is the most recent prior snapshot of the same slice name) and a
pure diff between two snapshots, so a curator sees exactly what changed between dataset versions. The diff is
pure over two commit-like dicts, so it is tested without infra.
"""

from __future__ import annotations

from core.logging import get_logger

log = get_logger("snapshots")


def diff_commits(a: dict, b: dict) -> dict:
    """Diff two snapshots a (older) -> b (newer). Reports the count deltas, whether the ontology version
    changed, and which slice_spec fields differ."""
    spec_a, spec_b = a.get("slice_spec") or {}, b.get("slice_spec") or {}
    slice_changes = {k: {"from": spec_a.get(k), "to": spec_b.get(k)}
                     for k in set(spec_a) | set(spec_b) if spec_a.get(k) != spec_b.get(k)}
    return {
        "from": a.get("commit_id"), "to": b.get("commit_id"),
        "object_count_delta": (b.get("object_count") or 0) - (a.get("object_count") or 0),
        "object_3d_delta": (b.get("object_3d_count") or 0) - (a.get("object_3d_count") or 0),
        "cloud_delta": (b.get("cloud_count") or 0) - (a.get("cloud_count") or 0),
        "ontology_changed": a.get("ontology_version") != b.get("ontology_version"),
        "slice_changes": slice_changes,
    }


async def resolve_parent(db, slice_name: str, this_commit_id: str) -> str | None:
    """The most recent prior snapshot of the same slice name, to chain as the new snapshot's parent."""
    from sqlalchemy import select

    from db.models import DatasetCommit
    rows = (await db.execute(
        select(DatasetCommit.commit_id).where(DatasetCommit.slice_spec["name"].astext == slice_name)
        .order_by(DatasetCommit.created_at.desc()))).scalars().all()
    for cid in rows:
        if cid != this_commit_id:
            return cid
    return None


async def lineage(commit_id: str, max_depth: int = 100) -> dict:
    """Walk the parent chain from a snapshot back to its root."""
    from db.models import DatasetCommit
    from db.session import get_sessionmaker
    chain = []
    async with get_sessionmaker()() as db:
        cur = await db.get(DatasetCommit, commit_id)
        if cur is None:
            return {"error": "commit not found"}
        seen = set()
        while cur is not None and len(chain) < max_depth and cur.commit_id not in seen:
            seen.add(cur.commit_id)
            chain.append({"commit_id": cur.commit_id, "parent_id": cur.parent_id,
                          "object_count": cur.object_count, "ontology_version": cur.ontology_version})
            cur = await db.get(DatasetCommit, cur.parent_id) if cur.parent_id else None
    return {"commit_id": commit_id, "depth": len(chain), "chain": chain}


async def compare_commits(a_id: str, b_id: str) -> dict:
    """Load two snapshots and diff them (a older, b newer)."""
    from db.models import DatasetCommit
    from db.session import get_sessionmaker
    async with get_sessionmaker()() as db:
        a = await db.get(DatasetCommit, a_id)
        b = await db.get(DatasetCommit, b_id)
    if a is None or b is None:
        return {"error": "commit not found"}

    def _d(c):
        return {"commit_id": c.commit_id, "slice_spec": c.slice_spec, "object_count": c.object_count,
                "object_3d_count": c.object_3d_count, "cloud_count": c.cloud_count,
                "ontology_version": c.ontology_version}
    return diff_commits(_d(a), _d(b))
