"""How good are the label-error detectors? Plant known errors, see which ones they find, put everything back.

Human confirm/dismiss verdicts (services/errordetect/queue.py) measure a detector's precision only as fast as
people judge candidates, and on this corpus they had judged almost none. Planting gives a ground truth today:

  1. In one transaction, run each class-error detector on a session as it is (the flags it already raises).
  2. Change the class of a random sample of that session's accepted labels, preferring a sibling class (same
     l1, the confusion a detector should catch: sedan for hatchback, not sedan for traffic light).
  3. Run the detectors again. A planted label it now flags is a catch; a new flag on a label that was not
     planted is a false alarm caused by the planting.
  4. Roll the transaction back. Nothing is written: commit is refused for the whole run.

Reported per detector: recall (planted errors caught) and precision of the new flags. Flags that existed
before planting are left out of both, since they may be real errors nobody has planted.

The detectors measured are the ones that judge a label's class: confident learning (cleanlab over DINO
embeddings), embedding outliers, and track consistency. The policy, near-duplicate and critic detectors check
machine labels for geometric or contextual nonsense, which a planted class change does not create, so a
number for them here would be about the benchmark rather than the detector.
"""

from __future__ import annotations

import random
import uuid
from collections.abc import Awaitable, Callable

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Frame, Object, ObjectEmbedding
from services.autolabel.ontology import get_ontology

_ACCEPTED = ("accepted", "auto_accept")


def _detectors() -> dict[str, Callable[..., Awaitable[list[dict]]]]:
    from services.errordetect.confident import detect_confident_learning
    from services.errordetect.consistency import detect_consistency
    from services.errordetect.embedding_outlier import detect_embedding_outliers

    return {"confident_learning": detect_confident_learning, "embedding_outlier": detect_embedding_outliers,
            "track_consistency": detect_consistency}


async def _flags(db: AsyncSession, session_id: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for name, fn in _detectors().items():
        out[name] = {str(c["object_id"]) for c in await fn(db, session_id) if c.get("object_id")}
    return out


def _swap_class(cid: int, present: list[int], rng: random.Random, onto) -> int | None:
    """A plausible wrong class: a sibling present in the session, else any other class present."""
    l1 = onto.by_id(cid).l1
    siblings = [c for c in present if c != cid and onto.by_id(c).l1 == l1]
    others = [c for c in present if c != cid]
    pool = siblings or others
    return rng.choice(pool) if pool else None


async def planted_error_benchmark(db: AsyncSession, session_id: str, *, rate: float = 0.1,
                                  seed: int = 7) -> dict:
    """Plant class errors in one session's accepted labels, measure each detector, roll back."""
    onto = get_ontology()
    rng = random.Random(seed)
    sid = uuid.UUID(str(session_id))

    async def _refuse(*_a, **_k):
        raise RuntimeError("the planted-error benchmark never commits")

    real_commit = db.commit
    db.commit = _refuse  # type: ignore[method-assign]
    try:
        rows = (await db.execute(
            select(Object.object_id, Object.class_id)
            .join(ObjectEmbedding, ObjectEmbedding.object_id == Object.object_id)
            .join(Frame, Frame.frame_id == Object.frame_id)
            .where(Frame.session_id == sid, Object.state.in_(_ACCEPTED))
            .order_by(Object.object_id))).all()
        if len(rows) < 20:
            return {"session_id": str(sid), "labels": len(rows),
                    "refused": "fewer than 20 accepted labels with embeddings; not enough to measure"}
        present = sorted({int(c) for _, c in rows})
        before = await _flags(db, str(sid))

        k = max(1, int(round(len(rows) * rate)))
        planted: dict[str, tuple[int, int]] = {}
        for oid, cid in rng.sample(rows, k):
            new = _swap_class(int(cid), present, rng, onto)
            if new is None:
                continue
            planted[str(oid)] = (int(cid), new)
            await db.execute(update(Object).where(Object.object_id == oid).values(class_id=new))
        await db.flush()
        after = await _flags(db, str(sid))
    finally:
        await db.rollback()
        db.commit = real_commit  # type: ignore[method-assign]

    planted_ids = set(planted)
    per: dict[str, dict] = {}
    for name in _detectors():
        new_flags = after[name] - before[name]
        caught = new_flags & planted_ids
        per[name] = {
            "caught": len(caught), "new_flags": len(new_flags),
            "recall": round(len(caught) / len(planted_ids), 4) if planted_ids else None,
            "precision": round(len(caught) / len(new_flags), 4) if new_flags else None,
            "flags_before": len(before[name]),
        }
    sibling = sum(1 for old, new in planted.values() if onto.by_id(old).l1 == onto.by_id(new).l1)
    return {"session_id": str(sid), "labels": len(rows), "classes": len(present), "planted": len(planted_ids),
            "planted_sibling": sibling, "rate": rate, "seed": seed, "detectors": per}


async def benchmark_sessions(db: AsyncSession, session_ids: list[str], *, rate: float = 0.1,
                             seed: int = 7) -> dict:
    """The benchmark over several sessions, pooled: total catches over total planted, per detector."""
    runs = [await planted_error_benchmark(db, s, rate=rate, seed=seed) for s in session_ids]
    measured = [r for r in runs if "refused" not in r]
    pooled: dict[str, dict] = {}
    for name in _detectors():
        caught = sum(r["detectors"][name]["caught"] for r in measured)
        new = sum(r["detectors"][name]["new_flags"] for r in measured)
        planted = sum(r["planted"] for r in measured)
        pooled[name] = {"caught": caught, "planted": planted, "new_flags": new,
                        "recall": round(caught / planted, 4) if planted else None,
                        "precision": round(caught / new, 4) if new else None}
    return {"sessions": runs, "pooled": pooled, "rate": rate, "seed": seed}


async def default_sessions(db: AsyncSession, n: int = 5) -> list[str]:
    """The sessions with the most accepted, embedded labels: where the detectors have the most to work with."""
    from sqlalchemy import func

    rows = (await db.execute(
        select(Frame.session_id, func.count())
        .join(Object, Object.frame_id == Frame.frame_id)
        .join(ObjectEmbedding, ObjectEmbedding.object_id == Object.object_id)
        .where(Object.state.in_(_ACCEPTED))
        .group_by(Frame.session_id).order_by(func.count().desc()).limit(n))).all()
    return [str(s) for s, _ in rows]


__all__ = ["benchmark_sessions", "default_sessions", "planted_error_benchmark"]
