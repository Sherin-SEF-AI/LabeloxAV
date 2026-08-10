"""Human productivity and QA analytics (M-F.4): the DataOps view of the annotation operation, built from the
real Review trail and the training-job durations. It reports per-reviewer throughput, correction rate, and
agreement; inter-annotator agreement where two people reviewed the same object; throughput trends over time;
and a cost view (human hours from actual review time, GPU hours from training jobs, and the savings from
auto-accept). Kept honest and non-punitive: it surfaces agreement and quality to improve the guidelines and the
pipeline, and deliberately does NOT rank people by raw speed, which would trade correctness for throughput.
"""

from __future__ import annotations

from collections import defaultdict

from sqlalchemy import func, select

from core.logging import get_logger
from db.models import Object, Review, TrainingJob
from db.session import get_sessionmaker

log = get_logger("analytics.productivity")

# actions that CHANGED an auto-label (a correction) vs CONFIRMED it. create is net-new annotation, not a review.
_CORRECTION = {"reclassify", "reclassify_track", "reject", "adjust_geometry", "error_fix", "split_track", "merge_track"}
_CONFIRM = {"accept", "confirm"}
# cost assumptions (documented, configurable via env LBX_ overrides on these defaults)
_HUMAN_USD_PER_HOUR = 8.0        # annotation labor rate
_MANUAL_LABEL_SECONDS = 20.0     # est. time to draw + class a fresh object by hand


def _system(reviewer: str) -> bool:
    return reviewer in ("system", "relabel", "gate", "autolabel", "error-detect", "vlm-qa", "inspector_health")


async def reviewer_metrics() -> list[dict]:
    """Per-reviewer throughput, correction rate, average review time, and confirm agreement, from real reviews.
    Sorted by agreement (a quality signal), never by raw speed."""
    maker = get_sessionmaker()
    async with maker() as db:
        rows = (await db.execute(
            select(Review.user_id, Review.reviewer, Review.action, Review.before, Review.after,
                   Review.time_spent_ms))).all()
    agg: dict = defaultdict(lambda: {"n": 0, "corrections": 0, "confirms": 0, "time_ms": 0, "class_reviews": 0,
                                     "class_confirms": 0})
    for uid, reviewer, action, before, after, time_ms in rows:
        if _system(reviewer):
            continue
        key = str(uid) if uid else reviewer
        a = agg[key]
        a["n"] += 1
        a["time_ms"] += int(time_ms or 0)
        if action in _CORRECTION:
            a["corrections"] += 1
        elif action in _CONFIRM:
            a["confirms"] += 1
        bid = (before or {}).get("class_id")
        aid = (after or {}).get("class_id")
        if bid is not None and aid is not None:
            a["class_reviews"] += 1
            if bid == aid:
                a["class_confirms"] += 1

    out = []
    for key, a in agg.items():
        touched = a["corrections"] + a["confirms"]
        hours = a["time_ms"] / 3.6e6
        out.append({
            "reviewer": key[:12], "reviews": a["n"],
            "correction_rate": round(a["corrections"] / touched, 4) if touched else None,
            "avg_review_ms": round(a["time_ms"] / a["n"], 1) if a["n"] else 0,
            "objects_per_hour": round(a["n"] / hours, 1) if hours > 0.01 else None,
            "agreement": round(a["class_confirms"] / a["class_reviews"], 4) if a["class_reviews"] else None,
        })
    # sort by agreement desc (quality first), None last; NOT by speed
    out.sort(key=lambda r: (r["agreement"] is None, -(r["agreement"] or 0)))
    return out


async def interannotator_agreement() -> dict:
    """Where two or more distinct reviewers touched the same object, do their final class decisions agree? This
    is the honeypot / shared-task signal for guideline quality."""
    maker = get_sessionmaker()
    async with maker() as db:
        rows = (await db.execute(
            select(Review.object_id, Review.user_id, Review.reviewer, Review.after))).all()
    by_obj: dict = defaultdict(dict)  # object_id -> {reviewer_key: class_id}
    for oid, uid, reviewer, after in rows:
        if _system(reviewer):
            continue
        cid = (after or {}).get("class_id")
        if cid is not None:
            by_obj[oid][str(uid) if uid else reviewer] = cid
    shared = [d for d in by_obj.values() if len(d) >= 2]
    agree = sum(1 for d in shared if len(set(d.values())) == 1)
    return {"shared_objects": len(shared), "agreed": agree,
            "agreement_rate": round(agree / len(shared), 4) if shared else None}


async def throughput_trend(days: int = 30) -> list[dict]:
    """Reviews per day and the confirm (agreement) rate per day, so the operation's trend is visible."""
    maker = get_sessionmaker()
    async with maker() as db:
        rows = (await db.execute(select(Review.ts_ns, Review.reviewer, Review.before, Review.after))).all()
    per_day: dict = defaultdict(lambda: {"reviews": 0, "class_reviews": 0, "confirms": 0})
    for ts, reviewer, before, after in rows:
        if _system(reviewer) or not ts:
            continue
        day = int(ts // 86_400_000_000_000)  # ns -> whole days since epoch
        d = per_day[day]
        d["reviews"] += 1
        bid = (before or {}).get("class_id")
        aid = (after or {}).get("class_id")
        if bid is not None and aid is not None:
            d["class_reviews"] += 1
            if bid == aid:
                d["confirms"] += 1
    out = []
    for day in sorted(per_day)[-days:]:
        d = per_day[day]
        out.append({"day": day, "reviews": d["reviews"],
                    "agreement": round(d["confirms"] / d["class_reviews"], 4) if d["class_reviews"] else None})
    return out


async def cost_metrics() -> dict:
    """Human hours (actual review time), GPU hours (training-job durations), object/frame counts, and the
    auto-accept savings. Cost uses documented labor-rate and manual-time assumptions, surfaced with the number."""
    maker = get_sessionmaker()
    async with maker() as db:
        human_ms = (await db.execute(select(func.coalesce(func.sum(Review.time_spent_ms), 0)))).scalar_one()
        n_objects = (await db.execute(select(func.count()).select_from(Object))).scalar_one()
        n_auto = (await db.execute(select(func.count()).select_from(Object).where(Object.state == "auto_accept"))).scalar_one()
        n_frames = (await db.execute(select(func.count(func.distinct(Object.frame_id))).select_from(Object))).scalar_one()
        jobs = (await db.execute(select(TrainingJob.created_at, TrainingJob.updated_at, TrainingJob.compute_target)
                                 .where(TrainingJob.status.in_(("succeeded", "completed", "failed"))))).all()
    gpu_hours = sum(max(0.0, (u - c).total_seconds()) for c, u, _t in jobs if c and u) / 3600.0
    human_hours = human_ms / 3.6e6
    human_cost = round(human_hours * _HUMAN_USD_PER_HOUR, 2)
    # savings: each auto-accepted object is an object a human did not have to draw by hand
    saved_hours = n_auto * _MANUAL_LABEL_SECONDS / 3600.0
    saved_cost = round(saved_hours * _HUMAN_USD_PER_HOUR, 2)
    return {
        "human_hours": round(human_hours, 2), "human_cost_usd": human_cost, "gpu_hours": round(gpu_hours, 2),
        "n_objects": n_objects, "n_frames": n_frames, "n_auto_accept": n_auto,
        "cost_per_object_usd": round(human_cost / n_objects, 4) if n_objects else None,
        "cost_per_frame_usd": round(human_cost / n_frames, 4) if n_frames else None,
        "auto_accept_saved_hours": round(saved_hours, 2), "auto_accept_saved_usd": saved_cost,
        "assumptions": {"human_usd_per_hour": _HUMAN_USD_PER_HOUR, "manual_label_seconds": _MANUAL_LABEL_SECONDS},
    }


async def productivity_report() -> dict:
    """The full DataOps operations view: reviewers, agreement, trend, and cost."""
    reviewers = await reviewer_metrics()
    return {
        "reviewers": reviewers, "n_reviewers": len(reviewers),
        "interannotator": await interannotator_agreement(),
        "trend": await throughput_trend(),
        "cost": await cost_metrics(),
        "note": "Agreement and correction rate are quality signals for the guidelines and the pipeline, "
                "not a speed ranking. Do not optimize reviewers for raw throughput.",
    }
