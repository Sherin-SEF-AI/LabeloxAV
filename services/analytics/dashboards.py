"""Analytics and eval dashboards (the sales sheet): aggregate read models over Postgres.

Each function opens its own async session and returns plain dicts/lists, ready to JSON-serialize.
These are the operator-facing numbers: class distribution and long-tail coverage, label-source
mix (how much was auto-accepted vs human-touched), scenario coverage, capture-density geo points,
and the review-agreement signal (the eval/loop health: did humans confirm or reclassify the auto
label). Nothing here mutates state.
"""

from __future__ import annotations

from uuid import UUID

from geoalchemy2 import Geometry
from sqlalchemy import cast, func, select

from core.logging import get_logger
from db.models import Frame, Object, PiiAudit, Review, Scenario, Track
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology

log = get_logger("analytics")

# The source/state vocabularies, kept explicit so the mix is reported with a fixed shape even when
# a bucket is empty (matches core.schemas GateState / ObjectSource).
_STATES = ["auto_accept", "review", "annotate", "accepted", "rejected"]
_SOURCES = ["fused", "auto_accept", "human"]


def _sid(session_id: str | None) -> UUID | None:
    return UUID(session_id) if session_id else None


async def class_distribution(session_id: str | None = None) -> list[dict]:
    """Per-class object counts joined to ontology, sorted desc, with long-tail coverage.

    Coverage is how many distinct ontology classes have at least one object versus the total in
    the ontology; the long tail is the set of (especially India) classes still at zero.
    """
    onto = get_ontology()
    maker = get_sessionmaker()
    sid = _sid(session_id)

    stmt = select(Object.class_id, func.count().label("count")).group_by(Object.class_id)
    if sid is not None:
        stmt = stmt.join(Frame, Object.frame_id == Frame.frame_id).where(Frame.session_id == sid)

    async with maker() as db:
        rows = (await db.execute(stmt)).all()

    counts = {class_id: count for class_id, count in rows}
    out: list[dict] = []
    for c in onto.classes:
        out.append(
            {
                "class_id": c.id,
                "name": c.name,
                "l0": c.l0,
                "l1": c.l1,
                "india": c.india,
                "count": int(counts.get(c.id, 0)),
            }
        )
    out.sort(key=lambda r: r["count"], reverse=True)
    return out


async def label_source_mix(session_id: str | None = None) -> dict:
    """Counts of objects by state and by source, plus auto-accepted / human-touched percentages."""
    maker = get_sessionmaker()
    sid = _sid(session_id)

    def _scoped(col):
        s = select(col, func.count().label("count")).group_by(col)
        if sid is not None:
            s = s.join(Frame, Object.frame_id == Frame.frame_id).where(Frame.session_id == sid)
        return s

    # Human-touched is a per-object OR (state settled by a reviewer OR sourced from a human), so it
    # is counted as DISTINCT objects, not by summing overlapping state/source buckets.
    human_pred = Object.state.in_(["accepted", "rejected"]) | (Object.source == "human")
    human_stmt = select(func.count()).select_from(Object)
    if sid is not None:
        human_stmt = human_stmt.join(Frame, Object.frame_id == Frame.frame_id).where(
            Frame.session_id == sid
        )
    human_stmt = human_stmt.where(human_pred)

    async with maker() as db:
        state_rows = (await db.execute(_scoped(Object.state))).all()
        source_rows = (await db.execute(_scoped(Object.source))).all()
        human = (await db.execute(human_stmt)).scalar_one()

    by_state = {s: 0 for s in _STATES}
    for state, count in state_rows:
        by_state[state] = by_state.get(state, 0) + int(count)

    by_source = {s: 0 for s in _SOURCES}
    for source, count in source_rows:
        by_source[source] = by_source.get(source, 0) + int(count)

    total = sum(by_state.values())
    auto = by_state.get("auto_accept", 0)
    human = int(human)

    return {
        "total": total,
        "by_state": by_state,
        "by_source": by_source,
        "auto_accepted_pct": round(100.0 * auto / total, 2) if total else 0.0,
        "human_touched_pct": round(100.0 * human / total, 2) if total else 0.0,
    }


async def scenario_coverage(session_id: str | None = None) -> list[dict]:
    """Counts per scenario type with mean criticality, sorted by count desc."""
    maker = get_sessionmaker()
    sid = _sid(session_id)

    stmt = (
        select(
            Scenario.type,
            func.count().label("count"),
            func.avg(Scenario.criticality).label("mean_criticality"),
        )
        .group_by(Scenario.type)
        .order_by(func.count().desc())
    )
    if sid is not None:
        stmt = stmt.where(Scenario.session_id == sid)

    async with maker() as db:
        rows = (await db.execute(stmt)).all()

    return [
        {
            "type": stype,
            "count": int(count),
            "mean_criticality": round(float(mean or 0.0), 4),
        }
        for stype, count, mean in rows
    ]


async def geo_points(session_id: str | None = None, limit: int = 2000) -> list[dict]:
    """Capture-density points from Frame.gnss as {lat, lon}, skipping frames without a fix."""
    maker = get_sessionmaker()
    sid = _sid(session_id)

    geom = cast(Frame.gnss, Geometry)
    stmt = (
        select(func.ST_Y(geom).label("lat"), func.ST_X(geom).label("lon"))
        .where(Frame.gnss.isnot(None))
        .limit(limit)
    )
    if sid is not None:
        stmt = stmt.where(Frame.session_id == sid)

    async with maker() as db:
        rows = (await db.execute(stmt)).all()

    return [{"lat": float(lat), "lon": float(lon)} for lat, lon in rows]


async def review_agreement() -> dict:
    """The eval/loop signal: over Review rows, the fraction where the human CONFIRMED the auto
    class (before.class_id == after.class_id) versus RECLASSIFIED it (changed), overall and
    per-class. A high confirm rate is a proxy that auto-accept is trustworthy and the loop is
    converging. Also reports total reviews and mean time_spent_ms.
    """
    onto = get_ontology()
    maker = get_sessionmaker()

    async with maker() as db:
        rows = (await db.execute(select(Review.before, Review.after, Review.time_spent_ms))).all()

    total = 0
    confirmed = 0
    time_spent_total = 0
    per_class: dict[int, dict] = {}

    for before, after, time_spent_ms in rows:
        before = before or {}
        after = after or {}
        before_id = before.get("class_id")
        after_id = after.get("class_id")
        # Only count reviews that recorded a before/after class (a class decision was made).
        if before_id is None or after_id is None:
            continue
        total += 1
        time_spent_total += int(time_spent_ms or 0)
        is_confirm = before_id == after_id
        if is_confirm:
            confirmed += 1
        bucket = per_class.setdefault(before_id, {"reviews": 0, "confirmed": 0})
        bucket["reviews"] += 1
        if is_confirm:
            bucket["confirmed"] += 1

    reclassified = total - confirmed
    per_class_out: list[dict] = []
    for class_id, bucket in per_class.items():
        try:
            name = onto.by_id(class_id).name
        except KeyError:
            name = str(class_id)
        per_class_out.append(
            {
                "class_id": class_id,
                "name": name,
                "reviews": bucket["reviews"],
                "confirmed": bucket["confirmed"],
                "confirmed_pct": round(100.0 * bucket["confirmed"] / bucket["reviews"], 2)
                if bucket["reviews"]
                else 0.0,
            }
        )
    per_class_out.sort(key=lambda r: r["reviews"], reverse=True)

    return {
        "total_reviews": total,
        "confirmed": confirmed,
        "reclassified": reclassified,
        "confirmed_pct": round(100.0 * confirmed / total, 2) if total else 0.0,
        "reclassified_pct": round(100.0 * reclassified / total, 2) if total else 0.0,
        "mean_time_spent_ms": round(time_spent_total / total, 1) if total else 0.0,
        "per_class": per_class_out,
    }


async def overview(session_id: str | None = None) -> dict:
    """Top-card rollups: corpus totals plus the source mix and long-tail coverage number."""
    onto = get_ontology()
    maker = get_sessionmaker()
    sid = _sid(session_id)

    async with maker() as db:
        if sid is not None:
            sessions = 1
            frames = (
                await db.execute(
                    select(func.count()).select_from(Frame).where(Frame.session_id == sid)
                )
            ).scalar_one()
            objects = (
                await db.execute(
                    select(func.count())
                    .select_from(Object)
                    .join(Frame, Object.frame_id == Frame.frame_id)
                    .where(Frame.session_id == sid)
                )
            ).scalar_one()
            tracks = (
                await db.execute(
                    select(func.count()).select_from(Track).where(Track.session_id == sid)
                )
            ).scalar_one()
            scenarios = (
                await db.execute(
                    select(func.count()).select_from(Scenario).where(Scenario.session_id == sid)
                )
            ).scalar_one()
            distinct_classes = (
                await db.execute(
                    select(func.count(func.distinct(Object.class_id)))
                    .select_from(Object)
                    .join(Frame, Object.frame_id == Frame.frame_id)
                    .where(Frame.session_id == sid)
                )
            ).scalar_one()
        else:
            sessions = (
                await db.execute(select(func.count()).select_from(DbSession))
            ).scalar_one()
            frames = (await db.execute(select(func.count()).select_from(Frame))).scalar_one()
            objects = (await db.execute(select(func.count()).select_from(Object))).scalar_one()
            tracks = (await db.execute(select(func.count()).select_from(Track))).scalar_one()
            scenarios = (
                await db.execute(select(func.count()).select_from(Scenario))
            ).scalar_one()
            distinct_classes = (
                await db.execute(select(func.count(func.distinct(Object.class_id))))
            ).scalar_one()

    total_classes = len(onto.classes)
    mix = await label_source_mix(session_id)

    return {
        "sessions": int(sessions),
        "frames": int(frames),
        "objects": int(objects),
        "tracks": int(tracks),
        "scenarios": int(scenarios),
        "auto_accepted_pct": mix["auto_accepted_pct"],
        "human_touched_pct": mix["human_touched_pct"],
        "long_tail": {
            "covered_classes": int(distinct_classes),
            "total_classes": total_classes,
            "coverage_pct": round(100.0 * int(distinct_classes) / total_classes, 2)
            if total_classes
            else 0.0,
        },
        "source_mix": mix,
    }


async def pii_coverage(session_id: str | None = None) -> dict:
    """Gate A (DPDPA) evidence: fraction of frames with a PII-anonymization audit row, totals of
    faces/plates blurred, and the method versions in play. This is the compliance attestation a
    buyer (and a regulator) asks to see."""
    sid = _sid(session_id)
    maker = get_sessionmaker()
    async with maker() as db:
        fstmt = select(func.count()).select_from(Frame)
        if sid is not None:
            fstmt = fstmt.where(Frame.session_id == sid)
        total_frames = (await db.execute(fstmt)).scalar_one()

        astmt = select(
            func.count().label("n"),
            func.coalesce(func.sum(PiiAudit.n_faces), 0),
            func.coalesce(func.sum(PiiAudit.n_plates), 0),
        )
        if sid is not None:
            astmt = astmt.where(PiiAudit.session_id == sid)
        n_audited, n_faces, n_plates = (await db.execute(astmt)).one()

        mstmt = select(PiiAudit.method_version, func.count()).group_by(PiiAudit.method_version)
        if sid is not None:
            mstmt = mstmt.where(PiiAudit.session_id == sid)
        methods = {m: c for m, c in (await db.execute(mstmt)).all()}

    return {
        "total_frames": int(total_frames),
        "frames_anonymized": int(n_audited),
        "coverage_pct": round(100.0 * n_audited / total_frames, 1) if total_frames else 0.0,
        "faces_blurred": int(n_faces),
        "plates_blurred": int(n_plates),
        "method_versions": methods,
    }


# ---- Data Intelligence Layer (M1.7): scene splits, dedup rate, growth, embedding cluster map ----
async def scene_splits(session_id: str | None = None) -> dict:
    """Frame counts per scene axis value (weather/time_of_day/road_type/density)."""
    from collections import Counter

    maker = get_sessionmaker()
    async with maker() as db:
        stmt = select(Frame.scene).where(Frame.scene.isnot(None))
        if session_id:
            stmt = stmt.where(Frame.session_id == UUID(session_id))
        rows = (await db.execute(stmt)).all()
    axes = {ax: Counter() for ax in ("weather", "time_of_day", "road_type", "density")}
    for (scene,) in rows:
        for ax, c in axes.items():
            v = (scene or {}).get(ax)
            if v:
                c[v] += 1
    return {ax: dict(c.most_common()) for ax, c in axes.items()}


async def dedup_rate(session_id: str | None = None) -> dict:
    """Duplicate rate: redundant non-canonical frames over total."""
    maker = get_sessionmaker()
    async with maker() as db:
        base = select(func.count()).select_from(Frame)
        red = select(func.count()).select_from(Frame).where(
            Frame.dup_group_id.isnot(None), Frame.is_dup_canonical.is_(False))
        grp = select(func.count(func.distinct(Frame.dup_group_id))).where(Frame.dup_group_id.isnot(None))
        if session_id:
            sid = UUID(session_id)
            base = base.where(Frame.session_id == sid)
            red = red.where(Frame.session_id == sid)
            grp = grp.where(Frame.session_id == sid)
        total = (await db.execute(base)).scalar_one()
        redundant = (await db.execute(red)).scalar_one()
        groups = (await db.execute(grp)).scalar_one()
    return {"total": int(total), "redundant": int(redundant), "groups": int(groups),
            "rate": round(redundant / total, 3) if total else 0.0}


async def dataset_growth() -> list[dict]:
    """Frames ingested per day (cumulative), from frame.created_at."""
    maker = get_sessionmaker()
    async with maker() as db:
        rows = (await db.execute(
            select(func.date_trunc("day", Frame.created_at).label("d"), func.count())
            .group_by("d").order_by("d"))).all()
    out, cum = [], 0
    for d, n in rows:
        cum += int(n)
        out.append({"date": d.date().isoformat() if d else None, "frames": int(n), "cumulative": cum})
    return out


async def cluster_map(limit: int = 1500) -> dict:
    """UMAP 2D projection of DINOv3 frame embeddings + HDBSCAN clusters, colored by scene/cluster, for
    the interactive map. A derived/cached artifact (recomputable), capped for responsiveness."""
    import numpy as np

    from db.models import FrameEmbedding

    maker = get_sessionmaker()
    async with maker() as db:
        rows = (await db.execute(
            select(Frame.frame_id, FrameEmbedding.dino_vec, Frame.scene)
            .join(FrameEmbedding, FrameEmbedding.frame_id == Frame.frame_id)
            .where(FrameEmbedding.dino_vec.isnot(None)).limit(limit))).all()
    if len(rows) < 10:
        return {"points": [], "n": len(rows)}
    mat = np.asarray([r[1] for r in rows], dtype=np.float32)
    import hdbscan
    import umap

    xy = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine").fit_transform(mat)
    labels = hdbscan.HDBSCAN(min_cluster_size=15).fit_predict(mat)
    points = []
    for (fid, _, scene), (x, y), lbl in zip(rows, xy, labels):
        s = scene or {}
        points.append({"frame_id": str(fid), "x": round(float(x), 3), "y": round(float(y), 3),
                       "cluster": int(lbl), "time_of_day": s.get("time_of_day"), "road_type": s.get("road_type")})
    return {"points": points, "n": len(points), "clusters": int(len(set(labels)) - (1 if -1 in labels else 0))}
