"""Recall recovery: manufacture the objects the primary detector never proposed.

The active-learning value score can only rank objects that already exist as rows, so a missed object
is invisible to triage, mining, and the flywheel. This layer sources candidate objects from three
channels that do not depend on the primary detector's recall, fuses them, and persists each as a
provisional review-state object so it flows through the existing gate, queue, and governance untouched.

The three channels:

  trackgap   a tracked object present before and after a frame but absent on it is almost certainly a
             per-frame miss. Interpolate it. Free, runs full-session, no model. Highest precision.
  openvocab  the open-vocab detector (Path B: YOLO-World + SAM) finds named ontology classes the
             primary detector missed. Medium precision.
  region     class-agnostic region proposals (SAM everything mode) classified by the VLM recover
             objects no named detector proposed. Lowest precision, the noisy channel, upside not
             foundation.

The pure tier (the functions below) is numpy only and imports with no torch; the orchestrator imports
the DB and the lazy model backends inside the function so tests load with no GPU stack.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np

from core.logging import get_logger

log = get_logger("recall")

# Per-channel precision priors, the probability a raw proposal from that channel is a real miss. These
# are corrected by the human verdict over time (fit_channel_reliability), the same way the isotonic gate
# corrects its calibration. Rank orders the channels for fusion (trackgap wins ties).
_CHANNEL_PRIOR = {"trackgap": 0.85, "openvocab": 0.60, "region": 0.40}
_CHANNEL_RANK = {"trackgap": 3, "openvocab": 2, "region": 1}


@dataclass(frozen=True)
class RecallProposal:
    bbox: tuple[float, float, float, float]
    channel: str
    conf: float
    class_id: int | None = None
    class_name: str | None = None
    track_id: uuid.UUID | None = None
    interp_source: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class FusedRecall:
    bbox: tuple[float, float, float, float]
    channels: set[str]
    conf: float
    class_id: int | None = None
    class_name: str | None = None
    track_id: uuid.UUID | None = None
    interp_source: str | None = None
    value: float = 0.0


# ---------------------------------------------------------------------------------------------------
# Pure geometry, mining, fusion, scoring (numpy only, no DB, no torch)
# ---------------------------------------------------------------------------------------------------

def iou_matrix(a, b) -> np.ndarray:
    """Pairwise IoU of two xyxy box sets, shape (len(a), len(b)). Empty inputs return a correctly
    shaped zero matrix."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=float)
    area_a = (a[:, 2] - a[:, 0]).clip(min=0) * (a[:, 3] - a[:, 1]).clip(min=0)
    area_b = (b[:, 2] - b[:, 0]).clip(min=0) * (b[:, 3] - b[:, 1]).clip(min=0)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)


def interp_box(box_a, box_b, frac: float) -> tuple[float, float, float, float]:
    """Linear corner interpolation. frac 0 returns a, frac 1 returns b."""
    a = np.asarray(box_a, dtype=float)
    b = np.asarray(box_b, dtype=float)
    out = a + (b - a) * float(frac)
    return (float(out[0]), float(out[1]), float(out[2]), float(out[3]))


def track_gap_proposals(track_id, class_id, observed, gap_frames, max_gap_frames: int = 5,
                        conf_decay: float = 0.9) -> list[tuple]:
    """Interpolate a tracked object across frames where it was not observed.

    observed: (ts_ns, bbox, conf) for frames where the track has an object.
    gap_frames: (ts_ns, frame_id) session frames where the track has no object.

    For each adjacent observation pair, interpolate any session frame whose ts falls strictly between
    them. A run longer than max_gap_frames is a likely exit or occlusion, not a per-frame miss, and is
    skipped. Confidence decays as min(c_a, c_b) * conf_decay**i across the run.
    """
    obs = sorted(observed, key=lambda o: o[0])
    gaps = sorted(gap_frames, key=lambda g: g[0])
    out: list[tuple] = []
    for (ts_a, box_a, conf_a), (ts_b, box_b, conf_b) in zip(obs, obs[1:], strict=False):
        if ts_b <= ts_a:
            continue
        interior = [(ts, fid) for ts, fid in gaps if ts_a < ts < ts_b]
        if not interior or len(interior) > max_gap_frames:
            continue
        base = min(conf_a, conf_b)
        for i, (ts, fid) in enumerate(interior):
            frac = (ts - ts_a) / (ts_b - ts_a)
            out.append((fid, RecallProposal(
                bbox=interp_box(box_a, box_b, frac), channel="trackgap",
                conf=float(base * (conf_decay ** i)), class_id=class_id, track_id=track_id,
                interp_source="recall_trackgap")))
    return out


def mine_unmatched(proposals, existing_boxes, iou_match: float = 0.45) -> list[RecallProposal]:
    """Keep only proposals whose max IoU against any existing object is below iou_match. A region
    already covered by some object (any state) is already known and is not a miss."""
    if not proposals:
        return []
    if not existing_boxes:
        return list(proposals)
    m = iou_matrix([p.bbox for p in proposals], existing_boxes)
    return [p for i, p in enumerate(proposals) if float(m[i].max()) < iou_match]


def fuse_channels(proposals, fuse_iou: float = 0.55) -> list[FusedRecall]:
    """Greedy NMS across channels. Sort by (channel rank, conf) descending; a suppressed proposal
    merges its channel into the kept candidate so multi-channel agreement is preserved for scoring, and
    a kept candidate without a class inherits one from a suppressed channel that has it."""
    ordered = sorted(proposals, key=lambda p: (_CHANNEL_RANK.get(p.channel, 0), p.conf), reverse=True)
    kept: list[FusedRecall] = []
    for p in ordered:
        merged = False
        for fc in kept:
            if float(iou_matrix([p.bbox], [fc.bbox])[0, 0]) >= fuse_iou:
                fc.channels.add(p.channel)
                if fc.class_id is None and p.class_id is not None:
                    fc.class_id = p.class_id
                if fc.class_name is None and p.class_name is not None:
                    fc.class_name = p.class_name
                if fc.track_id is None and p.track_id is not None:
                    fc.track_id = p.track_id
                if fc.interp_source is None and p.interp_source is not None:
                    fc.interp_source = p.interp_source
                merged = True
                break
        if not merged:
            kept.append(FusedRecall(bbox=p.bbox, channels={p.channel}, conf=p.conf, class_id=p.class_id,
                                    class_name=p.class_name, track_id=p.track_id,
                                    interp_source=p.interp_source))
    return kept


def fn_value(c: FusedRecall, is_rare: bool, in_rare_frame: bool) -> float:
    """The recovery value of a candidate. base is the strongest channel prior; multi-channel agreement,
    a rare class, and a rare-scenario frame each add. These are priors, corrected by the human verdict
    over time, the same way the isotonic gate is corrected by reviewed labels."""
    base = max((_CHANNEL_PRIOR.get(ch, 0.0) for ch in c.channels), default=0.0)
    v = base
    if len(c.channels) >= 2:
        v += 0.15
    if is_rare:
        v += 0.20
    if in_rare_frame:
        v += 0.10
    return float(min(1.0, max(0.0, v)))


# ---------------------------------------------------------------------------------------------------
# Orchestrator (async, DB plus lazy backends)
# ---------------------------------------------------------------------------------------------------

async def run_recall(db, session_id, *, backend=None, frame_ids=None) -> dict:
    """Source recall candidates over a session, fuse them, and persist each as a review-state object
    plus a RecallCandidate audit row. Imports the DB and the model backends here so the pure tier above
    stays torch-free."""
    from sqlalchemy import select

    from core.config import get_settings
    from core.storage import get_object_store
    from db.models import Frame, Object, RecallCandidate, ScenarioCandidate, Track
    from services.autolabel.gate import is_rare
    from services.autolabel.ontology import get_ontology

    cfg = get_settings().phase4.recall
    onto = get_ontology()
    sid = uuid.UUID(str(session_id))

    # 1. session frames ordered by ts, the rare-scenario frames, and the requested model-channel targets
    frames = (await db.execute(
        select(Frame).where(Frame.session_id == sid).order_by(Frame.ts_ns))).scalars().all()
    if not frames:
        return {"session_id": str(sid), "persisted": 0, "by_channel": {}, "frames": 0}
    frame_ts = {f.frame_id: int(f.ts_ns) for f in frames}
    frame_by_id = {f.frame_id: f for f in frames}
    ordered_fids = [f.frame_id for f in frames]
    rare_frames = set((await db.execute(
        select(ScenarioCandidate.frame_id).where(
            ScenarioCandidate.session_id == sid,
            ScenarioCandidate.kind.in_(("rare_class", "embedding_outlier"))))).scalars())

    if frame_ids is not None:
        want = {uuid.UUID(str(f)) for f in frame_ids}
        target_frames = [f for f in frames if f.frame_id in want]
    else:
        target_frames = list(frames)

    # 2. existing objects (any state) per frame, for unmatched mining
    objs = (await db.execute(select(Object).join(Frame, Frame.frame_id == Object.frame_id)
                             .where(Frame.session_id == sid))).scalars().all()
    existing_by_frame: dict = {}
    for o in objs:
        existing_by_frame.setdefault(o.frame_id, []).append(list(o.bbox))

    proposals_by_frame: dict = {}

    # 3. trackgap channel over the whole session, no model
    tracks = {t.track_id: t for t in (await db.execute(
        select(Track).where(Track.session_id == sid))).scalars()}
    by_track: dict = {}
    track_class: dict = {}
    track_present: dict = {}
    for o in objs:
        if o.track_id is None:
            continue
        by_track.setdefault(o.track_id, []).append((frame_ts[o.frame_id], list(o.bbox), float(o.conf)))
        track_present.setdefault(o.track_id, set()).add(o.frame_id)
        track_class.setdefault(o.track_id, o.class_id)
    for tid, observed in by_track.items():
        t = tracks.get(tid)
        class_id = t.class_id if t is not None else track_class.get(tid)
        if class_id is None:
            continue
        present = track_present.get(tid, set())
        gap = [(frame_ts[fid], fid) for fid in ordered_fids if fid not in present]
        for fid, prop in track_gap_proposals(tid, class_id, observed, gap,
                                              max_gap_frames=cfg.max_gap_frames, conf_decay=cfg.conf_decay):
            proposals_by_frame.setdefault(fid, []).append(prop)

    # 4. model channels (open-vocab named misses, region proposals classified by the VLM)
    if cfg.enable_model_channels and target_frames:
        store = get_object_store()
        be = backend if backend is not None else _build_backends()
        for f in target_frames:
            try:
                img = be.load_image(store, f.img_uri)
            except Exception as exc:  # noqa: BLE001  (a missing/unreadable frame must not abort the run)
                log.warning("recall.image_unreadable", frame_id=str(f.frame_id), error=str(exc))
                continue
            existing = existing_by_frame.get(f.frame_id, [])
            channel_props: list[RecallProposal] = [
                RecallProposal(bbox=bbox, channel="openvocab", conf=float(conf), class_name=cn)
                for bbox, cn, conf in be.openvocab(img)]
            if cfg.enable_region_channel:
                raw = [RecallProposal(bbox=rb, channel="region", conf=0.0) for rb in be.regions(img)]
                # mine before the VLM so we never spend a VLM call on a region an object already covers
                for rp in mine_unmatched(raw, existing, iou_match=cfg.iou_match):
                    cn, vconf = be.classify(img, rp.bbox)
                    if cn is None or vconf < cfg.region_min_vlm_conf:
                        continue
                    channel_props.append(RecallProposal(bbox=rp.bbox, channel="region", conf=float(vconf),
                                                        class_name=cn))
            for p in mine_unmatched(channel_props, existing, iou_match=cfg.iou_match):
                proposals_by_frame.setdefault(f.frame_id, []).append(p)

    # 5. fuse, score, persist
    counts = {"trackgap": 0, "openvocab": 0, "region": 0}
    persisted = 0
    for fid, props in proposals_by_frame.items():
        if fid not in frame_by_id:
            continue
        for fc in fuse_channels(props, fuse_iou=cfg.fuse_iou):
            cid = fc.class_id
            if cid is None and fc.class_name is not None and onto.has_name(fc.class_name):
                cid = onto.by_name(fc.class_name).id
            if cid is None:
                continue  # never persist a class-less object
            value = fn_value(fc, is_rare(cid, onto), fid in rare_frames)
            chans = sorted(fc.channels)
            provenance = {
                "proposals": [{"path": f"recall_{ch}", "verdict": "miss", "model_version": "recall-v1"}
                              for ch in chans],
                "agreement": False,
                "raw_conf": {"fn_value": round(value, 4)},
                "notes": [f"recall:{'+'.join(chans)}"],
            }
            obj = Object(frame_id=fid, track_id=fc.track_id, class_id=cid, bbox=[float(v) for v in fc.bbox],
                         conf=value, source="recall", state="review", interp_source=fc.interp_source,
                         provenance=provenance)
            db.add(obj)
            await db.flush()
            db.add(RecallCandidate(object_id=obj.object_id, frame_id=fid, channels=chans, fn_value=value,
                                   class_id=cid, status="pending"))
            persisted += 1
            for ch in chans:
                counts[ch] = counts.get(ch, 0) + 1
    await db.commit()
    log.info("recall.run", session_id=str(sid), persisted=persisted, frames=len(frames),
             trackgap=counts["trackgap"], openvocab=counts["openvocab"], region=counts["region"])
    return {"session_id": str(sid), "persisted": persisted, "by_channel": counts, "frames": len(frames)}


def _build_backends():
    from services.recall.backends import build_backends
    return build_backends()


def fit_channel_reliability(db):
    """STUB (closed-loop follow-on, not wired into fn_value yet). RecallCandidate.status carries the
    human verdict on each recovered miss. Given enough confirmed and rejected candidates per channel,
    this should replace _CHANNEL_PRIOR with measured per-channel confirmed-rates, closing the recall
    loop the same way the isotonic curve closes the precision loop. Left unwired by design."""
    raise NotImplementedError("fit_channel_reliability is a specified closed-loop stub, not yet wired")
