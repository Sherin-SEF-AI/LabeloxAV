"""Hard frames made harder: rain, dusk, night, fog and blur variants of real labelled frames, labels included.

The copy-paste generator adds positives a starved class lacks; this adds conditions a model is weak in. A frame
the model already struggles with (one a person had to correct) is re-rendered under a photometric perturbation
from services/verdyx/perturb.py. Photometric means the pixels change and nothing moves, so the source frame's
boxes, masks, polylines and keypoints are exactly right on the variant and are copied, not re-predicted.

What keeps it honest:

* Only `PHOTOMETRIC` perturbations. Occlusion would make a copied "visible" label a lie and is refused.
* A source frame must be real, and its label set complete enough to train on: at least one object in a trusted
  state (accepted, settled, auto_accept) and none still waiting for a person (`annotate`). Objects in `review`
  make the frame refuse too, unless the caller opts in with `include_review`, because a copied unreviewed
  label would reach a trainset as if it had been checked.
* Variants are `Frame.origin = 'perturbed'` on a session of the same origin, with `source_frame_id` set; the
  copied labels have state and source `synthetic` and say in provenance which object, state and perturbation
  they came from. Nothing that measures a model can see them (core/origin.py; tests/test_synthetic_variants.py
  proves it reader by reader), and the trainer takes them only with `BuildSpec.include_synthetic`.
* Copies never join a track: a track relabel or a 3D link reaching a synthetic copy would edit a label that is
  not a measurement. `track_id` is left empty.
* One run is one tool job; `revert_variants` deletes exactly the frames it made.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import cv2
import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.origin import PERTURBED, REAL, SYNTHETIC_SOURCE, SYNTHETIC_STATE
from core.storage import get_object_store

log = get_logger("synth.variants")

ProgressFn = Callable[[float, str], Awaitable[None]]
TRUSTED_STATES = ("accepted", "settled", "auto_accept")
MAX_FRAMES = 500
JPEG_QUALITY = 92


def check_perturbations(names: list[str]) -> list[str]:
    from services.verdyx.perturb import PHOTOMETRIC

    if not names:
        raise ValueError("choose at least one perturbation")
    bad = [n for n in names if n not in PHOTOMETRIC]
    if bad:
        raise ValueError(f"{bad} would move or hide what the labels describe; only photometric perturbations "
                         f"{list(PHOTOMETRIC)} carry labels over exactly")
    return list(dict.fromkeys(names))


async def eligibility(db: AsyncSession, frame_ids: list[uuid.UUID], *, include_review: bool = False) -> dict:
    """Which of these frames can be varied, and why each other one cannot. No writes."""
    from db.models import Frame, Object

    states = TRUSTED_STATES + (("review",) if include_review else ())
    rows = (await db.execute(select(Frame.frame_id, Frame.origin).where(Frame.frame_id.in_(frame_ids)))).all()
    found = {r[0]: r[1] for r in rows}
    counts: dict = dict((await db.execute(
        select(Object.frame_id, func.count()).where(Object.frame_id.in_(frame_ids), Object.state.in_(states))
        .group_by(Object.frame_id))).all())
    blocked: dict = dict((await db.execute(
        select(Object.frame_id, func.count()).where(
            Object.frame_id.in_(frame_ids),
            Object.state.in_(("annotate",) if include_review else ("annotate", "review")))
        .group_by(Object.frame_id))).all())
    ok, refused = [], {}
    for fid in frame_ids:
        if fid not in found:
            refused[str(fid)] = "frame not found"
        elif found[fid] != REAL:
            refused[str(fid)] = f"a {found[fid]} frame; only real frames are varied"
        elif blocked.get(fid):
            refused[str(fid)] = (f"{blocked[fid]} object(s) still waiting for a person" if include_review else
                                 f"{blocked[fid]} object(s) in annotate or review; their copies would train as checked")
        elif not counts.get(fid):
            refused[str(fid)] = "no trusted labels to carry over"
        else:
            ok.append(fid)
    return {"eligible": [str(f) for f in ok], "refused": refused,
            "objects": int(sum(counts.get(f, 0) for f in ok))}


async def hard_frames(db: AsyncSession, *, source: str = "errors", session_id: uuid.UUID | None = None,
                      limit: int = 100, include_review: bool = False) -> list[dict]:
    """Candidate source frames, hardest first, each with why it is hard and whether it can be varied.

    `errors`: frames with a label error a person confirmed in the error queue (the model or the auto-labeller
    got them wrong). `corrected`: frames with labels a person wrote or edited (source human, or lifecycle human_edited). `session`: a session's frames
    in order (for a chosen stretch of road). Only real frames, each at most once.
    """
    from db.models import ErrorCandidate, Frame, Object

    if source == "errors":
        stmt = (select(Frame.frame_id, Frame.session_id, Frame.cam_id, Frame.ts_ns,
                       func.count(ErrorCandidate.candidate_id).label("n"), func.max(ErrorCandidate.score))
                .join(Object, Object.frame_id == Frame.frame_id)
                .join(ErrorCandidate, ErrorCandidate.object_id == Object.object_id)
                .where(ErrorCandidate.status == "confirmed_error", Frame.origin == REAL)
                .group_by(Frame.frame_id).order_by(func.count(ErrorCandidate.candidate_id).desc(),
                                                   func.max(ErrorCandidate.score).desc()).limit(limit))
        why = "confirmed label error"
    elif source == "corrected":
        stmt = (select(Frame.frame_id, Frame.session_id, Frame.cam_id, Frame.ts_ns, func.count(Object.object_id),
                       func.max(Object.conf))
                .join(Object, Object.frame_id == Frame.frame_id)
                .where(Frame.origin == REAL, Object.state == "accepted",
                       (Object.lifecycle == "human_edited") | (Object.source == "human"))
                .group_by(Frame.frame_id).order_by(func.count(Object.object_id).desc()).limit(limit))
        why = "person-corrected labels"
    elif source == "session":
        if session_id is None:
            raise ValueError("source=session needs a session_id")
        stmt = (select(Frame.frame_id, Frame.session_id, Frame.cam_id, Frame.ts_ns, func.count(Object.object_id),
                       func.max(Object.conf))
                .join(Object, Object.frame_id == Frame.frame_id, isouter=True)
                .where(Frame.session_id == session_id, Frame.origin == REAL)
                .group_by(Frame.frame_id).order_by(Frame.ts_ns).limit(limit))
        why = "in the session"
    else:
        raise ValueError("source must be errors, corrected or session")
    rows = (await db.execute(stmt)).all()
    elig = await eligibility(db, [r[0] for r in rows], include_review=include_review) if rows else \
        {"eligible": [], "refused": {}}
    ok = set(elig["eligible"])
    return [{"frame_id": str(r[0]), "session_id": str(r[1]), "cam_id": r[2], "ts_ns": int(r[3]),
             "signal": int(r[4] or 0), "why": why, "eligible": str(r[0]) in ok,
             "refused": elig["refused"].get(str(r[0]))} for r in rows]


def _render(img_bytes: bytes, name: str, strength: float | None, seed: int) -> bytes | None:
    from services.verdyx.perturb import apply

    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    out = apply(name, img, strength=strength, seed=seed)
    ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return enc.tobytes() if ok else None


async def build_variants(frame_ids: list[uuid.UUID], perturbations: list[str], *, strengths: dict | None = None,
                         seed: int = 7, include_review: bool = False, job_id: uuid.UUID | None = None,
                         created_by: str | None = None, progress: ProgressFn | None = None) -> dict:
    """Make every (frame, perturbation) variant. Returns the new session and frame ids and what was refused."""
    from db.models import Frame, Object
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.verdyx.perturb import DEFAULT_STRENGTHS

    names = check_perturbations(perturbations)
    strengths = {k: float(v) for k, v in (strengths or {}).items() if k in names}
    if len(frame_ids) > MAX_FRAMES:
        raise ValueError(f"at most {MAX_FRAMES} source frames per run")
    maker = get_sessionmaker()
    store = get_object_store()
    store.ensure_bucket()
    async with maker() as db:
        elig = await eligibility(db, frame_ids, include_review=include_review)
        ok = [uuid.UUID(f) for f in elig["eligible"]]
        if not ok:
            return {"frames": 0, "frame_ids": [], "session_id": None, "refused": elig["refused"],
                    "reason": "no source frame can be varied"}

    # One perturbed session per source project: a members-only project's frames must not land in a session
    # that another project's members can list. A frame with no session at all goes to a project-less one.
    sessions: dict[uuid.UUID | None, uuid.UUID] = {}
    ts_by_session: dict[uuid.UUID, list[int]] = {}

    async def session_for(db: AsyncSession, src_session_id: uuid.UUID | None) -> uuid.UUID:
        from services.autolabel.ontology import get_ontology

        tmpl = await db.get(DbSession, src_session_id) if src_session_id else None
        key = tmpl.project_id if tmpl is not None else None
        if key in sessions:
            return sessions[key]
        new_sid = uuid.uuid4()
        db.add(DbSession(session_id=new_sid, pack_id=tmpl.pack_id if tmpl is not None else "av", project_id=key,
                         vehicle_id="perturbed", start_ts_ns=tmpl.start_ts_ns if tmpl is not None else 0,
                         end_ts_ns=tmpl.end_ts_ns if tmpl is not None else 0,
                         city=tmpl.city if tmpl is not None else None, route=f"variants:{','.join(names)}",
                         sensors={"synthetic": {"generator": "variants", "perturbations": names,
                                                "job_id": str(job_id) if job_id else None}},
                         ontology_version=tmpl.ontology_version if tmpl is not None else get_ontology().version,
                         origin=PERTURBED))
        await db.flush()
        sessions[key] = new_sid
        ts_by_session[new_sid] = []
        return new_sid

    states = TRUSTED_STATES + (("review",) if include_review else ())
    made: list[str] = []
    per: dict[str, int] = {n: 0 for n in names}
    copied = 0
    failed: dict[str, str] = {}
    total = len(ok) * len(names)
    step = 0
    for fid in ok:
        async with maker() as db:
            fr = await db.get(Frame, fid)
            assert fr is not None                       # eligibility found it a moment ago
            sid = await session_for(db, fr.session_id)
            await db.commit()
            fr = await db.get(Frame, fid)
            assert fr is not None
            objs = (await db.execute(select(Object).where(Object.frame_id == fid, Object.state.in_(states)))
                    ).scalars().all()
            src: dict[str, Any] = {"frame": {"session_id": fr.session_id, "cam_id": fr.cam_id, "ts_ns": fr.ts_ns,
                             "img_uri": fr.img_uri, "width": fr.width, "height": fr.height,
                             "ego_speed": fr.ego_speed},
                   "objects": [{c: getattr(o, c) for c in (
                       "object_id", "class_id", "bbox", "bbox_amodal", "mask_uri", "mask_encoding", "attrs", "conf",
                       "rot_deg", "keypoints", "polyline", "cuboid_3d", "sign_type", "sign_category", "ocr_text",
                       "ocr_lang", "ocr_conf", "state", "source")} for o in objs]}
        try:
            raw = await asyncio.to_thread(store.get_bytes, src["frame"]["img_uri"])
        except Exception as exc:  # noqa: BLE001 - one unreadable blob skips one frame
            failed[str(fid)] = f"image unreadable: {exc}"[:200]
            step += len(names)
            continue
        rows: list = []
        for name in names:
            step += 1
            data = await asyncio.to_thread(_render, raw, name, strengths.get(name), seed)
            if data is None:
                failed[str(fid)] = "image could not be decoded"
                continue
            f = src["frame"]
            new_id = uuid.uuid4()
            uri = await asyncio.to_thread(store.put_bytes, f"frames/{sid}/{f['cam_id']}/{f['ts_ns']}_{name}.jpg",
                                          data, "image/jpeg")
            frame = Frame(frame_id=new_id, session_id=sid, ts_ns=f["ts_ns"], cam_id=f["cam_id"], img_uri=uri,
                          width=f["width"], height=f["height"], ego_speed=f["ego_speed"], quality=0.0,
                          origin=PERTURBED, source_frame_id=fid, selected=True,
                          tags=["perturbed", f"perturb:{name}"])
            strength = strengths.get(name, DEFAULT_STRENGTHS.get(name))
            objs_new = []
            for o in src["objects"]:
                o2 = {k: v for k, v in o.items() if k not in ("object_id", "state", "source")}
                objs_new.append(Object(
                    frame_id=new_id, state=SYNTHETIC_STATE, source=SYNTHETIC_SOURCE, track_id=None,
                    provenance={"synthetic": {"generator": "variants", "perturbation": name, "strength": strength,
                                              "seed": seed, "source_frame_id": str(fid),
                                              "source_object_id": str(o["object_id"]), "source_state": o["state"],
                                              "source_source": o["source"],
                                              "job_id": str(job_id) if job_id else None}},
                    **{**o2, "attrs": dict(o2.get("attrs") or {})}))
            rows.append((frame, objs_new))
            per[name] += 1
            ts_by_session[sid].append(int(f["ts_ns"]))
            if progress is not None and step % 10 == 0:
                await progress(0.05 + 0.9 * step / total, f"variant {step} of {total}")
        if rows:
            async with maker() as db:
                for frame, _o in rows:
                    db.add(frame)
                await db.flush()
                for _f, objs_new in rows:
                    db.add_all(objs_new)
                    copied += len(objs_new)
                await db.commit()
            made.extend(str(fr_.frame_id) for fr_, _ in rows)
    async with maker() as db:
        for sid_, ts_seen in ts_by_session.items():
            s = await db.get(DbSession, sid_)
            if s is None:
                continue
            if not ts_seen:
                await db.delete(s)
            else:
                s.start_ts_ns, s.end_ts_ns = min(ts_seen), max(ts_seen)
        await db.commit()
    kept = [str(k) for k, v in ts_by_session.items() if v]
    log.info("variants.built", frames=len(made), objects=copied, sessions=len(kept), job_id=str(job_id))
    return {"frames": len(made), "objects": copied, "frame_ids": made, "session_id": kept[0] if kept else None,
            "session_ids": kept, "per_perturbation": per, "refused": elig["refused"], "failed": failed,
            "sources": len(ok), "created_by": created_by}


async def revert_variants(db: AsyncSession, frame_ids: list[str], session_ids: list[str] | str | None) -> dict:
    """Delete the variant frames a run made (their copied objects cascade) and their images; drop the run's
    sessions once they are empty. A frame that is somehow real is never touched."""
    from db.models import Frame, Object
    from db.models import Session as DbSession

    store = get_object_store()
    removed = skipped = 0
    gone: list[uuid.UUID] = []
    for f in frame_ids:
        frame = await db.get(Frame, uuid.UUID(f))
        if frame is None or frame.origin != PERTURBED:
            skipped += 1
            continue
        await asyncio.to_thread(store.remove, frame.img_uri)
        await db.delete(frame)
        gone.append(frame.frame_id)
        removed += 1
    await db.flush()
    if isinstance(session_ids, str):
        session_ids = [session_ids]
    for sid_s in session_ids or []:
        sid = uuid.UUID(sid_s)
        left = (await db.execute(select(func.count()).select_from(Frame).where(Frame.session_id == sid))).scalar_one()
        if left == 0:
            s = await db.get(DbSession, sid)
            if s is not None and s.origin == PERTURBED:
                await db.delete(s)
    await db.commit()
    still = 0
    if gone:
        still = int((await db.execute(select(func.count()).select_from(Object).where(
            Object.frame_id.in_(gone)))).scalar_one())
    return {"removed": removed, "skipped": skipped, "objects_left": still}


async def variant_summary(db: AsyncSession) -> dict:
    """How many perturbed frames and copied labels exist, by perturbation."""
    from db.models import Frame, Object

    rows = (await db.execute(select(Frame.tags, func.count()).where(Frame.origin == PERTURBED)
                             .group_by(Frame.tags))).all()
    by: dict[str, int] = {}
    for tags, n in rows:
        name = next((t.split(":", 1)[1] for t in (tags or []) if str(t).startswith("perturb:")), "unknown")
        by[name] = by.get(name, 0) + int(n)
    labels = int((await db.execute(select(func.count()).select_from(Object).join(
        Frame, Frame.frame_id == Object.frame_id).where(Frame.origin == PERTURBED))).scalar_one())
    return {"frames": int(sum(by.values())), "by_perturbation": by, "labels": labels}
