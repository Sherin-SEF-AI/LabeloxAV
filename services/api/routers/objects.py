"""Object detail, frame image proxy, and SAM click-to-segment."""

from __future__ import annotations

import json
import math
import uuid
from uuid import UUID

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.storage import get_object_store
from core.timebase import now_ns
from db.models import Frame, Object, ObjectRelationship, Review
from services.api.deps import (
    CreateObjectIn,
    MaskIn,
    ObjectDetail,
    RelateIn,
    SegmentIn,
    current_user,
    db_session,
    require_role,
)
from services.autolabel.ontology import get_ontology
from services.govern.audit import record as audit_record

router = APIRouter()

# Directed relationship kinds the editor offers (the India case is rider_of on a two-wheeler).
_RELATION_KINDS = {"rider_of", "towed_by", "part_of", "member_of", "occludes"}


@router.post("/objects/{object_id}/relate", dependencies=[Depends(require_role("reviewer"))])
async def relate_object(object_id: str, payload: RelateIn, db: AsyncSession = Depends(db_session)):
    """Create a directed relationship from this object to another on the same frame."""
    if payload.kind not in _RELATION_KINDS:
        raise HTTPException(400, f"unknown relation kind '{payload.kind}'")
    if object_id == payload.to_object_id:
        raise HTTPException(400, "cannot relate an object to itself")
    frm = await db.get(Object, UUID(object_id))
    to = await db.get(Object, UUID(payload.to_object_id))
    if frm is None or to is None:
        raise HTTPException(404, "object not found")
    rel = ObjectRelationship(from_object_id=frm.object_id, to_object_id=to.object_id,
                             frame_id=frm.frame_id, kind=payload.kind)
    db.add(rel)
    await db.commit()
    return {"relationship_id": str(rel.relationship_id), "from_object_id": object_id,
            "to_object_id": payload.to_object_id, "kind": payload.kind}


@router.delete("/relationships/{relationship_id}", dependencies=[Depends(require_role("reviewer"))])
async def delete_relationship(relationship_id: str, db: AsyncSession = Depends(db_session)):
    rel = await db.get(ObjectRelationship, UUID(relationship_id))
    if rel is not None:
        await db.delete(rel)
        await db.commit()
    return {"deleted": relationship_id}


@router.get("/frames/{frame_id}/relationships")
async def frame_relationships(frame_id: str, db: AsyncSession = Depends(db_session)):
    rows = (await db.execute(select(ObjectRelationship)
            .where(ObjectRelationship.frame_id == UUID(frame_id)))).scalars().all()
    return [{"relationship_id": str(r.relationship_id), "from_object_id": str(r.from_object_id),
             "to_object_id": str(r.to_object_id), "kind": r.kind} for r in rows]


@router.get("/frames/{frame_id}/cuboids")
async def frame_cuboids(frame_id: str, db: AsyncSession = Depends(db_session)):
    """Project every cuboid_3d on the frame onto the camera image, so the 3D box is visible (and editable)
    in the 2D editor. Uses the configured rig + nominal intrinsics, so it works without LiDAR calibration."""
    from services.lidar.boxes import project_cuboid

    frame = await db.get(Frame, UUID(frame_id))
    if frame is None:
        raise HTTPException(404, "frame not found")
    rows = (await db.execute(select(Object).where(
        Object.frame_id == frame.frame_id, Object.cuboid_3d.isnot(None)))).scalars().all()
    out = []
    for o in rows:
        c = o.cuboid_3d or {}
        center, size, yaw = c.get("center"), c.get("size"), float(c.get("yaw", 0.0))
        if not center or not size:
            continue
        dims = [size[1], size[0], size[2]]  # cuboid_3d size is [w,l,h]; project_cuboid wants [length,width,height]
        proj = project_cuboid(center, dims, yaw, frame.cam_id, frame.width, frame.height)
        out.append({"object_id": str(o.object_id), "corners_uv": proj["corners_uv"], "edges": proj["edges"],
                    "any_in_image": proj["any_in_image"]})
    return out


@router.get("/frames/{frame_id}/lift_ground")
async def lift_ground(frame_id: str, u: float, v: float, db: AsyncSession = Depends(db_session)):
    """The ego ground point (z=0) a pixel sees, for placing a cuboid on the road from an image click."""
    from services.lidar.project import camera_ray_to_ego

    frame = await db.get(Frame, UUID(frame_id))
    if frame is None:
        raise HTTPException(404, "frame not found")
    ray = camera_ray_to_ego(u, v, frame.cam_id, frame.width, frame.height)
    o, dvec = ray["origin"], ray["direction"]
    # A pixel above the horizon (or a ray parallel to the road) simply has no ground point. That is a
    # normal answer for a valid query, not a client error, so return ego=null with a reason instead of a
    # 400 the browser logs on every hover/click near the skyline.
    if abs(float(dvec[2])) < 1e-6:
        return {"ego": None, "reason": "ray is parallel to the ground"}
    t = -float(o[2]) / float(dvec[2])
    if t <= 0:
        return {"ego": None, "reason": "pixel is above the horizon (no ground ahead)"}
    return {"ego": [round(float(o[0] + t * dvec[0]), 3), round(float(o[1] + t * dvec[1]), 3), 0.0]}


def _mask_polygons(mask_uri: str | None) -> list[list[float]]:
    if not mask_uri:
        return []
    store = get_object_store()
    try:
        return json.loads(store.get_bytes(mask_uri)).get("polygons", [])
    except Exception:
        return []


def _mask_key(session_id, frame_id, object_id) -> str:
    return f"masks/{session_id}/{frame_id}/{object_id}.json"


def _write_mask(store, session_id, frame_id, object_id, polygons, width, height) -> str:
    # Same polygon-JSON shape services/autolabel/persist.py writes, so the read path is identical.
    payload = {"encoding": "polygon", "polygons": polygons, "height": height, "width": width}
    return store.put_bytes(_mask_key(session_id, frame_id, object_id),
                           json.dumps(payload).encode(), "application/json")


def _detail(obj: Object, frame: Frame, onto) -> ObjectDetail:
    return ObjectDetail(
        object_id=str(obj.object_id),
        frame_id=str(obj.frame_id),
        session_id=str(frame.session_id),
        track_id=str(obj.track_id) if obj.track_id else None,
        ts_ns=frame.ts_ns,
        cam_id=frame.cam_id,
        image_url=f"/api/frames/{frame.frame_id}/image",
        width=frame.width,
        height=frame.height,
        class_id=obj.class_id,
        class_name=onto.by_id(obj.class_id).name,
        bbox=list(obj.bbox),
        mask_polygons=_mask_polygons(obj.mask_uri),
        attrs=obj.attrs or {},
        conf=obj.conf,
        state=obj.state,
        source=obj.source,
        provenance=obj.provenance or {},
        version=obj.version,
        rot_deg=obj.rot_deg or 0.0,
        keypoints=obj.keypoints,
        polyline=obj.polyline,
        cuboid_3d=obj.cuboid_3d,
        # Sign typing and road text have lived on the object since 0016 and were served by nothing, so a
        # wrong sign_type was invisible to the one person able to correct it. Read-only here; correcting it
        # is a review action, not a field edit.
        sign_type=obj.sign_type,
        sign_category=obj.sign_category,
        ocr_text=obj.ocr_text,
        ocr_lang=obj.ocr_lang,
        ocr_conf=obj.ocr_conf,
    )


@router.get("/objects/{object_id}", response_model=ObjectDetail)
async def get_object(object_id: str, db: AsyncSession = Depends(db_session)):
    obj = await db.get(Object, UUID(object_id))
    if obj is None:
        raise HTTPException(404, "object not found")
    frame = await db.get(Frame, obj.frame_id)
    return _detail(obj, frame, get_ontology())


@router.get("/objects/{object_id}/explain")
async def explain_object_ep(object_id: str, db: AsyncSession = Depends(db_session)):
    """M-F.0: the plain-language decision story for an object, assembled from its real provenance."""
    from services.autolabel.explain import explain_object

    res = await explain_object(db, UUID(object_id))
    if "error" in res:
        raise HTTPException(404, res["error"])
    return res


@router.get("/objects/{object_id}/quality")
async def object_quality_ep(object_id: str, db: AsyncSession = Depends(db_session)):
    """M-F.1: the object's composite quality score with its factor breakdown."""
    from services.analytics.quality_score import score_object

    res = await score_object(db, UUID(object_id), persist=True)
    if res is None:
        raise HTTPException(404, "object not found")
    return res


@router.post("/objects/quality/backfill")
async def quality_backfill_ep(session_id: str | None = None):
    """M-F.1: compute and store quality scores across a session or the whole corpus."""
    from services.analytics.quality_score import backfill

    return await backfill(UUID(session_id) if session_id else None)


# M-F.5 scene-graph relations


@router.get("/scene-graph/vocab")
async def scene_graph_vocab():
    from services.intelligence.scene_graph import vocab

    return vocab()


@router.post("/frames/{frame_id}/relations/propose")
async def relations_propose(frame_id: str):
    """Propose geometric scene-graph relations (occluded_by, following, parked_near, crossing_in_front_of)."""
    from services.intelligence.scene_graph import propose_relations

    res = await propose_relations(UUID(frame_id))
    if "error" in res:
        raise HTTPException(404, res["error"])
    return res


@router.get("/frames/{frame_id}/relations")
async def relations_list(frame_id: str):
    from services.intelligence.scene_graph import frame_relations

    return await frame_relations(UUID(frame_id))


@router.post("/relations/{relationship_id}/status")
async def relation_status(relationship_id: str, status: str):
    from services.intelligence.scene_graph import set_relation_status

    res = await set_relation_status(UUID(relationship_id), status)
    if "error" in res:
        raise HTTPException(400, res["error"])
    return res


# M-F.5 VLM dataset generation


@router.post("/frames/{frame_id}/vlm-target/generate")
async def vlm_target_generate(frame_id: str):
    """Generate a grounded VLM training target for a labeled frame (awaits human review)."""
    from services.intelligence.vlm_dataset import generate_target

    res = await generate_target(UUID(frame_id))
    if "error" in res:
        raise HTTPException(400, res["error"])
    return res


@router.get("/frames/{frame_id}/vlm-targets")
async def vlm_targets_list(frame_id: str):
    from services.intelligence.vlm_dataset import list_targets

    return await list_targets(UUID(frame_id))


@router.post("/vlm-targets/{target_id}/status")
async def vlm_target_status(target_id: str, status: str):
    """Human review gate: approve or reject a generated target (only approved export)."""
    from services.intelligence.vlm_dataset import set_target_status

    res = await set_target_status(UUID(target_id), status)
    if "error" in res:
        raise HTTPException(400, res["error"])
    return res


@router.get("/vlm-dataset/export")
async def vlm_dataset_export(session_id: str | None = None):
    """Export the approved VLM targets in the multimodal per-frame format."""
    from services.intelligence.vlm_dataset import export_dataset

    return await export_dataset(UUID(session_id) if session_id else None)


@router.get("/frames/{frame_id}/objects")
async def frame_objects(frame_id: str, job_id: str | None = None,
                        db: AsyncSession = Depends(db_session)):
    """Every object on a frame, or only this job's when the job is a blind replica.

    A replica job exists to be compared against another annotator's independent answer. If the editor
    handed it the machine pre-labels, both annotators would be correcting the same proposals and the
    agreement between them would measure how well two people agree with a third party neither of them can
    see. 82.6% of frames here are pre-labelled, so this is the normal case.

    The filter is server-side deliberately: hiding the pre-labels in the browser still ships them to the
    browser, and a hidden label is one keystroke away from being an unhidden one.
    """
    from sqlalchemy import select

    from db.models import LabelJob

    onto = get_ontology()
    q = select(Object).where(Object.frame_id == UUID(frame_id))
    if job_id:
        job = await db.get(LabelJob, UUID(job_id))
        if job is None:
            raise HTTPException(404, "job not found")
        if job.replica_group is not None:
            q = q.where(Object.job_id == job.job_id)
    rows = (await db.execute(q)).scalars().all()
    return [
        {
            "object_id": str(o.object_id),
            "track_id": str(o.track_id) if o.track_id else None,
            "class_id": o.class_id,
            "class_name": onto.by_id(o.class_id).name,
            "bbox": list(o.bbox),
            "conf": o.conf,
            "quality_score": o.quality_score,
            "state": o.state,
            "mask_polygons": _mask_polygons(o.mask_uri),
            "version": o.version,
            "rot_deg": o.rot_deg or 0.0,
            "keypoints": o.keypoints,
            "polyline": o.polyline,
            "cuboid_3d": o.cuboid_3d,
        }
        for o in rows
    ]


# States that mean a person has dealt with the object, which is what "is this frame finished" asks. A
# rejection is as settled as an acceptance. auto_accept is deliberately not here: it is the machine's
# opinion, and counting it would draw a strip of finished frames nobody has looked at.
_SETTLED_STATES = ("accepted", "rejected")


@router.get("/frames/{frame_id}/filmstrip")
async def frame_filmstrip(frame_id: str, span: int = 12, db: AsyncSession = Depends(db_session)):
    """The frames on either side of this one, in capture order, for the editor's filmstrip.

    The editor could only step one frame at a time through prev/next, so a reviewer had no way to see what
    was coming or to jump several frames back to where an object first appeared. That made every temporal
    judgement (is this the same vehicle, when did the occlusion start) a sequence of blind single steps.

    Ordered by ts_ns and restricted to the same camera: a multi-camera rig interleaves frames from every
    camera at nearly the same timestamp, and mixing them would make the strip jump viewpoint every tile.
    """
    frame = await db.get(Frame, UUID(frame_id))
    if frame is None:
        raise HTTPException(404, "frame not found")
    span = max(1, min(span, 40))

    async def _side(newer: bool) -> list:
        cond = Frame.ts_ns > frame.ts_ns if newer else Frame.ts_ns < frame.ts_ns
        order = Frame.ts_ns.asc() if newer else Frame.ts_ns.desc()
        rows = (await db.execute(
            select(Frame.frame_id, Frame.ts_ns)
            .where(Frame.session_id == frame.session_id, Frame.cam_id == frame.cam_id, cond)
            .order_by(order).limit(span))).all()
        return list(rows)

    before = list(reversed(await _side(False)))
    after = await _side(True)
    tiles = [*before, (frame.frame_id, frame.ts_ns), *after]

    # Object counts in one query rather than one per tile: a 25-tile strip would otherwise issue 25 round
    # trips every time the reviewer moved a frame.
    ids = [fid for fid, _ in tiles]
    # Confirmed counts alongside the totals, in the same pass. The strip showed how many objects a
    # neighbouring frame holds but not whether anyone had finished with it, so a reviewer working a session
    # could not see which frames they had already done or where they had stopped, and reopened frames that
    # were finished. The counts have to be per state rather than a boolean because a partly confirmed frame
    # is the interesting case: it is the one somebody left halfway.
    rows = (await db.execute(
        select(Object.frame_id, Object.state, func.count()).where(Object.frame_id.in_(ids))
        .group_by(Object.frame_id, Object.state))).all()
    counts: dict[UUID, int] = {}
    confirmed: dict[UUID, int] = {}
    for fid, state, n in rows:
        counts[fid] = counts.get(fid, 0) + int(n)
        if state in _SETTLED_STATES:
            confirmed[fid] = confirmed.get(fid, 0) + int(n)

    return {"frame_id": frame_id, "cam_id": frame.cam_id, "frames": [
        {"frame_id": str(fid), "ts_ns": int(ts), "n_objects": int(counts.get(fid, 0)),
         "n_confirmed": int(confirmed.get(fid, 0)),
         "image_url": f"/api/frames/{fid}/image", "current": str(fid) == frame_id}
        for fid, ts in tiles]}


@router.get("/frames/{frame_id}")
async def get_frame(frame_id: str, db: AsyncSession = Depends(db_session)):
    """Frame meta for the editor: dimensions, image url, object count, and prev/next frame in the
    session (by ts_ns) for keyboard frame navigation."""
    frame = await db.get(Frame, UUID(frame_id))
    if frame is None:
        raise HTTPException(404, "frame not found")
    prev = (await db.execute(
        select(Frame.frame_id).where(Frame.session_id == frame.session_id, Frame.ts_ns < frame.ts_ns)
        .order_by(Frame.ts_ns.desc()).limit(1))).scalar_one_or_none()
    nxt = (await db.execute(
        select(Frame.frame_id).where(Frame.session_id == frame.session_id, Frame.ts_ns > frame.ts_ns)
        .order_by(Frame.ts_ns.asc()).limit(1))).scalar_one_or_none()
    n = (await db.execute(select(func.count()).select_from(Object).where(Object.frame_id == frame.frame_id))).scalar_one()
    # The dominant annotation source on this frame, so the editor can say plainly whether these labels are
    # imported from a public dataset (Mapillary / IDD / BDD) or produced in-app.
    src_rows = (await db.execute(
        select(Object.source, func.count()).where(Object.frame_id == frame.frame_id)
        .group_by(Object.source).order_by(func.count().desc()))).all()
    annotation_source = src_rows[0][0] if src_rows else None
    import_format = None
    if annotation_source == "imported":
        prov = (await db.execute(select(Object.provenance).where(
            Object.frame_id == frame.frame_id, Object.source == "imported").limit(1))).scalar()
        import_format = (prov or {}).get("import_format")
    # whether this session has an MCAP recording, so the editor only offers the Session Inspector when there is
    # a timeline to inspect (image/video/imagery sessions have no MCAP and the Inspector would 409).
    from db.models import Session as DbSession

    mcap_uri = (await db.execute(select(DbSession.mcap_uri).where(DbSession.session_id == frame.session_id))).scalar_one_or_none()
    return {
        "frame_id": str(frame.frame_id), "session_id": str(frame.session_id),
        "width": frame.width, "height": frame.height, "ts_ns": frame.ts_ns, "cam_id": frame.cam_id,
        "image_url": f"/api/frames/{frame.frame_id}/image", "n_objects": int(n),
        "has_mcap": mcap_uri is not None,
        "annotation_source": annotation_source, "import_format": import_format,
        "prev_frame_id": str(prev) if prev else None, "next_frame_id": str(nxt) if nxt else None,
        "is_lidar": bool(frame.lidar), "lidar_points": (frame.lidar or {}).get("n_points"),
        "lidar_res": ((frame.lidar or {}).get("bev") or {}).get("res"),  # metres per pixel, for the ruler
    }


@router.post("/frames/{frame_id}/objects", response_model=ObjectDetail)
async def create_object(frame_id: str, payload: CreateObjectIn, db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    """Create a human-drawn object on a frame (source=human, state=accepted). Optional mask."""
    frame = await db.get(Frame, UUID(frame_id))
    if frame is None:
        raise HTTPException(404, "frame not found")
    onto = get_ontology()
    if not onto.has_name(payload.class_name):
        raise HTTPException(400, f"unknown class '{payload.class_name}'")
    if len(payload.bbox) != 4:
        raise HTTPException(400, "bbox must be [x1,y1,x2,y2]")
    if payload.attrs:
        errors = onto.validate_attrs(payload.attrs, onto.by_name(payload.class_name).id)
        if errors:
            raise HTTPException(400, {"attr_errors": errors})

    # Idempotency: if this frame already carries an object for the client's idem_key, return it rather
    # than creating a duplicate (a retried or raced autosave from the editor).
    if payload.idem_key:
        existing = (await db.execute(
            select(Object).where(Object.frame_id == frame.frame_id,
                                  Object.provenance["idem_key"].astext == payload.idem_key))).scalars().first()
        if existing is not None:
            return _detail(existing, frame, onto)

    # A job in the payload is a claim that this label belongs to that job's work. It is checked, because an
    # annotator whose next/prev walked out of their job's frame set would otherwise stamp every box after
    # that with a job that does not contain the frame, and the agreement pass would compare work nobody was
    # asked to do.
    job = None
    if payload.job_id:
        from db.models import LabelJob

        job = await db.get(LabelJob, UUID(payload.job_id))
        if job is None:
            raise HTTPException(404, "job not found")
        if str(frame.frame_id) not in {str(f) for f in (job.frame_ids or [])}:
            raise HTTPException(400, f"frame {frame.frame_id} is not part of job {payload.job_id}")

    oid = uuid.uuid4()
    mask_uri = mask_encoding = None
    if payload.mask_polygons:
        mask_uri = _write_mask(get_object_store(), frame.session_id, frame.frame_id, oid,
                               payload.mask_polygons, frame.width, frame.height)
        mask_encoding = "polygon"
    obj = Object(
        object_id=oid, frame_id=frame.frame_id, class_id=onto.by_name(payload.class_name).id,
        bbox=payload.bbox, mask_uri=mask_uri, mask_encoding=mask_encoding, attrs=payload.attrs or {},
        conf=1.0, source="human", state=payload.state, rot_deg=payload.rot_deg, keypoints=payload.keypoints,
        polyline=payload.polyline, cuboid_3d=payload.cuboid_3d,
        # Who drew it and under which job. Both null for a label drawn outside a job, which is every
        # existing flow and stays exactly as it was.
        job_id=job.job_id if job is not None else None,
        annotator_id=user.user_id if user else None,
        provenance={"created_by": "human-annotation", "idem_key": payload.idem_key},
    )
    db.add(obj)
    db.add(Review(object_id=oid, reviewer=user.name if user else "anon", user_id=user.user_id if user else None,
                  action="create", before=None,
                  after={"class_id": obj.class_id, "bbox": list(obj.bbox), "attrs": obj.attrs, "state": obj.state},
                  time_spent_ms=0, ts_ns=now_ns()))
    await db.commit()
    return _detail(obj, frame, onto)


@router.put("/objects/{object_id}/mask", dependencies=[Depends(require_role("annotator"))])
async def update_mask(object_id: str, payload: MaskIn, db: AsyncSession = Depends(db_session),
                      user=Depends(current_user)):
    """Replace an object's mask, and record who replaced it.

    Every other geometry change writes a Review row; this one rewrote the mask in place with no reviewer,
    no before-state and no role check, so a segment could be redrawn by anyone the API let in and the change
    left no trace. The mask blob itself is content-addressed and the old key is not deleted, so the
    recorded uri is enough to go back and look at what was there.
    """
    obj = await db.get(Object, UUID(object_id))
    if obj is None:
        raise HTTPException(404, "object not found")
    frame = await db.get(Frame, obj.frame_id)
    before = {"mask_uri": obj.mask_uri, "mask_encoding": obj.mask_encoding,
              "source": obj.source, "conf": obj.conf, "state": obj.state, "version": obj.version}
    obj.mask_uri = _write_mask(get_object_store(), frame.session_id, frame.frame_id, obj.object_id,
                               payload.polygons, payload.width or frame.width, payload.height or frame.height)
    obj.mask_encoding = "polygon"
    obj.version = (obj.version or 0) + 1
    db.add(Review(object_id=obj.object_id, reviewer=user.name if user else "anon",
                  user_id=user.user_id if user else None, action="edit_mask", before=before,
                  after={"mask_uri": obj.mask_uri, "mask_encoding": "polygon",
                         "n_polygons": len(payload.polygons), "version": obj.version},
                  time_spent_ms=0, ts_ns=now_ns()))
    await db.commit()
    return {"object_id": str(obj.object_id), "mask_polygons": payload.polygons, "version": obj.version}


@router.delete("/objects/{object_id}", dependencies=[Depends(require_role("annotator"))])
async def delete_object(object_id: str, db: AsyncSession = Depends(db_session),
                        user=Depends(current_user)):
    """Delete an object, leaving a record that outlives it.

    Deleting an object cascades its Review rows, so the object and its entire history left the database
    together and nothing said anyone had done it: the audit trail was destroyed by the very action most
    worth auditing. AuditDecision carries no foreign key to the object, which is exactly why the record
    goes there and survives. The full state is written into the rationale, because after the delete there
    is nothing left to join against.
    """
    obj = await db.get(Object, UUID(object_id))
    if obj is None:
        raise HTTPException(404, "object not found")
    await audit_record(
        db, actor=(user.name if user else "anon"), decision="delete_object", subject=str(obj.object_id),
        rationale={"frame_id": str(obj.frame_id), "track_id": str(obj.track_id) if obj.track_id else None,
                   "class_id": obj.class_id, "bbox": list(obj.bbox or []), "conf": obj.conf,
                   "source": obj.source, "state": obj.state, "attrs": obj.attrs,
                   "mask_uri": obj.mask_uri, "provenance": obj.provenance, "version": obj.version,
                   "user_id": str(user.user_id) if user else None},
        commit=False)
    await db.delete(obj)  # Review rows cascade; the mask blob is left (harmless, content-addressed path)
    await db.commit()
    return {"deleted": object_id}


@router.post("/objects/{object_id}/propagate")
async def propagate_object(object_id: str, frames: int = 12, db: AsyncSession = Depends(db_session)):
    """Label once, carry forward: optical-flow propagate this object's box across the next `frames`
    frames as an annotate-state track the human confirms. Yields the GPU to training is moot (CPU)."""
    from services.intelligence.propagate import propagate_forward

    return await propagate_forward(UUID(object_id), frames)


@router.post("/objects/{object_id}/sam_propagate", dependencies=[Depends(require_role("annotator"))])
async def sam_propagate(object_id: str, frames: int = 12, direction: str = "both", refine: bool = True,
                        db: AsyncSession = Depends(db_session)):
    """Label once, carry both ways: propagate this keyframe object's box forward AND backward with optical
    flow, refining each into a mask with a SAM box prompt (interp_source=sam_propagated). Routed to review."""
    from services.temporal.sam_propagate import sam_propagate_object

    return await sam_propagate_object(UUID(object_id), frames, direction, refine)


@router.get("/frames/{frame_id}/image")
async def frame_image(frame_id: str, db: AsyncSession = Depends(db_session),
                      user=Depends(current_user)):
    """The frame image, and a record that somebody looked at it.

    A frame is where the personal data in this corpus actually lives: faces and registration marks are in
    the pixels. `pii_audit` records what the redactor found in a frame; until this, nothing recorded who
    then viewed it, so the system could describe its personal data and not say who had seen it. The record
    is written only when the frame is known to contain some, so ordinary empty road scenes do not bury the
    accesses that matter.
    """
    frame = await db.get(Frame, UUID(frame_id))
    if frame is None:
        raise HTTPException(404, "frame not found")
    try:
        data = get_object_store().get_bytes(frame.img_uri)
    except Exception as exc:  # noqa: BLE001  (missing/unreadable blob -> 404, never a 500 that breaks the editor)
        raise HTTPException(404, "frame image unavailable") from exc

    await _log_frame_view(db, frame, user)
    return Response(content=data, media_type="image/jpeg")


async def _log_frame_view(db: AsyncSession, frame, user) -> None:
    """Record the view when the frame carries personal data. Never raises: evidence about a request must
    not be able to fail the request."""
    try:
        from db.models import PiiAudit
        from services.govern.pii_access import record_access

        audit = await db.get(PiiAudit, frame.frame_id)
        if audit is None or (int(audit.n_faces or 0) + int(audit.n_plates or 0)) == 0:
            return
        kinds = (["face"] * bool(audit.n_faces)) + (["plate"] * bool(audit.n_plates))
        await record_access(
            db, subject_type="frame", subject_id=str(frame.frame_id), action="view",
            user=user, session_id=frame.session_id, pii_kinds=kinds,
            # The served image is the redacted one: the privacy plane blurs on ingest, so what a reviewer
            # sees is already masked. Recording it as unredacted would inflate every count that matters.
            redacted=True, route=f"/api/frames/{frame.frame_id}/image")
    except Exception:  # noqa: BLE001
        pass


class CropSheetIn(BaseModel):
    object_ids: list[str]
    cell: int = 128                  # each crop is letterboxed into a square cell of this many pixels
    pad: float = 0.15
    cols: int = 0                    # 0 = choose a near-square sheet


# A grid of 200 crops is 200 of these, and each one fetches a whole frame from the object store and decodes
# it. Objects cluster on frames (16 to a frame on this corpus), so the same JPEG is fetched and decoded
# dozens of times over. That is what makes a contact sheet unusable rather than slow.
MAX_SHEET_CROPS = 400


@router.post("/objects/crops")
async def object_crop_sheet(payload: CropSheetIn, db: AsyncSession = Depends(db_session)):
    """One sprite sheet holding many crops, plus the map of where each landed.

    Grouped by frame so every frame is fetched and decoded exactly once however many of its objects are
    requested, and the decode runs in a worker thread because it is CPU-bound work inside an async handler.

    Returned as a sheet rather than as N images because a grid wants one request, not N: at 200 tiles the
    per-request overhead alone dominates, and browsers cap concurrent connections per host, so the tail of
    the grid arrives long after the reviewer has looked at it.
    """
    import asyncio
    import base64

    ids = payload.object_ids[:MAX_SHEET_CROPS]
    if not ids:
        return {"cell": payload.cell, "cols": 0, "rows": 0, "count": 0, "placements": [], "sheet": None}
    cell = max(32, min(int(payload.cell), 512))

    rows = (await db.execute(
        select(Object.object_id, Object.bbox, Frame.frame_id, Frame.img_uri)
        .join(Frame, Frame.frame_id == Object.frame_id)
        .where(Object.object_id.in_([UUID(i) for i in ids])))).all()
    # Requested order is the reviewer's order, so the sheet must follow it rather than the database's.
    by_id = {str(r[0]): r for r in rows}
    ordered = [by_id[i] for i in ids if i in by_id]
    if not ordered:
        raise HTTPException(404, "none of those objects exist")

    cols = payload.cols or max(1, int(math.ceil(math.sqrt(len(ordered)))))
    n_rows = int(math.ceil(len(ordered) / cols))

    def _build() -> tuple[bytes | None, list[dict]]:
        store = get_object_store()
        sheet = np.zeros((n_rows * cell, cols * cell, 3), dtype=np.uint8)
        placements: list[dict] = []
        cache_uri, cache_img = None, None
        for i, (oid, bbox, _fid, uri) in enumerate(ordered):
            if uri != cache_uri:
                cache_uri = uri
                try:
                    buf = np.frombuffer(store.get_bytes(uri), dtype=np.uint8)
                    cache_img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                except Exception:  # noqa: BLE001  one missing frame must not lose the whole sheet
                    cache_img = None
            r, c = divmod(i, cols)
            # A tile is emitted even when its crop failed, so the client's index arithmetic still lines up
            # and the gap is visible as a blank cell rather than shifting every later tile by one.
            place = {"object_id": str(oid), "row": r, "col": c, "ok": False}
            if cache_img is not None:
                h, w = cache_img.shape[:2]
                x1, y1, x2, y2 = bbox
                px, py = (x2 - x1) * payload.pad, (y2 - y1) * payload.pad
                cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
                cx2, cy2 = min(w, int(x2 + px)), min(h, int(y2 + py))
                crop = cache_img[cy1:cy2, cx1:cx2]
                if crop.size:
                    ch, cw = crop.shape[:2]
                    s = min(cell / cw, cell / ch)
                    tw, th = max(1, int(cw * s)), max(1, int(ch * s))
                    resized = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)
                    # Letterboxed rather than stretched: a squashed crop changes the aspect ratio a
                    # reviewer uses to tell a rider from a pedestrian.
                    oy, ox = (cell - th) // 2, (cell - tw) // 2
                    sheet[r*cell + oy: r*cell + oy + th, c*cell + ox: c*cell + ox + tw] = resized
                    place.update(ok=True, w=tw, h=th)
            placements.append(place)
        ok, enc = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])
        return (enc.tobytes() if ok else None), placements

    data, placements = await asyncio.get_running_loop().run_in_executor(None, _build)
    frames_touched = len({r[3] for r in ordered})
    return {
        "cell": cell, "cols": cols, "rows": n_rows, "count": len(ordered),
        # How much work the grouping saved, so the claim that this is cheaper is checkable rather than
        # asserted: one decode per frame against one per crop.
        "frames_decoded": frames_touched, "crops": len(ordered),
        "placements": placements,
        "sheet": ("data:image/jpeg;base64," + base64.b64encode(data).decode()) if data else None,
    }


@router.get("/objects/{object_id}/crop")
async def object_crop(object_id: str, pad: float = 0.15, db: AsyncSession = Depends(db_session)):
    """A JPEG crop of the object's bbox (with padding) for the track timeline thumbnails."""
    obj = await db.get(Object, UUID(object_id))
    if obj is None:
        raise HTTPException(404, "object not found")
    frame = await db.get(Frame, obj.frame_id)
    try:
        buf = np.frombuffer(get_object_store().get_bytes(frame.img_uri), dtype=np.uint8)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(404, "frame image unavailable") from exc
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(404, "failed to decode frame image")
    h, w = img.shape[:2]
    x1, y1, x2, y2 = obj.bbox
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    cx2, cy2 = min(w, int(x2 + px)), min(h, int(y2 + py))
    crop = img[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        crop = img
    ok, out = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(content=out.tobytes(), media_type="image/jpeg")


@router.post("/segment")
async def segment(payload: SegmentIn, db: AsyncSession = Depends(db_session)):
    from services.api.sam_service import segment as run_segment
    from services.training.gpu_lease import gpu_busy_detail

    # Single-GPU discipline: interactive segmentation yields to an active training job. Loading SAM
    # on top of a running train would OOM and KILL the multi-hour job, so refuse cleanly (no GPU touch).
    # Box-level review (accept/reject/reclassify) needs no GPU and still works.
    #
    # Liveness comes from the job's heartbeat rather than its status column, because a run killed by a crash
    # or a stopped container leaves the column at "running" and would otherwise refuse every request from
    # then on, promising a GPU that is already free.
    busy = await gpu_busy_detail(db)
    if busy:
        raise HTTPException(503, busy)

    frame = await db.get(Frame, UUID(payload.frame_id))
    if frame is None:
        raise HTTPException(404, "frame not found")
    buf = np.frombuffer(get_object_store().get_bytes(frame.img_uri), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(500, "failed to decode frame image")
    try:
        return run_segment(img, points=payload.points, labels=payload.labels, box=payload.box, precise=payload.precise)
    except Exception as exc:  # noqa: BLE001
        # On a single GPU, a running training job can consume all VRAM. Surface that cleanly (503)
        # instead of an unhandled 500 so the UI can show a friendly "GPU busy" notice. Box-level
        # review (accept/reject/reclassify) does not need the GPU and still works.
        name = type(exc).__name__
        if "OutOfMemory" in name or "GpuCapacity" in name or "CUDA" in str(exc):
            raise HTTPException(503, "GPU busy (a training job is using the GPU). Interactive "
                                     "segmentation is unavailable until it finishes; box review still works.")
        raise


class ClassifyIn(BaseModel):
    frame_id: str
    box: list[float]                     # [x1, y1, x2, y2] in image pixels


@router.post("/objects/classify")
async def classify_object(payload: ClassifyIn, db: AsyncSession = Depends(db_session)):
    """Zero-shot: what class is the object in this box? Crops the region and scores it against the ontology
    with SigLIP 2, so a SAM box or wand click can auto-detect the class instead of the annotator picking it.
    Returns the top-k class suggestions with confidence; the first is the auto-assigned class."""
    frame = await db.get(Frame, UUID(payload.frame_id))
    if frame is None:
        raise HTTPException(404, "frame not found")
    if len(payload.box) != 4:
        raise HTTPException(400, "box must be [x1,y1,x2,y2]")
    img = cv2.imdecode(np.frombuffer(get_object_store().get_bytes(frame.img_uri), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(500, "failed to decode frame image")
    from services.autolabel.classify_crop import classify_crop
    from services.autolabel.paths.path_c_qwen3vl import crop_object
    crop = crop_object(img, tuple(payload.box), 0.08)
    if crop is None or crop.size == 0:
        raise HTTPException(400, "empty crop")
    try:
        preds = classify_crop(crop)
    except Exception as exc:  # noqa: BLE001
        if "CUDA" in str(exc) or "OutOfMemory" in type(exc).__name__:
            raise HTTPException(503, "GPU busy; auto-classify unavailable right now") from exc
        raise
    return {"predictions": preds}
