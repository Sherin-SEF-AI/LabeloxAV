"""Cross-camera label propagation: a label on one camera becomes labels on every other camera that can see
the same object at the same instant.

The bridge is 3D. An object's cuboid_3d is in the ego frame, shared by the whole rig, so projecting it with a
different camera's calibration lands it exactly where that camera sees it. So: take the source object's 3D
cuboid (fit one monocularly if it has none, reusing the 2D->3D agent), find the synchronized frames from the
other cameras (same session, timestamp within the PPS tolerance), project the cuboid into each, and where it
is actually visible create a 2D box carrying the same class and the same cuboid. Visibility (how many
corners fall in front of and inside the image) grades the box: fully visible auto-accepts, partial routes to
review, not-visible is skipped (e.g. a forward object is behind the rear camera). One reversible AgentRun;
revert deletes the created boxes.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import AgentRun, Frame, Object

log = get_logger("agent.crosscam")


def _project_box(cuboid: dict, cam_id: str, w: int, h: int):
    """Project an ego-frame cuboid into a camera; return (2D box of the visible corners, visibility 0..1)."""
    from services.lidar.boxes import project_cuboid

    size = cuboid["size"]
    dims = [size[1], size[0], size[2]]  # project_cuboid wants [length, width, height]
    proj = project_cuboid(cuboid["center"], dims, float(cuboid.get("yaw", 0.0)), cam_id, w, h)
    uv, infr, inim = proj["corners_uv"], proj["in_front"], proj["in_image"]
    vis = [uv[i] for i in range(len(uv)) if infr[i] and inim[i]]
    front = [uv[i] for i in range(len(uv)) if infr[i]]
    if len(front) < 4:
        return None, 0.0
    pts = vis if len(vis) >= 4 else front
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    box = [max(0.0, min(xs)), max(0.0, min(ys)), min(float(w), max(xs)), min(float(h), max(ys))]
    if box[2] - box[0] < 3 or box[3] - box[1] < 3:
        return None, 0.0
    visibility = sum(1 for i in range(len(uv)) if infr[i] and inim[i]) / 8.0
    return box, round(visibility, 3)


async def _cuboid_for(db: AsyncSession, src: Object, src_frame: Frame):
    if src.cuboid_3d:
        return src.cuboid_3d
    from services.agent.cuboid_agent import _fit_mono
    from services.autolabel.ontology import get_ontology

    fit = _fit_mono(src, get_ontology(), src_frame.cam_id, src_frame.width, src_frame.height)
    return fit[0] if fit else None


async def _sync_frames(db: AsyncSession, src_frame: Frame, tol_ns: int):
    rows = await db.execute(
        select(Frame).where(Frame.session_id == src_frame.session_id, Frame.cam_id != src_frame.cam_id,
                            Frame.ts_ns >= src_frame.ts_ns - tol_ns, Frame.ts_ns <= src_frame.ts_ns + tol_ns))
    # nearest frame per other camera
    by_cam: dict[str, Frame] = {}
    for f in rows.scalars().all():
        cur = by_cam.get(f.cam_id)
        if cur is None or abs(f.ts_ns - src_frame.ts_ns) < abs(cur.ts_ns - src_frame.ts_ns):
            by_cam[f.cam_id] = f
    return list(by_cam.values())


async def plan_cross_camera(db: AsyncSession, object_id: uuid.UUID, *, tol_ms: int = 20,
                            high: float = 0.75, min_vis: float = 0.5) -> dict:
    """Dry-run: which other cameras can see this object, and the box it would get there. No writes."""
    from services.autolabel.ontology import get_ontology

    src = await db.get(Object, object_id)
    if src is None:
        raise ValueError("object not found")
    src_frame = await db.get(Frame, src.frame_id)
    cuboid = await _cuboid_for(db, src, src_frame)
    if cuboid is None:
        return {"object_id": str(object_id), "reason": "no 3D cuboid and the box does not lift to the ground",
                "counts": {"targets": 0, "auto_accept": 0, "review": 0, "skip": 0}, "items": []}
    onto = get_ontology()
    cname = onto.by_id(int(src.class_id)).name if src.class_id is not None else "?"
    targets = await _sync_frames(db, src_frame, tol_ms * 1_000_000)
    items = []
    counts = {"targets": len(targets), "auto_accept": 0, "review": 0, "skip": 0}
    for tf in targets:
        box, vis = _project_box(cuboid, tf.cam_id, tf.width, tf.height)
        if box is None or vis < min_vis:
            counts["skip"] += 1
            items.append({"cam_id": tf.cam_id, "frame_id": str(tf.frame_id), "action": "skip",
                          "visibility": vis, "reason": "not visible in this camera"})
            continue
        action = "auto_accept" if vis >= high else "review"
        counts[action] += 1
        items.append({"cam_id": tf.cam_id, "frame_id": str(tf.frame_id), "action": action,
                      "visibility": vis, "box": [round(v, 1) for v in box], "class_name": cname})
    return {"object_id": str(object_id), "class_name": cname, "cuboid_source": "existing" if src.cuboid_3d else "fitted",
            "counts": counts, "items": items}


async def commit_cross_camera(db: AsyncSession, object_id: uuid.UUID, *, tol_ms: int = 20, high: float = 0.75,
                              min_vis: float = 0.5, created_by: str | None = None) -> dict:
    """Create the projected 2D boxes on the other cameras as one reversible run."""
    plan = await plan_cross_camera(db, object_id, tol_ms=tol_ms, high=high, min_vis=min_vis)
    src = await db.get(Object, object_id)
    src_frame = await db.get(Frame, src.frame_id)
    cuboid = await _cuboid_for(db, src, src_frame)
    run_id = uuid.uuid4()
    changes: dict[str, dict] = {}
    for item in plan["items"]:
        if item["action"] == "skip":
            continue
        oid = uuid.uuid4()
        db.add(Object(
            object_id=oid, frame_id=uuid.UUID(item["frame_id"]), track_id=src.track_id, class_id=src.class_id,
            bbox=[float(v) for v in item["box"]], conf=float(item["visibility"]), source="propagated",
            state=item["action"], cuboid_3d=cuboid, attrs={},
            provenance={"cross_camera_from": str(object_id), "source_cam": src_frame.cam_id,
                        "target_cam": item["cam_id"], "method": "3d-reprojection", "agent_run_id": str(run_id)},
        ))
        changes[str(oid)] = {"created": True}
    db.add(AgentRun(run_id=run_id, kind="crosscam", scope={"object_id": str(object_id)}, status="committed",
                    policy={"tol_ms": tol_ms, "high": high, "min_vis": min_vis}, counts=plan["counts"],
                    changes=changes, critic={}, created_by=created_by))
    await db.commit()
    log.info("agent.crosscam.commit", object_id=str(object_id), run_id=str(run_id), created=len(changes))
    return {"run_id": str(run_id), "object_id": str(object_id), "created": len(changes), "counts": plan["counts"]}
