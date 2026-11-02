"""Persist fused + gated objects: masks to MinIO (polygon JSON), object rows to Postgres, and an
object.gated event per object. The object row is the join hub for the provenance walk.

Re-run idempotency decision (explicit): re-running autolabel on a frame clears not only the fused and
auto_accept rows it produced last time, but also the interpolated / propagated / relabel rows DERIVED from
that prior machine pass (track interpolation, cross-camera propagation, machine re-inference). Those are
artifacts of a pass that no longer exists; keeping them leaves stale rows attributed to a model whose output
was replaced, polluting every corpus count. They are re-derived by the propagate/interpolate pass over the new
output, so clearing is safe. Human-reviewed objects (source='human') and recall candidates awaiting review are
never touched.
"""

from __future__ import annotations

import json
import uuid

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.bus import TOPIC_OBJECT_GATED, EventBus
from core.logging import get_logger
from core.schemas import FrameMeta, GateState, MaskEncoding, ObjectSource
from core.storage import ObjectStore
from core.timebase import now_ns
from db.models import Object
from services.autolabel.fusion import FusedObject
from services.autolabel.ontology import get_ontology
from services.autolabel.paths.path_b_openvocab import polygons_from_mask

log = get_logger("persist")

# Re-running autolabel on a frame must replace only its own machine output, never human work. These are
# the machine-written sources; an object a human has touched becomes source="human" and is preserved.
# Machine-derived rows cleared on an autolabel re-run (see the module docstring for the decision). The three
# string sources are written by the interpolate/propagate/relabel passes and are not ObjectSource enum members.
_MACHINE_SOURCES = (
    ObjectSource.fused.value, ObjectSource.auto_accept.value,
    "interpolated", "propagated", "relabel",
)


def _mask_key(session_id, frame_id, object_id) -> str:
    return f"masks/{session_id}/{frame_id}/{object_id}.json"


# States that only a person's verdict produces. An object in one of them carries a decision, whatever its source.
_VERDICT_STATES = ("accepted", "rejected", "submitted")

# A new detection this close to an object the re-run kept is the same object seen again, whatever class the
# model gives it: a person who corrected the class is exactly the case where the model still says the old one.
KEPT_OVERLAP_IOU = 0.6


def _carries_a_decision():
    """Machine rows a person has acted on. Not every human decision changes `source`: re-classing a track,
    reverting a relabel and the backfill of a track relabel all leave the row "fused" or "propagated" with a
    review row beside it, and a re-run that keyed on source alone deleted those corrections."""
    from sqlalchemy import exists, or_

    from db.models import Review

    return or_(Object.state.in_(_VERDICT_STATES),
               exists().where(Review.object_id == Object.object_id))


async def _clear_machine_objects(db: AsyncSession, store: ObjectStore, frame: FrameMeta) -> None:
    """Idempotency: drop this frame's prior machine objects (and their mask blobs) before re-inserting,
    so a re-run does not duplicate. Human work is never touched: source='human', and any machine row that
    carries a person's decision (a review row or a verdict state)."""
    clearable = (Object.frame_id == frame.frame_id) & Object.source.in_(_MACHINE_SOURCES) & ~_carries_a_decision()
    rows = (await db.execute(select(Object.mask_uri).where(clearable))).all()
    for (mask_uri,) in rows:
        if mask_uri:
            store.remove(mask_uri)
    await db.execute(delete(Object).where(clearable))


async def _kept_boxes(db: AsyncSession, frame: FrameMeta) -> list[list[float]]:
    """Boxes of the objects a re-run left in place on this frame (human work, decisions, recall candidates)."""
    rows = (await db.execute(select(Object.bbox).where(Object.frame_id == frame.frame_id))).scalars().all()
    return [list(map(float, b)) for b in rows if b is not None and len(b) == 4]


async def persist_frame_objects(
    db: AsyncSession,
    store: ObjectStore,
    bus: EventBus,
    frame: FrameMeta,
    fused: list[FusedObject],
) -> dict[str, int]:
    by_state: dict[str, int] = {}
    await _clear_machine_objects(db, store, frame)

    # Thing/stuff enforcement (panoptic split), the single chokepoint every machine path funnels through:
    # stuff classes (sky, road, every surface, vegetation, barriers, walls, buildings) are background and
    # belong to semantic segmentation, never an instance box. Drop them here so no detection path -- runner,
    # fresh-inference, re-detection -- can ever persist a boxed tree or barrier, by construction.
    onto = get_ontology()
    fused = [fo for fo in fused if onto.is_thing(fo.obj.class_id)]

    # What survived the clear is a person's work. A detection on top of it is the same object again, and
    # writing it would label that object twice, once as corrected and once as the model still sees it.
    kept = await _kept_boxes(db, frame)
    if kept and fused:
        from core.accel.boxes import box_iou_matrix

        iou = box_iou_matrix([fo.obj.bbox.as_list() for fo in fused], kept)
        fused = [fo for fo, row in zip(fused, iou, strict=True) if row.max() < KEPT_OVERLAP_IOU]

    for fo in fused:
        obj = fo.obj
        object_id = uuid.uuid4()
        obj.object_id = object_id

        mask_uri = None
        mask_encoding = None
        if fo.mask is not None:
            polys = polygons_from_mask(fo.mask)
            if polys:
                payload = {
                    "encoding": "polygon",
                    "polygons": polys,
                    "height": frame.height,
                    "width": frame.width,
                }
                mask_uri = store.put_bytes(
                    _mask_key(frame.session_id, frame.frame_id, object_id),
                    json.dumps(payload).encode(),
                    "application/json",
                )
                mask_encoding = MaskEncoding.polygon.value

        source = (
            ObjectSource.auto_accept.value
            if obj.state == GateState.auto_accept
            else ObjectSource.fused.value
        )

        db.add(
            Object(
                object_id=object_id,
                frame_id=frame.frame_id,
                track_id=None,
                class_id=obj.class_id,
                bbox=obj.bbox.as_list(),
                mask_uri=mask_uri,
                mask_encoding=mask_encoding,
                attrs=obj.attrs,
                conf=obj.conf,
                source=source,
                provenance=obj.provenance.model_dump(mode="json"),
                state=obj.state.value,
            )
        )
        by_state[obj.state.value] = by_state.get(obj.state.value, 0) + 1

        # Control sample (M4.4): mirror a small random fraction of the gate's own auto-accepts into the
        # always-reviewed control stream, so measured precision reflects the live gate, not a backfill.
        if obj.state == GateState.auto_accept:
            from services.govern.control_sample import maybe_sample

            await maybe_sample(db, str(object_id), True)

        await bus.emit(
            TOPIC_OBJECT_GATED,
            {
                "object_id": str(object_id),
                "frame_id": str(frame.frame_id),
                "session_id": str(frame.session_id),
                "class_id": obj.class_id,
                "class_name": obj.class_name,
                "conf": obj.conf,
                "state": obj.state.value,
                "ts_ns": now_ns(),
            },
            key=str(frame.session_id),
        )

    await db.flush()
    return by_state
