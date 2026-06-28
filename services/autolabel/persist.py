"""Persist fused + gated objects: masks to MinIO (polygon JSON), object rows to Postgres, and an
object.gated event per object. The object row is the join hub for the provenance walk.
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
from services.autolabel.paths.path_b_sam3 import polygons_from_mask

log = get_logger("persist")

# Re-running autolabel on a frame must replace only its own machine output, never human work. These are
# the machine-written sources; an object a human has touched becomes source="human" and is preserved.
_MACHINE_SOURCES = (ObjectSource.fused.value, ObjectSource.auto_accept.value)


def _mask_key(session_id, frame_id, object_id) -> str:
    return f"masks/{session_id}/{frame_id}/{object_id}.json"


async def _clear_machine_objects(db: AsyncSession, store: ObjectStore, frame: FrameMeta) -> None:
    """Idempotency: drop this frame's prior machine objects (and their mask blobs) before re-inserting,
    so a re-run does not duplicate. Human-reviewed objects (source='human') are never touched."""
    rows = (await db.execute(
        select(Object.mask_uri).where(Object.frame_id == frame.frame_id, Object.source.in_(_MACHINE_SOURCES))
    )).all()
    for (mask_uri,) in rows:
        if mask_uri:
            store.remove(mask_uri)
    await db.execute(delete(Object).where(Object.frame_id == frame.frame_id, Object.source.in_(_MACHINE_SOURCES)))


async def persist_frame_objects(
    db: AsyncSession,
    store: ObjectStore,
    bus: EventBus,
    frame: FrameMeta,
    fused: list[FusedObject],
) -> dict[str, int]:
    by_state: dict[str, int] = {}
    await _clear_machine_objects(db, store, frame)

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
