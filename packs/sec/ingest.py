"""Sec ingestion: a fixed-camera clip becomes a SessionDraft + FrameDrafts.

Reuses the engine's video reader (services/ingest/reader_video) - decoding a clip is domain-neutral - but
populates the schema the static-camera way: vehicle_id is None (there is no ego vehicle), ego_speed is None on
every frame (a fixed camera has no CAN speed), and the session is routed to the sec pack. The camera identity
lives in cam_id, exactly as the AV path already stores it.

DB persistence of these drafts lands in SEC-M3 alongside the Sec ontology (which supplies the session's
ontology binding); SEC-M2 delivers the decode + draft contract, tested on a procedurally generated clip.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast

from packs.base import FrameDraft, IngestSource, SessionDraft

PACK_ID = "sec"


class StaticCameraIngestionAdapter:
    """Turns a fixed-camera video into drafts. can_handle gates on the source media type."""

    media_types = frozenset({"video"})

    def can_handle(self, source: IngestSource) -> bool:
        return source.media_type in self.media_types

    def read(self, source: IngestSource) -> tuple[SessionDraft, Iterator[FrameDraft]]:
        if not self.can_handle(source):
            raise ValueError(f"StaticCameraIngestionAdapter cannot handle media_type={source.media_type!r}")
        draft = SessionDraft(
            pack_id=PACK_ID,
            cam_id=source.cam_id,
            vehicle_id=None,
            city=cast("str | None", source.meta.get("city")),
            route=cast("str | None", source.meta.get("route")),
        )
        return draft, self._frames(source)

    def _frames(self, source: IngestSource) -> Iterator[FrameDraft]:
        from services.ingest.reader_video import read_video

        target_fps = float(cast(float, source.meta.get("fps", 5.0)))
        start_ts_ns = int(cast(int, source.meta.get("start_ts_ns", 0)))
        # No sidecar: a fixed camera has no ego side-channel, and ego_speed is forced None regardless.
        for rf in read_video(source.uri, source.cam_id, start_ts_ns, target_fps, None):
            yield FrameDraft(ts_ns=rf.ts_ns, cam_id=rf.cam_id, image_bgr=rf.image_bgr, ego_speed=None)


async def persist(source: IngestSource, store: object | None = None,
                  ontology_version: str | None = None) -> dict:
    """Decode a static-camera source and write it as a Session (+ Frames) to the store and DB. The Session
    carries pack_id='sec' and a null vehicle_id; every Frame has a null ego_speed. This is the static-camera
    twin of services/ingest/run.ingest, minus the moving-camera machinery (MCAP/CAN, inertial, ego-hood).
    Returns {session_id, n_frames, pack_id}."""
    import uuid

    import cv2

    from core.storage import get_object_store
    from db.models import Frame
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    obj_store = store or get_object_store()
    obj_store.ensure_bucket()  # type: ignore[attr-defined]
    version = ontology_version or get_ontology("sec").version

    draft, frames = StaticCameraIngestionAdapter().read(source)
    session_id = uuid.uuid4()
    rows: list[Frame] = []
    ts_first: int | None = None
    ts_last: int | None = None
    for fd in frames:
        ok, buf = cv2.imencode(".jpg", fd.image_bgr)
        if not ok:
            continue
        key = f"frames/{session_id}/{fd.cam_id}/{fd.ts_ns}.jpg"
        uri = obj_store.put_bytes(key, buf.tobytes(), "image/jpeg")  # type: ignore[attr-defined]
        h, w = fd.image_bgr.shape[:2]
        rows.append(Frame(session_id=session_id, ts_ns=fd.ts_ns, cam_id=fd.cam_id, img_uri=uri,
                          width=int(w), height=int(h), ego_speed=fd.ego_speed))
        ts_first = fd.ts_ns if ts_first is None else ts_first
        ts_last = fd.ts_ns
    if not rows:
        raise RuntimeError(f"no frames decoded from {source.uri}")

    async with get_sessionmaker()() as db:
        db.add(DbSession(
            session_id=session_id, pack_id=draft.pack_id, vehicle_id=draft.vehicle_id,
            start_ts_ns=ts_first, end_ts_ns=ts_last, city=draft.city, route=draft.route,
            ontology_version=version,
        ))
        for fr in rows:
            db.add(fr)
        await db.commit()
    return {"session_id": str(session_id), "n_frames": len(rows), "pack_id": draft.pack_id}
