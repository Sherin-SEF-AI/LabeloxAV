"""Ingestion driver and CLI. Decodes a clip to frames at the labeling cadence, quality-gates,
writes frames to MinIO and rows to Postgres, writes the session manifest, emits frame.ready.

    python -m services.ingest.run --video clip.mp4 --vehicle TIGOR-07 --city BLR
    python -m services.ingest.run --mcap session.mcap --vehicle TIGOR-07 --city BLR
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from collections.abc import Iterator
from pathlib import Path

import click
import cv2
from geoalchemy2.elements import WKTElement

from core.bus import TOPIC_FRAME_READY, EventBus
from core.config import get_settings
from core.logging import get_logger, setup_logging
from core.schemas import SessionManifest
from core.storage import get_object_store
from core.timebase import now_ns
from db.models import Frame, PiiAudit
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.anonymize.anonymizer import get_anonymizer
from services.autolabel.ontology import get_ontology
from services.ingest.manifest import write_manifest
from services.ingest.quality import score_frame
from services.ingest.reader_mcap import read_mcap
from services.ingest.reader_video import read_video
from services.ingest.types import RawFrame

log = get_logger("ingest")


def _frame_key(session_id: uuid.UUID, cam_id: str, ts_ns: int) -> str:
    return f"frames/{session_id}/{cam_id}/{ts_ns}.jpg"


def _calib_hash(vehicle: str, cam_id: str) -> str:
    # Placeholder deterministic calibration hash. Real ingest reads it from the rig manifest;
    # the contract (a stable per-sensor hash) is what the provenance chain depends on.
    return hashlib.sha256(f"{vehicle}:{cam_id}:calib-v0".encode()).hexdigest()[:16]


async def ingest(
    *,
    frame_iter: Iterator[RawFrame],
    vehicle: str,
    city: str | None,
    route: str | None,
    raw_uri: str | None,
    mcap_uri: str | None,
    source_streams: list[str],
    anonymizer=None,
) -> dict:
    settings = get_settings()
    store = get_object_store()
    store.ensure_bucket()
    onto = get_ontology()
    maker = get_sessionmaker()

    # Gate A (DPDPA): blur faces + plates before any frame reaches storage. Mandatory unless the
    # audited opt-out (pii.enabled=false) is set. anonymizer is injectable for tests.
    anon = anonymizer or (get_anonymizer() if settings.pii.enabled else None)
    n_faces_total = 0
    n_plates_total = 0

    bus = EventBus()
    await bus.start()

    session_id = uuid.uuid4()
    n_frames = 0
    n_rejected = 0
    t_start = None
    t_end = None
    gps_track: list[list[float]] = []
    cams: set[str] = set()

    try:
        async with maker() as db:
            session_row: DbSession | None = None

            for rf in frame_iter:
                # Downscale wide frames (e.g. 4K dashcam) to the configured label resolution before
                # anything touches them: detectors run at imgsz regardless, and review/masks only
                # need 1080p. This is a large, cheap throughput win.
                mw = settings.ingest.max_width
                if mw and rf.image_bgr.shape[1] > mw:
                    h0, w0 = rf.image_bgr.shape[:2]
                    rf.image_bgr = cv2.resize(rf.image_bgr, (mw, int(round(h0 * mw / w0))), interpolation=cv2.INTER_AREA)

                q = score_frame(rf.image_bgr, settings.ingest)
                if not q.accepted:
                    n_rejected += 1
                    log.debug("frame.rejected", ts_ns=rf.ts_ns, reasons=q.reasons)
                    continue

                if session_row is None:
                    session_row = DbSession(
                        session_id=session_id,
                        vehicle_id=vehicle,
                        start_ts_ns=rf.ts_ns,
                        end_ts_ns=rf.ts_ns,
                        city=city,
                        route=route,
                        sensors={},
                        raw_uri=raw_uri,
                        mcap_uri=mcap_uri,
                        ontology_version=onto.version,
                    )
                    db.add(session_row)
                    await db.flush()
                    t_start = rf.ts_ns

                # PII anonymization in place, before encode: no clean frame ever reaches storage.
                pii = anon.anonymize(rf.image_bgr) if anon else None

                ok, buf = cv2.imencode(".jpg", rf.image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if not ok:
                    log.warning("frame.encode_failed", ts_ns=rf.ts_ns)
                    continue
                key = _frame_key(session_id, rf.cam_id, rf.ts_ns)
                img_uri = store.put_bytes(key, buf.tobytes(), "image/jpeg")

                h, w = rf.image_bgr.shape[:2]
                gnss = (
                    WKTElement(f"POINT({rf.lon} {rf.lat})", srid=4326)
                    if rf.lat is not None and rf.lon is not None
                    else None
                )
                frame_row = Frame(
                    session_id=session_id,
                    ts_ns=rf.ts_ns,
                    cam_id=rf.cam_id,
                    img_uri=img_uri,
                    width=w,
                    height=h,
                    gnss=gnss,
                    ego_speed=rf.ego_speed,
                    quality=q.score,
                )
                db.add(frame_row)
                await db.flush()

                if pii is not None:
                    db.add(PiiAudit(
                        frame_id=frame_row.frame_id, session_id=session_id,
                        n_faces=pii.n_faces, n_plates=pii.n_plates, regions=pii.regions,
                        method_version=pii.method_version, ts_ns=rf.ts_ns,
                    ))
                    n_faces_total += pii.n_faces
                    n_plates_total += pii.n_plates

                cams.add(rf.cam_id)
                n_frames += 1
                t_end = rf.ts_ns
                if rf.lat is not None and rf.lon is not None:
                    gps_track.append([rf.lat, rf.lon, rf.ts_ns])

                await bus.publish(
                    TOPIC_FRAME_READY,
                    {
                        "session_id": str(session_id),
                        "frame_id": str(frame_row.frame_id),
                        "ts_ns": rf.ts_ns,
                        "cam_id": rf.cam_id,
                        "img_uri": img_uri,
                    },
                    key=str(session_id),
                )

            if session_row is None:
                raise RuntimeError("no acceptable frames ingested (all rejected by quality gate?)")

            sensors = {
                cam: {"serial": f"{vehicle}-{cam}", "calibration_hash": _calib_hash(vehicle, cam)}
                for cam in sorted(cams)
            }
            session_row.end_ts_ns = t_end or t_start
            session_row.sensors = sensors

            manifest = SessionManifest(
                session_id=session_id,
                vehicle_id=vehicle,
                t_start_ns=t_start or now_ns(),
                t_end_ns=t_end or now_ns(),
                city=city,
                route=route,
                streams=source_streams,
                sensors=sensors,
                gps_track=gps_track,
                n_frames=n_frames,
                raw_uri=raw_uri,
                mcap_uri=mcap_uri,
                ontology_version=onto.version,
                qa={
                    "accepted": n_frames,
                    "rejected": n_rejected,
                    "pii": {
                        "enabled": anon is not None,
                        "method_version": anon.method_version if anon else None,
                        "n_faces": n_faces_total,
                        "n_plates": n_plates_total,
                    },
                },
            )
            manifest_uri = write_manifest(store, manifest)
            session_row.manifest_uri = manifest_uri

            await db.commit()
    finally:
        await bus.stop()

    result = {
        "session_id": str(session_id),
        "n_frames": n_frames,
        "n_rejected": n_rejected,
        "cams": sorted(cams),
        "manifest_uri": manifest_uri,
        "pii": {"enabled": anon is not None, "n_faces": n_faces_total, "n_plates": n_plates_total},
    }
    log.info("ingest.done", **result)
    return result


def _upload_raw(path: Path, vehicle: str) -> str:
    store = get_object_store()
    store.ensure_bucket()
    data = path.read_bytes()
    return store.put_content_addressed(f"raw/{vehicle}", data, path.suffix.lower())


@click.command()
@click.option("--video", "video", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--mcap", "mcap_file", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--sidecar", type=click.Path(exists=True, dir_okay=False), default=None, help="CSV/JSON side channels for --video")
@click.option("--cam", "cam_id", default="cam_f", help="camera id for --video")
@click.option("--vehicle", required=True)
@click.option("--city", default=None)
@click.option("--route", default=None)
@click.option("--target-fps", type=float, default=None)
@click.option("--start-ts-ns", type=int, default=None, help="UTC ns for video frame 0 (default: now)")
def main(video, mcap_file, sidecar, cam_id, vehicle, city, route, target_fps, start_ts_ns) -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    fps = target_fps or settings.ingest.target_fps

    if not video and not mcap_file:
        raise click.UsageError("provide either --video or --mcap")
    if video and mcap_file:
        raise click.UsageError("provide only one of --video or --mcap")

    if video:
        start = start_ts_ns if start_ts_ns is not None else now_ns()
        raw_uri = _upload_raw(Path(video), vehicle)
        frame_iter = read_video(video, cam_id, start, fps, sidecar)
        result = asyncio.run(
            ingest(
                frame_iter=frame_iter,
                vehicle=vehicle,
                city=city,
                route=route,
                raw_uri=raw_uri,
                mcap_uri=None,
                source_streams=[cam_id],
            )
        )
    else:
        mcap_uri = _upload_raw(Path(mcap_file), vehicle)
        frame_iter = read_mcap(mcap_file, fps)
        result = asyncio.run(
            ingest(
                frame_iter=frame_iter,
                vehicle=vehicle,
                city=city,
                route=route,
                raw_uri=None,
                mcap_uri=mcap_uri,
                source_streams=["mcap"],
            )
        )

    click.echo(result)


if __name__ == "__main__":
    main()
