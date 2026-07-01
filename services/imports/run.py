"""Import driver: external dataset -> Session/Frame/Object rows. The mirror image of services.export.

Annotation formats (coco/yolo/pascalvoc/openlabel/nuscenes/parquet) are parsed to ImportFrame[], the
image is anonymized (Gate A) and stored, and objects land as source="imported", state="review" so they
flow through the existing triage/gate UI. Raw formats (video/mcap/images) reuse services.ingest.run so
the full ingest plane (PII + quality + manifest) applies. Names remap to the ontology via remap.py.

    python -m services.imports.run --format coco --source /path/to/dataset --vehicle IMPORT-01 --city BLR
"""

from __future__ import annotations

import asyncio
import json
import uuid
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import click
import cv2
import numpy as np

from core.config import get_settings
from core.logging import get_logger, setup_logging
from core.storage import get_object_store
from core.timebase import now_ns
from db.models import Frame, ImportJob, Object, PiiAudit
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.anonymize.anonymizer import get_anonymizer
from services.autolabel.ontology import get_ontology
from services.imports import (
    adapter_bdd,
    adapter_coco,
    adapter_kitti,
    adapter_mapillary,
    adapter_nuscenes,
    adapter_openlabel,
    adapter_parquet,
    adapter_pascalvoc,
    adapter_yolo,
)
from services.imports._util import resolve_image
from services.imports.records import ImportSpec
from services.imports.remap import remap_name

log = get_logger("import")

ADAPTERS = {
    "coco": adapter_coco.parse,
    "yolo": adapter_yolo.parse,
    "pascalvoc": adapter_pascalvoc.parse,
    "openlabel": adapter_openlabel.parse,
    "nuscenes": adapter_nuscenes.parse,
    "parquet": adapter_parquet.parse,
    "mapillary": adapter_mapillary.parse,
    "kitti": adapter_kitti.parse,
    "bdd": adapter_bdd.parse,
}
RAW_FORMATS = {"video", "mcap", "images"}
ALL_FORMATS = sorted(set(ADAPTERS) | RAW_FORMATS)


def _acquire_source(source_uri: str, workdir: Path) -> Path:
    """Return a local directory containing the dataset. Downloads s3 objects and extracts zips."""
    store = get_object_store()
    if source_uri.startswith("s3://"):
        data = store.get_bytes(source_uri)
        name = source_uri.split("/")[-1] or "download"
        local = workdir / name
        local.write_bytes(data)
        src = local
    else:
        src = Path(source_uri)
        if src.is_dir():
            return src
    if src.suffix.lower() == ".zip":
        ext = workdir / "extracted"
        ext.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(src) as zf:
            zf.extractall(ext)
        return ext
    return src.parent


def _load_image(image_ref: str, root: Path) -> np.ndarray | None:
    store = get_object_store()
    try:
        if image_ref.startswith("s3://"):
            buf = np.frombuffer(store.get_bytes(image_ref), dtype=np.uint8)
            return cv2.imdecode(buf, cv2.IMREAD_COLOR)
        resolved = resolve_image(root, image_ref)
        if resolved is None:
            return None
        if isinstance(resolved, str) and resolved.startswith("s3://"):
            buf = np.frombuffer(store.get_bytes(resolved), dtype=np.uint8)
            return cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return cv2.imread(str(resolved))
    except Exception as exc:  # noqa: BLE001
        log.warning("import.image_load_failed", ref=image_ref, error=str(exc))
        return None


async def _bump_job(job_id, **fields) -> None:
    if not job_id:
        return
    async with get_sessionmaker()() as db:
        j = await db.get(ImportJob, uuid.UUID(str(job_id)))
        if j:
            for k, v in fields.items():
                setattr(j, k, v)
            await db.commit()


async def _import_raw(spec: ImportSpec, job_id, root: Path) -> dict:
    from services.imports.raw_media import read_image_folder
    from services.ingest.reader_mcap import read_mcap
    from services.ingest.reader_video import read_video
    from services.ingest.run import ingest

    if spec.format == "images":
        frame_iter = read_image_folder(root)
        raw_uri, mcap_uri = spec.source_uri, None
    elif spec.format == "video":
        vid = next((p for p in root.rglob("*") if p.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi")), None)
        if vid is None:
            raise FileNotFoundError("no video file found in the source")
        frame_iter = read_video(str(vid), "cam_front", now_ns(), get_settings().ingest.target_fps)
        raw_uri, mcap_uri = spec.source_uri, None
    else:  # mcap
        mc = next((p for p in root.rglob("*.mcap")), None)
        if mc is None:
            raise FileNotFoundError("no .mcap file found in the source")
        frame_iter, raw_uri, mcap_uri = (read_mcap(str(mc), get_settings().ingest.target_fps), None,
                                         spec.source_uri)

    result = await ingest(
        frame_iter=frame_iter, vehicle=spec.target_vehicle, city=spec.city,
        route=spec.route or f"import:{spec.format}", raw_uri=raw_uri, mcap_uri=mcap_uri,
        source_streams=[spec.format],
    )
    counts = {"sessions": 1, "frames": result["n_frames"], "objects": 0,
              "faces_blurred": result.get("pii", {}).get("n_faces", 0)}
    await _bump_job(job_id, status="done", progress=1.0, counts=counts, session_id=uuid.UUID(result["session_id"]))
    return {"session_id": result["session_id"], "counts": counts}


async def import_dataset(spec: ImportSpec, job_id=None) -> dict:
    settings = get_settings()
    onto = get_ontology()
    store = get_object_store()
    store.ensure_bucket()
    anon = get_anonymizer() if settings.pii.enabled else None
    await _bump_job(job_id, status="running", progress=0.0)

    with TemporaryDirectory() as tmp:
        root = _acquire_source(spec.source_uri, Path(tmp))

        if spec.format in RAW_FORMATS:
            return await _import_raw(spec, job_id, root)

        if spec.format not in ADAPTERS:
            raise ValueError(f"unknown import format: {spec.format} (choose from {ALL_FORMATS})")
        frames = ADAPTERS[spec.format](root)
        total = max(1, len(frames))
        session_id = uuid.uuid4()
        counts = {"sessions": 1, "frames": 0, "objects": 0, "unmapped": 0, "faces_blurred": 0}
        ts_lo, ts_hi = None, None

        async with get_sessionmaker()() as db:
            base = now_ns()
            session_row = DbSession(
                session_id=session_id, vehicle_id=spec.target_vehicle, start_ts_ns=base, end_ts_ns=base,
                city=spec.city, route=spec.route or f"import:{spec.format}", sensors={},
                raw_uri=spec.source_uri, ontology_version=onto.version,
            )
            db.add(session_row)
            await db.flush()

            for i, fr in enumerate(frames):
                img = _load_image(fr.image_ref, root)
                if img is None:
                    continue
                # Downscale very wide frames to the label resolution (keeps PII fast + disk bounded)
                # and scale the boxes by the SAME factor so they still line up.
                scale = 1.0
                mw = settings.ingest.max_width
                if mw and img.shape[1] > mw:
                    scale = mw / img.shape[1]
                    img = cv2.resize(img, (mw, int(round(img.shape[0] * scale))), interpolation=cv2.INTER_AREA)
                pii = anon.anonymize(img) if anon else None
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if not ok:
                    continue
                h, w = img.shape[:2]
                ts = fr.ts_ns or (base + i * 1000)
                ts_lo = ts if ts_lo is None else min(ts_lo, ts)
                ts_hi = ts if ts_hi is None else max(ts_hi, ts)
                key = f"frames/{session_id}/{fr.cam_id}/{ts}.jpg"
                img_uri = store.put_bytes(key, buf.tobytes(), "image/jpeg")

                frame_row = Frame(session_id=session_id, ts_ns=ts, cam_id=fr.cam_id, img_uri=img_uri,
                                  width=w, height=h, quality=1.0)
                db.add(frame_row)
                await db.flush()
                if pii is not None:
                    db.add(PiiAudit(frame_id=frame_row.frame_id, session_id=session_id, n_faces=pii.n_faces,
                                    n_plates=pii.n_plates, regions=pii.regions, method_version=pii.method_version, ts_ns=ts))
                    counts["faces_blurred"] += pii.n_faces

                for o in fr.objects:
                    cid, cname, mapped = None, None, False
                    if o.ontology_class_id is not None:
                        try:
                            c = onto.by_id(o.ontology_class_id)
                            cid, cname, mapped = c.id, c.name, True
                        except Exception:  # noqa: BLE001
                            pass
                    if cid is None:
                        cid, cname, mapped = remap_name(o.name, onto)
                    if not mapped:
                        counts["unmapped"] += 1
                    prov = {**(o.provenance or {}), "import_format": spec.format,
                            "import_job": str(job_id) if job_id else None, "original_name": o.name}
                    # Mask round-trip: carry a lossless export's mask uri through, or materialize polygons
                    # (COCO/OpenLABEL) into a fresh mask blob, so segmentation is not silently dropped.
                    oid = uuid.uuid4()
                    mask_uri, mask_encoding = o.mask_uri, o.mask_encoding
                    if mask_uri is None and o.mask_polygons:
                        scaled = [[float(v) * scale for v in poly] for poly in o.mask_polygons]
                        payload = {"encoding": "polygon", "polygons": scaled, "height": h, "width": w}
                        mask_uri = store.put_bytes(f"masks/{session_id}/{frame_row.frame_id}/{oid}.json",
                                                   json.dumps(payload).encode(), "application/json")
                        mask_encoding = "polygon"
                    db.add(Object(object_id=oid, frame_id=frame_row.frame_id, class_id=cid,
                                  bbox=[float(x) * scale for x in o.bbox], conf=float(o.conf),
                                  source="imported", state="review", provenance=prov, attrs=o.attrs or {},
                                  mask_uri=mask_uri, mask_encoding=mask_encoding, rot_deg=o.rot_deg,
                                  keypoints=o.keypoints))
                    counts["objects"] += 1
                counts["frames"] += 1

                if i % 100 == 0:
                    await db.commit()
                    await _bump_job(job_id, progress=round(i / total, 3), counts=counts)

            session_row.start_ts_ns = ts_lo or base
            session_row.end_ts_ns = ts_hi or base
            await db.commit()

    await _bump_job(job_id, status="done", progress=1.0, counts=counts, session_id=session_id)
    result = {"session_id": str(session_id), "format": spec.format, "counts": counts}
    log.info("import.done", **{k: result[k] for k in ("session_id", "format")}, **counts)
    return result


async def run_import_guarded(spec: ImportSpec, job_id) -> None:
    """Background entrypoint: run the import on the API event loop and record failures on the job row
    (so a crashed import is visible as status=error rather than a silent hang)."""
    try:
        await import_dataset(spec, job_id)
    except Exception as exc:  # noqa: BLE001
        log.error("import.failed", job_id=str(job_id), error=str(exc))
        await _bump_job(job_id, status="error", error=str(exc))


@click.command()
@click.option("--format", "fmt", required=True, type=click.Choice(ALL_FORMATS))
@click.option("--source", "source_uri", required=True, help="local path / zip / s3:// uri")
@click.option("--vehicle", "vehicle", default="IMPORT-01")
@click.option("--city", default=None)
@click.option("--route", default=None)
def main(fmt, source_uri, vehicle, city, route) -> None:
    setup_logging(get_settings().log_level)
    spec = ImportSpec(format=fmt, source_uri=source_uri, target_vehicle=vehicle, city=city, route=route)
    click.echo(asyncio.run(import_dataset(spec)))


if __name__ == "__main__":
    main()
