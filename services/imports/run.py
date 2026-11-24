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
from db.models import Frame, ImportJob, Object, PiiAudit, Track
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.anonymize.anonymizer import get_anonymizer
from services.autolabel.ontology import get_ontology
from services.imports import (
    adapter_bdd,
    adapter_coco,
    adapter_cvat,
    adapter_encord,
    adapter_kitti,
    adapter_labelbox,
    adapter_labelstudio,
    adapter_mapillary,
    adapter_nuscenes,
    adapter_openlabel,
    adapter_parquet,
    adapter_pascalvoc,
    adapter_scale,
    adapter_superannotate,
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
    "cvat": adapter_cvat.parse,
    "labelstudio": adapter_labelstudio.parse,
    # Migration formats: a competitor's export is the cheapest path in for a team that already
    # has years of labels somewhere else.
    "labelbox": adapter_labelbox.parse,
    "scale": adapter_scale.parse,
    "superannotate": adapter_superannotate.parse,
    "encord": adapter_encord.parse,
}
RAW_FORMATS = {"video", "mcap", "images"}
# Every container the upload page accepts as video and OpenCV's FFmpeg backend decodes. The upload page
# offered .webm and .m4v while this list stopped at .avi, so those uploads were stored and then failed with
# "no video file found"; the two lists must match (tests/test_video_import_formats.py).
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")
ALL_FORMATS = sorted(set(ADAPTERS) | RAW_FORMATS)


class SourceNotAllowed(ValueError):
    """A request named a source this server will not read. The message is fit to show the caller."""


def import_roots() -> list[Path]:
    out = []
    for raw in get_settings().paths.import_roots:
        if raw:
            out.append(Path(raw).expanduser().resolve(strict=False))
    return out


def check_request_source(source_uri: str) -> None:
    """Refuse a caller-supplied source the import pipeline must not open.

    `_acquire_source` reads whatever it is given, which is correct for an operator at a shell and wrong
    for an HTTP request: `source_uri` arrives from a reviewer or an outside workforce, and a bare
    `Path(source_uri)` let either of them read any directory or archive the API process can. Object
    storage is the product's own bucket and stays open. A local path is allowed only inside a directory
    an operator listed in LBX_PATHS__IMPORT_ROOTS, checked AFTER symlinks are resolved, because a link
    inside an allowed directory that points at /etc passes a prefix test on the string and reads
    something else. Called by the API routes; the CLI does not call it.
    """
    uri = (source_uri or "").strip()
    if not uri:
        raise SourceNotAllowed("no source given")
    if uri.startswith("s3://"):
        return
    if "://" in uri:
        raise SourceNotAllowed(f"unsupported source scheme in {uri!r}; upload the data or use an s3:// uri")
    path = Path(uri).expanduser()
    if not path.is_absolute():
        raise SourceNotAllowed("a local source must be an absolute path")
    roots = import_roots()
    if not roots:
        raise SourceNotAllowed(
            "this server does not import from local paths. Upload the data instead, or have an operator "
            "list the directory in LBX_PATHS__IMPORT_ROOTS")
    resolved = path.resolve(strict=False)
    if not any(resolved == r or r in resolved.parents for r in roots):
        raise SourceNotAllowed("that path is outside the directories this server may import from "
                               "(LBX_PATHS__IMPORT_ROOTS)")


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
        ext_root = ext.resolve()
        with zipfile.ZipFile(src) as zf:
            # zip-slip guard: a member named "../../x" or "/abs/path" escapes ext. Resolve every target and
            # refuse anything outside ext before extracting a single byte.
            for member in zf.namelist():
                target = (ext / member).resolve()
                if not (target == ext_root or ext_root in target.parents):
                    raise ValueError(f"unsafe zip member escapes extract dir: {member!r}")
            # Zip-bomb guard: the declared uncompressed total, checked before extracting anything.
            cap = get_settings().paths.import_max_extract_bytes
            total = sum(zi.file_size for zi in zf.infolist())
            if cap and total > cap:
                raise ValueError(f"archive expands to {total / 1024 ** 3:.1f} GB, over the "
                                 f"{cap / 1024 ** 3:.0f} GB limit (LBX_PATHS__IMPORT_MAX_EXTRACT_BYTES)")
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
    """Write a job's progress, and stop the run if somebody cancelled it.

    The conditional update in `services/job_control.py` is what makes a cancel real: without it the row
    would say cancelled while this import carried on writing frames for another twenty minutes.
    """
    if not job_id:
        return
    from services.job_control import raise_if_canceled

    async with get_sessionmaker()() as db:
        await raise_if_canceled(db, ImportJob, uuid.UUID(str(job_id)), **fields)


async def _import_raw(spec: ImportSpec, job_id, root: Path) -> dict:
    from services.imports.raw_media import read_image_folder
    from services.ingest.reader_mcap import read_mcap
    from services.ingest.reader_video import read_video
    from services.ingest.run import ingest

    if spec.format == "images":
        frame_iter = read_image_folder(root)
        raw_uri, mcap_uri = spec.source_uri, None
    elif spec.format == "video":
        vid = next((p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS), None)
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
        project_id=uuid.UUID(spec.project_id) if spec.project_id else None,
    )
    counts = {"sessions": 1, "frames": result["n_frames"], "objects": 0,
              "faces_blurred": result.get("pii", {}).get("n_faces", 0)}
    await _bump_job(job_id, status="done", progress=1.0, counts=counts, session_id=uuid.UUID(result["session_id"]))
    return {"session_id": result["session_id"], "counts": counts}


class ImportNotResumable(RuntimeError):
    """A resumed import found a source that no longer lists the frames the first run saw."""


def _fingerprint(frames) -> str:
    """Identity of a parsed source: the frames it lists, in the order it lists them.

    A resumed run recognises the frames an earlier run stored by camera and timestamp, and a frame with
    no timestamp of its own gets one from its position. That only works if the position means the same
    thing both times, so the order is part of what is compared.
    """
    import hashlib

    h = hashlib.sha1(usedforsecurity=False)
    for fr in frames:
        h.update(f"{fr.cam_id}|{fr.image_ref}|{fr.ts_ns}\n".encode())
    return h.hexdigest()


def _prepare_frame(fr, root: Path, key_prefix: str, ts: int, anon, max_width: int | None, store):
    """Decode, downscale, redact, encode and store one frame. Blocking; run in a thread.

    Every step here is CPU or network bound. Run on the event loop, as it was, each frame of an import
    stalled every other request the API was serving for as long as a decode, two detector passes and an
    upload took.
    """
    img = _load_image(fr.image_ref, root)
    if img is None:
        return None
    # Downscale very wide frames to the label resolution (keeps PII fast + disk bounded); the caller scales
    # the boxes by the SAME factor so they still line up.
    scale = 1.0
    if max_width and img.shape[1] > max_width:
        scale = max_width / img.shape[1]
        img = cv2.resize(img, (max_width, int(round(img.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    pii = anon.anonymize(img) if anon else None
    if pii is not None and pii.withheld:
        # Plate-like text survived both blur passes (services/anonymize/text_regions.py): this frame would
        # leak, so it is never stored, rather than stored with a flag nobody reads.
        log.warning("frame.withheld_redaction_failed", reason=pii.redaction_reason)
        return None
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return None
    h, w = img.shape[:2]
    img_uri = store.put_bytes(f"{key_prefix}/{fr.cam_id}/{ts}.jpg", buf.tobytes(), "image/jpeg")
    return img_uri, w, h, scale, pii


async def _load_resume(job_id) -> dict:
    if not job_id:
        return {}
    async with get_sessionmaker()() as db:
        j = await db.get(ImportJob, uuid.UUID(str(job_id)))
        return dict((j.spec or {}).get("resume") or {}) if j else {}


async def _save_resume(job_id, resume: dict) -> None:
    if not job_id:
        return
    async with get_sessionmaker()() as db:
        j = await db.get(ImportJob, uuid.UUID(str(job_id)))
        if j is not None:
            j.spec = {**(j.spec or {}), "resume": resume}
            await db.commit()


async def _import_nuscenes_lidar(root: Path, session_id: uuid.UUID) -> int:
    """Store the LIDAR_TOP keyframe sweeps of a standard nuScenes release beside its camera frames.

    The camera half of a nuScenes import brought the boxes and left the point clouds on disk, so a
    dataset whose whole point is its 3D sensor arrived as pictures. Each annotated sweep is stored in the
    ego frame at its sample's timestamp, which is the timestamp its camera frames were imported at, so
    the cloud and the frames line up without any matching step. Sweeps whose files are absent are
    skipped (a download is often a subset), a directory that is not a standard release has none, and a
    resumed import does not store a sweep twice. Returns how many clouds were written.
    """
    import dataclasses

    from sqlalchemy import select

    from db.models import PointCloud
    from services.lidar.ingest import iter_nuscenes_lidar_clouds
    from services.lidar.ingest.store import store_cloud

    async with get_sessionmaker()() as db:
        have = {int(t) for (t,) in (await db.execute(
            select(PointCloud.ts_ns).where(PointCloud.session_id == session_id))).all()}

    def _read() -> list:
        try:
            return list(iter_nuscenes_lidar_clouds(root))
        except Exception as exc:  # noqa: BLE001 - our own 2D export has no lidar tables; that is not an error
            log.info("import.nuscenes_no_lidar", reason=str(exc)[:160])
            return []

    written = 0
    for sweep, cloud in await asyncio.to_thread(_read):
        ts = int(sweep.sample_ts_ns or sweep.ts_ns)
        if ts in have or cloud.n == 0:
            continue
        await store_cloud(dataclasses.replace(cloud, ts_ns=ts), session_id, source="dataset")
        have.add(ts)
        written += 1
    if written:
        log.info("import.nuscenes_lidar", session_id=str(session_id), clouds=written)
    return written


async def import_dataset(spec: ImportSpec, job_id=None) -> dict:
    """Import a dataset. A job that an earlier run started is continued, not repeated.

    The first run records the session it created, its time base and a fingerprint of the frames the
    source listed. A later run of the same job writes into that session and skips every frame already
    there (frames and their objects commit together, so a stored frame is a complete one). If the source
    no longer lists the same frames the run refuses, because position is what gives an untimed frame its
    timestamp and a changed listing would store the same picture twice.
    """
    from sqlalchemy import func, select

    settings = get_settings()
    onto = get_ontology()
    store = get_object_store()
    store.ensure_bucket()
    anon = get_anonymizer() if settings.pii.enabled else None
    await _bump_job(job_id, status="running", progress=0.0)

    with TemporaryDirectory() as tmp:
        root = await asyncio.to_thread(_acquire_source, spec.source_uri, Path(tmp))

        if spec.format in RAW_FORMATS:
            return await _import_raw(spec, job_id, root)

        if spec.format not in ADAPTERS:
            raise ValueError(f"unknown import format: {spec.format} (choose from {ALL_FORMATS})")
        frames = await asyncio.to_thread(ADAPTERS[spec.format], root)
        total = max(1, len(frames))
        fingerprint = _fingerprint(frames)
        resume = await _load_resume(job_id)
        counts = {"sessions": 1, "frames": 0, "objects": 0, "unmapped": 0, "faces_blurred": 0}
        ts_lo, ts_hi = None, None
        have: set[tuple[str, int]] = set()

        async with get_sessionmaker()() as db:
            session_row = None
            if resume.get("session_id"):
                if resume.get("fingerprint") != fingerprint:
                    raise ImportNotResumable(
                        "this import was interrupted and its source no longer lists the same frames in "
                        f"the same order, so it cannot be continued safely. The partial session "
                        f"{resume['session_id']} was left as it is; start a new import")
                session_row = await db.get(DbSession, uuid.UUID(resume["session_id"]))
            if session_row is not None:
                session_id = session_row.session_id
                base = int(resume["base_ns"])
                have = {(c, int(t)) for c, t in (await db.execute(
                    select(Frame.cam_id, Frame.ts_ns).where(Frame.session_id == session_id))).all()}
                counts["frames"] = len(have)
                counts["objects"] = int((await db.execute(
                    select(func.count()).select_from(Object).join(Frame, Frame.frame_id == Object.frame_id)
                    .where(Frame.session_id == session_id))).scalar() or 0)
                if have:
                    ts_lo, ts_hi = min(t for _c, t in have), max(t for _c, t in have)
                log.info("import.resumed", job_id=str(job_id), session_id=str(session_id),
                         frames_already_stored=len(have), of=len(frames))
            else:
                session_id = uuid.uuid4()
                base = now_ns()
                from services.api.tenancy import default_project_id

                session_row = DbSession(
                    session_id=session_id, vehicle_id=spec.target_vehicle, start_ts_ns=base, end_ts_ns=base,
                    city=spec.city, route=spec.route or f"import:{spec.format}", sensors={},
                    raw_uri=spec.source_uri, ontology_version=onto.version,
                    project_id=uuid.UUID(spec.project_id) if spec.project_id
                    else await default_project_id(db),
                )
                db.add(session_row)
                # Committed before the first frame, with the job told where it is: a run that dies on
                # frame one must still leave behind something the next run can find and continue.
                await db.commit()
                await _save_resume(job_id, {"session_id": str(session_id), "base_ns": base,
                                            "fingerprint": fingerprint})

            # Tracks this import has made, by (camera, the source's track id). Rebuilt from the rows on a
            # resume, so a track that started before the interruption continues instead of splitting.
            tracks: dict[tuple[str, str], Track] = {}
            if have:
                seen = (await db.execute(
                    select(Frame.cam_id, Object.provenance["track_ref"].astext, Object.track_id)
                    .join(Frame, Frame.frame_id == Object.frame_id)
                    .where(Frame.session_id == session_id, Object.track_id.is_not(None))
                    .distinct())).all()
                for cam, ref, tid in seen:
                    if ref and (cam, ref) not in tracks:
                        existing = await db.get(Track, tid)
                        if existing is not None:
                            tracks[(cam, ref)] = existing
            stored_any = bool(have)
            try:
                for i, fr in enumerate(frames):
                    ts = fr.ts_ns or (base + i * 1000)
                    if (fr.cam_id, int(ts)) in have:
                        continue
                    prepared = await asyncio.to_thread(_prepare_frame, fr, root, f"frames/{session_id}", ts,
                                                       anon, settings.ingest.max_width, store)
                    if prepared is None:
                        continue
                    img_uri, w, h, scale, pii = prepared
                    ts_lo = ts if ts_lo is None else min(ts_lo, ts)
                    ts_hi = ts if ts_hi is None else max(ts_hi, ts)

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
                        # The source's own track identity becomes a track here. Nine adapters read it and
                        # nothing used it, so a tracked export from CVAT, Scale or nuScenes arrived as a pile of
                        # unrelated boxes. One track per camera, which is what a track means in this schema; the
                        # source id stays in provenance so a cross-camera link can be rebuilt.
                        track_id = None
                        if o.track_ref:
                            prov["track_ref"] = str(o.track_ref)
                            tkey = (fr.cam_id, str(o.track_ref))
                            track = tracks.get(tkey)
                            if track is None:
                                track = Track(track_id=uuid.uuid4(), session_id=session_id, class_id=cid,
                                              first_ts_ns=ts, last_ts_ns=ts, tracker_version="imported")
                                db.add(track)
                                await db.flush()
                                tracks[tkey] = track
                                counts["tracks"] = counts.get("tracks", 0) + 1
                            else:
                                track.first_ts_ns = min(track.first_ts_ns, ts)
                                track.last_ts_ns = max(track.last_ts_ns, ts)
                            track_id = track.track_id
                        cuboid = prov.get("cuboid_3d") if isinstance(prov.get("cuboid_3d"), dict) else None
                        # Mask round-trip: carry a lossless export's mask uri through, or materialize polygons
                        # (COCO/OpenLABEL) into a fresh mask blob, so segmentation is not silently dropped.
                        oid = uuid.uuid4()
                        mask_uri, mask_encoding = o.mask_uri, o.mask_encoding
                        if mask_uri is None and o.mask_polygons:
                            scaled = [[float(v) * scale for v in poly] for poly in o.mask_polygons]
                            payload = {"encoding": "polygon", "polygons": scaled, "height": h, "width": w}
                            mask_uri = await asyncio.to_thread(
                                store.put_bytes, f"masks/{session_id}/{frame_row.frame_id}/{oid}.json",
                                json.dumps(payload).encode(), "application/json")
                            mask_encoding = "polygon"
                        # Polyline geometry is scaled with the same factor as the bbox, so a linear feature does
                        # not drift off the image when the import resizes.
                        polyline = ([[float(px) * scale, float(py) * scale] for px, py in o.polyline]
                                    if o.polyline else None)
                        db.add(Object(object_id=oid, frame_id=frame_row.frame_id, class_id=cid,
                                      bbox=[float(x) * scale for x in o.bbox], conf=float(o.conf),
                                      source="imported", state="review", provenance=prov, attrs=o.attrs or {},
                                      mask_uri=mask_uri, mask_encoding=mask_encoding, rot_deg=o.rot_deg,
                                      keypoints=o.keypoints, polyline=polyline, track_id=track_id,
                                      # A 3D box the source carried (nuScenes, our own exports), in this
                                      # schema's ego-frame shape. It rode in provenance and was dropped.
                                      cuboid_3d=cuboid))
                        counts["objects"] += 1
                    counts["frames"] += 1

                    if i % 100 == 0:
                        await db.commit()
                        stored_any = True
                        # The progress write doubles as the cancel check, so a cancelled import stops here
                        # rather than at the end of a corpus somebody has already changed their mind about.
                        await _bump_job(job_id, progress=round(i / total, 3), counts=counts)
            except Exception:
                if not stored_any:
                    # Nothing reached the database, so the session this run created is an empty shell.
                    # Before sessions were committed up front it would have rolled back with the frames;
                    # remove it so a failed import still leaves nothing behind.
                    await db.rollback()
                    await db.delete(session_row)
                    await db.commit()
                    await _save_resume(job_id, {})
                raise

            session_row.start_ts_ns = ts_lo or base
            session_row.end_ts_ns = ts_hi or base
            await db.commit()

        if spec.format == "nuscenes" and (spec.options or {}).get("lidar", True):
            counts["point_clouds"] = await _import_nuscenes_lidar(root, session_id)

    await _bump_job(job_id, status="done", progress=1.0, counts=counts, session_id=session_id)
    result = {"session_id": str(session_id), "format": spec.format, "counts": counts}
    log.info("import.done", **{k: result[k] for k in ("session_id", "format")}, **counts)
    return result


async def run_import_guarded(spec: ImportSpec, job_id) -> None:
    """Background entrypoint: run the import and record failures on the job row (so a crashed import is
    visible as status=error rather than a silent hang).

    Runs holding the job's lock (services/job_lock.py). That is what lets the next process tell this
    import apart from one whose process died, and what stops two processes continuing the same job.
    """
    from services.job_control import JobCanceled
    from services.job_lock import job_lock

    async with job_lock(job_id) as mine:
        if not mine:
            log.info("import.already_running_elsewhere", job_id=str(job_id))
            return
        try:
            await import_dataset(spec, job_id)
        except JobCanceled:
            # Not a failure. The row already says `canceled`, and writing `error` over it would turn
            # somebody's deliberate decision into a fault report.
            log.info("import.canceled", job_id=str(job_id))
        except Exception as exc:  # noqa: BLE001
            log.error("import.failed", job_id=str(job_id), error=str(exc))
            try:
                await _bump_job(job_id, status="error", error=str(exc))
            except JobCanceled:
                # Cancelled while this failure was being recorded. The cancel is the newer fact and wins.
                log.info("import.canceled_during_failure", job_id=str(job_id))


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
