"""Dataset versioning, curation slice, seal/commit, and the export driver.

A dataset is a query, not a dump: a SliceSpec selects objects, the selection is sealed into an
immutable content-addressed dataset_commit (P0 versioning; the lakeFS seam), and the adapters
render it. Every legacy export carries the Parquet provenance sidecar so the full-fidelity object
is one join from any exported file.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from uuid import UUID

import click
from pydantic import BaseModel, Field
from sqlalchemy import select

from core.config import get_settings
from core.logging import get_logger, setup_logging
from core.storage import get_object_store
from db.models import DatasetCommit, Frame, Object, ObjectRelationship
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology
from services.export.adapter_bdd import write_bdd
from services.export.adapter_coco import write_coco
from services.export.adapter_cvat import write_cvat
from services.export.adapter_kitti import write_kitti
from services.export.adapter_labelstudio import write_labelstudio
from services.export.adapter_mapillary import write_mapillary
from services.export.adapter_nuscenes import write_nuscenes
from services.export.adapter_openlabel import write_openlabel
from services.export.adapter_parquet import write_parquet
from services.export.adapter_pascalvoc import write_pascalvoc
from services.export.adapter_scene import (
    write_drivable,
    write_hdmap,
    write_lanes,
    write_masks,
    write_panoptic,
)
from services.export.adapter_yolo import write_yolo
from services.export.records import ExportRecord

log = get_logger("export")


class SliceSpec(BaseModel):
    name: str = "dataset"
    states: list[str] | None = None        # e.g. ["accepted", "auto_accept"]
    class_names: list[str] | None = None
    cities: list[str] | None = None
    vehicle_ids: list[str] | None = None  # export a whole fleet (all its sessions) in one commit
    min_conf: float | None = None
    has_mask: bool | None = None
    session_id: str | None = None
    limit: int | None = None
    formats: list[str] = Field(default_factory=lambda: ["coco", "parquet"])


async def fetch_records(spec: SliceSpec) -> list[ExportRecord]:
    onto = get_ontology()
    maker = get_sessionmaker()
    async with maker() as db:
        stmt = (
            select(Object, Frame, DbSession)
            .join(Frame, Object.frame_id == Frame.frame_id)
            .join(DbSession, Frame.session_id == DbSession.session_id)
            .order_by(Frame.ts_ns, Object.object_id)
        )
        if spec.states:
            stmt = stmt.where(Object.state.in_(spec.states))
        if spec.min_conf is not None:
            stmt = stmt.where(Object.conf >= spec.min_conf)
        if spec.has_mask is True:
            stmt = stmt.where(Object.mask_uri.isnot(None))
        if spec.cities:
            stmt = stmt.where(DbSession.city.in_(spec.cities))
        if spec.vehicle_ids:
            stmt = stmt.where(DbSession.vehicle_id.in_(spec.vehicle_ids))
        if spec.session_id:
            stmt = stmt.where(DbSession.session_id == UUID(spec.session_id))
        if spec.class_names:
            ids = [onto.by_name(n).id for n in spec.class_names]
            stmt = stmt.where(Object.class_id.in_(ids))
        if spec.limit:
            stmt = stmt.limit(spec.limit)

        rows = (await db.execute(stmt)).all()
        # attach each object's outgoing relationships (rider_of, towed_by, member_of, ...) for export.
        # Batch the id list: a whole-fleet export has hundreds of thousands of objects, and a single IN clause
        # over all of them blows past Postgres's 65535-parameter limit (the query simply fails). Chunk it.
        rel_map: dict[str, list] = {}
        oids = [o.object_id for o, _, _ in rows]
        for i in range(0, len(oids), 10000):
            rel_rows = (await db.execute(select(ObjectRelationship).where(
                ObjectRelationship.from_object_id.in_(oids[i:i + 10000])))).scalars().all()
            for r in rel_rows:
                rel_map.setdefault(str(r.from_object_id), []).append(
                    {"to_object_id": str(r.to_object_id), "kind": r.kind})

    records: list[ExportRecord] = []
    for obj, frame, sess in rows:
        records.append(
            ExportRecord(
                object_id=obj.object_id,
                frame_id=obj.frame_id,
                session_id=frame.session_id,
                ts_ns=frame.ts_ns,
                cam_id=frame.cam_id,
                img_uri=frame.img_uri,
                width=frame.width,
                height=frame.height,
                vehicle_id=sess.vehicle_id,
                city=sess.city,
                class_id=obj.class_id,
                class_name=onto.by_id(obj.class_id).name,
                bbox=list(obj.bbox),
                conf=obj.conf,
                quality_score=obj.quality_score,
                state=obj.state,
                source=obj.source,
                mask_uri=obj.mask_uri,
                mask_encoding=obj.mask_encoding,
                track_id=obj.track_id,
                attrs=obj.attrs or {},
                provenance=obj.provenance or {},
                cuboid_3d=obj.cuboid_3d,
                rot_deg=obj.rot_deg or 0.0,
                keypoints=obj.keypoints,
                polyline=obj.polyline,
                relationships=rel_map.get(str(obj.object_id), []),
                sign_type=obj.sign_type,
                sign_category=obj.sign_category,
                ocr_text=obj.ocr_text,
                ocr_lang=obj.ocr_lang,
                ocr_conf=obj.ocr_conf,
            )
        )
    return records


def _fp_dicts(records: list[ExportRecord]) -> list[dict]:
    """Project export records onto the fields the content fingerprint hashes (class, geometry, mask, state)."""
    return [{"object_id": str(r.object_id), "class_id": r.class_id, "bbox": r.bbox,
             "mask_uri": r.mask_uri, "state": r.state, "version": getattr(r, "version", None)}
            for r in records]


def seal_content_fingerprint(spec: SliceSpec, records: list[ExportRecord], ontology_version: str) -> str:
    """Content hash of a release (class/geometry/state), so a mutated annotation yields a distinct id."""
    from services.release.fingerprint import content_fingerprint

    return content_fingerprint(_fp_dicts(records), spec.model_dump(), ontology_version)


def seal_commit_id(spec: SliceSpec, records: list[ExportRecord], ontology_version: str) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(spec.model_dump(), sort_keys=True).encode())
    h.update(ontology_version.encode())
    for oid in sorted(str(r.object_id) for r in records):
        h.update(oid.encode())
    return f"lbx-{h.hexdigest()[:16]}"


def _upload_dir(store, prefix: str, root: Path) -> dict[str, str]:
    uris: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            uris[rel] = store.put_file(f"{prefix}/{rel}", p)
    return uris


class DpdpaRefusal(Exception):
    """Raised when the unified DPDPA pre-sale gate refuses an export (un-redacted face, plate, or speech)."""


class UnknownExportFormat(ValueError):
    """Raised when an export requests a format no adapter implements. The old dispatch chain ignored these
    silently, so the caller got a 200 and a commit that claimed a format the archive never contained."""


# One entry per supported target. The adapters have slightly different signatures, so each is wrapped to the
# uniform (records, onto, store, out_dir) shape; the wrapper owns the per-format subdirectory name. Adding a
# format means adding it here, which is also what makes it accepted by validate_formats.
_WRITERS = {
    "coco": lambda rec, onto, store, d: write_coco(rec, onto, store, d / "coco"),
    "yolo": lambda rec, onto, store, d: write_yolo(rec, onto, d),
    "openlabel": lambda rec, onto, store, d: write_openlabel(rec, onto, store, d / "openlabel"),
    "nuscenes": lambda rec, onto, store, d: write_nuscenes(rec, onto, d / "nuscenes"),
    "kitti": lambda rec, onto, store, d: write_kitti(rec, onto, d / "kitti"),
    "bdd": lambda rec, onto, store, d: write_bdd(rec, onto, d / "bdd"),
    "cvat": lambda rec, onto, store, d: write_cvat(rec, onto, store, d / "cvat"),
    "labelstudio": lambda rec, onto, store, d: write_labelstudio(rec, onto, store, d / "labelstudio"),
    "pascalvoc": lambda rec, onto, store, d: write_pascalvoc(rec, onto, d / "pascalvoc"),
    "mapillary": lambda rec, onto, store, d: write_mapillary(rec, onto, d / "mapillary"),
    # Masks are object-derived like the rest, so this one fits the same signature.
    "masks": lambda rec, onto, store, d: write_masks(rec, onto, store, d / "masks"),
}

# Scene-level targets. Separate registry because these are not derived from ExportRecord at all: a lane, a
# drivable surface and a map element are not Objects, which is exactly why none of them could ever leave the
# system. They take the frame ids of the slice instead, and they are async because they read their own rows.
_SCENE_WRITERS = {
    "lanes": lambda fids, store, d, commit: write_lanes(fids, d / "lanes"),
    "drivable": lambda fids, store, d, commit: write_drivable(fids, store, d / "drivable"),
    "hdmap": lambda fids, store, d, commit: write_hdmap(d / "hdmap", commit),
    "panoptic": lambda fids, store, d, commit: write_panoptic(fids, store, d / "panoptic"),
}

# Parquet is always written (lossless provenance) and so is accepted but never dispatched.
SUPPORTED_EXPORT_FORMATS = frozenset(_WRITERS) | frozenset(_SCENE_WRITERS) | {"parquet"}


def validate_formats(formats: list[str]) -> None:
    """Refuse an export naming a format no adapter implements, before any work is done. Fail loud: a silently
    dropped format ships a dataset that claims contents it does not have."""
    unknown = [f for f in formats if f not in SUPPORTED_EXPORT_FORMATS]
    if unknown:
        raise UnknownExportFormat(
            f"unsupported export format(s): {sorted(unknown)}; "
            f"supported: {sorted(SUPPORTED_EXPORT_FORMATS)}"
        )


def _requested(formats: list[str]) -> list[str]:
    """The dispatchable formats in request order, de-duplicated. Parquet is filtered out (always written)."""
    seen: list[str] = []
    for f in formats:
        if f in _WRITERS and f not in seen:
            seen.append(f)
    return seen


def _requested_scene(formats: list[str]) -> list[str]:
    """The scene-level targets in request order, de-duplicated."""
    seen: list[str] = []
    for f in formats:
        if f in _SCENE_WRITERS and f not in seen:
            seen.append(f)
    return seen


async def _dpdpa_pre_sale_gate(records) -> None:
    """The single fail-closed compliance gate in the export path. Refuses, does not warn."""
    from collections import defaultdict
    from uuid import UUID

    from services.anonymize.compliance import dpdpa_export_gate
    by_session: dict = defaultdict(set)
    for r in records:
        if r.session_id and r.frame_id:
            by_session[str(r.session_id)].add(r.frame_id)
    refused = []
    for sid, fids in by_session.items():
        verdict = await dpdpa_export_gate(UUID(sid), list(fids))
        if not verdict["pass"]:
            refused.append({"session_id": sid, "blockers": verdict["blockers"]})
    if refused:
        raise DpdpaRefusal({"detail": "DPDPA pre-sale gate refused export", "refused": refused})


async def export_dataset(spec: SliceSpec, out_root: Path | None = None) -> dict:
    validate_formats(spec.formats)   # refuse an unsupported target before any work is done
    settings = get_settings()
    onto = get_ontology()
    store = get_object_store()
    store.ensure_bucket()

    records = await fetch_records(spec)
    await _dpdpa_pre_sale_gate(records)   # fail-closed: refuse any clip with un-redacted face, plate, or speech
    commit_id = seal_commit_id(spec, records, onto.version)
    out_dir = (out_root or settings.scratch_path() / "exports") / spec.name / commit_id
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    # The Parquet sidecar is always emitted (lossless provenance), regardless of requested formats.
    written.append(write_parquet(records, out_dir / "parquet"))
    # Dispatch through the writer registry rather than a chain of ifs. The chain silently ignored any format
    # it had no branch for, so a request for an unsupported target returned 200, wrote only the Parquet
    # sidecar, and then recorded the never-written format on the commit as if it had been delivered. The
    # registry is validated up front (validate_formats) so an unsupported target is refused, and only the
    # formats actually written are recorded below.
    for fmt in _requested(spec.formats):
        written.append(_WRITERS[fmt](records, onto, store, out_dir))

    # Scene-level targets read their own rows for the frames in this slice, so they cannot be driven from
    # the object records the object adapters share.
    frame_ids = sorted({str(r.frame_id) for r in records if r.frame_id})
    for fmt in _requested_scene(spec.formats):
        written.append(await _SCENE_WRITERS[fmt](frame_ids, store, out_dir, commit_id))

    delivered = ["parquet", *_requested(spec.formats), *_requested_scene(spec.formats)]

    prefix = f"datasets/{spec.name}/{commit_id}"
    export_uris = _upload_dir(store, prefix, out_dir)

    maker = get_sessionmaker()
    async with maker() as db:
        existing = await db.get(DatasetCommit, commit_id)
        if existing is None:
            from services.export.snapshots import resolve_parent
            parent_id = await resolve_parent(db, spec.name, commit_id)   # chain lineage to the prior snapshot
            db.add(
                DatasetCommit(
                    commit_id=commit_id,
                    parent_id=parent_id,
                    slice_spec=spec.model_dump(),
                    object_count=len(records),
                    ontology_version=onto.version,
                    export_uris={k: v for k, v in list(export_uris.items())[:50]},
                    content_fingerprint=seal_content_fingerprint(spec, records, onto.version),
                    notes=f"slice '{spec.name}' formats={delivered}",   # what was written, not what was asked for
                )
            )
            await db.commit()

    result = {
        "commit_id": commit_id,
        "object_count": len(records),
        "ontology_version": onto.version,
        "out_dir": str(out_dir),
        "formats": spec.formats,
        "dataset_prefix": store.uri(prefix),
    }

    # Meter the delivery. Here rather than in the API router because every path that produces a dataset
    # ships one: the buyer agent, the ops agent, the resumable exporter and the CLI all call this function,
    # and metering at the router would have missed four of the five. Idempotent on the commit id, so a
    # re-export of the same content is recorded once and charged once.
    #
    # Marked uncertified: a certificate needs a sealed gold set and a scored evaluation run, and an export
    # is usually shipped before it has been evaluated. certify_delivery attaches one later. Metering it as
    # measured because a certificate might arrive would be the dishonest default.
    async with maker() as db:
        from services.billing.meter import record_delivery

        await record_delivery(db, kind="export", subject_id=commit_id, quantity=float(len(records)),
                              detail={"slice": spec.name, "formats": delivered,
                                      "ontology_version": onto.version})

    log.info("export.done", **result)
    from services.integrations.webhooks import emit

    await emit("export.completed", {k: result[k] for k in
                                    ("commit_id", "object_count", "formats", "dataset_prefix")})
    return result


def reimport_sanity(out_dir: Path) -> dict:
    """Read the exported COCO + Parquet back and confirm counts agree. Returns a report."""
    import pyarrow.parquet as pq

    report: dict = {"ok": True}
    parquet = out_dir / "parquet" / "objects.parquet"
    n_parquet = pq.read_table(parquet).num_rows
    report["parquet_rows"] = n_parquet

    coco_path = out_dir / "coco" / "annotations.json"
    if coco_path.exists():
        coco = json.loads(coco_path.read_text())
        report["coco_annotations"] = len(coco["annotations"])
        report["coco_images"] = len(coco["images"])
        report["coco_categories"] = len(coco["categories"])
        report["ok"] = report["ok"] and (len(coco["annotations"]) == n_parquet)

    ol_path = out_dir / "openlabel" / "openlabel.json"
    if ol_path.exists():
        ol = json.loads(ol_path.read_text())["openlabel"]
        ann = sum(len(f["objects"]) for f in ol["frames"].values())
        report["openlabel_objects"] = len(ol["objects"])
        report["openlabel_annotations"] = ann
        report["ok"] = report["ok"] and (ann == n_parquet)

    nusc_ann = out_dir / "nuscenes" / "sample_annotation.json"
    if nusc_ann.exists():
        report["nuscenes_annotations"] = len(json.loads(nusc_ann.read_text()))
        report["ok"] = report["ok"] and (report["nuscenes_annotations"] == n_parquet)
    return report


@click.command()
@click.option("--name", default="dataset")
@click.option("--state", "states", multiple=True, help="filter by object state (repeatable)")
@click.option("--klass", "class_names", multiple=True, help="filter by ontology class name (repeatable)")
@click.option("--city", "cities", multiple=True)
@click.option("--min-conf", type=float, default=None)
@click.option("--has-mask", is_flag=True, default=False)
@click.option("--session", "session_id", default=None)
@click.option("--formats", default="coco,parquet", help="comma list: coco,yolo,parquet")
@click.option("--limit", type=int, default=None)
def main(name, states, class_names, cities, min_conf, has_mask, session_id, formats, limit) -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    spec = SliceSpec(
        name=name,
        states=list(states) or None,
        class_names=list(class_names) or None,
        cities=list(cities) or None,
        min_conf=min_conf,
        has_mask=has_mask or None,
        session_id=session_id,
        limit=limit,
        formats=[f.strip() for f in formats.split(",") if f.strip()],
    )
    result = asyncio.run(export_dataset(spec))
    report = reimport_sanity(Path(result["out_dir"]))
    result["reimport_sanity"] = report
    click.echo(result)


if __name__ == "__main__":
    main()
