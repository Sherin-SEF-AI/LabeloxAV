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
from db.models import DatasetCommit, Frame, Object
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology
from services.export.adapter_coco import write_coco
from services.export.adapter_nuscenes import write_nuscenes
from services.export.adapter_openlabel import write_openlabel
from services.export.adapter_parquet import write_parquet
from services.export.adapter_yolo import write_yolo
from services.export.records import ExportRecord

log = get_logger("export")


class SliceSpec(BaseModel):
    name: str = "dataset"
    states: list[str] | None = None        # e.g. ["accepted", "auto_accept"]
    class_names: list[str] | None = None
    cities: list[str] | None = None
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
        if spec.session_id:
            stmt = stmt.where(DbSession.session_id == UUID(spec.session_id))
        if spec.class_names:
            ids = [onto.by_name(n).id for n in spec.class_names]
            stmt = stmt.where(Object.class_id.in_(ids))
        if spec.limit:
            stmt = stmt.limit(spec.limit)

        rows = (await db.execute(stmt)).all()

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
                state=obj.state,
                source=obj.source,
                mask_uri=obj.mask_uri,
                mask_encoding=obj.mask_encoding,
                track_id=obj.track_id,
                attrs=obj.attrs or {},
                provenance=obj.provenance or {},
            )
        )
    return records


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


async def export_dataset(spec: SliceSpec, out_root: Path | None = None) -> dict:
    settings = get_settings()
    onto = get_ontology()
    store = get_object_store()
    store.ensure_bucket()

    records = await fetch_records(spec)
    commit_id = seal_commit_id(spec, records, onto.version)
    out_dir = (out_root or settings.scratch_path() / "exports") / spec.name / commit_id
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    # The Parquet sidecar is always emitted (lossless provenance), regardless of requested formats.
    written.append(write_parquet(records, out_dir / "parquet"))
    if "coco" in spec.formats:
        written.append(write_coco(records, onto, store, out_dir / "coco"))
    if "yolo" in spec.formats:
        written.append(write_yolo(records, onto, out_dir))
    if "openlabel" in spec.formats:
        written.append(write_openlabel(records, onto, store, out_dir / "openlabel"))
    if "nuscenes" in spec.formats:
        written.append(write_nuscenes(records, onto, out_dir / "nuscenes"))

    prefix = f"datasets/{spec.name}/{commit_id}"
    export_uris = _upload_dir(store, prefix, out_dir)

    maker = get_sessionmaker()
    async with maker() as db:
        existing = await db.get(DatasetCommit, commit_id)
        if existing is None:
            db.add(
                DatasetCommit(
                    commit_id=commit_id,
                    parent_id=None,
                    slice_spec=spec.model_dump(),
                    object_count=len(records),
                    ontology_version=onto.version,
                    export_uris={k: v for k, v in list(export_uris.items())[:50]},
                    notes=f"slice '{spec.name}' formats={spec.formats}",
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
    log.info("export.done", **result)
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
