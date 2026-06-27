"""Gate B, part (a): a frozen, content-addressed gold set sealed from the fleet's OWN human-verified
frames (Object.source=="human" AND state=="accepted"). This is the distribution the dataset of record
lives in (NOT IDD, which is a different rig). Sealing freezes the exact object list and materializes a
YOLO val split the eval harness points at.

Seam: gold objects carry no guaranteed track_id, so MOTA/IDF1 are documented-not-built here.

    python -m services.training.gold --name fleet-v1 --city BLR
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from uuid import UUID

import click
import cv2
import numpy as np
import yaml
from pydantic import BaseModel, Field
from sqlalchemy import select

from core.config import get_settings
from core.logging import get_logger, setup_logging
from core.storage import get_object_store
from db.models import Frame, GoldSet, Object
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology

log = get_logger("gold")


class GoldSpec(BaseModel):
    name: str = "gold"
    states: list[str] = Field(default_factory=lambda: ["accepted"])
    class_names: list[str] | None = None
    cities: list[str] | None = None
    session_id: str | None = None
    limit: int | None = None


async def _fetch_gold(spec: GoldSpec) -> list[dict]:
    onto = get_ontology()
    maker = get_sessionmaker()
    async with maker() as db:
        stmt = (
            select(Object, Frame.frame_id, Frame.img_uri, Frame.width, Frame.height)
            .join(Frame, Object.frame_id == Frame.frame_id)
            .join(DbSession, Frame.session_id == DbSession.session_id)
            .where(Object.source == "human", Object.state.in_(spec.states))
        )
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
    return [
        {"object_id": str(o.object_id), "frame_id": str(fid), "img_uri": uri, "w": w, "h": h,
         "class_id": o.class_id, "bbox": list(o.bbox)}
        for o, fid, uri, w, h in rows
    ]


def _gold_id(spec: GoldSpec, object_ids: list[str], ontology_version: str) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(spec.model_dump(), sort_keys=True).encode())
    h.update(ontology_version.encode())
    for oid in sorted(object_ids):
        h.update(oid.encode())
    return f"gold-{h.hexdigest()[:16]}"


def _materialize(gold_id: str, objs: list[dict], onto) -> tuple[str, int]:
    """Write a frozen YOLO val split for the eval harness. Returns (data_yaml_path, n_frames)."""
    store = get_object_store()
    out = get_settings().scratch_path() / "gold" / gold_id
    for sub in ("images/val", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    present = sorted({o["class_id"] for o in objs})
    idx_of = {cid: i for i, cid in enumerate(present)}
    names = {i: onto.by_id(cid).name for cid, i in idx_of.items()}

    by_frame: dict[str, list[dict]] = {}
    for o in objs:
        by_frame.setdefault(o["frame_id"], []).append(o)

    for fid, group in by_frame.items():
        first = group[0]
        try:
            buf = np.frombuffer(store.get_bytes(first["img_uri"]), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                continue
        except Exception:
            continue
        cv2.imwrite(str(out / f"images/val/{fid}.jpg"), img)
        lines = []
        for o in group:
            x1, y1, x2, y2 = o["bbox"]
            w, h = max(1, o["w"]), max(1, o["h"])
            lines.append(f"{idx_of[o['class_id']]} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
        (out / f"labels/val/{fid}.txt").write_text("\n".join(lines) + "\n")

    n_frames = len(list((out / "images/val").glob("*.jpg")))
    data_yaml = out / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(
        {"path": str(out), "train": "images/val", "val": "images/val", "nc": len(names), "names": names},
        sort_keys=False,
    ))
    return str(data_yaml), n_frames


def _materialize_aligned(gold_id: str, objs: list[dict], onto, names_list: list[str]) -> tuple[str, int]:
    """Like _materialize, but the label class indices follow the MODEL's class order (names_list), so
    evaluate()'s index-based matching is correct. Objects whose ontology class the model does not know
    are dropped (uncomparable). This is what makes the M9 numbers real rather than a vocab mismatch."""
    store = get_object_store()
    out = get_settings().scratch_path() / "gold" / gold_id / "aligned"
    for sub in ("images/val", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    name_to_idx = {n: i for i, n in enumerate(names_list)}

    by_frame: dict[str, list[dict]] = {}
    for o in objs:
        by_frame.setdefault(o["frame_id"], []).append(o)

    for fid, group in by_frame.items():
        rows = [o for o in group if onto.by_id(o["class_id"]).name in name_to_idx]
        if not rows:
            continue
        try:
            buf = np.frombuffer(store.get_bytes(rows[0]["img_uri"]), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                continue
        except Exception:
            continue
        cv2.imwrite(str(out / f"images/val/{fid}.jpg"), img)
        lines = []
        for o in rows:
            x1, y1, x2, y2 = o["bbox"]
            w, h = max(1, o["w"]), max(1, o["h"])
            idx = name_to_idx[onto.by_id(o["class_id"]).name]
            lines.append(f"{idx} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
        (out / f"labels/val/{fid}.txt").write_text("\n".join(lines) + "\n")

    n_frames = len(list((out / "images/val").glob("*.jpg")))
    names = {i: n for i, n in enumerate(names_list)}
    data_yaml = out / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(
        {"path": str(out), "train": "images/val", "val": "images/val", "nc": len(names), "names": names},
        sort_keys=False,
    ))
    return str(data_yaml), n_frames


async def materialize_for_model(gold_id: str, model_names) -> str:
    """Re-materialize the sealed gold objects into a val split aligned to a model's class order."""
    names_list = (
        [model_names[i] for i in range(len(model_names))] if isinstance(model_names, dict) else list(model_names)
    )
    async with get_sessionmaker()() as db:
        g = await db.get(GoldSet, gold_id)
    if g is None:
        raise RuntimeError(f"gold set {gold_id} not found")
    objs = await _fetch_gold(GoldSpec(**g.spec))
    objs = [o for o in objs if o["object_id"] in set(g.object_ids)]
    data_yaml, _ = _materialize_aligned(gold_id, objs, get_ontology(), names_list)
    return data_yaml


async def seal_gold(spec: GoldSpec) -> dict:
    onto = get_ontology()
    objs = await _fetch_gold(spec)
    if not objs:
        raise RuntimeError("no human-accepted objects match this gold spec")
    object_ids = [o["object_id"] for o in objs]
    gold_id = _gold_id(spec, object_ids, onto.version)
    data_yaml, n_frames = _materialize(gold_id, objs, onto)

    maker = get_sessionmaker()
    async with maker() as db:
        existing = await db.get(GoldSet, gold_id)
        if existing is None:
            db.add(GoldSet(
                gold_id=gold_id, name=spec.name, spec=spec.model_dump(), object_ids=object_ids,
                n_objects=len(object_ids), n_frames=n_frames, ontology_version=onto.version,
                data_yaml_uri=data_yaml, notes=f"sealed from {len(object_ids)} human-accepted objects",
            ))
            await db.commit()

    result = {"gold_id": gold_id, "name": spec.name, "n_objects": len(object_ids),
              "n_frames": n_frames, "data_yaml": data_yaml, "ontology_version": onto.version}
    log.info("gold.sealed", **{k: result[k] for k in ("gold_id", "n_objects", "n_frames")})
    return result


async def gold_data_yaml(gold_id: str) -> str:
    async with get_sessionmaker()() as db:
        g = await db.get(GoldSet, gold_id)
    if g is None or not g.data_yaml_uri or not Path(g.data_yaml_uri).exists():
        # re-materialize from the frozen object_ids if the local split is gone
        if g is None:
            raise RuntimeError(f"gold set {gold_id} not found")
        objs = await _fetch_gold(GoldSpec(**g.spec))
        objs = [o for o in objs if o["object_id"] in set(g.object_ids)]
        data_yaml, _ = _materialize(gold_id, objs, get_ontology())
        return data_yaml
    return g.data_yaml_uri


@click.command()
@click.option("--name", default="fleet-v1")
@click.option("--city", "cities", multiple=True)
@click.option("--session", "session_id", default=None)
@click.option("--klass", "class_names", multiple=True)
@click.option("--limit", type=int, default=None)
def main(name, cities, session_id, class_names, limit) -> None:
    setup_logging(get_settings().log_level)
    spec = GoldSpec(name=name, cities=list(cities) or None, session_id=session_id,
                    class_names=list(class_names) or None, limit=limit)
    click.echo(asyncio.run(seal_gold(spec)))


if __name__ == "__main__":
    main()
