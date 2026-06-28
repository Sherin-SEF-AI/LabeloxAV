"""BDD100K detection adapter: a single `bdd_det.json` holding a list of per-image entries

    {"name": <image>, "labels": [{"id", "category", "box2d": {"x1","y1","x2","y2"}, "attributes": {...}}]}

The ontology class_name is used verbatim as the BDD category. Typed attributes ride in the per-label
`attributes` block; the fields BDD cannot natively carry (track id, conf, state, source, full
provenance) are preserved in a `labelox` extension on each label so our own export is a lossless
round-trip, while the Parquet sidecar remains the source of truth.
"""

from __future__ import annotations

import json
from pathlib import Path

from services.autolabel.ontology import Ontology
from services.export.records import ExportRecord


def _name(r: ExportRecord) -> str:
    # Stable per-frame image name: prefer the real filename, fall back to cam + timestamp.
    tail = r.img_uri.split("/")[-1] if r.img_uri else ""
    return tail or f"{r.cam_id}_{r.ts_ns}.jpg"


def write_bdd(records: list[ExportRecord], onto: Ontology, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    by_frame: dict[str, list[ExportRecord]] = {}
    names: dict[str, str] = {}
    order: list[str] = []
    for r in records:
        fkey = str(r.frame_id)
        if fkey not in by_frame:
            order.append(fkey)
        by_frame.setdefault(fkey, []).append(r)
        names[fkey] = _name(r)

    label_id = 0
    entries: list[dict] = []
    for fkey in order:
        labels = []
        for r in by_frame[fkey]:
            label_id += 1
            x1, y1, x2, y2 = r.bbox
            labels.append(
                {
                    "id": label_id,
                    "category": r.class_name,
                    "box2d": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "attributes": r.attrs,
                    # extension block: what BDD cannot natively express, kept for fidelity
                    "labelox": {
                        "object_id": str(r.object_id),
                        "track_id": str(r.track_id) if r.track_id else None,
                        "conf": r.conf,
                        "state": r.state,
                        "source": r.source,
                        "provenance": r.provenance,
                    },
                }
            )
        entries.append({"name": names[fkey], "labels": labels})

    path = out_dir / "bdd_det.json"
    path.write_text(json.dumps(entries, indent=2))
    return path
