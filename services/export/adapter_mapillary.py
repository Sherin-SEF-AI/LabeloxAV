"""Mapillary Vistas (v2.0) polygon-JSON adapter: one `<stem>.json` per frame carrying {width, height,
objects:[{label, polygon}]}. Closes a round-trip hole: Mapillary was importable
(services/imports/adapter_mapillary.py) but not exportable.

The export is the inverse of the import mapping: the LabeloxAV ontology name is mapped back to the
hierarchical Mapillary label ("sedan" -> "object--vehicle--car") using the importer's own table, so a slice
imported from Vistas and exported again round-trips its labels rather than emitting our internal names. A
class with no Mapillary equivalent is emitted under the closest generic instance label, recorded explicitly
in unmapped.json so the loss is visible rather than silent.

Geometry: where a real mask polygon exists we would prefer it, but masks live in object storage and are not
resident on the ExportRecord, so the box is emitted as its 4-point polygon (a valid Vistas polygon). The
Parquet sidecar remains the lossless record (Principle 10).
"""

from __future__ import annotations

import json
from pathlib import Path

from services.autolabel.ontology import Ontology
from services.export.adapter_pascalvoc import unique_stems
from services.export.records import ExportRecord
from services.imports.adapter_mapillary import MAPILLARY_TO_ONTOLOGY

# Invert the importer's table. Several Mapillary labels map onto one ontology class (car/truck/bus all have
# distinct entries, but rider has three), so the first Mapillary label wins for a stable, deterministic
# inverse; the alternatives are recorded in the sidecar for auditability.
_ONTOLOGY_TO_MAPILLARY: dict[str, str] = {}
_ALTERNATIVES: dict[str, list[str]] = {}
for _mv, _onto_name in MAPILLARY_TO_ONTOLOGY.items():
    if _onto_name not in _ONTOLOGY_TO_MAPILLARY:
        _ONTOLOGY_TO_MAPILLARY[_onto_name] = _mv
    else:
        _ALTERNATIVES.setdefault(_onto_name, []).append(_mv)

# Fallback instance labels for ontology classes Vistas has no term for, chosen by l0 superclass so the export
# stays inside Mapillary's instance namespace instead of inventing labels.
_FALLBACK_BY_L0 = {
    "vehicle": "object--vehicle--other-vehicle",
    "vru": "human--person--individual",
    "animal": "animal--ground-animal",
}
_GENERIC_FALLBACK = "object--other"


def _stem(r: ExportRecord) -> str:
    return f"{r.cam_id}_{r.ts_ns}"


def _label_for(r: ExportRecord, onto: Ontology) -> tuple[str, bool]:
    """Return (mapillary_label, mapped_exactly)."""
    exact = _ONTOLOGY_TO_MAPILLARY.get(r.class_name)
    if exact:
        return exact, True
    cls = onto.by_name(r.class_name) if hasattr(onto, "by_name") else None
    l0 = getattr(cls, "l0", None)
    return _FALLBACK_BY_L0.get(l0 or "", _GENERIC_FALLBACK), False


def _polygon(r: ExportRecord) -> list[list[float]]:
    """The box as a closed 4-point polygon, which is what Vistas' polygon field expects."""
    x1, y1, x2, y2 = r.bbox
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def write_mapillary(records: list[ExportRecord], onto: Ontology, out_dir: Path) -> Path:
    poly_dir = out_dir / "polygons"
    poly_dir.mkdir(parents=True, exist_ok=True)

    by_frame: dict[str, list[ExportRecord]] = {}
    for r in records:
        by_frame.setdefault(str(r.frame_id), []).append(r)

    stem_of = unique_stems(by_frame)   # collision-safe across sessions, same rule as the VOC adapter
    unmapped: dict[str, int] = {}
    for fkey, recs in by_frame.items():
        first = recs[0]
        objects = []
        for r in recs:
            label, exact = _label_for(r, onto)
            if not exact:
                unmapped[r.class_name] = unmapped.get(r.class_name, 0) + 1
            objects.append({"label": label, "polygon": _polygon(r)})
        payload = {"width": first.width, "height": first.height, "objects": objects}
        (poly_dir / f"{stem_of[fkey]}.json").write_text(json.dumps(payload, indent=2))

    # Make the lossy edges of the mapping explicit rather than silent.
    (out_dir / "unmapped.json").write_text(json.dumps({
        "note": "ontology classes with no exact Mapillary Vistas label, emitted under a generic instance label",
        "counts": dict(sorted(unmapped.items())),
        "ambiguous_inverse": {k: v for k, v in sorted(_ALTERNATIVES.items())},
    }, indent=2))
    return out_dir
