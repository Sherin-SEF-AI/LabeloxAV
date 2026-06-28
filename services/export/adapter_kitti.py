"""KITTI adapter: the classic 2D object-detection label format, one `<image_stem>.txt` per frame
with one space-separated line per object:

    type truncated occluded alpha x1 y1 x2 y2 h w l x y z rotation_y

This build is 2D single-camera, so truncated/occluded/alpha and every 3D field (h w l x y z
rotation_y) are emitted as KITTI's "unknown" sentinels; the real value is the 2D bbox from the
record. The ontology class_name is used verbatim as the KITTI type (KITTI is case + whitespace
sensitive), and a classes.txt records the type vocabulary. The Parquet sidecar stays the lossless
source of truth (Principle 10: never block an export on format limits).
"""

from __future__ import annotations

from pathlib import Path

from services.autolabel.ontology import Ontology
from services.export.records import ExportRecord

# KITTI "don't care / unknown" sentinels for the 3D + camera-geometry fields we cannot fill in 2D.
_ALPHA = -10.0
_DIM3 = [-1.0, -1.0, -1.0]            # h w l
_LOC3 = [-1000.0, -1000.0, -1000.0]  # x y z
_ROT_Y = -10.0


def _stem(r: ExportRecord) -> str:
    # Stable image stem independent of img_uri availability: cam + timestamp uniquely names a frame.
    return f"{r.cam_id}_{r.ts_ns}"


def _occluded(r: ExportRecord) -> int:
    # KITTI occlusion state: 0 visible .. 3 largely occluded. Map our occlusion attr if present.
    occ = r.attrs.get("occlusion")
    if isinstance(occ, bool):
        return 3 if occ else 0
    if isinstance(occ, (int, float)):
        if occ >= 75:
            return 3
        if occ >= 40:
            return 2
        if occ >= 10:
            return 1
    return 0


def write_kitti(records: list[ExportRecord], onto: Ontology, out_dir: Path) -> Path:
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    by_frame: dict[str, list[ExportRecord]] = {}
    stems: dict[str, str] = {}
    for r in records:
        fkey = str(r.frame_id)
        by_frame.setdefault(fkey, []).append(r)
        stems[fkey] = _stem(r)

    for fkey, recs in by_frame.items():
        lines = []
        for r in recs:
            x1, y1, x2, y2 = r.bbox
            h3, w3, l3 = _DIM3
            lx, ly, lz = _LOC3
            lines.append(
                f"{r.class_name} 0.00 {_occluded(r)} {_ALPHA:.2f} "
                f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
                f"{h3:.2f} {w3:.2f} {l3:.2f} {lx:.2f} {ly:.2f} {lz:.2f} {_ROT_Y:.2f}"
            )
        (labels_dir / f"{stems[fkey]}.txt").write_text("\n".join(lines) + "\n")

    # classes.txt: the KITTI type vocabulary, id-sorted over the full ontology (stable per export).
    ordered = sorted(onto.classes, key=lambda c: c.id)
    (out_dir / "classes.txt").write_text("\n".join(c.name for c in ordered) + "\n")
    return out_dir
