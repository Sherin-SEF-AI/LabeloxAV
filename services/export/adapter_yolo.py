"""YOLO adapter: boxes + class only, one label .txt per image (normalized cx cy w h) plus a
data.yaml. Tracks, masks, attributes and ts_ns are lossy here by design and live in the Parquet
sidecar keyed by image (Principle 10: never block an export on format limits).
"""

from __future__ import annotations

from pathlib import Path

from services.autolabel.ontology import Ontology
from services.export.records import ExportRecord


def write_yolo(records: list[ExportRecord], onto: Ontology, out_dir: Path) -> Path:
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Contiguous YOLO class index over the full ontology, id-sorted, stable per export.
    ordered = sorted(onto.classes, key=lambda c: c.id)
    idx_of = {c.id: i for i, c in enumerate(ordered)}

    by_frame: dict[str, list[ExportRecord]] = {}
    stems: dict[str, str] = {}
    for r in records:
        fkey = str(r.frame_id)
        by_frame.setdefault(fkey, []).append(r)
        stems[fkey] = Path(r.img_uri.split("/")[-1]).stem

    for fkey, recs in by_frame.items():
        lines = []
        for r in recs:
            x1, y1, x2, y2 = r.bbox
            cx = ((x1 + x2) / 2) / r.width
            cy = ((y1 + y2) / 2) / r.height
            bw = (x2 - x1) / r.width
            bh = (y2 - y1) / r.height
            lines.append(f"{idx_of[r.class_id]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (labels_dir / f"{stems[fkey]}.txt").write_text("\n".join(lines) + "\n")

    names = "\n".join(f"  {i}: {c.name}" for i, c in enumerate(ordered))
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"# LabeloxAV YOLO export\n"
        f"# ontology: {onto.version}\n"
        f"path: .\n"
        f"train: images\n"
        f"val: images\n"
        f"nc: {len(ordered)}\n"
        f"names:\n{names}\n"
    )
    return out_dir
