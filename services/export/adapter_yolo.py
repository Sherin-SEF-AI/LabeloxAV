"""YOLO export: one label file per frame, plus the list files a YOLO trainer reads.

Two bugs lived here. `data.yaml` declared `train: images` and `val: images` — the same directory, so the
declared split was a fiction — and no `images/` directory was ever written, so both pointed at nothing. And
label files were named from the image's basename, which is not unique across sessions: two drives whose
frames are both `000001.jpg` wrote the same `labels/000001.txt` and one silently overwrote the other.

The list-file form is the honest fix. This adapter has no object store (the writer registry passes it none),
so it cannot fetch images and must not pretend to: `train.txt` and its siblings carry image URIs, which is a
form the trainer supports and which says plainly that the pixels live elsewhere.
"""

from __future__ import annotations

from pathlib import Path

from services.autolabel.ontology import Ontology
from services.export.records import ExportRecord
from services.export.splits import SPLITS


def _stem(r: ExportRecord) -> str:
    """A label-file name that is unique across sessions.

    The frame id alone would do, but keeping the original basename in front of it means a person looking at
    a label file can still tell which capture it came from.
    """
    base = Path(r.img_uri.split("/")[-1]).stem
    # The whole frame id, not a prefix of it. A prefix is only as unique as the id generator happens to be
    # at its front, and a filename that collides loses a label silently.
    return f"{base}-{r.frame_id}"


def write_yolo(records: list[ExportRecord], onto: Ontology, out_dir: Path) -> Path:
    # Contiguous YOLO class index over the full ontology, id-sorted, stable per export.
    ordered = sorted(onto.classes, key=lambda c: c.id)
    idx_of = {c.id: i for i, c in enumerate(ordered)}

    by_frame: dict[str, list[ExportRecord]] = {}
    for r in records:
        by_frame.setdefault(str(r.frame_id), []).append(r)

    listed: dict[str, list[str]] = {s: [] for s in SPLITS}
    for _fkey, recs in by_frame.items():
        split = recs[0].split
        labels_dir = out_dir / "labels" / split
        labels_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        for r in recs:
            x1, y1, x2, y2 = r.bbox
            cx = ((x1 + x2) / 2) / r.width
            cy = ((y1 + y2) / 2) / r.height
            bw = (x2 - x1) / r.width
            bh = (y2 - y1) / r.height
            lines.append(f"{idx_of[r.class_id]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        stem = _stem(recs[0])
        (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        listed.setdefault(split, []).append(recs[0].img_uri)

    # Only the splits that carry frames get a list file, so an unsplit export writes train.txt alone rather
    # than two empty files a trainer would read as "validation found nothing".
    present = [s for s in SPLITS if listed.get(s)]
    for split in present:
        (out_dir / f"{split}.txt").write_text("\n".join(sorted(listed[split])) + "\n")

    names = "\n".join(f"  {i}: {c.name}" for i, c in enumerate(ordered))
    split_lines = "".join(f"{s}: {s}.txt\n" for s in present)
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"# LabeloxAV YOLO export\n"
        f"# ontology: {onto.version}\n"
        f"# images are referenced by uri in the list files; this export carries labels, not pixels\n"
        f"path: .\n"
        f"{split_lines}"
        f"nc: {len(ordered)}\n"
        f"names:\n{names}\n"
    )
    return out_dir
