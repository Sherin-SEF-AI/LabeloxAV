"""Convert IDD-Detection (India Driving Dataset, Pascal-VOC XML) to a YOLO dataset whose class names
are LabeloxAV ontology names, ready for `services.training.dataset_builder --idd-dir` to merge as the
cold-start anchor.

IDD-Detection layout (from idd.insaan.iiit.ac.in, the IDD_Detection.tar.gz):
    IDD_Detection/
      JPEGImages/<scene>/<id>.jpg          (or .png)
      Annotations/<scene>/<id>.xml         (Pascal VOC: <size>, <object><name><bndbox>)
      train.txt  val.txt  test.txt         (one relative path per line, no extension)

We remap IDD's class names onto our ontology (by name) and write YOLO labels. Names IDD has that we
do not (e.g. generic 'animal', 'train') map to the nearest ontology class or object_fallback; edit
IDD_TO_ONTOLOGY below to taste. License: IDD is research-oriented and must be confirmed before any
buyer-bound use (idd.insaan.iiit.ac.in terms).

    python scripts/idd_to_yolo.py --idd-root /data/IDD_Detection --out .scratch/idd_yolo --splits train,val
"""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

from core.logging import get_logger, setup_logging
from services.autolabel.ontology import get_ontology

log = get_logger("idd2yolo")

# IDD detection class name -> LabeloxAV ontology class name. IDD names are normalized first
# (lowercased, spaces/hyphens -> underscores), then this override map is applied.
IDD_TO_ONTOLOGY: dict[str, str] = {
    "car": "sedan",
    "person": "pedestrian",
    "rider": "rider",
    "motorcycle": "motorcycle",
    "bicycle": "cycle",
    "autorickshaw": "autorickshaw",
    "auto_rickshaw": "autorickshaw",
    "truck": "truck",
    "bus": "bus",
    "vehicle_fallback": "vehicle_fallback",
    "traffic_sign": "traffic_sign",
    "traffic_light": "traffic_signal",
    "animal": "cattle",          # IDD 'animal' is generic; cattle is the dominant road animal
    "caravan": "minivan",
    "trailer": "multi_axle_trailer",
    "train": "object_fallback",  # no rail-vehicle leaf in the ontology
}


def _normalize(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _map_name(idd_name: str, onto) -> str | None:
    norm = _normalize(idd_name)
    mapped = IDD_TO_ONTOLOGY.get(norm, norm)
    if onto.has_name(mapped):
        return mapped
    return "object_fallback" if onto.has_name("object_fallback") else None


def _read_split(idd_root: Path, split: str) -> list[str]:
    f = idd_root / f"{split}.txt"
    if f.exists():
        return [ln.strip() for ln in f.read_text().splitlines() if ln.strip()]
    # fall back to every annotation if no split list
    return [str(p.relative_to(idd_root / "Annotations").with_suffix("")) for p in (idd_root / "Annotations").rglob("*.xml")]


def _find_image(idd_root: Path, rel: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png"):
        p = idd_root / "JPEGImages" / f"{rel}{ext}"
        if p.exists():
            return p
    return None


def _parse_voc(xml_path: Path) -> tuple[int, int, list[tuple[str, float, float, float, float]]]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = int(float(size.findtext("width", "0"))) if size is not None else 0
    h = int(float(size.findtext("height", "0"))) if size is not None else 0
    objs = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()
        bb = obj.find("bndbox")
        if not name or bb is None:
            continue
        objs.append((
            name,
            float(bb.findtext("xmin", "0")), float(bb.findtext("ymin", "0")),
            float(bb.findtext("xmax", "0")), float(bb.findtext("ymax", "0")),
        ))
    return w, h, objs


def convert(idd_root: Path, out: Path, splits: list[str], link: bool) -> dict:
    onto = get_ontology()
    for s in splits:
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)

    # collect the ontology names that actually appear, for a stable contiguous index
    present: set[str] = set()
    pending: list[tuple[str, Path, Path, int, int, list]] = []
    unmapped: dict[str, int] = {}

    for split in splits:
        for rel in _read_split(idd_root, split):
            xml_path = idd_root / "Annotations" / f"{rel}.xml"
            img_path = _find_image(idd_root, rel)
            if not xml_path.exists() or img_path is None:
                continue
            w, h, objs = _parse_voc(xml_path)
            if w <= 0 or h <= 0:
                import cv2

                im = cv2.imread(str(img_path))
                if im is None:
                    continue
                h, w = im.shape[:2]
            mapped_objs = []
            for name, x1, y1, x2, y2 in objs:
                m = _map_name(name, onto)
                if m is None:
                    unmapped[name] = unmapped.get(name, 0) + 1
                    continue
                present.add(m)
                mapped_objs.append((m, x1, y1, x2, y2))
            if mapped_objs:
                pending.append((split, img_path, xml_path, w, h, mapped_objs))

    names_sorted = sorted(present, key=lambda n: onto.by_name(n).id)
    idx_of = {n: i for i, n in enumerate(names_sorted)}

    n_written = 0
    for split, img_path, _xml, w, h, objs in pending:
        # IDD numbers frames per scene (0000060.jpg recurs in every scene), so the destination name
        # must include the scene path or images/labels collide and overwrite across scenes.
        rel = img_path.relative_to(idd_root / "JPEGImages").with_suffix("")
        stem = "idd_" + str(rel).replace("/", "_").replace("\\", "_")
        dst_img = out / "images" / split / f"{stem}{img_path.suffix}"
        if link:
            if not dst_img.exists():
                dst_img.symlink_to(img_path.resolve())
        else:
            shutil.copy(img_path, dst_img)
        lines = []
        for name, x1, y1, x2, y2 in objs:
            cx = max(0.0, min(1.0, (x1 + x2) / 2 / w))
            cy = max(0.0, min(1.0, (y1 + y2) / 2 / h))
            bw = max(0.0, min(1.0, (x2 - x1) / w))
            bh = max(0.0, min(1.0, (y2 - y1) / h))
            lines.append(f"{idx_of[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        n_written += 1

    (out / "data.yaml").write_text(yaml.safe_dump({
        "path": str(out), "train": "images/train", "val": "images/val" if "val" in splits else "images/train",
        "nc": len(names_sorted), "names": {i: n for n, i in idx_of.items()},
    }, sort_keys=False))

    result = {
        "out": str(out), "images": n_written, "classes": len(names_sorted),
        "ontology_classes": names_sorted, "unmapped": unmapped,
    }
    log.info("idd2yolo.done", images=n_written, classes=len(names_sorted), unmapped=len(unmapped))
    if unmapped:
        log.warning("idd2yolo.unmapped_names", names=unmapped)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--idd-root", required=True, help="path to extracted IDD_Detection/")
    ap.add_argument("--out", default=".scratch/idd_yolo")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    args = ap.parse_args()
    setup_logging("INFO")
    res = convert(Path(args.idd_root), Path(args.out), [s.strip() for s in args.splits.split(",")], link=not args.copy)
    print(res)


if __name__ == "__main__":
    main()
