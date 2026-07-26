"""Turn an export directory into a self-contained, portable dataset bundle.

The COCO/YOLO adapters reference each image by its object-store URI rather than copying the pixels, which keeps
an export small but means the folder is not usable offline: a consumer would have to re-fetch every image from
MinIO. This fetches the images once into an `images/` directory (whose basenames already match the YOLO label
files and the COCO file_name field), so the result is a ready-to-train YOLO dataset AND a valid COCO dataset
in one folder, then zips it.

    python scripts/bundle_export.py <export_dir> [--zip]

Idempotent: an image already present is not re-fetched, so an interrupted bundle resumes.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from core.logging import get_logger, setup_logging
from core.storage import get_object_store

log = get_logger("bundle_export")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir", help="an export dir written by export_dataset (contains coco/annotations.json)")
    ap.add_argument("--zip", action="store_true", help="also write a .zip next to the bundle")
    args = ap.parse_args()
    setup_logging()

    root = Path(args.export_dir)
    coco_path = root / "coco" / "annotations.json"
    if not coco_path.exists():
        raise SystemExit(f"no coco/annotations.json under {root}")

    coco = json.loads(coco_path.read_text())
    images = coco.get("images", [])
    store = get_object_store()

    img_dir = root / "images"
    img_dir.mkdir(exist_ok=True)

    fetched = skipped = missing = 0
    for rec in images:
        name = rec.get("file_name")
        uri = rec.get("uri")
        if not name or not uri:
            continue
        dst = img_dir / name
        if dst.exists() and dst.stat().st_size > 0:
            skipped += 1
            continue
        try:
            data = store.get_bytes(uri)
            dst.write_bytes(data)
            fetched += 1
        except Exception as exc:  # noqa: BLE001 - a missing frame must not abort the whole bundle
            missing += 1
            log.warning("bundle.image_missing", uri=uri, error=str(exc)[:120])

    # A self-contained COCO copy at the root whose file_name resolves against ./images, next to the YOLO
    # data.yaml that already points train/val at images/. Both trainers now read the folder with no network.
    shutil.copy(coco_path, root / "annotations.coco.json")
    (root / "README.txt").write_text(
        "LabeloxAV portable dataset bundle\n"
        f"images: {len(images)} frames in ./images\n"
        f"annotations (COCO): ./annotations.coco.json ({len(coco.get('annotations', []))} boxes, "
        f"{len(coco.get('categories', []))} classes)\n"
        "annotations (YOLO): ./labels/*.txt with ./data.yaml\n"
        "provenance: ./parquet\n",
    )

    print(f"images fetched={fetched} skipped(existing)={skipped} missing={missing} total={len(images)}")
    size_mb = sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) / (1024 * 1024)
    print(f"bundle: {root}  ({size_mb:.1f} MB)")

    if args.zip:
        # shutil.make_archive wants the archive path without the extension; write it beside the bundle.
        archive = shutil.make_archive(str(root), "zip", root_dir=root)
        print(f"zip: {archive}  ({Path(archive).stat().st_size / (1024 * 1024):.1f} MB)")


if __name__ == "__main__":
    main()
