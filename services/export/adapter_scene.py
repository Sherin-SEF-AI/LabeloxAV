"""Scene-level export: masks, lanes, drivable surfaces, and the HD map.

Every existing adapter is object-shaped, because `ExportRecord` is one row per `Object`. That was fine while
the only thing worth exporting was a box, and it is why four annotation types could be created, corrected,
propagated and gated inside the system and never leave it in any format:

- **Masks** left only as COCO polygons on an annotation. A segmentation consumer wants label maps.
- **Lanes** could not leave at all. `Lane` is not an `Object`, so no record ever carried one.
- **Drivable surfaces** likewise: `DrivableMask` is per frame, not per object.
- **The HD map** is per corpus, in world coordinates, and has no per-frame representation at all.

Each writer here follows the convention its consumers already read, rather than a house format that would
need a converter on the far side:

- Masks: Cityscapes-style `labelIds` PNGs, one per frame, plus a `labels.json` giving the id-to-name map.
  Written as a paletted greyscale image so a class id is a pixel value, not a colour to be matched.
- Lanes: CULane-style, one text file per frame with one lane per line as x y pairs, plus a JSON sidecar
  carrying the lane type and the ego flag, which CULane's own format has nowhere to put.
- Drivable: BDD-style ternary masks, one PNG per frame.
- HD map: GeoJSON `FeatureCollection` in WGS84, which is what the map elements are actually stored in, so
  nothing is reprojected and nothing loses precision on the way out.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from core.logging import get_logger
from core.storage import ObjectStore
from services.autolabel.ontology import Ontology
from services.export.records import ExportRecord

log = get_logger("adapter_scene")

DRIVABLE_VALUES = {"non_drivable": 0, "drivable": 1, "fallback": 2}


def _kind_rank():
    """Raster precedence for a frame carrying more than one kind: richest first.

    A function rather than a module constant because building it imports the model, and adapter_scene is
    imported by the export driver at module load.
    """
    from sqlalchemy import case

    from db.models import FrameSegmentation

    return case({"panoptic": 0, "semantic": 1}, value=FrameSegmentation.kind, else_=2)


def _load_polygons(store: ObjectStore, mask_uri: str | None) -> list[list[float]]:
    if not mask_uri:
        return []
    try:
        return json.loads(store.get_bytes(mask_uri)).get("polygons", [])
    except Exception as exc:  # noqa: BLE001
        log.warning("scene.mask_unreadable", uri=mask_uri, error=str(exc))
        return []


def write_masks(records: list[ExportRecord], onto: Ontology, store: ObjectStore,
                out_dir: Path) -> Path:
    """Cityscapes-style label maps, one PNG per frame.

    A single-channel image whose pixel value is the ontology class id, which is what a segmentation trainer
    reads directly. A colour image would need the consumer to invert a palette, and any two classes that
    happened to render similarly would be silently merged.

    Overlap is resolved by drawing smaller objects last, so a pedestrian inside a bus is still a pedestrian.
    Painting in arbitrary order would let the larger object swallow the smaller one, which is the common and
    quiet way a mask export loses its rarest classes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "labelIds").mkdir(parents=True, exist_ok=True)

    by_frame: dict[str, list[ExportRecord]] = {}
    for r in records:
        if r.mask_uri:
            by_frame.setdefault(str(r.frame_id), []).append(r)

    written = 0
    classes: dict[int, str] = {}
    for frame_id, group in by_frame.items():
        h, w = int(group[0].height or 0), int(group[0].width or 0)
        if h <= 0 or w <= 0:
            continue
        canvas = np.zeros((h, w), dtype=np.uint16)

        drawn: list[tuple[float, int, list[list[float]]]] = []
        for r in group:
            polys = _load_polygons(store, r.mask_uri)
            if not polys:
                continue
            area = max(0.0, (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]))
            drawn.append((area, r.class_id, polys))
            classes[r.class_id] = r.class_name

        # Largest first, so the smallest object is painted last and survives the overlap.
        for _area, class_id, polys in sorted(drawn, key=lambda t: -t[0]):
            for flat in polys:
                if len(flat) < 6:
                    continue
                pts = np.array(flat, dtype=np.float32).reshape(-1, 2).round().astype(np.int32)
                cv2.fillPoly(canvas, [pts], color=int(class_id))

        if canvas.any():
            cv2.imwrite(str(out_dir / "labelIds" / f"{frame_id}.png"), canvas)
            written += 1

    (out_dir / "labels.json").write_text(json.dumps({
        "format": "cityscapes_labelids",
        "encoding": "uint16 single channel; pixel value is the ontology class id, 0 is unlabelled",
        "ontology_version": onto.version,
        "classes": {str(k): v for k, v in sorted(classes.items())},
        "frames": written,
    }, indent=2))
    log.info("export.masks_written", frames=written, classes=len(classes))
    return out_dir


async def write_lanes(frame_ids: list[str], out_dir: Path) -> Path:
    """CULane-style lane geometry, one text file per frame, plus the attributes CULane cannot carry.

    The text file is what a lane trainer reads. The sidecar exists because CULane's format is purely
    geometric: it has no field for the lane type or the ego flag, and dropping them on the way out would
    make it impossible to train the type classifier the corpus has labels for.
    """
    import uuid as _uuid

    from sqlalchemy import select

    from db.models import Lane
    from db.session import get_sessionmaker

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lines").mkdir(parents=True, exist_ok=True)

    if not frame_ids:
        (out_dir / "lanes.json").write_text(json.dumps({"format": "culane", "frames": 0}, indent=2))
        return out_dir

    async with get_sessionmaker()() as db:
        rows = (await db.execute(
            select(Lane).where(Lane.frame_id.in_([_uuid.UUID(f) for f in frame_ids])))).scalars().all()

    by_frame: dict[str, list] = {}
    for lane in rows:
        by_frame.setdefault(str(lane.frame_id), []).append(lane)

    meta: dict[str, list[dict]] = {}
    for frame_id, lanes in by_frame.items():
        lines = []
        entries = []
        for lane in lanes:
            pts = [p for p in (lane.control_points or []) if len(p) >= 2]
            if len(pts) < 2:
                continue
            lines.append(" ".join(f"{float(p[0]):.3f} {float(p[1]):.3f}" for p in pts))
            entries.append({"lane_type": lane.lane_type, "is_ego": bool(lane.is_ego),
                            "source": lane.source, "track_ref": str(lane.track_ref) if lane.track_ref else None,
                            "n_points": len(pts)})
        if not lines:
            continue
        (out_dir / "lines" / f"{frame_id}.lines.txt").write_text("\n".join(lines) + "\n")
        meta[frame_id] = entries

    (out_dir / "lanes.json").write_text(json.dumps({
        "format": "culane",
        "geometry": "one lane per line in <frame>.lines.txt, as x y pairs in image pixels",
        "note": "lane_type and is_ego live here because the CULane text format has nowhere to put them",
        "frames": len(meta),
        "lanes": meta,
    }, indent=2))
    log.info("export.lanes_written", frames=len(meta))
    return out_dir


async def write_drivable(frame_ids: list[str], store: ObjectStore, out_dir: Path) -> Path:
    """BDD-style drivable-area masks, one PNG per frame, with the ternary encoding stated.

    Copied through rather than re-rendered. The stored mask is the artifact the annotator corrected, and
    re-deriving it from coverage fractions would export a different mask from the one that was approved.
    """
    import uuid as _uuid

    from sqlalchemy import select

    from db.models import DrivableMask
    from db.session import get_sessionmaker

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    if not frame_ids:
        (out_dir / "drivable.json").write_text(json.dumps({"format": "bdd_drivable", "frames": 0}, indent=2))
        return out_dir

    async with get_sessionmaker()() as db:
        rows = (await db.execute(
            select(DrivableMask).where(
                DrivableMask.frame_id.in_([_uuid.UUID(f) for f in frame_ids])))).scalars().all()

    written = 0
    coverage: dict[str, dict] = {}
    unreadable = 0
    for dm in rows:
        try:
            data = store.get_bytes(dm.mask_uri)
        except Exception:  # noqa: BLE001
            unreadable += 1
            continue
        (out_dir / "masks" / f"{dm.frame_id}.png").write_bytes(data)
        coverage[str(dm.frame_id)] = {"coverage": dm.coverage or {}, "source": dm.source}
        written += 1

    (out_dir / "drivable.json").write_text(json.dumps({
        "format": "bdd_drivable",
        "encoding": {v: k for k, v in DRIVABLE_VALUES.items()},
        "frames": written,
        # Surfaced rather than dropped: a consumer counting files against frames would otherwise see a
        # shortfall with no explanation.
        "unreadable": unreadable,
        "per_frame": coverage,
    }, indent=2))
    log.info("export.drivable_written", frames=written, unreadable=unreadable)
    return out_dir


async def write_hdmap(out_dir: Path, commit_id: str | None = None) -> Path:
    """The HD map as GeoJSON in WGS84.

    GeoJSON because the elements are already stored as geographies in 4326, so nothing is reprojected and
    no precision is lost on the way out. OpenDRIVE or Lanelet2 would be a richer target and would require
    inventing the road topology those formats demand and this corpus does not carry; emitting them with
    guessed topology would produce a file that loads and is wrong.
    """
    from sqlalchemy import func, select

    from db.models import MapElement
    from db.session import get_sessionmaker

    out_dir.mkdir(parents=True, exist_ok=True)

    async with get_sessionmaker()() as db:
        stmt = select(MapElement, func.ST_AsGeoJSON(MapElement.geometry))
        if commit_id:
            stmt = stmt.where(MapElement.commit_id == commit_id)
        rows = (await db.execute(stmt)).all()

    features = []
    no_geometry = 0
    for el, geojson in rows:
        if not geojson:
            no_geometry += 1
            continue
        features.append({
            "type": "Feature",
            "geometry": json.loads(geojson),
            "properties": {
                "element_id": str(el.element_id), "kind": el.kind,
                "confidence": float(el.confidence or 0.0),
                "calibration_version": el.calibration_version,
                "source_sessions": list(el.source_sessions or []),
                "commit_id": el.commit_id,
                **(el.attrs or {}),
            },
        })

    (out_dir / "hdmap.geojson").write_text(json.dumps({
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
    }, indent=2))
    (out_dir / "hdmap.json").write_text(json.dumps({
        "format": "geojson_wgs84",
        "elements": len(features),
        "elements_without_geometry": no_geometry,
        "commit_id": commit_id,
        "note": ("OpenDRIVE and Lanelet2 need road topology this corpus does not carry; emitting them with "
                 "guessed topology would produce a file that loads and is wrong"),
    }, indent=2))
    log.info("export.hdmap_written", elements=len(features), skipped=no_geometry)
    return out_dir


async def write_panoptic(frame_ids: list[str], store: ObjectStore, out_dir: Path) -> Path:
    """COCO panoptic segmentation: an id-map PNG per frame, plus segments_info naming what each id is.

    `FrameSegmentation` holds semantic and panoptic rasters that a person can create, correct and gate, and
    until now could not leave the system in any format at all. That is the same failure adapter_scene was
    written to fix for masks, lanes and drivable surfaces: a layer with a `human` source, no write path out,
    and therefore a visualisation rather than a deliverable.

    **The format is an id map, not a class map**, and the difference is the whole point of panoptic. Each
    segment, whether a countable thing or a region of stuff, gets an id unique within its frame, encoded
    into the PNG as `id = R + G*256 + B*65536`. `segments_info` maps each id back to a category. Emitting a
    class-id raster and calling it panoptic would produce a file that loads and scores nonsense, because
    every car in the frame would be one segment.

    **`isthing` comes from the ontology's own thing/stuff split** rather than being guessed from the class
    name. That split already exists because the persist chokepoint uses it to drop stuff by construction,
    and it is the same distinction COCO panoptic needs, so the two cannot drift.

    **A semantic row is not a panoptic row, and saying so matters.** A semantic raster has no instance
    channel, so converting one gives a single segment per class. For stuff that is exactly right and loses
    nothing. For a thing class it merges every instance into one blob, which is a real difference a consumer
    computing PQ would otherwise never see. Those frames are converted, marked per frame, and counted in the
    summary, because refusing them would drop three quarters of the rasters this corpus holds and silently
    merging them would corrupt somebody's benchmark.
    """
    import uuid as _uuid

    from sqlalchemy import select

    from db.models import FrameSegmentation
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    out_dir.mkdir(parents=True, exist_ok=True)
    png_dir = out_dir / "panoptic"
    png_dir.mkdir(parents=True, exist_ok=True)
    onto = get_ontology()

    if not frame_ids:
        (out_dir / "panoptic.json").write_text(json.dumps(
            {"format": "coco_panoptic", "images": [], "annotations": [], "categories": []}, indent=2))
        return out_dir

    async with get_sessionmaker()() as db:
        rows = (await db.execute(
            select(FrameSegmentation)
            .where(FrameSegmentation.frame_id.in_([_uuid.UUID(f) for f in frame_ids]))
            # Panoptic first, so a frame carrying both kinds exports the richer one and the semantic
            # duplicate is skipped rather than overwriting it.
            #
            # Ordered by an explicit rank rather than by the string. `kind.desc()` was the first attempt and
            # is wrong for a reason that looks like a typo: "semantic" sorts after "panoptic", so descending
            # put the poorer raster first and quietly dropped the instances. Alphabetical order agreeing
            # with precedence would also have been a coincidence, and a third kind could break it again.
            .order_by(_kind_rank(), FrameSegmentation.created_at.desc()))).scalars().all()

    images: list[dict] = []
    annotations: list[dict] = []
    used_categories: dict[int, dict] = {}
    seen_frames: set[str] = set()
    unreadable = 0
    merged_from_semantic = 0

    for row in rows:
        fid = str(row.frame_id)
        if fid in seen_frames:
            continue

        labels = _load_npz(store, row.labels_uri)
        if labels is None:
            unreadable += 1
            continue
        instances = _load_npz(store, row.instance_uri) if row.instance_uri else None
        if instances is not None and instances.shape != labels.shape:
            # Mismatched rasters cannot be combined into segments, and pairing them by position anyway
            # would assign pixels to the wrong instances. Treated as semantic instead, and counted.
            instances = None
        seen_frames.add(fid)

        h, w = labels.shape[:2]
        id_map = np.zeros((h, w, 3), dtype=np.uint8)
        segments: list[dict] = []
        next_id = 1
        merged_here: list[str] = []

        # One segment per (class, instance) pair when there is an instance channel, and per class when
        # there is not.
        if instances is not None:
            keys = np.stack([labels.astype(np.int64), instances.astype(np.int64)], axis=-1)
            pairs = np.unique(keys.reshape(-1, 2), axis=0)
            groups = [(int(c), (labels == c) & (instances == i)) for c, i in pairs]
        else:
            groups = [(int(c), labels == c) for c in np.unique(labels)]

        for class_id, mask in groups:
            if class_id < 0:
                continue
            area = int(mask.sum())
            if area == 0:
                continue
            try:
                cls = onto.by_id(class_id)
            except Exception:  # noqa: BLE001
                continue

            is_thing = onto.is_thing(class_id)
            if instances is None and is_thing:
                # Recorded rather than silently accepted: this segment is every instance of the class at
                # once, and a PQ score computed over it is not the number the consumer thinks it is.
                merged_here.append(cls.name)

            seg_id = next_id
            next_id += 1
            # COCO panoptic packs the segment id into the pixel colour, little-endian across RGB.
            id_map[mask] = (seg_id % 256, (seg_id // 256) % 256, (seg_id // 65536) % 256)

            ys, xs = np.nonzero(mask)
            segments.append({
                "id": seg_id,
                "category_id": class_id,
                "area": area,
                "bbox": [int(xs.min()), int(ys.min()),
                         int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
                "iscrowd": 0,
            })
            used_categories.setdefault(class_id, {
                "id": class_id, "name": cls.name, "supercategory": cls.l1 or cls.l0 or "object",
                "isthing": 1 if is_thing else 0,
            })

        png_name = f"{fid}.png"
        # RGB written as BGR, because cv2 orders channels that way and a silent swap here would corrupt
        # every segment id above 255.
        cv2.imwrite(str(png_dir / png_name), id_map[:, :, ::-1])

        images.append({"id": fid, "file_name": f"{fid}.jpg", "width": w, "height": h})
        annotations.append({
            "image_id": fid, "file_name": png_name, "segments_info": segments,
            # Carried so a consumer can tell a corrected raster from a proposed one without a second query.
            "source": row.source, "kind": row.kind,
            "instances_merged_from_semantic": merged_here or None,
        })
        if merged_here:
            merged_from_semantic += 1

    (out_dir / "panoptic.json").write_text(json.dumps({
        "format": "coco_panoptic",
        "encoding": "segment id per pixel, packed as id = R + G*256 + B*65536",
        "images": images,
        "annotations": annotations,
        "categories": [used_categories[k] for k in sorted(used_categories)],
        # Stated at the top level, not only per frame. A consumer computing panoptic quality over a mixed
        # export needs to know some frames have merged instances before they compute anything, and a flag
        # buried per annotation is one nobody reads until the number looks wrong.
        "frames_with_merged_instances": merged_from_semantic,
        "unreadable": unreadable,
        "note": ("frames exported from a semantic raster have one segment per class; for thing classes that "
                 "merges every instance, which is listed per annotation in "
                 "instances_merged_from_semantic"),
    }, indent=2))
    log.info("export.panoptic_written", frames=len(images), categories=len(used_categories),
             merged_from_semantic=merged_from_semantic, unreadable=unreadable)
    return out_dir


def _load_npz(store: ObjectStore, uri: str | None):
    """A stored raster, or None when it cannot be read. Never raises: one unreadable frame should cost that
    frame, not the export."""
    import io

    if not uri:
        return None
    try:
        return np.load(io.BytesIO(store.get_bytes(uri)))["arr"]
    except Exception as exc:  # noqa: BLE001
        log.warning("export.panoptic.unreadable", uri=uri, error=str(exc)[:120])
        return None
