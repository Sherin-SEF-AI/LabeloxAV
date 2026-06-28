"""Intermediate records the import adapters produce. The mirror image of services.export.records:
an adapter parses an external dataset into ImportFrame[] (image reference + parsed objects with their
ORIGINAL class names), and services.imports.run remaps names to the ontology, anonymizes the image
(Gate A), and writes Session/Frame/Object rows.

Note the package is `imports` (plural): `import` is a Python keyword and cannot be a module name.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ImportObject:
    name: str                      # original external class name (pre-remap)
    bbox: list[float]              # xyxy pixel
    attrs: dict = field(default_factory=dict)
    track_ref: str | None = None
    conf: float = 1.0
    # If the source is one of our own lossless exports, these carry through verbatim.
    ontology_class_id: int | None = None
    provenance: dict = field(default_factory=dict)
    # Segmentation survives the round-trip: a Parquet export carries the mask blob uri verbatim, while
    # polygon-bearing formats (COCO segmentation, OpenLABEL poly2d) provide flattened pixel polygons that
    # services.imports.run writes to a fresh mask blob.
    mask_uri: str | None = None
    mask_encoding: str | None = None
    mask_polygons: list[list[float]] | None = None  # flattened [x,y,x,y,...] per polygon, pixel coords
    rot_deg: float = 0.0  # oriented-box rotation about the box centre (0 = axis-aligned)
    keypoints: dict | None = None  # COCO-style {"skeleton","points":[[x,y,v],...]} pose


@dataclass
class ImportFrame:
    image_ref: str                 # local path (relative to dataset root) or s3:// uri
    width: int | None = None       # filled from the image if missing
    height: int | None = None
    ts_ns: int | None = None
    cam_id: str = "cam_front"
    objects: list[ImportObject] = field(default_factory=list)


@dataclass
class ImportSpec:
    format: str                    # coco | yolo | pascalvoc | openlabel | nuscenes | parquet | images | video | mcap
    source_uri: str                # s3://uploads/... or a local path (dir/zip/file)
    target_vehicle: str = "IMPORT-01"
    city: str | None = None
    route: str | None = None
    options: dict = field(default_factory=dict)
