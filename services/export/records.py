"""ExportRecord: the flattened object + frame + session context every adapter consumes. It mirrors
the UnifiedObject plus the lineage fields needed for the provenance sidecar (Principle 10)."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID


@dataclass
class ExportRecord:
    object_id: UUID
    frame_id: UUID
    session_id: UUID
    ts_ns: int
    cam_id: str
    img_uri: str
    width: int
    height: int
    vehicle_id: str
    city: str | None
    class_id: int
    class_name: str
    bbox: list[float]            # xyxy pixel
    conf: float
    state: str
    source: str
    mask_uri: str | None = None
    mask_encoding: str | None = None
    track_id: UUID | None = None
    attrs: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
