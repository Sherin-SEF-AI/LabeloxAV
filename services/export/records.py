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
    quality_score: float | None = None  # M-F.1 composite label-quality QA signal [0,1], for buyers to filter on
    mask_uri: str | None = None
    mask_encoding: str | None = None
    track_id: UUID | None = None
    attrs: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    cuboid_3d: dict | None = None  # ego-frame {center,size,yaw} when a 3D label exists
    rot_deg: float = 0.0           # oriented-box rotation about the box centre (0 = axis-aligned)
    keypoints: dict | None = None  # COCO-style {"skeleton","points":[[x,y,v],...]} pose, when present
    polyline: list | None = None   # open linear feature [[x,y],...] (curb/road_edge/barrier), when present
    relationships: list = field(default_factory=list)  # outgoing [{"to_object_id","kind"},...] groupings
    # Frame-level scene context from Frame.scene: weather, density, road_type, time_of_day from ingest plus
    # whatever a person set. A property of the frame, so every record on one frame carries the same dict and
    # the writers put it on the image rather than on each annotation.
    context: dict = field(default_factory=dict)
    # Accepted track events overlapping this object's frame, as [{"event_type","start_ts_ns","end_ts_ns"}].
    # Per record rather than per track because an event covers part of a track, and a consumer asking "was
    # this object braking in this frame" cannot answer that from a track-level list.
    track_events: list = field(default_factory=list)
    sign_type: str | None = None       # Indian RTO taxonomy type, when a sign was typed and not rejected
    sign_category: str | None = None   # mandatory | cautionary | informatory
    ocr_text: str | None = None        # road text read off a sign or board; never a license plate
    ocr_lang: str | None = None
    ocr_conf: float | None = None      # None means unmeasured, never assume a number
    # train | val | test. Stamped by the export, not read from the database: a split is a property of a
    # delivered dataset, not of the object, and the same object belongs to different splits in different
    # releases.
    split: str = "train"
