"""Pydantic interchange schemas. UnifiedObject is the single shape every export adapter consumes.

These mirror the Postgres tables (db/models.py) but are the in-memory contract between planes.
"""

from __future__ import annotations

from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class GateState(str, Enum):
    auto_accept = "auto_accept"
    review = "review"
    annotate = "annotate"
    accepted = "accepted"
    rejected = "rejected"


class ObjectSource(str, Enum):
    fused = "fused"
    auto_accept = "auto_accept"
    human = "human"
    recall = "recall"  # recovered by the recall layer (a detector miss); always persisted in review


class MaskEncoding(str, Enum):
    rle = "rle"
    polygon = "polygon"


class BBox(BaseModel):
    """Axis-aligned box in pixel coordinates, xyxy."""

    model_config = ConfigDict(frozen=True)
    x1: float
    y1: float
    x2: float
    y2: float

    def as_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @classmethod
    def from_list(cls, v: list[float]) -> BBox:
        return cls(x1=v[0], y1=v[1], x2=v[2], y2=v[3])

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


class Detection(BaseModel):
    """A single path proposal, before fusion."""

    path: str  # path_a_yolo26 | path_b_sam3 | path_c_qwen3vl
    class_name: str | None = None
    class_id: int | None = None
    bbox: BBox
    conf: float
    mask_uri: str | None = None
    mask_encoding: MaskEncoding | None = None
    model_version: str
    extra: dict = Field(default_factory=dict)


class PathProposal(BaseModel):
    """What a path contributed to a fused cluster, retained in provenance."""

    path: str
    class_name: str | None = None
    conf: float | None = None
    verdict: str  # proposed | agree | overruled | confirm | miss | unsure
    model_version: str


class Provenance(BaseModel):
    proposals: list[PathProposal] = Field(default_factory=list)
    agreement: bool = False
    mask_box_disagree: bool = False
    raw_conf: dict[str, float] = Field(default_factory=dict)
    calibrated_from: float | None = None
    ontology_version: str | None = None
    notes: list[str] = Field(default_factory=list)
    # M-Q.4 quality reviewer demotion reasons (above_horizon, impossible_size, part_of_vehicle, ...). Empty
    # when the object passed; persisted so confirmed demotions feed the correction and retrain loop.
    quality_flags: list[str] = Field(default_factory=list)


class UnifiedObject(BaseModel):
    """The internal object contract. One per fused cluster."""

    object_id: UUID | None = None
    frame_id: UUID | None = None  # a 3D proposal on a cloud may have no synchronized camera frame
    track_id: UUID | None = None
    class_id: int
    class_name: str
    bbox: BBox
    mask_uri: str | None = None
    mask_encoding: MaskEncoding | None = None
    attrs: dict = Field(default_factory=dict)
    conf: float
    source: ObjectSource = ObjectSource.fused
    provenance: Provenance = Field(default_factory=Provenance)
    state: GateState = GateState.review


class FrameMeta(BaseModel):
    frame_id: UUID
    session_id: UUID
    ts_ns: int
    cam_id: str
    img_uri: str
    width: int
    height: int
    lat: float | None = None
    lon: float | None = None
    ego_speed: float | None = None
    quality: float


class SessionManifest(BaseModel):
    session_id: UUID
    vehicle_id: str
    t_start_ns: int
    t_end_ns: int
    city: str | None = None
    route: str | None = None
    streams: list[str] = Field(default_factory=list)
    sensors: dict = Field(default_factory=dict)  # per-sensor serial + calibration hash
    gps_track: list[list[float]] = Field(default_factory=list)  # [lat, lon, ts_ns]
    n_frames: int = 0
    raw_uri: str | None = None
    mcap_uri: str | None = None
    ontology_version: str
    qa: dict = Field(default_factory=dict)
