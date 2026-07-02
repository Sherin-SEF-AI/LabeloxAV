"""SQLAlchemy 2.0 models. Postgres is the system of record; blobs live in MinIO and tables hold
URIs plus structured truth. The object table is the join hub for the provenance walk.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from geoalchemy2 import Geography
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import (
    ARRAY as PGARRAY,  # noqa: F401  (scenario_candidate.rare_classes)
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class OntologyVersion(Base):
    __tablename__ = "ontology_version"

    version: Mapped[str] = mapped_column(String(64), primary_key=True)
    hierarchy_levels: Mapped[int] = mapped_column(Integer, nullable=False)
    attributes: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    classes: Mapped[list[OntologyClass]] = relationship(back_populates="ontology", cascade="all, delete-orphan")


class OntologyClass(Base):
    __tablename__ = "ontology_class"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[str] = mapped_column(ForeignKey("ontology_version.version", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    l0: Mapped[str] = mapped_column(String(32), nullable=False)
    l1: Mapped[str] = mapped_column(String(32), nullable=False)
    india: Mapped[bool] = mapped_column(Boolean, default=False)
    map_to: Mapped[dict] = mapped_column(JSONB, default=dict)  # COCO/KITTI/nuScenes crosswalk

    ontology: Mapped[OntologyVersion] = relationship(back_populates="classes")

    __table_args__ = (Index("ix_ontology_class_name", "name"),)


class Session(Base):
    __tablename__ = "session"

    session_id: Mapped[uuid.UUID] = _uuid_pk()
    vehicle_id: Mapped[str] = mapped_column(String(64), nullable=False)
    start_ts_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    end_ts_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    city: Mapped[str | None] = mapped_column(String(64))
    route: Mapped[str | None] = mapped_column(String(128))
    sensors: Mapped[dict] = mapped_column(JSONB, default=dict)  # per-sensor serial + calib hash
    raw_uri: Mapped[str | None] = mapped_column(Text)
    mcap_uri: Mapped[str | None] = mapped_column(Text)
    manifest_uri: Mapped[str | None] = mapped_column(Text)
    ontology_version: Mapped[str] = mapped_column(String(64), nullable=False)
    commit_id: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    frames: Mapped[list[Frame]] = relationship(back_populates="session")

    __table_args__ = (Index("ix_session_start_ts", "start_ts_ns"),)


class Frame(Base):
    __tablename__ = "frame"

    frame_id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("session.session_id", ondelete="CASCADE"))
    ts_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    cam_id: Mapped[str] = mapped_column(String(32), nullable=False)
    img_uri: Mapped[str] = mapped_column(Text, nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    gnss: Mapped[str | None] = mapped_column(Geography(geometry_type="POINT", srid=4326))
    ego_speed: Mapped[float | None] = mapped_column(Float)
    quality: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Data Intelligence Layer (Phase 1), all nullable + additive:
    scene: Mapped[dict | None] = mapped_column(JSONB)  # {weather,time_of_day,road_type,density,confidence_per_axis}
    dup_group_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    is_dup_canonical: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    dup_score: Mapped[float | None] = mapped_column(Float)
    selected: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)  # intelligent-extraction keep flag
    novelty_score: Mapped[float | None] = mapped_column(Float)

    # Phase 3 map context (from map-matching, all nullable since Indian OSM coverage is variable):
    road_segment_id: Mapped[str | None] = mapped_column(Text)
    road_class: Mapped[str | None] = mapped_column(Text)
    lane_count: Mapped[int | None] = mapped_column(Integer)
    speed_limit: Mapped[int | None] = mapped_column(Integer)

    # LiDAR BEV frames: img_uri is the rasterized bird's-eye view; this holds the point-cloud uri and the
    # BEV projection params so an oriented box drawn on the image lifts back to a metric 3D cuboid.
    lidar: Mapped[dict | None] = mapped_column(JSONB)

    session: Mapped[Session] = relationship(back_populates="frames")
    objects: Mapped[list[Object]] = relationship(back_populates="frame")

    __table_args__ = (
        Index("ix_frame_session_ts", "session_id", "ts_ns"),
        Index("ix_frame_ts", "ts_ns"),
    )


class Track(Base):
    __tablename__ = "track"

    track_id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("session.session_id", ondelete="CASCADE"))
    class_id: Mapped[int] = mapped_column(ForeignKey("ontology_class.id"))
    first_ts_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    last_ts_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    trajectory: Mapped[dict | None] = mapped_column(JSONB)  # per-frame centroids (image + ego frame)
    id_switch_flags: Mapped[dict | None] = mapped_column(JSONB)  # M2.0: flagged re-id/occlusion events
    tracker_version: Mapped[str | None] = mapped_column(String(48))  # M2.0: tracker backend + version

    __table_args__ = (Index("ix_track_session", "session_id"),)


class Object(Base):
    __tablename__ = "object"

    object_id: Mapped[uuid.UUID] = _uuid_pk()
    frame_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("frame.frame_id", ondelete="CASCADE"))
    track_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("track.track_id", ondelete="SET NULL"))
    class_id: Mapped[int] = mapped_column(ForeignKey("ontology_class.id"))
    bbox: Mapped[list[float]] = mapped_column(ARRAY(Float), nullable=False)  # xyxy pixel
    mask_uri: Mapped[str | None] = mapped_column(Text)
    mask_encoding: Mapped[str | None] = mapped_column(String(16))
    attrs: Mapped[dict] = mapped_column(JSONB, default=dict)
    conf: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String(16), nullable=False, default="fused")
    provenance: Mapped[dict] = mapped_column(JSONB, default=dict)
    state: Mapped[str] = mapped_column(String(16), nullable=False, default="review")
    # Optimistic-concurrency version: bumped on every human edit; a stale write is rejected (409) so two
    # annotators on the same object do not silently overwrite each other (last-write-wins).
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    # Optional 3D cuboid (ego frame, metres): {"center":[x,y,z], "size":[w,l,h], "yaw":rad}. Present only
    # when a 3D label exists (LiDAR/cuboid tool); enables a real nuScenes/KITTI 3D export.
    cuboid_3d: Mapped[dict | None] = mapped_column(JSONB)
    # Oriented-box rotation (degrees, clockwise about the box centre). 0 = axis-aligned. Additive: bbox
    # stays the unrotated AABB so export/IPM/dynamics are unchanged; consumers that support oriented boxes
    # use this angle on top of bbox.
    rot_deg: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, server_default="0")
    # Optional keypoints/skeleton (COCO-style, image pixels): {"skeleton": str, "points": [[x,y,v],...]}
    # with v in {0 not-labeled, 1 occluded, 2 visible}. For pedestrian/cyclist pose.
    keypoints: Mapped[dict | None] = mapped_column(JSONB)
    # Open polyline geometry (ordered [[x,y],...], image pixels) for linear features (curb, road_edge,
    # barrier). When present, bbox is the points AABB so export/gate stay consistent (like rot_deg).
    polyline: Mapped[list | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Phase 2 perception (additive, nullable):
    is_keyframe: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)  # M2.5 keyframe
    interp_source: Mapped[str | None] = mapped_column(String(16))  # linear|cubic|sam_propagated (M2.5)
    sign_type: Mapped[str | None] = mapped_column(Text)            # M2.3
    sign_category: Mapped[str | None] = mapped_column(String(16))  # mandatory|cautionary|informatory
    ocr_text: Mapped[str | None] = mapped_column(Text)            # M2.4 (never a license plate)
    ocr_lang: Mapped[str | None] = mapped_column(String(16))
    ocr_conf: Mapped[float | None] = mapped_column(Float)
    rig_track_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))  # M3.1 same object across cameras
    cross_cam_links: Mapped[dict | None] = mapped_column(JSONB)   # M3.1 the same object seen in other views

    frame: Mapped[Frame] = relationship(back_populates="objects")

    __table_args__ = (
        Index("ix_object_frame", "frame_id"),
        Index("ix_object_state", "state"),
        Index("ix_object_class", "class_id"),
        Index("ix_object_track", "track_id"),
    )


class ObjectRelationship(Base):
    # A directed relationship between two objects on a frame: the join hub for grouping that track_id
    # cannot express (rider on a two-wheeler, trailer to truck, parent-child, herd/group membership).
    __tablename__ = "object_relationship"

    relationship_id: Mapped[uuid.UUID] = _uuid_pk()
    from_object_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("object.object_id", ondelete="CASCADE"))
    to_object_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("object.object_id", ondelete="CASCADE"))
    frame_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("frame.frame_id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(24), nullable=False)  # rider_of|towed_by|part_of|member_of|occludes
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_object_relationship_from", "from_object_id"),
                      Index("ix_object_relationship_to", "to_object_id"),
                      Index("ix_object_relationship_frame", "frame_id"))


class Lane(Base):
    # M2.1: a lane line per frame (linked across frames by track_ref). Bezier/B-spline control points in
    # image coordinates; never a raster mask. Implicit/fallback lanes are hand-drawn on unmarked roads.
    __tablename__ = "lane"

    lane_id: Mapped[uuid.UUID] = _uuid_pk()
    frame_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("frame.frame_id", ondelete="CASCADE"))
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("session.session_id", ondelete="CASCADE"))
    track_ref: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))  # same lane across frames
    control_points: Mapped[list] = mapped_column(JSONB, nullable=False)  # [[x,y],...] control points
    lane_type: Mapped[str] = mapped_column(String(16), nullable=False)   # solid|dashed|double|road_edge|implicit|fallback
    is_ego: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    source: Mapped[str] = mapped_column(String(16), nullable=False, default="proposed")  # proposed|human|propagated
    model_version: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_lane_frame", "frame_id"), Index("ix_lane_track_ref", "track_ref"))


class DrivableMask(Base):
    # M2.2: ternary surface mask per frame (drivable / non-drivable / fallback). Mask blob lives in MinIO;
    # only the uri + per-class coverage fractions are in Postgres.
    __tablename__ = "drivable_mask"

    frame_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("frame.frame_id", ondelete="CASCADE"), primary_key=True)
    mask_uri: Mapped[str] = mapped_column(Text, nullable=False)
    coverage: Mapped[dict] = mapped_column(JSONB, default=dict)  # {drivable: f, non_drivable: f, fallback: f}
    source: Mapped[str] = mapped_column(String(16), nullable=False, default="proposed")
    model_version: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class User(Base):
    # "user" is a reserved word in Postgres, so the table is app_user. Lightweight: no password (the
    # current user is chosen client-side); role gates the QA workflow (annotator submits, reviewer approves).
    __tablename__ = "app_user"

    user_id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False, default="annotator")  # admin|reviewer|annotator
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Review(Base):
    __tablename__ = "review"

    review_id: Mapped[uuid.UUID] = _uuid_pk()
    object_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("object.object_id", ondelete="CASCADE"))
    reviewer: Mapped[str] = mapped_column(String(64), nullable=False)  # user name (kept for compat)
    user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("app_user.user_id", ondelete="SET NULL"))
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    before: Mapped[dict | None] = mapped_column(JSONB)
    after: Mapped[dict | None] = mapped_column(JSONB)
    time_spent_ms: Mapped[int] = mapped_column(Integer, default=0)
    ts_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)

    __table_args__ = (Index("ix_review_object", "object_id"),)


class Scenario(Base):
    __tablename__ = "scenario"

    scenario_id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("session.session_id", ondelete="CASCADE"))
    type: Mapped[str] = mapped_column(String(32), nullable=False)  # cut_in, near_miss, wrong_side, ...
    t_in_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    t_out_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    actors: Mapped[list] = mapped_column(JSONB, default=list)  # track_id strings
    criticality: Mapped[float] = mapped_column(Float, default=0.0)  # 0..1, TTC/PET-derived
    geo: Mapped[str | None] = mapped_column(Geography(geometry_type="POINT", srid=4326))
    tags: Mapped[list] = mapped_column(JSONB, default=list)  # [dusk, wet, metro, ...]
    clip_ref: Mapped[str | None] = mapped_column(Text)  # mcap/frame ref
    meta: Mapped[dict] = mapped_column(JSONB, default=dict)  # actor_classes, signals, etc.
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_scenario_session", "session_id"),
        Index("ix_scenario_type", "type"),
        Index("ix_scenario_criticality", "criticality"),
    )


class Embedding(Base):
    __tablename__ = "embedding"

    object_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("object.object_id", ondelete="CASCADE"), primary_key=True
    )
    model: Mapped[str] = mapped_column(String(48), nullable=False)
    dim: Mapped[int] = mapped_column(Integer, nullable=False)
    vec: Mapped[list[float]] = mapped_column(ARRAY(Float), nullable=False)  # L2-normalized


class FrameEmbedding(Base):
    # Whole-frame embeddings on pgvector (Data Intelligence Layer). DINOv3 (visual: dedup, novelty,
    # clustering) + SigLIP 2 (text-aligned: semantic search, zero-shot scene). HNSW cosine on both.
    __tablename__ = "frame_embedding"

    frame_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("frame.frame_id", ondelete="CASCADE"), primary_key=True
    )
    dino_vec: Mapped[list[float] | None] = mapped_column(Vector(768))     # DINOv3 ViT-B/16
    siglip_vec: Mapped[list[float] | None] = mapped_column(Vector(1152))  # SigLIP 2 so400m image
    model_versions: Mapped[dict] = mapped_column(JSONB, default=dict)     # exact checkpoints used
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ObjectEmbedding(Base):
    # Per-object-crop DINOv3 features on pgvector, for object-level similarity (find-similar, the
    # correction loop). Supersedes the legacy CLIP `embedding` table.
    __tablename__ = "object_embedding"

    object_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("object.object_id", ondelete="CASCADE"), primary_key=True
    )
    dino_vec: Mapped[list[float]] = mapped_column(Vector(768), nullable=False)
    model_versions: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ScenarioCandidate(Base):
    # Rare-scenario discovery output (M1.5): unusual frames surfaced by embedding novelty or rare-class,
    # routed to a human confirm/dismiss/tag queue. Feeds active learning and sellable rare slices.
    __tablename__ = "scenario_candidate"

    candidate_id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("session.session_id", ondelete="CASCADE"))
    frame_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("frame.frame_id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(24), nullable=False)  # embedding_outlier|sparse_cluster|rare_class
    score: Mapped[float] = mapped_column(Float, nullable=False)
    cluster_id: Mapped[int | None] = mapped_column(Integer)
    rare_classes: Mapped[list[str] | None] = mapped_column(PGARRAY(Text))
    state: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")  # pending|confirmed|dismissed
    tag: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_scenario_candidate_state", "state", "score"),)


class ModelRun(Base):
    __tablename__ = "model_run"

    run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    base_weights: Mapped[str] = mapped_column(String(128), nullable=False)
    weights_uri: Mapped[str | None] = mapped_column(Text)  # MinIO uri of the fine-tuned weights
    dataset_name: Mapped[str] = mapped_column(String(128), nullable=False)
    n_train: Mapped[int] = mapped_column(Integer, default=0)
    n_val: Mapped[int] = mapped_column(Integer, default=0)
    epochs: Mapped[int] = mapped_column(Integer, default=0)
    metrics: Mapped[dict] = mapped_column(JSONB, default=dict)           # candidate eval
    baseline_metrics: Mapped[dict] = mapped_column(JSONB, default=dict)  # base eval on same val
    gate: Mapped[dict] = mapped_column(JSONB, default=dict)              # promote decision + reasons
    promoted: Mapped[bool] = mapped_column(Boolean, default=False)
    ontology_version: Mapped[str] = mapped_column(String(64), nullable=False)
    # Registry generalization: a model "line" is (purpose, task_type). The active model for a purpose
    # is the latest promoted row for that purpose. job_id links back to the training_job that made it.
    purpose: Mapped[str] = mapped_column(String(64), nullable=False, default="perception")
    task_type: Mapped[str] = mapped_column(String(32), nullable=False, default="detection")
    job_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DatasetCommit(Base):
    __tablename__ = "dataset_commit"

    commit_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    parent_id: Mapped[str | None] = mapped_column(String(128))
    slice_spec: Mapped[dict] = mapped_column(JSONB, default=dict)
    object_count: Mapped[int] = mapped_column(Integer, default=0)
    object_3d_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    cloud_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    ontology_version: Mapped[str] = mapped_column(String(64), nullable=False)
    export_uris: Mapped[dict] = mapped_column(JSONB, default=dict)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PiiAudit(Base):
    __tablename__ = "pii_audit"

    frame_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("frame.frame_id", ondelete="CASCADE"), primary_key=True
    )
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("session.session_id", ondelete="CASCADE"))
    n_faces: Mapped[int] = mapped_column(Integer, default=0)
    n_plates: Mapped[int] = mapped_column(Integer, default=0)
    regions: Mapped[list] = mapped_column(JSONB, default=list)  # [{type, bbox:[x1,y1,x2,y2], score}]
    method_version: Mapped[str] = mapped_column(String(64), nullable=False)
    ts_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_pii_audit_session", "session_id"),)


class GoldSet(Base):
    __tablename__ = "gold_set"

    gold_id: Mapped[str] = mapped_column(String(128), primary_key=True)  # content-addressed
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    spec: Mapped[dict] = mapped_column(JSONB, default=dict)
    object_ids: Mapped[list] = mapped_column(JSONB, default=list)  # frozen, sealed
    n_objects: Mapped[int] = mapped_column(Integer, default=0)
    n_frames: Mapped[int] = mapped_column(Integer, default=0)
    ontology_version: Mapped[str] = mapped_column(String(64), nullable=False)
    metrics: Mapped[dict] = mapped_column(JSONB, default=dict)  # eval cached at seal time (optional)
    data_yaml_uri: Mapped[str | None] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ImportJob(Base):
    __tablename__ = "import_job"

    job_id: Mapped[uuid.UUID] = _uuid_pk()
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")  # pending|running|done|error
    format: Mapped[str] = mapped_column(String(32), nullable=False)
    source_uri: Mapped[str | None] = mapped_column(Text)
    target_vehicle: Mapped[str] = mapped_column(String(64), nullable=False)
    city: Mapped[str | None] = mapped_column(String(64))
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    counts: Mapped[dict] = mapped_column(JSONB, default=dict)  # sessions, frames, objects, unmapped, dedup_hits
    error: Mapped[str | None] = mapped_column(Text)
    session_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (Index("ix_import_job_status", "status"),)


class TrainingJob(Base):
    __tablename__ = "training_job"

    job_id: Mapped[uuid.UUID] = _uuid_pk()
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")  # pending|running|done|error|canceled
    purpose: Mapped[str] = mapped_column(String(64), nullable=False)
    task_type: Mapped[str] = mapped_column(String(32), nullable=False, default="detection")
    compute_target: Mapped[str] = mapped_column(String(16), nullable=False, default="local")  # local|cloud
    config: Mapped[dict] = mapped_column(JSONB, default=dict)    # the full TrainJobSpec
    stage: Mapped[str | None] = mapped_column(String(24))        # build|train|evaluate|gate|done
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    counts: Mapped[dict] = mapped_column(JSONB, default=dict)    # epoch, total_epochs, n_train, n_val
    metrics: Mapped[dict] = mapped_column(JSONB, default=dict)   # live/candidate eval cache
    result: Mapped[dict] = mapped_column(JSONB, default=dict)    # run_id, weights_uri, gate, promoted
    error: Mapped[str | None] = mapped_column(Text)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    run_id: Mapped[str | None] = mapped_column(String(128))      # link to model_run once recorded
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (Index("ix_training_job_status", "status"),)


class ExportJob(Base):
    __tablename__ = "export_job"

    job_id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")  # pending|running|done|error
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    spec: Mapped[dict] = mapped_column(JSONB, default=dict)
    commit_id: Mapped[str | None] = mapped_column(String(128))  # set on completion
    object_count: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (Index("ix_export_job_status", "status"),)


class AutolabelJob(Base):
    __tablename__ = "autolabel_job"

    job_id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")  # pending|running|done|error
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    counts: Mapped[dict] = mapped_column(JSONB, default=dict)  # frames, objects, by_state
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (Index("ix_autolabel_job_status", "status"),)


class CloudSession(Base):
    # A warm cloud-GPU session: one RunPod pod held up across a work session and torn down on disconnect
    # (distinct from the ephemeral per-job burst flow). At most one row is in a live state at a time. The
    # row is the source of truth for the cost meter, the idle/max-session guards, and orphan detection on
    # app load, so a connected GPU can never silently run: started_at + idle_since drive auto-terminate.
    __tablename__ = "cloud_session"

    session_id: Mapped[uuid.UUID] = _uuid_pk()
    pod_id: Mapped[str | None] = mapped_column(Text)
    mode: Mapped[str] = mapped_column(String(8), nullable=False, default="warm")
    # disconnected | provisioning | connected | running_job | pausing | terminating
    state: Mapped[str] = mapped_column(String(16), nullable=False, default="provisioning")
    gpu_type: Mapped[str | None] = mapped_column(String(64))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))  # when the pod went RUNNING
    idle_since: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))  # null while a job runs
    gpu_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    est_cost: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    max_session_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_job_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_cloud_session_state", "state"),)


# ---- Phase 3 Multi-Sensor and Spatial ----
class CameraRig(Base):
    # The rig layout for a vehicle config: per-camera lens type + intrinsics/extrinsics references.
    __tablename__ = "camera_rig"

    rig_id: Mapped[uuid.UUID] = _uuid_pk()
    vehicle_id: Mapped[str] = mapped_column(String(64), nullable=False)
    cameras: Mapped[dict] = mapped_column(JSONB, default=dict)  # {cam_id: {lens, intrinsics, extrinsics}}
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class CalibrationValidation(Base):
    # One row per camera per session: the validation verdict that gates 3D + multi-camera work.
    __tablename__ = "calibration_validation"

    id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("session.session_id", ondelete="CASCADE"))
    cam_id: Mapped[str] = mapped_column(String(32), nullable=False)
    model: Mapped[str] = mapped_column(String(16), nullable=False)  # pinhole | fisheye
    reproj_error_px: Mapped[float | None] = mapped_column(Float)
    fov_check: Mapped[dict] = mapped_column(JSONB, default=dict)            # implied vs configured FOV
    extrinsic_consistency: Mapped[dict | None] = mapped_column(JSONB)      # epipolar + IMU residuals
    time_offset_ns: Mapped[int | None] = mapped_column(BigInteger)
    status: Mapped[str] = mapped_column(String(8), nullable=False, default="pass")  # pass | warn | fail
    report_uri: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_calib_session", "session_id"),)


class CameraCalibration(Base):
    # M-CAL.1: the resolved per-session, per-camera calibration the 3D pipeline reads. Intrinsics are stored
    # at ref_width and scaled to the actual image; extrinsics are the full 6-DOF camera->ego mount pose
    # (rpy + xyz), generalizing the nominal yaw + height. source records how it was obtained (measured |
    # dataset | estimated | nominal) so a cuboid's trust follows its calibration. Absent -> nominal fallback.
    __tablename__ = "camera_calibration"

    id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("session.session_id", ondelete="CASCADE"))
    cam_id: Mapped[str] = mapped_column(String(32), nullable=False)
    model: Mapped[str] = mapped_column(String(16), nullable=False)             # pinhole | fisheye
    fx: Mapped[float] = mapped_column(Float, nullable=False)
    fy: Mapped[float] = mapped_column(Float, nullable=False)
    cx: Mapped[float] = mapped_column(Float, nullable=False)
    cy: Mapped[float] = mapped_column(Float, nullable=False)
    dist: Mapped[list] = mapped_column(JSONB, default=list)                    # distortion coefficients
    ref_width: Mapped[int] = mapped_column(Integer, nullable=False)            # image width the intrinsics fit
    rpy_deg: Mapped[list] = mapped_column(JSONB, default=list)                 # [roll, pitch, yaw] ego->cam
    xyz_m: Mapped[list] = mapped_column(JSONB, default=list)                   # camera mount position in ego
    source: Mapped[str] = mapped_column(String(16), nullable=False, default="nominal")
    quality: Mapped[float] = mapped_column(Float, nullable=False, default=0.3)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_camera_calibration_session_cam", "session_id", "cam_id", unique=True),)


class TimelineEvent(Base):
    # Milestone B: a human or auto event on the canonical session timeline. modality is which signal it lives
    # on (imu, audio, scene, geo, crossmodal); a crossmodal event binds an inertial spike, a frame, and an
    # audio region at one instant. source=auto events are unconfirmed candidates (state=review), never
    # auto-accepted. Optimistic concurrency via version, the same as Object.
    __tablename__ = "timeline_event"

    event_id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("session.session_id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(40), nullable=False)
    modality: Mapped[str] = mapped_column(String(16), nullable=False)        # imu|audio|scene|geo|crossmodal
    t_start_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    t_end_ns: Mapped[int | None] = mapped_column(BigInteger)                 # null = a point event
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    source: Mapped[str] = mapped_column(String(16), nullable=False, default="human")  # human|auto|correlated
    state: Mapped[str] = mapped_column(String(16), nullable=False, default="review")  # review|confirmed|rejected
    provenance: Mapped[dict] = mapped_column(JSONB, default=dict)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_timeline_event_session_t", "session_id", "t_start_ns"),)


class SpeechSegment(Base):
    # Milestone D: a detected human-speech region on a session's audio, the third DPDPA modality alongside
    # face and plate. is_personal defaults True (speech is personal until confirmed otherwise); redacted is
    # False until the audio is masked. The unified export gate refuses any clip with a personal, un-redacted
    # speech segment, the same fail-closed posture as un-redacted faces and plates.
    __tablename__ = "speech_segment"

    segment_id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("session.session_id", ondelete="CASCADE"))
    t_start_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    t_end_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    is_personal: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="true")
    redacted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    method_version: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_speech_segment_session", "session_id"),)


class CurationSlice(Base):
    # Milestone I: a named, persisted dataset cohort. predicate is a query over the SigLIP2 scene axes
    # (weather, time_of_day, road_type, density) plus class / state / city / confidence, so a cohort like
    # "rare-class at night in rain" is defined once and reused for export, training, and review instead of
    # re-typing an ad-hoc export SliceSpec each time. version carries optimistic concurrency.
    __tablename__ = "curation_slice"

    slice_id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(String(240))
    predicate: Mapped[dict] = mapped_column(JSONB, default=dict)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class MapCommit(Base):
    # A fused, versioned HD-map output (content-addressed, like DatasetCommit).
    __tablename__ = "map_commit"

    commit_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    region: Mapped[str] = mapped_column(String(128), nullable=False)
    session_ids: Mapped[list[str]] = mapped_column(PGARRAY(Text), default=list)
    element_count: Mapped[int] = mapped_column(Integer, default=0)
    formats: Mapped[dict] = mapped_column(JSONB, default=dict)  # {lanelet2: uri, opendrive: uri}
    calibration_version: Mapped[str | None] = mapped_column(String(64))
    fusion_job_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class MapElement(Base):
    # A geo-referenced map element in world space. Provenance: calibration + source frames + fusion run.
    __tablename__ = "map_element"

    element_id: Mapped[uuid.UUID] = _uuid_pk()
    kind: Mapped[str] = mapped_column(String(16), nullable=False)  # lane|road_edge|sign|signal|crossing
    geometry: Mapped[str | None] = mapped_column(Geography(srid=4326))  # LineString or Point
    attrs: Mapped[dict] = mapped_column(JSONB, default=dict)
    source_frames: Mapped[list[str] | None] = mapped_column(PGARRAY(Text))
    source_sessions: Mapped[list[str] | None] = mapped_column(PGARRAY(Text))
    calibration_version: Mapped[str | None] = mapped_column(String(64))
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    fusion_job_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    commit_id: Mapped[str | None] = mapped_column(ForeignKey("map_commit.commit_id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_map_element_kind", "kind"), Index("ix_map_element_commit", "commit_id"))


class MapFusionJob(Base):
    # The HD-map multi-drive fusion job; compute_target=cloud bursts to the A100 (GTSAM) via the seam.
    __tablename__ = "map_fusion_job"

    job_id: Mapped[uuid.UUID] = _uuid_pk()
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    compute_target: Mapped[str] = mapped_column(String(16), nullable=False, default="local")
    region: Mapped[str] = mapped_column(String(128), nullable=False)
    session_ids: Mapped[list[str]] = mapped_column(PGARRAY(Text), default=list)
    stage: Mapped[str | None] = mapped_column(String(24))
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    counts: Mapped[dict] = mapped_column(JSONB, default=dict)
    result: Mapped[dict] = mapped_column(JSONB, default=dict)
    commit_id: Mapped[str | None] = mapped_column(String(128))
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (Index("ix_map_fusion_status", "status"),)


# ----------------------------------------------------------------------------------------------------
# Phase 4: closed loop and governance (M4.0 to M4.4). All additive. The human becomes a governor.
# ----------------------------------------------------------------------------------------------------


class AlSelection(Base):
    # An active-learning batch: a value-ranked set of items chosen within a human-hour budget (M4.0).
    __tablename__ = "al_selection"

    batch_id: Mapped[uuid.UUID] = _uuid_pk()
    strategy: Mapped[dict] = mapped_column(JSONB, default=dict)        # the weights used
    item_ids: Mapped[list[str]] = mapped_column(PGARRAY(Text), default=list)
    budget_hours: Mapped[float] = mapped_column(Float, default=0.0)
    expected_value: Mapped[dict] = mapped_column(JSONB, default=dict)  # per-item value + totals
    status: Mapped[str] = mapped_column(String(16), default="open")    # open|assigned|done
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_al_selection_status", "status"),)


class ErrorCandidate(Base):
    # A suspected label error on already-accepted data (M4.1): cleanlab, embedding-outlier, or consistency.
    __tablename__ = "error_candidate"

    candidate_id: Mapped[uuid.UUID] = _uuid_pk()
    object_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("object.object_id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(24), nullable=False)      # confident_learning|embedding_outlier|track_inconsistent|cross_cam_inconsistent
    score: Mapped[float] = mapped_column(Float, default=0.0)
    proposed_label: Mapped[dict | None] = mapped_column(JSONB)         # {class_id, class_name} if a fix is suggested
    detail: Mapped[dict] = mapped_column(JSONB, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending|confirmed_error|dismissed
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_error_candidate_status", "status"), Index("ix_error_candidate_object", "object_id"))


class RelabelRun(Base):
    # A bulk relabeling pass with the champion model (M4.2). Lands on its own lakeFS branch, reversible.
    __tablename__ = "relabel_run"

    run_id: Mapped[uuid.UUID] = _uuid_pk()
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    lakefs_branch: Mapped[str | None] = mapped_column(String(128))
    proposed: Mapped[int] = mapped_column(Integer, default=0)
    auto_applied: Mapped[int] = mapped_column(Integer, default=0)
    routed_to_review: Mapped[int] = mapped_column(Integer, default=0)
    regressions_flagged: Mapped[int] = mapped_column(Integer, default=0)
    reason: Mapped[str | None] = mapped_column(Text)                   # e.g. "ontology promotion: vehicle_fallback -> water_tanker"
    job_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RelabelJob(Base):
    # The A100 relabel re-inference burst job (M4.2); compute_target=cloud bursts via the seam.
    __tablename__ = "relabel_job"

    job_id: Mapped[uuid.UUID] = _uuid_pk()
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending|running|done|error
    compute_target: Mapped[str] = mapped_column(String(16), default="local")  # local|cloud
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    session_ids: Mapped[list[str]] = mapped_column(PGARRAY(Text), default=list)
    ontology_promotion: Mapped[dict | None] = mapped_column(JSONB)     # {from_class, to_class} for ontology relabel
    stage: Mapped[str | None] = mapped_column(String(24))              # build|infer|diff|apply|done|queued-cloud
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    counts: Mapped[dict] = mapped_column(JSONB, default=dict)
    result: Mapped[dict] = mapped_column(JSONB, default=dict)          # run_id, lakefs_branch, applied, routed
    run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (Index("ix_relabel_job_status", "status"),)


class ModelRegistry(Base):
    # The champion and challengers (M4.4). References a ModelRun by version; gold_metrics carry Safe-mIoU.
    __tablename__ = "model_registry"

    model_version: Mapped[str] = mapped_column(String(128), primary_key=True)
    task: Mapped[str] = mapped_column(String(32), default="detection")
    gold_metrics: Mapped[dict] = mapped_column(JSONB, default=dict)    # map, per_class, safe_miou, mask_iou, MOTA, IDF1
    is_champion: Mapped[bool] = mapped_column(Boolean, default=False)
    promoted_from: Mapped[str | None] = mapped_column(String(128))     # the prior champion it beat
    dataset_commit: Mapped[str | None] = mapped_column(String(128))
    weights_uri: Mapped[str | None] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_model_registry_champion", "task", "is_champion"),)


class ControlSample(Base):
    # The always-reviewed random stream (M4.4): even auto-accepted, so we measure true auto-accept precision.
    __tablename__ = "control_sample"

    sample_id: Mapped[uuid.UUID] = _uuid_pk()
    object_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("object.object_id", ondelete="CASCADE"))
    was_auto_accepted: Mapped[bool] = mapped_column(Boolean, default=False)
    human_verdict: Mapped[str | None] = mapped_column(String(16))      # correct|incorrect (null until reviewed)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_control_sample_verdict", "human_verdict"),)


class DriftMetric(Base):
    # Drift over time (M4.4): input embeddings, label distribution, control-sample precision.
    __tablename__ = "drift_metric"

    id: Mapped[uuid.UUID] = _uuid_pk()
    metric: Mapped[str] = mapped_column(String(24), nullable=False)    # input_embedding|label_distribution|control_precision
    window: Mapped[dict] = mapped_column(JSONB, default=dict)          # {ref, cur} descriptors
    value: Mapped[float] = mapped_column(Float, default=0.0)
    breach: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_drift_metric_created", "metric", "created_at"),)


class Assignment(Base):
    # Collaboration (M4.3): an item assigned to a user, worked on an isolated branch.
    __tablename__ = "assignment"

    assignment_id: Mapped[uuid.UUID] = _uuid_pk()
    item_id: Mapped[str] = mapped_column(String(128), nullable=False)  # object_id or al batch item ref
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("app_user.user_id", ondelete="CASCADE"))
    branch: Mapped[str | None] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(16), default="assigned")  # assigned|in_progress|submitted|done
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_assignment_user", "user_id", "status"),)


class MergeRequest(Base):
    # Collaboration (M4.3): a reviewed merge of an annotator/experiment branch to main, with attribution.
    __tablename__ = "merge_request"

    mr_id: Mapped[uuid.UUID] = _uuid_pk()
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    source_branch: Mapped[str] = mapped_column(String(128), nullable=False)
    target_branch: Mapped[str] = mapped_column(String(128), default="main")
    author_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    reviewer_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    status: Mapped[str] = mapped_column(String(16), default="open")    # open|approved|merged|rejected|reverted
    merge_commit: Mapped[str | None] = mapped_column(String(128))
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (Index("ix_merge_request_status", "status"),)


class AuditDecision(Base):
    # The audit trail (M4.4): every automated decision, for unattended-run safety and buyer diligence.
    __tablename__ = "audit_decision"

    audit_id: Mapped[uuid.UUID] = _uuid_pk()
    actor: Mapped[str] = mapped_column(String(32), default="controller")  # controller|champion|relabel|gate|drift|killswitch
    decision: Mapped[str] = mapped_column(String(48), nullable=False)     # promote|reject|auto_apply|route_review|pause|rollback|select|...
    subject: Mapped[str | None] = mapped_column(String(128))              # model_version|object_id|batch_id|...
    rationale: Mapped[dict] = mapped_column(JSONB, default=dict)          # the inputs and reasons (deterministic, replayable)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_audit_created", "created_at"), Index("ix_audit_actor", "actor"))


class GovernanceState(Base):
    # Singleton control row (M4.4): the kill switch and autonomy flags the controller reads each tick.
    __tablename__ = "governance_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    loop_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    auto_accept_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    auto_promote_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    champion_version: Mapped[str | None] = mapped_column(String(128))
    paused_reason: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class ObjectDynamics(Base):
    # Derived per-object motion state (P3): distance, speed, heading, closing speed, time-to-collision, and
    # a risk level, computed from the M2.0 track + ego CAN speed + the Phase 3 IPM ground-plane. One row per
    # object (a detection in a frame). Monocular estimate (no LiDAR): distance is approximate, so method and
    # confidence record how it was derived. Computed, never hand-labeled.
    __tablename__ = "object_dynamics"

    object_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("object.object_id", ondelete="CASCADE"), primary_key=True)
    track_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    frame_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    ts_ns: Mapped[int | None] = mapped_column(BigInteger)
    distance_m: Mapped[float | None] = mapped_column(Float)
    lateral_m: Mapped[float | None] = mapped_column(Float)
    speed_kmh: Mapped[float | None] = mapped_column(Float)
    closing_speed_kmh: Mapped[float | None] = mapped_column(Float)
    heading_deg: Mapped[float | None] = mapped_column(Float)
    ttc_s: Mapped[float | None] = mapped_column(Float)
    risk_level: Mapped[str | None] = mapped_column(String(8))  # low|medium|high
    method: Mapped[str] = mapped_column(String(32), default="ipm_mono_v1")
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_object_dynamics_track", "track_id"), Index("ix_object_dynamics_frame", "frame_id"))


# ---- LiDAR module (3D) ----
class PointCloud(Base):
    """One row per scan (real LiDAR) or per synthesized cloud (pseudo-LiDAR), from any source. ts_ns is on
    the PPS base, so a cloud and the camera frames captured at the same ts_ns in the session are one query."""
    __tablename__ = "point_cloud"

    cloud_id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("session.session_id", ondelete="CASCADE"))
    ts_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    source: Mapped[str] = mapped_column(String(16), nullable=False)        # lidar | pseudo | dataset
    cloud_uri: Mapped[str] = mapped_column(Text, nullable=False)            # compressed npz in the object store
    point_count: Mapped[int] = mapped_column(Integer, nullable=False)
    depth_model: Mapped[str | None] = mapped_column(String(96))             # pinned checkpoint, for pseudo-LiDAR
    calibration_version: Mapped[str | None] = mapped_column(String(64))
    bounds: Mapped[dict | None] = mapped_column(JSONB)                      # 3D extent {min,max,n}
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_point_cloud_session_ts", "session_id", "ts_ns"),)


class PointCloudDerived(Base):
    """A cleaned or ground-removed variant of a cloud. Raw is immutable: derived variants never overwrite it."""
    __tablename__ = "point_cloud_derived"

    derived_id: Mapped[uuid.UUID] = _uuid_pk()
    cloud_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("point_cloud.cloud_id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(24), nullable=False)          # ground_removed | denoised | ground_plane
    uri: Mapped[str] = mapped_column(Text, nullable=False)
    method: Mapped[str] = mapped_column(String(48), nullable=False)
    params: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_point_cloud_derived_cloud", "cloud_id"),)


class LidarCalibrationValidation(Base):
    """Extends the Phase 3 calibration concept to the LiDAR triple. A failing session is flagged and excluded
    from 3D work until fixed, exactly as the 2D calibration validation does."""
    __tablename__ = "lidar_calibration_validation"

    id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("session.session_id", ondelete="CASCADE"))
    pair: Mapped[str] = mapped_column(String(24), nullable=False)          # lidar_camera | lidar_imu | lidar_radar
    reproj_error: Mapped[float | None] = mapped_column(Float)
    consistency: Mapped[dict] = mapped_column(JSONB, default=dict)
    drift_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    status: Mapped[str] = mapped_column(String(8), nullable=False, default="pass")  # pass | warn | fail
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_lidar_calib_session", "session_id"),)


# ---- LiDAR module Phase 2 (3D annotation) ----
class Track3D(Base):
    """A 3D track, linked to the M2.0 2D track (track_id) so the 3D and 2D tracks are the same physical
    object. trajectory holds per-frame 3D centroids; dynamic_state is moving/stopped/parked/turning/braking."""
    __tablename__ = "track_3d"

    track_3d_id: Mapped[uuid.UUID] = _uuid_pk()
    track_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("track.track_id", ondelete="SET NULL"))
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("session.session_id", ondelete="CASCADE"))
    class_id: Mapped[int] = mapped_column(ForeignKey("ontology_class.id"))
    first_ts_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    last_ts_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    trajectory: Mapped[dict | None] = mapped_column(JSONB)          # per-frame 3D centroids + yaw
    dynamic_state: Mapped[str | None] = mapped_column(String(16))   # moving|stopped|parked|turning|braking

    __table_args__ = (Index("ix_track_3d_session", "session_id"), Index("ix_track_3d_track", "track_id"))


class Object3D(Base):
    """One 3D cuboid. object_id links it to the 2D Object (the unifying identity, one physical object across
    its 2D box, mask, 3D cuboid, and multi-camera views). The same governed ontology and gate apply: class_id
    is an ontology class, conf is calibrated, box_source records lifted vs native, provenance is one walk."""
    __tablename__ = "object_3d"

    object_3d_id: Mapped[uuid.UUID] = _uuid_pk()
    cloud_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("point_cloud.cloud_id", ondelete="CASCADE"))
    frame_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("frame.frame_id", ondelete="SET NULL"))
    object_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("object.object_id", ondelete="SET NULL"))
    track_3d_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("track_3d.track_3d_id", ondelete="SET NULL"))
    class_id: Mapped[int] = mapped_column(ForeignKey("ontology_class.id"))
    center: Mapped[list[float]] = mapped_column(ARRAY(Float), nullable=False)   # [x, y, z] ego metres
    dims: Mapped[list[float]] = mapped_column(ARRAY(Float), nullable=False)     # [L, W, H] metres
    yaw: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    pitch: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    roll: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    conf: Mapped[float] = mapped_column(Float, nullable=False)                  # calibrated
    box_source: Mapped[str] = mapped_column(String(8), nullable=False)         # lifted | native
    is_keyframe: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    interp_source: Mapped[str | None] = mapped_column(String(16))              # linear | slerp
    source: Mapped[str] = mapped_column(String(16), nullable=False, default="fused")  # fused|auto_accept|human
    state: Mapped[str] = mapped_column(String(16), nullable=False, default="review")
    attrs: Mapped[dict] = mapped_column(JSONB, default=dict)                    # occlusion, dynamics, auto props
    provenance: Mapped[dict] = mapped_column(JSONB, default=dict)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_object_3d_cloud", "cloud_id"),
        Index("ix_object_3d_frame", "frame_id"),
        Index("ix_object_3d_object", "object_id"),
        Index("ix_object_3d_track", "track_3d_id"),
    )


class PointSegmentation(Base):
    """Per-point semantic and instance labels on a cloud. labels_uri points to the arrays in the object store
    (semantic class id and instance id per point); low_conf_frac flags how much was uncertain on pseudo-LiDAR,
    which is surfaced for review rather than trusted blindly."""
    __tablename__ = "point_segmentation"

    seg_id: Mapped[uuid.UUID] = _uuid_pk()
    cloud_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("point_cloud.cloud_id", ondelete="CASCADE"))
    labels_uri: Mapped[str] = mapped_column(Text, nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)              # semantic | panoptic
    method: Mapped[str | None] = mapped_column(String(32))                     # ptv3 | projected_2d
    n_points: Mapped[int | None] = mapped_column(Integer)
    low_conf_frac: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_point_segmentation_cloud", "cloud_id"),)


# ---- LiDAR module Phase 3 (3D scene intelligence + export) ----
class StaticElement(Base):
    """An extracted persistent 3D map element (pole, road edge, building, vegetation, marking). Geo-referenced
    into world space and fed to the existing HD map pipeline as a MapElement. Provenance: source clouds, the
    extraction method, and the calibration that placed it."""
    __tablename__ = "static_element"

    element_id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("session.session_id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(20), nullable=False)   # pole|road_edge|curb|median|building|...
    geometry: Mapped[str | None] = mapped_column(Geography(srid=4326))   # Point or LineString or Polygon
    attrs: Mapped[dict] = mapped_column(JSONB, default=dict)
    source_clouds: Mapped[list[uuid.UUID] | None] = mapped_column(PGARRAY(UUID(as_uuid=True)))
    method: Mapped[str | None] = mapped_column(String(40))
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    calibration_version: Mapped[str | None] = mapped_column(String(64))
    map_element_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))   # the fed HD map element
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_static_element_session", "session_id"), Index("ix_static_element_kind", "kind"))


class Traversability(Base):
    """3D free space, drivable surface, road-surface class, and elevation profile for a cloud or an aggregated
    tile. Grids live in the object store; the surface and elevation summaries are inline."""
    __tablename__ = "traversability"

    id: Mapped[uuid.UUID] = _uuid_pk()
    cloud_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("point_cloud.cloud_id", ondelete="CASCADE"))
    tile_id: Mapped[str | None] = mapped_column(String(64))
    freespace_uri: Mapped[str | None] = mapped_column(Text)
    drivable_uri: Mapped[str | None] = mapped_column(Text)
    surface_class: Mapped[dict] = mapped_column(JSONB, default=dict)
    elevation_profile: Mapped[dict] = mapped_column(JSONB, default=dict)
    method: Mapped[str | None] = mapped_column(String(40))
    calibration_version: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_traversability_cloud", "cloud_id"),)


class AggregatedMap(Base):
    """A registered multi-scan, multi-drive map: scans aligned and accumulated into a dense cloud, with the
    pose graph and any loop closures that corrected it."""
    __tablename__ = "aggregated_map"

    agg_id: Mapped[uuid.UUID] = _uuid_pk()
    region: Mapped[str | None] = mapped_column(String(64))
    session_ids: Mapped[list[uuid.UUID] | None] = mapped_column(PGARRAY(UUID(as_uuid=True)))
    cloud_uri: Mapped[str | None] = mapped_column(Text)
    pose_graph: Mapped[dict] = mapped_column(JSONB, default=dict)
    loop_closures: Mapped[dict] = mapped_column(JSONB, default=dict)
    method: Mapped[str | None] = mapped_column(String(40))
    n_scans: Mapped[int | None] = mapped_column(Integer)
    mean_reg_fitness: Mapped[float | None] = mapped_column(Float)   # low -> flagged low-confidence registration
    input_calibrations: Mapped[dict | None] = mapped_column(JSONB)  # cloud_id -> calibration_version provenance
    job_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_aggregated_map_region", "region"),)


class QualityFlag3D(Base):
    """A detected 3D label problem (floating, below ground, impossible dims, duplicate, misaligned, missing
    neighbour). Feeds the same review and active-learning loop as the 2D quality reviewer."""
    __tablename__ = "quality_flag_3d"

    flag_id: Mapped[uuid.UUID] = _uuid_pk()
    object_3d_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("object_3d.object_3d_id", ondelete="CASCADE"))
    cloud_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("point_cloud.cloud_id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(20), nullable=False)   # floating|below_ground|impossible_dims|...
    score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    detail: Mapped[dict] = mapped_column(JSONB, default=dict)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="open")  # open|confirmed|dismissed
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_quality_flag_3d_object", "object_3d_id"),
                      Index("ix_quality_flag_3d_status", "status"))


class RecallCandidate(Base):
    # Recall recovery audit row: one per recovered miss, linking the provisional review-state Object to the
    # channels that proposed it and the human verdict (status). The verdict recalibrates each channel's
    # precision prior, closing the recall loop the way the isotonic curve closes the precision loop.
    __tablename__ = "recall_candidate"

    candidate_id: Mapped[uuid.UUID] = _uuid_pk()
    object_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("object.object_id", ondelete="CASCADE"))
    frame_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("frame.frame_id", ondelete="CASCADE"))
    channels: Mapped[list[str]] = mapped_column(PGARRAY(String(16)))  # trackgap|openvocab|region
    fn_value: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    class_id: Mapped[int] = mapped_column(ForeignKey("ontology_class.id"))
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")  # pending|confirmed|rejected
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_recall_candidate_status", "status"),
                      Index("ix_recall_candidate_frame", "frame_id"))


class AdverseRegion(Base):
    # A tagged image region affected by an adverse condition (glare, reflection, shadow, rain, fog,
    # lowlight). Frame-level and multi-region (unlike the single drivable mask), each a polygon plus a
    # condition label, so a model knows which pixels to distrust.
    __tablename__ = "adverse_region"

    region_id: Mapped[uuid.UUID] = _uuid_pk()
    frame_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("frame.frame_id", ondelete="CASCADE"))
    geometry: Mapped[list] = mapped_column(JSONB)  # polygon, flattened [x,y,x,y,...] image pixels
    condition: Mapped[str] = mapped_column(String(16), nullable=False)  # glare|reflection|shadow|rain|fog|lowlight
    source: Mapped[str] = mapped_column(String(16), nullable=False, default="human")  # human|proposed
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_adverse_region_frame", "frame_id"),)


class FrameSegmentation(Base):
    # Full-frame dense segmentation: a per-pixel class-id raster (semantic) plus an optional per-pixel
    # instance-id raster (panoptic). Rasters live in MinIO; this row holds the uris, the colored display
    # overlay, per-class coverage, and lineage. One row per frame per kind (semantic|panoptic).
    __tablename__ = "frame_segmentation"

    seg_id: Mapped[uuid.UUID] = _uuid_pk()
    frame_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("frame.frame_id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(16), nullable=False)  # semantic|panoptic
    labels_uri: Mapped[str] = mapped_column(Text, nullable=False)  # class-id per pixel (npz)
    instance_uri: Mapped[str | None] = mapped_column(Text)         # instance-id per pixel (npz), panoptic only
    overlay_uri: Mapped[str | None] = mapped_column(Text)          # colored RGBA png for display
    coverage: Mapped[dict] = mapped_column(JSONB, default=dict)    # {class_name: pixel_fraction}
    segments: Mapped[dict] = mapped_column(JSONB, default=dict)    # panoptic: instance_id -> {class_id, object_id}
    source: Mapped[str] = mapped_column(String(16), nullable=False, default="proposed")  # proposed|human
    model_version: Mapped[str | None] = mapped_column(String(64))
    ontology_version: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_frame_segmentation_frame_kind", "frame_id", "kind"),)


class AgentRun(Base):
    # An auditable, reversible unit of autonomous work by the annotation agent. The agent never mutates
    # objects silently: every run records the policy it applied, per-object state transitions (so a run can
    # be reverted exactly), the critic's findings, and roll-up counts. This is the guardrail that makes
    # auto-accept safe at scale -- a bad run is one row to revert, and provenance never lies about who
    # (which run, which model) touched a label.
    __tablename__ = "agent_run"

    run_id: Mapped[uuid.UUID] = _uuid_pk()
    kind: Mapped[str] = mapped_column(String(32), nullable=False)  # frame|session|flywheel|overnight_auditor|...
    scope: Mapped[dict] = mapped_column(JSONB, default=dict)       # {frame_id?, session_id?, ...}
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="planned")  # planned|committed|reverted|error
    policy: Mapped[dict] = mapped_column(JSONB, default=dict)      # thresholds + toggles the run used
    counts: Mapped[dict] = mapped_column(JSONB, default=dict)      # {auto_accepted, routed_review, escalated, demoted, ...}
    changes: Mapped[dict] = mapped_column(JSONB, default=dict)     # {object_id: {from_state, to_state, from_source, to_source}}
    critic: Mapped[dict] = mapped_column(JSONB, default=dict)      # critic findings summary (by check, by object)
    error: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[str | None] = mapped_column(String(64))     # user id that launched it, or "flywheel"
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    reverted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (Index("ix_agent_run_status", "status"), Index("ix_agent_run_kind", "kind"))


class PromotionProposal(Base):
    """An Ontology Steward evidence packet: a fallback cluster that has grown past the promotion threshold and
    is proposed as a new named class, awaiting a one-click approve/reject. Approval mints the class and
    relabels the cluster (reversibly); rejection records the decision. This is how the ontology grows from a
    reviewed pipeline instead of ad-hoc governance."""

    __tablename__ = "promotion_proposal"

    proposal_id: Mapped[uuid.UUID] = _uuid_pk()
    from_class: Mapped[int] = mapped_column(Integer, nullable=False)   # the fallback class id it split out of
    member_count: Mapped[int] = mapped_column(Integer, nullable=False)
    rep_object_ids: Mapped[list] = mapped_column(JSONB, default=list)  # cluster members (capped) to relabel on approve
    suggested_name: Mapped[str | None] = mapped_column(String(64))     # nearest existing-class hint (human names it)
    confusion_classes: Mapped[list] = mapped_column(JSONB, default=list)  # [{class, share}] visual neighbours
    evidence_uri: Mapped[str | None] = mapped_column(Text)             # crop-grid image in the object store
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="proposed")  # proposed|approved|rejected
    approved_class: Mapped[int | None] = mapped_column(Integer)        # the minted class id, once approved
    run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))  # the reversible relabel run
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (Index("ix_promotion_proposal_status", "status"),)


class CollectionOrder(Base):
    """A Fleet Dispatch proposal: a vehicle sent to a place, in a window, under a forecast, to collect the
    data the corpus is starved of. This closes the acquisition loop the way the labeling agents close the
    labeling loop, and only a platform that owns the fleet can act on it. Proposed by the agent; a human
    dispatches."""

    __tablename__ = "collection_order"

    order_id: Mapped[uuid.UUID] = _uuid_pk()
    vehicle_id: Mapped[str] = mapped_column(String(64), nullable=False)
    city: Mapped[str | None] = mapped_column(String(64))
    area: Mapped[str | None] = mapped_column(String(128))     # route / junction descriptor
    window: Mapped[str | None] = mapped_column(String(32))    # capture time window, e.g. "18:00-22:00"
    target: Mapped[str] = mapped_column(Text, nullable=False)  # the gap it fills, human-readable
    gap_kind: Mapped[str | None] = mapped_column(String(24))  # weather | time_of_day | road_type | class
    forecast: Mapped[str | None] = mapped_column(String(32))  # weather forecast for the window, if known
    priority: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="proposed")  # proposed|dispatched|done
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    created_by: Mapped[str | None] = mapped_column(String(64))

    __table_args__ = (Index("ix_collection_order_status", "status"),)
