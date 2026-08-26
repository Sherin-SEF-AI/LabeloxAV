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
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)

# Aliased: Asset defines a column literally named `text`, which would shadow the bare sqlalchemy.text
# helper inside that class body.
from sqlalchemy import text as sql_text
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
    # The domain pack this session belongs to (routes it to its ontology/scene model/etc). Nullable with an
    # 'av' server default so every pre-pack row backfills to the AV pack.
    pack_id: Mapped[str | None] = mapped_column(String(32), server_default="av")
    # The tenant this drive belongs to. Frames, tracks and objects reach it through session_id, which is why
    # this is the only column in the corpus spine: three tables inherit their tenant from one hop that is
    # already indexed, rather than from a backfill of 576,393 rows.
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("label_project.project_id", ondelete="SET NULL"))
    # Nullable since SEC-M2: a static-camera (CCTV) session has no ego vehicle. The AV ingestion still sets it.
    vehicle_id: Mapped[str | None] = mapped_column(String(64))
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
    # Who set each key in `scene`. The classifier fills scene at ingest with a confidence per axis;
    # without this a value a person corrected is indistinguishable from one a model guessed, and the
    # next classifier pass overwrites the correction with no way to know it did.
    scene_provenance: Mapped[dict | None] = mapped_column(JSONB)
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

    # Free-form curation tags (explorer). Distinct from `scene`, which is model-derived: these are human or
    # bulk-applied marks ("needs_relabel", "golden", "night_rain") used to slice and act on the corpus.
    tags: Mapped[list] = mapped_column(JSONB, nullable=False, default=list, server_default="[]")

    session: Mapped[Session] = relationship(back_populates="frames")
    objects: Mapped[list[Object]] = relationship(back_populates="frame")

    __table_args__ = (
        Index("ix_frame_session_ts", "session_id", "ts_ns"),
        Index("ix_frame_ts", "ts_ns"),
        Index("ix_frame_tags_gin", "tags", postgresql_using="gin"),
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
    # M-F.2 track-level typed intents (proposed/confirmed). The same idea as TrackEvent below without a
    # time extent, and empty: 0 rows across 11,406 tracks. TrackEvent is what an annotator writes; this is
    # kept because services/intelligence/intent.py and the VLM dataset generator still read it, and its
    # vocabulary is the base of the event vocabulary so the two cannot diverge.
    intents: Mapped[list] = mapped_column(JSONB, default=list)

    __table_args__ = (Index("ix_track_session", "session_id"),)


class TrackEvent(Base):
    """A typed span within a track: what this object did, and over which frames.

    The extent is the whole point. `Track.intents` can say a track cut in; it cannot say that a 93-frame
    track cut in over frames 40 to 55, so nothing downstream can crop the clip, count the exposure, or
    compare two events on the same track. An event is a row rather than another JSONB entry because
    "every stopping_in_live_lane in the corpus" is a query the export and the datasheet both need, and
    scanning a JSONB list on 11,406 tracks is not that query.

    The vocabulary is the pack's, through `track_event_schema()`. Validated in the router rather than by a
    check constraint: the vocabulary belongs to whichever domain pack the session was captured under, and a
    constraint would freeze the AV one into the schema for every domain.
    """

    __tablename__ = "track_event"

    event_id: Mapped[uuid.UUID] = _uuid_pk()
    track_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("track.track_id", ondelete="CASCADE"))
    event_type: Mapped[str] = mapped_column(String(48), nullable=False)
    # The span, as frames rather than timestamps. An annotator drags across the crop strip, which is frames,
    # and resolving to a timestamp here would make the stored value depend on a frame's ts_ns being right.
    start_frame_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("frame.frame_id", ondelete="CASCADE"))
    end_frame_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("frame.frame_id", ondelete="CASCADE"))
    # Denormalised from the frames at write time so ordering and overlap are answerable without a join.
    # Written by the router, never by the client.
    start_ts_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    end_ts_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # human | heuristic | vlm. Same vocabulary as Object.source, minus the ones that cannot produce an event.
    source: Mapped[str] = mapped_column(String(16), nullable=False, default="human")
    # proposed | accepted | rejected. A proposer writes `proposed` and only a person moves it off that.
    state: Mapped[str] = mapped_column(String(16), nullable=False, default="proposed")
    confidence: Mapped[float | None] = mapped_column(Float)
    # What the proposer measured, so a reviewer can see why it fired rather than only that it did.
    evidence: Mapped[dict] = mapped_column(JSONB, default=dict)
    notes: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[str | None] = mapped_column(String(128))
    ontology_version: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_track_event_track", "track_id"),
        Index("ix_track_event_type_state", "event_type", "state"),
        CheckConstraint("end_ts_ns >= start_ts_ns", name="ck_track_event_span"),
        CheckConstraint("state in ('proposed','accepted','rejected')", name="ck_track_event_state"),
        CheckConstraint("source in ('human','heuristic','vlm')", name="ck_track_event_source"),
    )


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
    # Who made this label and under which job. Null for everything that predates job attribution and for
    # anything not made inside a job, which is most of the corpus: an autolabel run has no annotator.
    # Without these two, two people's boxes on one frame are one undifferentiated pile and agreement
    # between them cannot be measured at all.
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("label_job.job_id", ondelete="SET NULL"))
    annotator_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("app_user.user_id", ondelete="SET NULL"))
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
    quality_score: Mapped[float | None] = mapped_column(Float)  # M-F.1 composite label-quality QA signal [0,1]
    rig_object_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))  # M-MC one physical object across views at one instant
    rig_track_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))  # M3.1 same object across cameras
    cross_cam_links: Mapped[dict | None] = mapped_column(JSONB)   # M3.1 the same object seen in other views

    # Free-form curation tags (explorer): human or bulk-applied marks used to slice and act on labels.
    # Distinct from `attrs`, which is the ontology-typed attribute schema for this class.
    # How far along the machine-to-human path this label has travelled.
    #
    #   machine_proposed  a model put it here and nobody has looked
    #   machine_accepted  the gate accepted it on confidence, still unseen by a person
    #   human_edited      a person changed its class or geometry
    #   human_confirmed   a person looked and said it is right
    #   track_confirmed   confirmed by spot-checking the track it belongs to
    #
    # Separate from `state`, which is the queue, and from `source`, which is who last wrote the row and
    # collapses to "human" on any touch. None of the three is derivable from the others: an object can be
    # state=accepted, source=human and still never have been confirmed by the person who nudged its box.
    lifecycle: Mapped[str | None] = mapped_column(String(24))
    # Append-only [(state, actor, at)], so a badge that looks wrong can be traced to the write behind it.
    lifecycle_history: Mapped[list] = mapped_column(
        JSONB, nullable=False, default=list, server_default="[]")

    tags: Mapped[list] = mapped_column(JSONB, nullable=False, default=list, server_default="[]")

    # Drawn during a blind audit, where the annotator was shown the pixels and nothing else. Set, this row
    # is one of the two independent observations a capture-recapture estimate is computed from, and it must
    # never be pooled with ordinary review labels: an ordinary label is usually a confirmed machine box, so
    # it carries no information about what the model missed, which is the only thing an audit measures.
    blind_audit_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("blind_audit.audit_id", ondelete="SET NULL"))

    frame: Mapped[Frame] = relationship(back_populates="objects")

    __table_args__ = (
        Index("ix_object_frame", "frame_id"),
        Index("ix_object_state", "state"),
        Index("ix_object_class", "class_id"),
        Index("ix_object_track", "track_id"),
        Index("ix_object_tags_gin", "tags", postgresql_using="gin"),
        # Partial: null on all 576,469 existing objects and on nearly every future one, so indexing the
        # nulls would cover most of the corpus and answer no query anybody asks.
        Index("ix_object_blind_audit", "blind_audit_id",
              postgresql_where=sql_text("blind_audit_id IS NOT NULL")),
    )


class ObjectRelationship(Base):
    # A directed relationship between two objects on a frame: the join hub for grouping that track_id
    # cannot express (rider on a two-wheeler, trailer to truck, parent-child, herd/group membership).
    __tablename__ = "object_relationship"

    relationship_id: Mapped[uuid.UUID] = _uuid_pk()
    from_object_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("object.object_id", ondelete="CASCADE"))
    to_object_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("object.object_id", ondelete="CASCADE"))
    frame_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("frame.frame_id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(32), nullable=False)  # structural OR scene-graph relation (M-F.5)
    status: Mapped[str] = mapped_column(String(16), default="confirmed")  # M-F.5 proposed | confirmed
    source: Mapped[str] = mapped_column(String(16), default="human")      # M-F.5 human | geometry | vlm
    evidence: Mapped[dict | None] = mapped_column(JSONB)                  # M-F.5 why it was proposed
    # How strongly the proposer believes it, in [0, 1]. Numeric strength lived untyped inside `evidence`,
    # where every writer chose its own key and no query could rank or threshold a proposal. Null for a
    # human-drawn edge, which is not a belief.
    conf: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_object_relationship_from", "from_object_id"),
                      Index("ix_object_relationship_to", "to_object_id"),
                      Index("ix_object_relationship_frame", "frame_id"))


class VlmTarget(Base):
    """A generated multimodal training target for a labeled frame (M-F.5): a scene description, hazard list,
    per-agent intent, or ego-action justification, GROUNDED in the frame's actual labels (objects, intents,
    relations) rather than free-hallucinated. Every target records the label ids it was produced from, so it is
    fully traceable, and it must pass human review (status approved) before it can enter a sellable dataset."""

    __tablename__ = "vlm_target"

    target_id: Mapped[uuid.UUID] = _uuid_pk()
    frame_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("frame.frame_id", ondelete="CASCADE"), nullable=False)
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("session.session_id", ondelete="CASCADE"), nullable=False)
    kind: Mapped[str] = mapped_column(String(24), nullable=False)  # scene_description|hazards|agent_intents|ego_action
    content: Mapped[dict] = mapped_column(JSONB, default=dict)     # the structured target
    grounding: Mapped[dict] = mapped_column(JSONB, default=dict)   # {object_ids, track_ids, relation_ids} it came from
    model: Mapped[str | None] = mapped_column(String(48))
    status: Mapped[str] = mapped_column(String(16), default="generated")  # generated | approved | rejected
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_vlm_target_frame", "frame_id"), Index("ix_vlm_target_status", "status"))


class Lane(Base):
    # M2.1: a lane line per frame (linked across frames by track_ref). Bezier/B-spline control points in
    # image coordinates; never a raster mask. Implicit/fallback lanes are hand-drawn on unmarked roads.
    __tablename__ = "lane"

    lane_id: Mapped[uuid.UUID] = _uuid_pk()
    frame_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("frame.frame_id", ondelete="CASCADE"))
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("session.session_id", ondelete="CASCADE"))
    track_ref: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))  # same lane across frames
    control_points: Mapped[list] = mapped_column(JSONB, nullable=False)  # [[x,y],...] control points
    # unknown is a real value and the honest one: the type was measured and the paint did not say. It is not
    # in illegal_to_cross, so a crossing of an unmeasurable line derives as a manoeuvre rather than an
    # offence, which is the safe direction when the evidence is a smear of grey pixels.
    lane_type: Mapped[str] = mapped_column(String(16), nullable=False)   # solid|dashed|double|road_edge|implicit|fallback|unknown
    is_ego: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    source: Mapped[str] = mapped_column(String(16), nullable=False, default="proposed")  # proposed|human|propagated
    model_version: Mapped[str | None] = mapped_column(String(64))
    # How strongly the paint supported the type. Null means nobody measured it, which is different from
    # measured-and-uncertain and has to stay distinguishable.
    marking_conf: Mapped[float | None] = mapped_column(Float)
    provenance: Mapped[dict] = mapped_column(JSONB, default=dict, server_default=sql_text("'{}'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_lane_frame", "frame_id"), Index("ix_lane_track_ref", "track_ref"),
                      Index("ix_lane_session_conf", "session_id", "marking_conf"))


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


class ServiceAccount(Base):
    """A machine credential: an API key bound to an app_user row.

    Every auth path in this system was human, so an integration had to hold a person's password and act as
    them. A service account is deliberately still a user, so role floors, audit trails and `created_by`
    provenance keep working unchanged and a machine's actions are attributable the same way a person's are.

    The key itself is never stored. What is stored is the sha256 of the secret half plus the public prefix,
    which is what the lookup indexes on, so a leaked database yields no usable credential.

    Revocation is a column rather than a token version, because a machine credential has to die the moment
    somebody presses the button. A bearer token is checked against a version and is otherwise valid until it
    expires, which is the wrong trade for something that lives in a CI config for a year.
    """

    __tablename__ = "service_account"

    service_account_id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    # The identity it acts as. CASCADE: deleting the user removes the way to act as them.
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("app_user.user_id", ondelete="CASCADE"), nullable=False)
    # Public half of the key, shown in the UI and used to find the row. Unique so a lookup is exact.
    key_prefix: Mapped[str] = mapped_column(String(24), unique=True, nullable=False, index=True)
    key_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    # Reserved for narrowing a key below its user's role later; empty means the role alone governs.
    scopes: Mapped[list] = mapped_column(JSONB, default=list, server_default="[]")
    description: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("app_user.user_id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # Throttled on write, so "is this key still in use?" is answerable without a write per request.
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class User(Base):
    # "user" is a reserved word in Postgres, so the table is app_user. Lightweight: no password (the
    # current user is chosen client-side); role gates the QA workflow (annotator submits, reviewer approves).
    __tablename__ = "app_user"

    user_id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False, default="annotator")  # admin|reviewer|annotator
    # Per-user token revocation with no session store: every token carries this version, and the verifier
    # rejects a token whose version does not match. Incrementing it invalidates every token for this user at
    # once, without rotating the global signing key (which would log out everyone).
    token_version: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
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


class MachineVerdict(Base):
    """A machine's judgement on an existing label. Deliberately not a Review.

    `review` means a person ruled on this object, and three things depend on that meaning: precision
    sampling excludes reviewed objects, corpus precision reads the states a human moved, and annotator
    scorecards count rows there. A VLM opinion written into that table would corrupt all three invisibly,
    because the rows would look identical.

    Keeping the planes separate is also what makes the method work. A judge has its own error rate, and the
    only way to measure it is to have humans rule on a subsample and compare. There is nothing to compare
    against once the two are mixed.
    """

    __tablename__ = "machine_verdict"

    verdict_id: Mapped[uuid.UUID] = _uuid_pk()
    object_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("object.object_id", ondelete="CASCADE"))
    judge: Mapped[str] = mapped_column(String(32), nullable=False)          # kind of judge, e.g. "vlm"
    provider: Mapped[str] = mapped_column(String(32), nullable=False)       # anthropic | groq | ollama
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    verdict: Mapped[str] = mapped_column(String(16), nullable=False)        # correct | incorrect | unsure
    proposed_class_id: Mapped[int | None] = mapped_column(Integer)
    confidence: Mapped[float | None] = mapped_column(Float)
    agreement: Mapped[float | None] = mapped_column(Float)                  # multi-vote agreement fraction
    detail: Mapped[dict] = mapped_column(JSONB, default=dict)
    # In the uniqueness key: the same object can appear in two populations being measured for different
    # reasons, and every reader here filters by batch. Without it, judging a detector sample stole nine
    # verdicts out of the calibration set and nothing errored.
    batch_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    ts_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint("verdict in ('correct','incorrect','unsure')", name="ck_machine_verdict_verdict"),
        UniqueConstraint("object_id", "judge", "model_version", "batch_id",
                         name="uq_machine_verdict_object_judge_batch"),
        Index("ix_machine_verdict_batch", "batch_id"),
        Index("ix_machine_verdict_object", "object_id"),
    )


class UsageRecord(Base):
    """One metered, billable delivery.

    Unique on (kind, subject_id) because a commit id is content-addressed: exporting the same slice twice
    returns the same commit, and metering per call would bill twice for one artifact.

    Prices are stamped here at write time rather than joined from config at read time, so recomputing an old
    invoice cannot silently rewrite what a customer was quoted.
    """

    __tablename__ = "usage_record"

    record_id: Mapped[uuid.UUID] = _uuid_pk()
    kind: Mapped[str] = mapped_column(String(32), nullable=False)      # export | inference | judge
    account: Mapped[str] = mapped_column(String(128), nullable=False, default="default")
    user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("app_user.user_id", ondelete="SET NULL"))
    subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    unit: Mapped[str] = mapped_column(String(32), nullable=False)
    unit_price_inr: Mapped[float | None] = mapped_column(Float)
    amount_inr: Mapped[float | None] = mapped_column(Float)
    # Whether this delivery carried a measured quality claim. On the row rather than looked up later, so an
    # invoice can show which lines were sold uncertified instead of hiding them.
    certified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    certificate_signature: Mapped[str | None] = mapped_column(String(128))
    detail: Mapped[dict] = mapped_column(JSONB, default=dict)
    ts_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("kind", "subject_id", name="uq_usage_record_kind_subject"),
        Index("ix_usage_record_account", "account"),
        Index("ix_usage_record_kind", "kind"),
    )


class Workforce(Base):
    """A team that labels for a living: an outside vendor, or an internal team routed the same way.

    Internal teams go through the same machinery on purpose. Exempting "our people" from measurement is how
    a quality bar quietly becomes a quality bar for suppliers only.
    """

    __tablename__ = "workforce"

    workforce_id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    kind: Mapped[str] = mapped_column(String(16), nullable=False, default="vendor")  # vendor | internal
    endpoint: Mapped[str | None] = mapped_column(Text)      # where a dispatch is POSTed; null means pull-only
    secret: Mapped[str] = mapped_column(String(128), nullable=False)   # HMAC key for this workforce's callbacks
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    capabilities: Mapped[dict] = mapped_column(JSONB, default=dict)    # {"classes": [...], "modalities": [...]}
    capacity_jobs_per_day: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # The acceptance bar is a commercial term negotiated per vendor, not a global constant.
    min_honeypot_accuracy: Mapped[float] = mapped_column(Float, nullable=False, default=0.9)
    contact: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class WorkforceAssignment(Base):
    """One dispatch of one job to one workforce, with its whole life.

    Separate from label_job.assignee_id because a dispatch can be sent, returned, rejected on quality and
    sent again, possibly elsewhere. A column on the job would keep only the last of those and lose exactly
    the history that says which workforce is worth using.
    """

    __tablename__ = "workforce_assignment"

    assignment_id: Mapped[uuid.UUID] = _uuid_pk()
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("label_job.job_id", ondelete="CASCADE"))
    workforce_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workforce.workforce_id", ondelete="CASCADE"))
    state: Mapped[str] = mapped_column(String(16), nullable=False, default="dispatched")
    external_ref: Mapped[str | None] = mapped_column(String(128))
    dispatched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    returned_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    honeypot_accuracy: Mapped[float | None] = mapped_column(Float)
    objects_returned: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    reason: Mapped[str | None] = mapped_column(Text)
    detail: Mapped[dict] = mapped_column(JSONB, default=dict)

    __table_args__ = (
        CheckConstraint("state in ('dispatched','returned','accepted','rejected','expired')",
                        name="ck_workforce_assignment_state"),
        Index("ix_workforce_assignment_workforce", "workforce_id", "state"),
    )


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
    # SigLIP2 is a joint image-text space, so this is what makes a crop retrievable by a text query. DINOv3
    # has no text tower, so with only that vector "cattle at night" could match frames but never objects.
    # Nullable: the embedding daemon backfills it, and an object without one is still image-searchable.
    siglip_vec: Mapped[list[float] | None] = mapped_column(Vector(1152), nullable=True)
    model_versions: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EmbeddingProjection(Base):
    """A fitted 2D projection of an embedding space, for the explorer's embeddings map. The high-dimensional
    vectors already live in pgvector (object_embedding / frame_embedding); this stores the 2D layout so the
    map opens instantly and the same picture is reproducible rather than re-fit (and re-shuffled) per visit.
    `method` records what actually produced it (umap or the deterministic pca fallback), so a map is never
    silently a different algorithm than the operator asked for."""

    __tablename__ = "embedding_projection"

    projection_id: Mapped[uuid.UUID] = _uuid_pk()
    kind: Mapped[str] = mapped_column(String(8), nullable=False)      # object | frame
    space: Mapped[str] = mapped_column(String(8), nullable=False)     # dino | siglip
    method: Mapped[str] = mapped_column(String(8), nullable=False)    # umap | pca
    params: Mapped[dict] = mapped_column(JSONB, default=dict)         # n_neighbors, min_dist, seed, filters
    n: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("session.session_id", ondelete="CASCADE"))         # null = whole corpus
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EmbeddingProjectionPoint(Base):
    """One point of a fitted projection: the 2D coordinate for an object or frame. Kept in its own table (not
    a JSONB blob on the projection) so the explorer can page and filter points in SQL."""

    __tablename__ = "embedding_projection_point"

    projection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("embedding_projection.projection_id", ondelete="CASCADE"), primary_key=True)
    ref_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)  # object_id or frame_id
    x: Mapped[float] = mapped_column(Float, nullable=False)
    y: Mapped[float] = mapped_column(Float, nullable=False)
    # Density-cluster label from HDBSCAN over the source vectors (-1 = noise/outlier). Persisted alongside the
    # coordinates so "colour by cluster" and "select this cluster" need no re-fit.
    cluster: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (Index("ix_projection_point_projection", "projection_id"),)


class EvalPatch(Base):
    """One prediction-vs-gold outcome from a model evaluation, so a confusion-matrix cell can be opened and
    the actual crops inspected (the "why did it confuse these" drill-down). outcome is tp | fp | fn; for a tp
    the two class ids match, for a confusion they differ. object_id is the prediction (fp/tp) or the missed
    gold object (fn), and is what the patch grid renders through /api/objects/{id}/crop."""

    __tablename__ = "eval_patch"

    patch_id: Mapped[uuid.UUID] = _uuid_pk()
    eval_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)  # one evaluation run
    gold_id: Mapped[str | None] = mapped_column(String(128))       # the sealed gold set it scored against
    # model_version is now derived from the InferenceRun the patch was scored against, never caller-supplied
    # (a caller-supplied version was the misattribution source: it named a model the scored rows never came from).
    model_version: Mapped[str | None] = mapped_column(String(128))
    run_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("inference_run.run_id", ondelete="CASCADE"))
    # A false positive / true positive patch is now a Prediction row; a false negative is the missed gold Object.
    prediction_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("prediction.prediction_id", ondelete="CASCADE"))
    object_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("object.object_id", ondelete="CASCADE"))
    frame_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("frame.frame_id", ondelete="CASCADE"))
    outcome: Mapped[str] = mapped_column(String(4), nullable=False)  # tp | fp | fn
    gt_class_id: Mapped[int | None] = mapped_column(Integer)
    pred_class_id: Mapped[int | None] = mapped_column(Integer)
    iou: Mapped[float | None] = mapped_column(Float)
    conf: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_eval_patch_eval", "eval_id"),
        Index("ix_eval_patch_cell", "eval_id", "gt_class_id", "pred_class_id"),
    )


class InferenceRun(Base):
    """The immutable prediction plane (measurement spine). A named model scores a set of frames ONCE, and every
    detection it produced is persisted verbatim under this run. This exists because the old harness drew
    "predictions" from live corpus rows (Object.source in the machine sources); but human review mutates that
    row in place (source becomes "human"), so every correct detection a human confirmed was erased from the
    prediction population, and eval scored only the residue humans rejected. Predictions must therefore live in
    their own append-only plane that review never touches: inference writes Prediction rows, review writes
    Object rows, and the two never share a row. A run is keyed by (model_version, gold_id, code_sha, params)
    so the same evaluation is reproducible and de-duplicated rather than recomputed against drifting state."""

    __tablename__ = "inference_run"

    run_id: Mapped[uuid.UUID] = _uuid_pk()
    model_version: Mapped[str] = mapped_column(
        String(128), ForeignKey("model_registry.model_version", ondelete="CASCADE"), nullable=False)
    gold_id: Mapped[str | None] = mapped_column(String(128))   # set when the run scores a sealed gold set
    frame_count: Mapped[int] = mapped_column(Integer, default=0)
    params: Mapped[dict] = mapped_column(JSONB, default=dict)   # imgsz, conf_floor, nms_iou, device, pack_id, ontology_version
    code_sha: Mapped[str | None] = mapped_column(String(40))    # git sha of the tree that produced the run
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="running")  # running | complete | failed

    __table_args__ = (Index("ix_inference_run_model_gold", "model_version", "gold_id"),)


class Prediction(Base):
    """One raw model detection under an InferenceRun. Append-only: the hard invariant is that no code path
    outside the inference writer (services/verdyx/inference_run.py) may UPDATE or DELETE a Prediction row.
    Human review never writes here; it writes Object. Keeping the full raw score (conf, unthresholded) is
    deliberate: evaluation needs the whole distribution to compute a PR curve and pick an operating point, so
    inference runs at a low conf floor and gating happens at scoring time, not at inference time."""

    __tablename__ = "prediction"

    prediction_id: Mapped[uuid.UUID] = _uuid_pk()
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("inference_run.run_id", ondelete="CASCADE"), nullable=False)
    frame_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("frame.frame_id", ondelete="CASCADE"), nullable=False)
    class_id: Mapped[int] = mapped_column(ForeignKey("ontology_class.id"), nullable=False)
    bbox: Mapped[list[float]] = mapped_column(ARRAY(Float), nullable=False)  # xyxy pixel, same as Object.bbox
    # Raw model score. Nullable ONLY for a reconstructed run (predictions backfilled from review history, where
    # the original confidence was never captured); a real inference run always writes it. A null conf makes a
    # PR curve / AP uncomputable, which is exactly why the eval refuses AP for a reconstructed run.
    conf: Mapped[float | None] = mapped_column(Float)
    conf_calibrated: Mapped[float | None] = mapped_column(Float)             # post-isotonic, when calibrated
    rot_deg: Mapped[float] = mapped_column(Float, default=0.0)
    mask_uri: Mapped[str | None] = mapped_column(Text)
    mask_encoding: Mapped[str | None] = mapped_column(String(16))
    cuboid_3d: Mapped[dict | None] = mapped_column(JSONB)
    # The identity a tracker assigned, when the run was a tracker rather than a per-frame detector. Null for
    # a detection run, and that nullness is what tells the tracking evaluator there is nothing to associate,
    # rather than it scoring a detector as a tracker with an identity switch on every frame.
    track_id: Mapped[str | None] = mapped_column(String(64))
    # The top-k class distribution behind the argmax, as {class_id: prob}. class_id + conf is a hard
    # argmax and throws away the only signal that distinguishes "confidently a scooter" from "torn
    # between scooter and motorcycle at the same confidence". Those two need completely different things
    # from a labelling budget, and nothing downstream could tell them apart.
    #
    # Nullable, and null for every prediction written before this existed. Top-5 rather than the full
    # distribution: over a 192-class ontology the tail is numerically zero and storing it would multiply
    # the table's size for no signal.
    class_probs: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_prediction_run", "run_id"),
        Index("ix_prediction_track", "run_id", "track_id"),
        Index("ix_prediction_frame", "frame_id"),
        Index("ix_prediction_run_frame", "run_id", "frame_id"),
    )


class PropagationConflict(Base):
    """Two ways of propagating one box disagreed by more than tolerance, so neither was written.

    A label can be carried to the next frame by ego geometry (the ground homography) or by the tracker.
    When they agree, the box is trustworthy and cheap. When they disagree the honest move is to write
    NEITHER and record why: picking one silently would propagate the wrong box, and picking the average
    would propagate a box neither method proposed.

    This is a queue of the frames where geometry and tracking see different worlds, which is a more useful
    artifact than a propagated label, because it is where the calibration or the ego pose is wrong.
    """

    __tablename__ = "propagation_conflict"

    conflict_id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    from_frame_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("frame.frame_id", ondelete="CASCADE"), nullable=False)
    to_frame_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("frame.frame_id", ondelete="CASCADE"), nullable=False)
    object_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("object.object_id", ondelete="SET NULL"))
    class_id: Mapped[int | None] = mapped_column(Integer)
    motion_model: Mapped[str | None] = mapped_column(String(24))
    geometry_box: Mapped[list | None] = mapped_column(JSONB)
    tracker_box: Mapped[list | None] = mapped_column(JSONB)
    iou: Mapped[float | None] = mapped_column(Float)
    tolerance: Mapped[float | None] = mapped_column(Float)
    reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_propagation_conflict_frames", "from_frame_id", "to_frame_id"),
        Index("ix_propagation_conflict_session", "session_id"),
    )


class ThresholdFit(Base):
    """A per-class auto-accept threshold fitted from measured outcomes, rather than picked.

    The gate ships 0.95 for benign classes and 0.99 for safety ones and calls them calibrated precision
    floors. A threshold is only a precision floor if somebody measured the precision at that score, and
    nobody did: two classes at the same nominal 0.95 can sit at very different real precisions, and a
    recalibration moves both without moving either constant.

    One row per (fit, class). `fit_id` groups the classes fitted together from one run, so a fit is
    replaced wholesale rather than per class: a threshold set where half the classes came from one
    evaluation and half from another is not an operating point, it is two.

    A class that could not be fitted gets a row with `measured = false` and a reason, never no row. The
    gate has to be able to tell "this class earned no threshold" from "nobody looked at this class", and
    fall back loudly in the first case rather than silently in both.

    `score_field` records whether the fit read calibrated or raw confidence. A threshold fitted on one and
    applied to the other is not conservative, it is arbitrary, and the two are only equal for a model that
    was never calibrated.
    """

    __tablename__ = "threshold_fit"

    row_id: Mapped[uuid.UUID] = _uuid_pk()
    fit_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("inference_run.run_id", ondelete="CASCADE"), nullable=False)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    gold_id: Mapped[str | None] = mapped_column(String(128))
    class_id: Mapped[int] = mapped_column(Integer, nullable=False)
    class_name: Mapped[str] = mapped_column(String(128), nullable=False)
    # conf_calibrated | conf. Which column the (score, outcome) pairs were read from.
    score_field: Mapped[str] = mapped_column(String(24), nullable=False, default="conf")
    alpha: Mapped[float] = mapped_column(Float, nullable=False)

    measured: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    reason: Mapped[str | None] = mapped_column(Text)
    threshold: Mapped[float | None] = mapped_column(Float)
    threshold_lo: Mapped[float | None] = mapped_column(Float)
    threshold_hi: Mapped[float | None] = mapped_column(Float)
    far_at: Mapped[float | None] = mapped_column(Float)
    accept_rate: Mapped[float | None] = mapped_column(Float)
    n_accept: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_pairs: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_positive: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_boot_fit: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # The threshold the gate would have used without this fit, so the delta is readable in one row rather
    # than reconstructed from config at the time the fit was made.
    config_threshold: Mapped[float | None] = mapped_column(Float)
    # Fitted and stored is not the same as in force. gold_calibrate sets the precedent: fit, report
    # whether it is trustworthy, and leave activating it to a human.
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("fit_id", "class_id", name="uq_threshold_fit_class"),
        # Partial: all but one fit per model is inactive, and the gate only ever asks for the active one.
        Index("ix_threshold_fit_model", "model_version", "active",
              postgresql_where=sql_text("active")),
        Index("ix_threshold_fit_run", "run_id"),
        Index("ix_threshold_fit_fit", "fit_id"),
    )


class CliqueBandit(Base):
    """The posterior for one confusion clique: how much labelling it has earned.

    Active learning has to divide a fixed labelling budget across the ways a model is confused, and the
    right split is not knowable in advance: it depends on which confusions labelling actually fixes. A
    Thompson-sampled Beta posterior per clique makes that a measurement rather than a constant, and it
    explores on its own without an epsilon anybody has to tune.

    `alpha`/`beta` are the Beta parameters. A reward is "labelling this clique moved gold recall for its
    classes"; there is no labelling history yet, so every clique starts at the prior and this table is
    honest about having learned nothing. `n_pulls` and `n_rewards` are carried so a posterior can be told
    from a prior at a glance, which the parameters alone do not do.
    """

    __tablename__ = "clique_bandit"

    clique: Mapped[str] = mapped_column(String(64), primary_key=True)
    pack_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    alpha: Mapped[float] = mapped_column(Float, nullable=False, default=1.0, server_default="1")
    beta: Mapped[float] = mapped_column(Float, nullable=False, default=1.0, server_default="1")
    n_pulls: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_rewards: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # What the last cycle actually allocated and what it measured afterwards, so a posterior that moved
    # can be traced to the batch that moved it.
    last_allocated: Mapped[int | None] = mapped_column(Integer)
    last_reward: Mapped[float | None] = mapped_column(Float)
    last_recall_before: Mapped[float | None] = mapped_column(Float)
    last_recall_after: Mapped[float | None] = mapped_column(Float)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class BlindAudit(Base):
    """A second, independent observation of a set of frames, for estimating what BOTH observers missed.

    Every recall number this engine reports is recall against a denominator somebody already found, and the
    two ways of finding things are not equally likely: confirming a machine box is one click, drawing a
    missed object is thirty seconds of work. So the gold set is biased toward what the model already sees,
    and gold recall is an overestimate by an unknown amount. Fixing the prediction plane made the numerator
    honest; it did nothing about the denominator.

    A blind audit is the denominator's fix. The annotator is served the pixels with every prediction and
    every existing object withheld SERVER-SIDE (services/api/routers/objects.py, the same filter a replica
    job uses, tightened to hide this job's own history too), labels the frames from scratch, and the result
    is a human observation that is genuinely independent of the model's. Capture-recapture over the two
    then estimates the population neither of them saw.

    The blindness is not a UI preference. If the predictions reach the browser at all the audit is void,
    because a hidden label is one keystroke from being an unhidden one and nothing afterwards could tell
    whether it had been. That is why the filter is in the fetch handler and not in the editor.

    status:  seeded (frames chosen, nobody has labelled) -> labeling -> scored | abandoned
    """

    __tablename__ = "blind_audit"

    audit_id: Mapped[uuid.UUID] = _uuid_pk()
    # The run being audited. The model's observation is this run's Prediction rows on the audit frames, at
    # score_thr, and never anything from Object.
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("inference_run.run_id", ondelete="CASCADE"), nullable=False)
    gold_id: Mapped[str | None] = mapped_column(String(128))
    # The annotation job serving the frames. Nullable so an audit can be seeded and scored headlessly (a
    # backfill, a test) without a labelling queue, but the blindness filter keys on it, so an audit that a
    # human is meant to label must have one.
    job_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("label_job.job_id", ondelete="SET NULL"))
    n_frames: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # How the frames were stratified, and why. Capture probability is not constant across the corpus (a
    # crowded junction and an empty highway do not share a detection rate), and pooling over a single
    # collapsed count assumes it is.
    stratify_by: Mapped[str] = mapped_column(String(32), nullable=False, default="density")
    strata: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    # The operating point the model's observation is taken at. Stored because the estimate is meaningless
    # without it: the same run at a lower threshold is a different observer.
    score_thr: Mapped[float] = mapped_column(Float, nullable=False, default=0.25)
    iou_thr: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="seeded")
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    scored_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("ix_blind_audit_run", "run_id"),
        Index("ix_blind_audit_gold", "gold_id"),
        Index("ix_blind_audit_job", "job_id"),
    )


class BlindAuditFrame(Base):
    """One frame in a blind audit, and the three capture counts it contributed.

    Counts are stored per frame rather than only rolled up so a suspicious estimate can be opened: an
    audit whose whole n_human_only comes from four frames is an annotator finding one thing repeatedly,
    not a model with a systematic blind spot, and the pooled number cannot tell those apart.

    labeled_at is what makes an audit scoreable. An unlabelled frame is not a frame where the human found
    nothing, and counting it as one would report the model's recall as far better than it is.
    """

    __tablename__ = "blind_audit_frame"

    audit_frame_id: Mapped[uuid.UUID] = _uuid_pk()
    audit_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("blind_audit.audit_id", ondelete="CASCADE"), nullable=False)
    frame_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("frame.frame_id", ondelete="CASCADE"), nullable=False)
    stratum: Mapped[str] = mapped_column(String(64), nullable=False, default="all")
    labeled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Null until scored. Zero and null mean different things here and the distinction is the whole point.
    n_both: Mapped[int | None] = mapped_column(Integer)
    n_model_only: Mapped[int | None] = mapped_column(Integer)
    n_human_only: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (
        UniqueConstraint("audit_id", "frame_id", name="uq_blind_audit_frame"),
        Index("ix_blind_audit_frame_audit", "audit_id"),
        # Keyed on frame because the blindness check in the frame fetch handler asks "is this frame under
        # an audit" on every editor request. The unique constraint's index leads with audit_id and cannot
        # answer that, so without this the guard would sequentially scan on the editor's hot path.
        Index("ix_blind_audit_frame_frame", "frame_id"),
    )


class RecaptureEstimateRow(Base):
    """A capture-recapture population estimate, durable and keyed on (run_id, gold_id).

    One row per (audit, stratum, class): stratum null means pooled across strata, class_id null means
    pooled across classes, so the pooled corpus-wide estimate is the row with both null.

    measured is False, with a reason, when the counts cannot support an estimate at all (nothing found by
    both observers leaves the population unbounded above). Storing that as a row rather than omitting it is
    deliberate: a missing row reads as "not computed", and "computed, and the answer is that we cannot
    tell" is a different and more useful statement.

    gold_recall is carried alongside so the two denominators sit in one row. The gap between gold_recall
    and model_recall IS the measurement, and putting them in separate tables would make the comparison a
    join nobody performs.
    """

    __tablename__ = "recapture_estimate"

    estimate_id: Mapped[uuid.UUID] = _uuid_pk()
    audit_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("blind_audit.audit_id", ondelete="CASCADE"), nullable=False)
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("inference_run.run_id", ondelete="CASCADE"), nullable=False)
    gold_id: Mapped[str | None] = mapped_column(String(128))
    stratum: Mapped[str | None] = mapped_column(String(64))
    class_id: Mapped[int | None] = mapped_column(Integer)

    n_both: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_model_only: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_human_only: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    measured: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    reason: Mapped[str | None] = mapped_column(Text)
    population: Mapped[float | None] = mapped_column(Float)
    population_lo: Mapped[float | None] = mapped_column(Float)
    population_hi: Mapped[float | None] = mapped_column(Float)
    variance: Mapped[float | None] = mapped_column(Float)
    model_recall: Mapped[float | None] = mapped_column(Float)
    recall_lo: Mapped[float | None] = mapped_column(Float)
    recall_hi: Mapped[float | None] = mapped_column(Float)
    human_recall: Mapped[float | None] = mapped_column(Float)
    # Recall against the sealed gold denominator, for the same run and slice. The number the engine
    # reported before this table existed.
    gold_recall: Mapped[float | None] = mapped_column(Float)
    estimator: Mapped[str] = mapped_column(String(32), nullable=False, default="chapman-lp-v1")
    n_strata_pooled: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        # NULLS NOT DISTINCT because null is a meaningful value in both key columns here (pooled), and the
        # SQL default would let the pooled row be inserted twice while refusing a duplicate on every other
        # slice. Postgres 15+; this engine requires 16 for the PostGIS and pgvector versions it pins.
        UniqueConstraint("audit_id", "stratum", "class_id", name="uq_recapture_estimate_slice",
                         postgresql_nulls_not_distinct=True),
        Index("ix_recapture_estimate_run_gold", "run_id", "gold_id"),
        Index("ix_recapture_estimate_audit", "audit_id"),
    )


class LabelProject(Base):
    """A labeling programme: the unit a team is organised around, holding the schema and the QA policy every
    task under it inherits. Named LabelProject rather than Project to keep it unambiguous against the compute
    jobs (ImportJob / TrainingJob), which are a different notion of "job" entirely.

    label_config is reserved for the per-project configurable interface; an AV project leaves it empty and
    inherits the 170-class ontology, which stays the default source of truth."""

    __tablename__ = "label_project"

    project_id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    modality: Mapped[str] = mapped_column(String(16), nullable=False, default="image")
    label_config: Mapped[dict] = mapped_column(JSONB, default=dict)
    # QA policy inherited by every task: what fraction of a job is hidden gold, and the accuracy an annotator
    # must hold against it.
    honeypot_frac: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, server_default="0")
    min_honeypot_accuracy: Mapped[float] = mapped_column(Float, nullable=False, default=0.9,
                                                         server_default="0.9")
    gold_id: Mapped[str | None] = mapped_column(String(128))   # the sealed gold set honeypots are drawn from
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("app_user.user_id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class LabelTask(Base):
    """A batch of work inside a project, usually one session or one curated slice. Splits into jobs, which are
    what a person actually picks up."""

    __tablename__ = "label_task"

    task_id: Mapped[uuid.UUID] = _uuid_pk()
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("label_project.project_id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    session_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("session.session_id", ondelete="CASCADE"))
    slice_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("curation_slice.slice_id", ondelete="SET NULL"))
    predicate: Mapped[dict] = mapped_column(JSONB, default=dict)   # the explorer predicate that defined it
    # How many independent annotators each chunk of frames goes to. 1 is ordinary work; more than 1 buys a
    # measurement of how much they agree, at a proportional cost in annotation spend.
    replicas: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_label_task_project", "project_id"),)


class LabelJob(Base):
    """The unit of assignable work: a bounded set of frames one person annotates, reviews, or accepts.

    stage and state are separate on purpose, the same split CVAT uses. stage is WHERE in the pipeline the work
    sits (annotation -> validation -> acceptance); state is how far along it is within that stage. Collapsing
    them into one enum loses the ability to say "in validation, not yet started".

    Frames are referenced by id, never copied, so a job is a view over the corpus and cannot drift from it."""

    __tablename__ = "label_job"

    job_id: Mapped[uuid.UUID] = _uuid_pk()
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("label_task.task_id", ondelete="CASCADE"), nullable=False)
    frame_ids: Mapped[list] = mapped_column(JSONB, default=list)
    assignee_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("app_user.user_id", ondelete="SET NULL"))
    stage: Mapped[str] = mapped_column(String(12), nullable=False, default="annotation")
    state: Mapped[str] = mapped_column(String(12), nullable=False, default="new")
    # Two jobs over the same frames, labelled independently so their agreement can be measured. The pairing
    # cannot be recovered by comparing frame_ids: seed_honeypots appends gold frames chosen per job id, so
    # replicas diverge the moment they are created.
    #
    # A job in a replica group is also a BLIND job: it hides existing labels, because two annotators
    # correcting the same machine proposals produce one label set with two editors, not two label sets, and
    # agreement over that measures nothing. 82.6% of this corpus is pre-labelled, so this is the normal case.
    replica_group: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    replica_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    # Hidden gold frames mixed into this job, and the measured accuracy against them once submitted.
    honeypot_frame_ids: Mapped[list] = mapped_column(JSONB, default=list)
    honeypot_accuracy: Mapped[float | None] = mapped_column(Float)
    honeypot_detail: Mapped[dict] = mapped_column(JSONB, default=dict)
    # Optimistic concurrency, matching Object: two reviewers acting on one job cannot silently clobber.
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_label_job_task", "task_id"),
        Index("ix_label_job_assignee", "assignee_id", "state"),
    )


class JobAgreement(Base):
    """How much two annotators agreed on one frame, and how much they did not.

    Stored per frame per pair rather than rolled up, because the rolled-up number tells you a task is at
    0.7 and nothing about which frames to look at. The pair is ordered by the caller so recomputing a
    group updates these rows rather than accumulating a second opinion beside the first.
    """

    __tablename__ = "job_agreement"

    agreement_id: Mapped[uuid.UUID] = _uuid_pk()
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("label_task.task_id", ondelete="CASCADE"), nullable=False)
    replica_group: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    frame_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("frame.frame_id", ondelete="CASCADE"), nullable=False)
    job_a_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("label_job.job_id", ondelete="CASCADE"), nullable=False)
    job_b_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("label_job.job_id", ondelete="CASCADE"), nullable=False)
    # The iaa_score dict verbatim: detection_agreement, class_agreement, mean_iou, cohen_kappa, n_matched,
    # n_a, n_b. Stored whole rather than as columns so the measurement can gain a term without a migration.
    metrics: Mapped[dict] = mapped_column(JSONB, default=dict)
    n_disagreements: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("frame_id", "job_a_id", "job_b_id", name="uq_job_agreement_pair"),
        Index("ix_job_agreement_task", "task_id"),
        Index("ix_job_agreement_group", "replica_group"),
    )


class Issue(Base):
    """A review thread pinned to a specific annotation or region: the feedback loop that lets a reviewer say
    "this box is wrong, here" instead of rejecting silently. Anchored to an object when the complaint is about
    a label, or to a frame plus a box region when it is about something missing."""

    __tablename__ = "issue"

    issue_id: Mapped[uuid.UUID] = _uuid_pk()
    job_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("label_job.job_id", ondelete="CASCADE"))
    frame_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("frame.frame_id", ondelete="CASCADE"))
    object_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("object.object_id", ondelete="CASCADE"))
    region: Mapped[list | None] = mapped_column(JSONB)          # optional [x1,y1,x2,y2] pin on the frame
    kind: Mapped[str] = mapped_column(String(24), nullable=False, default="comment")
    status: Mapped[str] = mapped_column(String(12), nullable=False, default="open")   # open | resolved
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("app_user.user_id", ondelete="SET NULL"))
    resolved_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("app_user.user_id", ondelete="SET NULL"))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_issue_frame", "frame_id", "status"),
        Index("ix_issue_job", "job_id", "status"),
    )


class IssueComment(Base):
    """One message in an issue thread."""

    __tablename__ = "issue_comment"

    comment_id: Mapped[uuid.UUID] = _uuid_pk()
    issue_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("issue.issue_id", ondelete="CASCADE"),
                                                nullable=False)
    author_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("app_user.user_id", ondelete="SET NULL"))
    body: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_issue_comment_issue", "issue_id"),)


class Asset(Base):
    """A labelable item in a project, for any modality.

    LabeloxAV has two annotation spines, kept deliberately separate:

      AV spine       Session -> Frame -> Object (+ TimelineEvent)   the driving corpus, untouched
      project spine  LabelProject -> Asset -> Annotation            everything a project labels

    Forcing text spans and audio regions into Object (which is welded to a bbox, an ontology class id and the
    confidence gate) would have made both worse. An Asset can REFERENCE an existing frame or session, so an AV
    project reuses the corpus through the same job machinery without copying a single row.
    """

    __tablename__ = "asset"

    asset_id: Mapped[uuid.UUID] = _uuid_pk()
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("label_project.project_id", ondelete="CASCADE"), nullable=False)
    # image | video | audio | text | timeseries | document | pointcloud | dialogue
    media_type: Mapped[str] = mapped_column(String(16), nullable=False)
    uri: Mapped[str | None] = mapped_column(Text)          # object-store uri for binary media
    text: Mapped[str | None] = mapped_column(Text)         # inline body for text/dialogue assets
    external_id: Mapped[str | None] = mapped_column(String(200))   # caller's own id, for idempotent import
    # Optional links back into the AV spine, so an Asset can be a view of an existing frame or session.
    frame_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("frame.frame_id", ondelete="CASCADE"))
    session_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("session.session_id", ondelete="CASCADE"))
    meta: Mapped[dict] = mapped_column(JSONB, default=dict)  # duration_s, sample_rate, width/height, channels
    state: Mapped[str] = mapped_column(String(16), nullable=False, default="new")  # new|in_progress|labeled|skipped
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_asset_project", "project_id", "state"),
        Index("ix_asset_external", "project_id", "external_id", unique=True,
              postgresql_where=sql_text("external_id IS NOT NULL")),
    )


class Annotation(Base):
    """One annotation on an Asset, of any shape.

    `kind` selects how `payload` is read, and payload is validated against the kind AND against the project's
    label_config before it is stored (services/assets/labelconfig.py). Keeping the shape in JSONB rather than
    as columns is what lets one table serve boxes, text spans, audio regions and preference rankings; keeping
    the validation strict is what stops that flexibility from becoming a junk drawer.

    Carries the same discipline as Object: source, state, version for optimistic concurrency, and provenance.
    """

    __tablename__ = "annotation"

    annotation_id: Mapped[uuid.UUID] = _uuid_pk()
    asset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("asset.asset_id", ondelete="CASCADE"),
                                                nullable=False)
    # bbox | polygon | polyline | keypoints | mask | span | relation | region | classification |
    # transcription | preference | rubric | ranking
    kind: Mapped[str] = mapped_column(String(20), nullable=False)
    label: Mapped[str | None] = mapped_column(String(120))   # the label config entry this instance carries
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    fields: Mapped[dict] = mapped_column(JSONB, default=dict)  # typed per-label fields from the label config
    conf: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(16), nullable=False, default="human")  # human|model|imported
    state: Mapped[str] = mapped_column(String(16), nullable=False, default="accepted")
    provenance: Mapped[dict] = mapped_column(JSONB, default=dict)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("app_user.user_id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_annotation_asset", "asset_id", "kind"),
        Index("ix_annotation_label", "label"),
    )


class Webhook(Base):
    """An outbound HTTP subscription, so an external pipeline can react to what happens here instead of
    polling for it.

    Deliveries are signed with an HMAC over the body using the subscription's own secret, the same scheme as
    GitHub and Stripe. Without a signature a receiver cannot tell a real delivery from anyone who learned the
    URL, which makes a webhook an unauthenticated write into whatever it triggers.
    """

    __tablename__ = "webhook"

    webhook_id: Mapped[uuid.UUID] = _uuid_pk()
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("label_project.project_id", ondelete="CASCADE"))   # null = all projects
    url: Mapped[str] = mapped_column(Text, nullable=False)
    events: Mapped[list] = mapped_column(JSONB, default=list)         # [] = every event
    secret: Mapped[str | None] = mapped_column(String(128))
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="true")
    # Delivery health, so a silently dead endpoint is visible rather than merely absent.
    last_status: Mapped[int | None] = mapped_column(Integer)
    last_error: Mapped[str | None] = mapped_column(Text)
    last_delivery_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    failure_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_webhook_project", "project_id", "active"),)


class StorageSource(Base):
    """A registered external bucket a project imports from.

    Credentials are NOT stored here: the row holds only the locator and which server-side credential profile
    to use. Persisting per-source keys would put long-lived cloud credentials in the application database,
    where a single read of one table becomes a breach of every connected bucket.
    """

    __tablename__ = "storage_source"

    source_id: Mapped[uuid.UUID] = _uuid_pk()
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("label_project.project_id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    provider: Mapped[str] = mapped_column(String(12), nullable=False)   # s3 | gcs | azure | minio
    bucket: Mapped[str] = mapped_column(String(200), nullable=False)
    prefix: Mapped[str | None] = mapped_column(Text)
    region: Mapped[str | None] = mapped_column(String(32))
    endpoint_url: Mapped[str | None] = mapped_column(Text)             # for s3-compatible stores
    credential_profile: Mapped[str | None] = mapped_column(String(64))  # names a server-side credential
    last_scanned_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_object_count: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_storage_source_project", "project_id"),)


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
    # Content hash (class/geometry/state), distinct from the object-id-only commit_id, so a mutated
    # annotation is detectable on /release/{id}/verify. Nullable: pre-0061 commits were not fingerprinted.
    content_fingerprint: Mapped[str | None] = mapped_column(String(64))
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
    # Sealed track ids, present only when the set was sealed with track continuity guaranteed. A tracking
    # metric scored against a set without this is scored against association labels that may not exist, so
    # the evaluator refuses rather than reporting a number built on absent ground truth.
    track_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    tracks_sealed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
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
    # What has already been written and verified. An export was all or nothing, so a failure at ninety
    # percent restarted from zero, which on a large corpus means hours of repeated work.
    checkpoint: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    resumed_from: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
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
    drift_delta: Mapped[dict | None] = mapped_column(JSONB)        # CALYX: SE(3) drift delta per sensor pair
    severity: Mapped[str | None] = mapped_column(String(16))       # CALYX: ok | drift_detected | block
    confidence: Mapped[float | None] = mapped_column(Float)        # M11: calibration confidence 0..1 (uncertainty)
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
    modality: Mapped[str] = mapped_column(String(16), nullable=False)  # imu|audio|scene|geo|crossmodal|driving
    t_start_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    t_end_ns: Mapped[int | None] = mapped_column(BigInteger)                 # null = a point event
    # What the event is about. Nullable because an inertial spike or an audio region is genuinely about the
    # session alone; SET NULL rather than CASCADE because a confirmed event is a finding about the drive and
    # should outlive the track that suggested it.
    track_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("track.track_id", ondelete="SET NULL"))
    frame_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("frame.frame_id", ondelete="SET NULL"))
    conf: Mapped[float | None] = mapped_column(Float)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    source: Mapped[str] = mapped_column(String(16), nullable=False, default="human")  # human|auto|correlated
    state: Mapped[str] = mapped_column(String(16), nullable=False, default="review")  # review|confirmed|rejected
    provenance: Mapped[dict] = mapped_column(JSONB, default=dict)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_timeline_event_session_t", "session_id", "t_start_ns"),
                      Index("ix_timeline_event_track", "track_id"),
                      Index("ix_timeline_event_session_kind", "session_id", "kind"))


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
    # Who ruled and when. Dismissals used to leave no trace at all, which made them useless as calibration:
    # confirmed over confirmed-plus-dismissed is the detector's precision, and an untimestamped verdict
    # cannot be attributed to a detector version or a period.
    decided_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("app_user.user_id", ondelete="SET NULL"))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    decision_note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_error_candidate_status", "status"), Index("ix_error_candidate_object", "object_id"),
                      Index("ix_error_candidate_kind_status", "kind", "status"))


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
    # Written as the job progresses. A `running` row whose heartbeat has gone stale is a job whose process
    # died, which is the only way to tell it from live work: the task lives in the API process and leaves no
    # trace when that process goes away.
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # The resume cursor, shaped by the job. Opaque on purpose: the only thing every job shares is needing to
    # say how much is done, and a common shape would make each one lie about its unit of work.
    progress: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
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


class SessionIndex(Base):
    """Per-session MCAP index (M-I.1): every topic with its schema, message count, measured rate, time range,
    and gap windows, read cheaply from the MCAP summary + message index. The raw material for the Inspector
    timeline, the topic browser, and the session-health checks. Stamped with the indexer version."""

    __tablename__ = "session_index"

    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("session.session_id", ondelete="CASCADE"), primary_key=True)
    mcap_uri: Mapped[str | None] = mapped_column(Text)
    topics: Mapped[dict] = mapped_column(JSONB, default=dict)      # {topic: {name, schema, count, rate, first_ts, last_ts}}
    time_range: Mapped[list] = mapped_column(ARRAY(BigInteger))    # [first_ts_ns, last_ts_ns] across all topics
    gaps: Mapped[dict] = mapped_column(JSONB, default=dict)        # {topic: [[gap_start_ns, gap_end_ns], ...]}
    indexer_version: Mapped[str] = mapped_column(String(32), nullable=False)
    built_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class SessionHealth(Base):
    """Per-session health-check run (M-I.2): the per-check results over the index and a pass/warn/fail verdict.
    A fail flags the session and gates it from auto-ingestion until a human reviews, exactly as calibration
    validation does."""

    __tablename__ = "session_health"

    id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("session.session_id", ondelete="CASCADE"), nullable=False)
    checks: Mapped[list] = mapped_column(JSONB, default=list)      # [{name, status, detail, evidence, score}]
    verdict: Mapped[str] = mapped_column(String(8), nullable=False)  # pass | warn | fail
    score: Mapped[float | None] = mapped_column(Float)             # SANYX overall 0..100 health score
    decision: Mapped[str | None] = mapped_column(String(12))       # SANYX: pass | degraded | quarantine
    root_cause: Mapped[str | None] = mapped_column(String(48))     # M10: named fault (loose_gmsl2_connector, ...)
    remediation: Mapped[str | None] = mapped_column(Text)          # M10: operator remediation hint
    indexer_version: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_session_health_session", "session_id"),)


class InspectorLayout(Base):
    """A saveable Inspector panel layout (M-I.3): panel types, sources, and arrangement, per user."""

    __tablename__ = "inspector_layout"

    layout_id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("app_user.user_id"))
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    panels: Mapped[list] = mapped_column(JSONB, default=list)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class FrameGroup(Base):
    """A synchronized set of frames across the rig at one instant (M-MC.0): the frames whose timestamps fall
    inside the sync tolerance, one per camera. The unit the multi-camera canvas navigates and the reviewer
    confirms as a whole. missing_cams records a camera that dropped a frame in this window (a dropout the
    surround view must show as an empty tile, not silently omit); sync_spread_ns is the max pairwise timestamp
    difference within the group, which must stay inside tolerance for the group to be trustworthy."""

    __tablename__ = "frame_group"

    group_id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("session.session_id", ondelete="CASCADE"), nullable=False)
    ts_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)         # group reference time (earliest member)
    frame_ids: Mapped[dict] = mapped_column(JSONB, default=dict)          # {cam_id: frame_id}
    missing_cams: Mapped[list] = mapped_column(ARRAY(Text), default=list)  # cameras with no frame in this window
    sync_spread_ns: Mapped[int] = mapped_column(BigInteger, default=0)     # max pairwise ts diff across members
    n_cams: Mapped[int] = mapped_column(Integer, default=0)                # members present (for quick filters)
    confirmed: Mapped[bool] = mapped_column(Boolean, default=False)        # reviewer confirmed the whole group
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_frame_group_session_ts", "session_id", "ts_ns"),)


class RigObject(Base):
    """One physical object seen across views at a single instant (M-MC.2): the rig-level identity that binds the
    per-camera Object rows (member_object_ids) which are the same real thing (the rickshaw visible in both the
    front and right cameras). class_id is the voted class across members; conflict marks members that disagree
    and route the rig object to review. Links carry their source (manual, appearance, or projection) so a
    reversible agent run can undo exactly the links it made."""

    __tablename__ = "rig_object"

    rig_object_id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("session.session_id", ondelete="CASCADE"), nullable=False)
    group_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("frame_group.group_id"), nullable=False)
    class_id: Mapped[int | None] = mapped_column(Integer)                 # voted class across members
    member_object_ids: Mapped[list] = mapped_column(ARRAY(UUID(as_uuid=True)), default=list)
    link_sources: Mapped[dict] = mapped_column(JSONB, default=dict)       # {object_id: "manual"|"appearance"|"projection"}
    rig_track_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))  # M-MC.4 same rig object across time
    conflict: Mapped[bool] = mapped_column(Boolean, default=False)        # members disagree on class -> review
    provenance: Mapped[dict | None] = mapped_column(JSONB)                # {agent_run_id, ...} for reversibility
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_rig_object_group", "group_id"), Index("ix_rig_object_track", "rig_track_id"))


# ---------------------------------------------------------------------------
# Data Engine spine additions (migration 0049). These complete the Model side of the canonical spine for the
# VERDYX (evaluation) and FORGYX (edge optimization) planes, and make ORACLYX offline-fusion consensus explicit
# without touching the hot object table. All additive; nothing above changes shape.
# ---------------------------------------------------------------------------


class Evaluation(Base):
    """VERDYX: a first-class per-slice evaluation of a model against a lineage-locked release or gold set, plus
    the champion-challenger verdict LabeloxAV governance consumes. Lifts eval out of model_registry.gold_metrics
    so a model can carry many slice-resolved evals over time, not one flat metrics blob."""

    __tablename__ = "evaluation"

    eval_id: Mapped[uuid.UUID] = _uuid_pk()
    model_version: Mapped[str] = mapped_column(
        ForeignKey("model_registry.model_version", ondelete="CASCADE"), nullable=False)
    release_commit: Mapped[str | None] = mapped_column(ForeignKey("dataset_commit.commit_id", ondelete="SET NULL"))
    gold_id: Mapped[str | None] = mapped_column(ForeignKey("gold_set.gold_id", ondelete="SET NULL"))
    per_slice: Mapped[dict] = mapped_column(JSONB, default=dict)        # slice_id -> {map, precision, recall, confusion}
    failure_clusters: Mapped[dict] = mapped_column(JSONB, default=dict)  # cluster_id -> {condition, member_object_ids, size}
    aggregate: Mapped[dict] = mapped_column(JSONB, default=dict)        # map50, map, precision, recall, safe_miou
    safety: Mapped[dict] = mapped_column(JSONB, default=dict)           # M15: track/scenario safety metrics + CIs + significance
    verdict: Mapped[str] = mapped_column(String(16), nullable=False, default="needs_review")  # promote|reject|needs_review
    challenger_of: Mapped[str | None] = mapped_column(String(128))     # the champion version this challenges
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_evaluation_model", "model_version"),)


class Benchmark(Base):
    """FORGYX: how a model behaves on one target silicon. One row per (model, target) with latency/throughput/
    power and a pointer to the accuracy re-verification (Evaluation), so a model that got faster but quietly lost
    a protected slice is visible, not hidden."""

    __tablename__ = "benchmark"

    benchmark_id: Mapped[uuid.UUID] = _uuid_pk()
    model_version: Mapped[str] = mapped_column(
        ForeignKey("model_registry.model_version", ondelete="CASCADE"), nullable=False)
    target: Mapped[str] = mapped_column(String(32), nullable=False)   # sentrixai_litert|agx_orin_trt|orin_nano_trt|pi_hailo
    latency_ms: Mapped[dict] = mapped_column(JSONB, default=dict)     # {p50, p95, p99}
    throughput_fps: Mapped[float | None] = mapped_column(Float)
    power_w: Mapped[float | None] = mapped_column(Float)
    accuracy_ref: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("evaluation.eval_id", ondelete="SET NULL"))
    per_layer_uri: Mapped[str | None] = mapped_column(Text)          # profile blob in the object store
    pareto_rank: Mapped[int | None] = mapped_column(Integer)
    artifact_uri: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_benchmark_model_target", "model_version", "target"),)


class Deployment(Base):
    """FORGYX: a deployable, verified artifact per target, with lineage back to the release it was built from and
    the VERDYX verdict and FORGYX benchmark that gated it. The SentrixAI row is a LiteRT artifact verified on the
    mobile latency budget and the FCW-relevant slices."""

    __tablename__ = "deployment"

    deployment_id: Mapped[uuid.UUID] = _uuid_pk()
    model_version: Mapped[str] = mapped_column(
        ForeignKey("model_registry.model_version", ondelete="CASCADE"), nullable=False)
    target: Mapped[str] = mapped_column(String(32), nullable=False)
    artifact_uri: Mapped[str] = mapped_column(Text, nullable=False)   # .onnx | .engine | .tflite | .hef in the object store
    export_format: Mapped[str] = mapped_column(String(16), nullable=False)  # onnx|tensorrt|litert|hailo
    release_commit: Mapped[str | None] = mapped_column(ForeignKey("dataset_commit.commit_id", ondelete="SET NULL"))
    verdict_ref: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("evaluation.eval_id", ondelete="SET NULL"))
    benchmark_ref: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("benchmark.benchmark_id", ondelete="SET NULL"))
    status: Mapped[str] = mapped_column(String(12), nullable=False, default="built")  # built|verified|blocked|deployed|retired
    signature: Mapped[str | None] = mapped_column(String(128))         # M16: HMAC over the signed package manifest
    package_uri: Mapped[str | None] = mapped_column(Text)              # M16: signed deployment package in the object store
    thermal_envelope: Mapped[dict] = mapped_column(JSONB, default=dict)  # M16: {sustained_fps, throttle_temp_c, power_w, headroom}
    rollout_state: Mapped[str] = mapped_column(String(12), nullable=False, default="none")  # none|canary|full|rolled_back
    superseded_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("deployment.deployment_id", ondelete="SET NULL"))
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_deployment_model", "model_version"),)


class PseudoLabel(Base):
    """ORACLYX: offline-fusion consensus over a fused object, in a side table so the hot object row is untouched.
    consensus True means fusion and the three auto-label paths agreed within tolerance (auto-accept); False routes
    the sample to the human queue, which is exactly where human attention has the highest marginal value."""

    __tablename__ = "pseudo_label"

    object_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("object.object_id", ondelete="CASCADE"), primary_key=True)
    consensus: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    consensus_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    voters: Mapped[dict] = mapped_column(JSONB, default=dict)         # path -> {agree: bool, conf: float}
    fusion_run_id: Mapped[str | None] = mapped_column(String(128))
    uncertainty: Mapped[float | None] = mapped_column(Float)          # M14: calibrated pseudo-GT uncertainty 0..1
    info_gain: Mapped[float | None] = mapped_column(Float)            # M14: expected training value of reviewing it
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SanyxRigAlert(Base):
    """SANYX predictive maintenance (M10): a per-component health trend across a vehicle's sessions that flags a
    module degrading toward failure before it fails (a camera whose exposure or lens sub-score is monotonically
    falling, an IMU whose saturation is climbing). Distinct from a single session's HealthReport; this is the
    rig-level trend over time."""

    __tablename__ = "sanyx_rig_alert"

    alert_id: Mapped[uuid.UUID] = _uuid_pk()
    vehicle_id: Mapped[str] = mapped_column(String(64), nullable=False)
    component: Mapped[str] = mapped_column(String(32), nullable=False)   # cam_ft | cam_rt | imu | gnss | ...
    metric: Mapped[str] = mapped_column(String(48), nullable=False)      # the check/sub-score that is trending
    trend: Mapped[str] = mapped_column(String(8), nullable=False)        # rising | falling
    severity: Mapped[str] = mapped_column(String(8), nullable=False)     # watch | warn | critical
    evidence: Mapped[dict] = mapped_column(JSONB, default=dict)          # slope, n_sessions, projected sessions to threshold
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_sanyx_rig_alert_vehicle", "vehicle_id"),)


class CalibrationOverride(Base):
    """CALYX data recovery (M11): a corrected calibration that makes a mildly drifted session usable instead of
    quarantined, kept as a versioned override on the session (raw calibration is never mutated). source records
    how it was derived (self-cal from the drift delta, targetless from natural-scene cues, or a cross-session
    consensus prior); confidence and provenance let ORACLYX and the workspace weight how much to trust it."""

    __tablename__ = "calibration_override"

    override_id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("session.session_id", ondelete="CASCADE"))
    cam_id: Mapped[str] = mapped_column(String(32), nullable=False)
    source: Mapped[str] = mapped_column(String(16), nullable=False)   # self_cal | targetless | consensus
    corrected: Mapped[dict] = mapped_column(JSONB, default=dict)      # {rpy_deg, xyz_m, fx, fy, cx, cy, dist}
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    provenance: Mapped[dict] = mapped_column(JSONB, default=dict)     # {method, drift_delta, residual_before/after, n}
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    applied: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_calibration_override_session", "session_id"),)


class ClipManeuver(Base):
    """SIEVYX clip-level maneuver (M12): a maneuver recognized over a track's trajectory (cut-in, unprotected
    turn, U-turn, jaywalk, ...), so mining works at the scenario level, not just the frame level. features holds
    the trajectory descriptor the embedding and classifier rest on."""

    __tablename__ = "clip_maneuver"

    id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("session.session_id", ondelete="CASCADE"))
    track_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    t_in_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    t_out_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    maneuver: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    features: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_clip_maneuver_session", "session_id"),)


class ScenarioCluster(Base):
    """SIEVYX long-tail discovery (M12): an auto-discovered rare scenario group, surfaced for a human to name so
    the scenario ontology grows from data. rarity is the cluster's isolation in embedding space; status walks
    discovered -> named -> dismissed."""

    __tablename__ = "scenario_cluster"

    cluster_id: Mapped[uuid.UUID] = _uuid_pk()
    method: Mapped[str] = mapped_column(String(24), nullable=False, default="dino_hdbscan")
    size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rarity: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    rep_frame_ids: Mapped[list] = mapped_column(JSONB, default=list)
    name: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(12), nullable=False, default="discovered")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AnnotationCheckpoint(Base):
    # A named save of a frame's annotations. Undo is capped, tab-local and dies with a refresh; this is the
    # state a person can always get back to. Stored as a full snapshot rather than a diff, because a diff
    # chain is only as good as its weakest link and this table exists precisely to not be that.
    __tablename__ = "annotation_checkpoint"

    checkpoint_id: Mapped[uuid.UUID] = _uuid_pk()
    frame_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("frame.frame_id", ondelete="CASCADE"))
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("session.session_id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    note: Mapped[str | None] = mapped_column(Text)
    objects: Mapped[list] = mapped_column(JSONB, nullable=False)
    object_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    created_by: Mapped[str | None] = mapped_column(String(64))
    # True on the checkpoint a restore takes of what it is about to replace, so undoing a restore is itself
    # a restore rather than a hope.
    auto: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False,
                                       server_default=sql_text("false"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_checkpoint_frame_created", "frame_id", "created_at"),
                      Index("ix_checkpoint_session", "session_id"))


class AnnotationQuality(Base):
    """LabeloxAV label quality layer (M13): per-annotation quality score, inter-annotator agreement, and gold
    audit verdict, in a side table so the hot object row is untouched. Surfaced in the workspace so a reviewer
    sees which labels to trust."""

    __tablename__ = "annotation_quality"

    object_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("object.object_id", ondelete="CASCADE"), primary_key=True)
    quality: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    agreement: Mapped[float | None] = mapped_column(Float)          # inter-annotator agreement 0..1
    flags: Mapped[list] = mapped_column(JSONB, default=list)        # [tiny_box, off_screen, class_conflict, ...]
    audit_verdict: Mapped[str | None] = mapped_column(String(12))   # null | pass | fail (gold-set audit)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PlaneSLO(Base):
    """M19 hardening: a recorded per-plane SLO evaluation over an observation window. Each plane declares its
    latency/error/coverage budgets; a tick measures them and records whether the plane met its SLO and which
    objectives breached. This is the observability ledger the fleet-scale operator reads to see which plane is
    the bottleneck, rather than inferring it from scattered logs."""

    __tablename__ = "plane_slo"

    slo_id: Mapped[uuid.UUID] = _uuid_pk()
    plane: Mapped[str] = mapped_column(String(16), nullable=False)      # labelox|sanyx|calyx|sievyx|oraclyx|verdyx|forgyx
    window_s: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    met: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    metrics: Mapped[dict] = mapped_column(JSONB, default=dict)          # measured values
    breaches: Mapped[list] = mapped_column(JSONB, default=list)         # [{metric, value, threshold, op}]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_plane_slo_plane", "plane", "created_at"),)


class RedactionProof(Base):
    """M18 governance: a signed attestation that PII redaction (Gate A) ran over every frame of a release. The
    per-frame PiiAudit rows are the evidence; this rolls them up per release into a coverage number and an HMAC
    signature a buyer can verify. A release whose coverage is below the floor cannot pass, so an unredacted
    frame cannot hide inside a sold dataset."""

    __tablename__ = "redaction_proof"

    proof_id: Mapped[uuid.UUID] = _uuid_pk()
    release_commit: Mapped[str] = mapped_column(String(128), nullable=False)
    n_frames: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_covered: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    coverage: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    verdict: Mapped[str] = mapped_column(String(12), nullable=False, default="fail")  # pass|fail
    signature: Mapped[str | None] = mapped_column(String(128))
    uncovered: Mapped[list] = mapped_column(JSONB, default=list)        # frame ids missing a PII audit
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_redaction_proof_release", "release_commit"),)


class ConsentRecord(Base):
    """M18 governance: the consent basis and retention deadline for a session's data. Export is refused unless
    consent is granted; a session past its retention_until must be purged. One row per session, so DPDPA
    consent and retention are enforced at the spine, not left to policy documents."""

    __tablename__ = "consent_record"

    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("session.session_id", ondelete="CASCADE"), primary_key=True)
    consent_status: Mapped[str] = mapped_column(String(12), nullable=False, default="unknown")  # granted|denied|unknown
    legal_basis: Mapped[str | None] = mapped_column(String(64))         # consent|legitimate_interest|contract
    retention_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class FlywheelCycle(Base):
    """The adaptive flywheel controller (M17): one recorded cycle where VERDYX safety failures and SIEVYX ODD
    coverage gaps are turned into a label-budget allocation across problem slices and a set of collection tasks.
    This is the ledger of how the data engine steered its own attention, so a spend can be traced to the failure
    or gap that justified it."""

    __tablename__ = "flywheel_cycle"

    cycle_id: Mapped[uuid.UUID] = _uuid_pk()
    label_budget: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    signals: Mapped[dict] = mapped_column(JSONB, default=dict)          # {regressions, odd_gaps, safety_slices}
    allocation: Mapped[list] = mapped_column(JSONB, default=list)       # [{slice, labels, weight, reason}]
    collection_tasks: Mapped[list] = mapped_column(JSONB, default=list)  # [{cell, priority, target_count, reason}]
    rationale: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PlateWatchlist(Base):
    """A registration mark a security deployment is watching for (LabeloxSec).

    Deployment state, not reference data: a stolen-vehicle list or an access allow-list belongs to the site,
    not to the product. Stored on the normalised form because that is the only thing matching can be done on
    reliably ("KA 01 AB 1234", "ka-01-ab-1234" and "KA01AB1234" are one mark), with the raw text kept so an
    operator sees what they typed.
    """

    __tablename__ = "plate_watchlist"

    entry_id: Mapped[uuid.UUID] = _uuid_pk()
    plate_normalized: Mapped[str] = mapped_column(String(16), nullable=False, unique=True, index=True)
    plate_raw: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)
    # info: log it. warn: surface it. critical: page someone. The severity travels with the hit so a
    # downstream consumer does not have to re-derive urgency from the reason text.
    severity: Mapped[str] = mapped_column(String(16), nullable=False, default="warn")
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    added_by: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PlateRead(Base):
    """One plate read by the ANPR path (LabeloxSec).

    This is personal data. It exists only under a pack that declares the `anpr` capability; the AV pack does
    the opposite and blurs plates without ever reading them, and the capability gate refuses ANPR there. The
    session FK cascades so an erasure request removes these with the rest of the session's data rather than
    leaving plate text behind, which would defeat the erasure.

    ocr_conf is nullable on purpose: a generative-VLM reader exposes no calibrated score, and a fabricated
    number would make a confidence filter look meaningful when it is not.
    """

    __tablename__ = "plate_read"

    read_id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("session.session_id", ondelete="CASCADE"), index=True)
    frame_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("frame.frame_id", ondelete="CASCADE"), index=True)
    camera_id: Mapped[str | None] = mapped_column(String(64), index=True)

    plate_normalized: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    plate_raw: Mapped[str] = mapped_column(Text, nullable=False)
    plate_type: Mapped[str] = mapped_column(String(16), nullable=False)   # standard|bh_series|diplomatic|invalid
    state_code: Mapped[str | None] = mapped_column(String(4))
    rto_district: Mapped[str | None] = mapped_column(String(8))
    valid: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    det_conf: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    ocr_conf: Mapped[float | None] = mapped_column(Float)                # None = unmeasured, never faked
    format_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    bbox: Mapped[list[float] | None] = mapped_column(ARRAY(Float))

    watchlist_hit: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    watchlist_severity: Mapped[str | None] = mapped_column(String(16))
    pack_id: Mapped[str] = mapped_column(String(32), nullable=False, default="sec")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


# ---------------------------------------------------------------------------------------------------
# Identity, notification, and access-record tables.
#
# Until now the only credential was an admin-minted token: there was no way for a person to obtain access
# themselves, no second factor, and no route to a corporate directory. That is the single largest blocker to
# a real deployment, and it is why these tables exist.
# ---------------------------------------------------------------------------------------------------


class UserCredential(Base):
    """A password and its second factor, kept off the User row.

    Separate from `app_user` on purpose. The user row is read on every authenticated request to resolve a
    role, and a hash that expensive to compare has no business travelling with it. Keeping the secret in its
    own table also means a query that lists users cannot accidentally select it.
    """

    __tablename__ = "user_credential"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("app_user.user_id", ondelete="CASCADE"), primary_key=True)
    # scrypt, with its parameters and salt encoded in the string, so a future cost increase can be rolled
    # per user rather than forcing a global reset.
    password_hash: Mapped[str | None] = mapped_column(Text)
    password_set_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Failed-attempt throttling lives here rather than in a cache: a lockout that a process restart clears
    # is not a lockout.
    failed_attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # TOTP. The secret is only usable once confirmed: a half-enrolled factor that already blocked sign-in
    # would lock a user out of their own account with an authenticator they never finished setting up.
    totp_secret: Mapped[str | None] = mapped_column(Text)
    totp_confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Single-use recovery codes, stored hashed for the same reason the password is.
    recovery_hashes: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    # The last TOTP step accepted, so a code cannot be replayed inside its own validity window.
    last_totp_step: Mapped[int | None] = mapped_column(BigInteger)

    # Federated identity. Set when the account signs in through OIDC; a matching issuer+subject is the
    # authority, never the email, which a directory can reassign to a different person.
    oidc_issuer: Mapped[str | None] = mapped_column(Text)
    oidc_subject: Mapped[str | None] = mapped_column(Text)
    email: Mapped[str | None] = mapped_column(String(320))

    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_user_credential_oidc", "oidc_issuer", "oidc_subject", unique=True),
        Index("ix_user_credential_email", "email"),
    )


class PasswordReset(Base):
    """A single-use, expiring reset token.

    Stored as a hash, so a database read cannot be turned into an account takeover, and consumed on use so a
    token recovered from a mailbox later is inert.
    """

    __tablename__ = "password_reset"

    reset_id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("app_user.user_id", ondelete="CASCADE"), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Notification(Base):
    """One thing that happened which somebody needs to know about.

    Issue comments, job completions, gate blocks, drift breaches, and SLO alarms were all silent: the system
    knew, and no one was told unless they happened to be looking at the right page. A notification is
    addressed to a user, or to a role when the audience is "whoever is on duty".
    """

    __tablename__ = "notification"

    notification_id: Mapped[uuid.UUID] = _uuid_pk()
    # Exactly one of these is set. A user-addressed notification is personal; a role-addressed one is a duty
    # queue, and is marked read per user in `notification_read` rather than on the row itself.
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("app_user.user_id", ondelete="CASCADE"), index=True)
    role: Mapped[str | None] = mapped_column(String(16), index=True)

    kind: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(16), nullable=False, default="info")  # info|warn|critical
    title: Mapped[str] = mapped_column(Text, nullable=False)
    body: Mapped[str | None] = mapped_column(Text)
    # Where clicking it goes. A notification you cannot act on is just noise.
    href: Mapped[str | None] = mapped_column(Text)
    # What it is about, so a second event on the same subject can supersede rather than pile up.
    subject_type: Mapped[str | None] = mapped_column(String(32))
    subject_id: Mapped[str | None] = mapped_column(String(64))
    meta: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    read_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 server_default=func.now(), index=True)

    __table_args__ = (
        Index("ix_notification_user_unread", "user_id", "read_at"),
        Index("ix_notification_subject", "subject_type", "subject_id"),
    )


class NotificationRead(Base):
    """Per-user read state for a role-addressed notification, which has no single owner to mark."""

    __tablename__ = "notification_read"

    notification_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("notification.notification_id", ondelete="CASCADE"), primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("app_user.user_id", ondelete="CASCADE"), primary_key=True)
    read_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PiiAccessLog(Base):
    """Who looked at personal data, and at what.

    `PiiAudit` records what the redactor found. This records what a human saw. A DPDPA or GDPR enquiry asks
    both, and the second question had no answer: an unredacted frame could be fetched by any authenticated
    reviewer and nothing recorded that it happened.
    """

    __tablename__ = "pii_access_log"

    access_id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("app_user.user_id", ondelete="SET NULL"), index=True)
    user_name: Mapped[str | None] = mapped_column(String(64))
    # Kept even when the subject row is deleted, because the fact of access outlives the data accessed and
    # is exactly what an erasure enquiry needs to see.
    subject_type: Mapped[str] = mapped_column(String(32), nullable=False)   # frame|plate_read|speech|export
    subject_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("session.session_id", ondelete="SET NULL"), index=True)
    action: Mapped[str] = mapped_column(String(32), nullable=False)         # view|download|export|read_plate
    pii_kinds: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)  # ["face","plate","speech"]
    redacted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    route: Mapped[str | None] = mapped_column(Text)
    pack_id: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 server_default=func.now(), index=True)

    __table_args__ = (Index("ix_pii_access_user_time", "user_id", "created_at"),)


class ActivityEvent(Base):
    """What a person did, in order, so they can answer "what did I do today" without reading five tables.

    Reviews, objects, jobs, and exports each already record their own history, but none of them is a
    timeline: there was no way to see a shift's work as one sequence, and no way for a lead to see a team's.
    """

    __tablename__ = "activity_event"

    event_id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("app_user.user_id", ondelete="CASCADE"), index=True)
    user_name: Mapped[str | None] = mapped_column(String(64))
    verb: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    subject_type: Mapped[str | None] = mapped_column(String(32))
    subject_id: Mapped[str | None] = mapped_column(String(64))
    summary: Mapped[str | None] = mapped_column(Text)
    href: Mapped[str | None] = mapped_column(Text)
    meta: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 server_default=func.now(), index=True)

    __table_args__ = (Index("ix_activity_user_time", "user_id", "created_at"),)


class Experiment(Base):
    """A named line of training work, and the runs under it.

    The per-job metric curve already answered "how did this run go". It could not answer "is this family of
    runs getting better", which is the question a person actually asks between iterations, because nothing
    tied runs together: comparing two meant reading two job rows and remembering which hyperparameters went
    with which. An external tracker (wandb, mlflow) is the usual answer and would put the loop's own history
    outside the loop, where the gate cannot read it.
    """

    __tablename__ = "experiment"

    experiment_id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
    task_type: Mapped[str] = mapped_column(String(32), nullable=False, default="detection")
    description: Mapped[str | None] = mapped_column(Text)
    # What is being varied and what is held fixed, so a comparison between two runs is interpretable rather
    # than a diff of every hyperparameter at once.
    hypothesis: Mapped[str | None] = mapped_column(Text)
    tags: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    created_by: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ExperimentRun(Base):
    """One training job's place in an experiment, with the numbers that make it comparable.

    Denormalised from `training_job` on purpose. A job row is mutable operational state (status, progress,
    the live metric), and an experiment record is a fixed claim about a finished run: what it scored, on
    which gold set, against which baseline. Reading the comparison off mutable rows would let a later job
    edit-in-place change what an earlier comparison said.
    """

    __tablename__ = "experiment_run"

    run_id: Mapped[uuid.UUID] = _uuid_pk()
    experiment_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("experiment.experiment_id", ondelete="CASCADE"), index=True)
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("training_job.job_id", ondelete="SET NULL"), index=True)
    label: Mapped[str | None] = mapped_column(String(128))
    hparams: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    dataset_spec: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    metrics: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    # The whole curve, not just the endpoint, so a run that peaked and then overfit is distinguishable from
    # one that never learned.
    curve: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    gold_id: Mapped[str | None] = mapped_column(String(128))
    baseline_run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="running")
    notes: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (Index("ix_experiment_run_exp_started", "experiment_id", "started_at"),)


# ---------------------------------------------------------------------------------------------------
# Campaigns, security scene analytics, and edge telemetry.
# ---------------------------------------------------------------------------------------------------


class Campaign(Base):
    """A standing intent to improve one class, and the autonomous loop that pursues it.

    Every piece of the improvement loop already existed and every transition between them was a person
    remembering to do it: read the gate's per-class deficit, build a review batch, run the VLM judge over
    it, launch a retrain, attempt promotion, read the result, decide whether to go again. That is the work
    the flywheel was built to remove and it stayed manual, so a class stalled whenever nobody was watching.

    A campaign is the missing orchestration. It holds the target, the budget, and where it has got to; it
    stops on its own terms rather than running forever, because an autonomous loop with no stopping
    condition is a way to spend a GPU budget on a class that is not improving.
    """

    __tablename__ = "campaign"

    campaign_id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
    class_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    task_type: Mapped[str] = mapped_column(String(32), nullable=False, default="detection")

    # What "done" means. A campaign that cannot succeed also cannot stop, so a target is required.
    target_metric: Mapped[str] = mapped_column(String(32), nullable=False, default="recall")
    target_value: Mapped[float] = mapped_column(Float, nullable=False, default=0.6)
    # Labels this campaign may spend, total. The single hardest limit: the review batches it builds are
    # human time, and an autonomous loop must not be able to commission unbounded amounts of it.
    label_budget: Mapped[int] = mapped_column(Integer, nullable=False, default=2000)
    labels_spent: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    max_iterations: Mapped[int] = mapped_column(Integer, nullable=False, default=6)
    # Consecutive iterations without improvement before giving up. Two, not one: a single flat iteration is
    # noise, and stopping on it would abandon a class that was about to move.
    patience: Mapped[int] = mapped_column(Integer, nullable=False, default=2)

    # pending | running | blocked | succeeded | exhausted | stopped
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending", index=True)
    iteration: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    stalled_iterations: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    best_value: Mapped[float | None] = mapped_column(Float)
    # Whether a person must approve each gate crossing. On by default: a loop that can promote a model with
    # no human in it is a different product with a different risk profile.
    require_approval: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    autopilot_stages: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    created_by: Mapped[str | None] = mapped_column(String(64))
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class CampaignStep(Base):
    """One stage of one iteration, with what it produced.

    Recorded per step rather than summarised on the campaign, because the interesting question when a
    campaign stalls is which stage stopped paying: a campaign whose batches keep coming back all-correct
    has a mining problem, and one whose retrains never gate has a data problem, and the two look identical
    from the campaign row alone.
    """

    __tablename__ = "campaign_step"

    step_id: Mapped[uuid.UUID] = _uuid_pk()
    campaign_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("campaign.campaign_id", ondelete="CASCADE"), index=True)
    iteration: Mapped[int] = mapped_column(Integer, nullable=False)
    # mine | judge | label | train | evaluate | promote
    stage: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="running")
    detail: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    metrics: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    # What this step needs a person to do, when it needs one.
    awaiting: Mapped[str | None] = mapped_column(String(64))
    job_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (Index("ix_campaign_step_campaign_iter", "campaign_id", "iteration"),)


class CameraZone(Base):
    """A named region of a fixed camera's view, and what crossing it means.

    A static camera's whole value is that its frame does not move, which makes a polygon drawn on it a
    permanent statement about the world: this rectangle is the loading bay, this line is the gate. The Sec
    pack had the static-camera scene model and no way to say either, so every detection was a detection
    somewhere in the picture and no rule could be written about place.
    """

    __tablename__ = "camera_zone"

    zone_id: Mapped[uuid.UUID] = _uuid_pk()
    camera_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("session.session_id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False, default="area")   # area | line
    # Image-pixel geometry: a closed polygon for an area, two points for a line.
    points: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    # What entering, dwelling in, or crossing it should raise.
    rule: Mapped[str] = mapped_column(String(24), nullable=False, default="enter")  # enter|exit|dwell|cross
    dwell_seconds: Mapped[float | None] = mapped_column(Float)
    # Which classes the rule applies to. Empty means every class, which is almost never what is wanted and
    # is therefore not the default anyone gets by accident: the UI requires a choice.
    classes: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    severity: Mapped[str] = mapped_column(String(16), nullable=False, default="warn")
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    pack_id: Mapped[str] = mapped_column(String(32), nullable=False, default="sec")
    created_by: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SecurityIncident(Base):
    """One thing that happened, stitched from everything that evidenced it.

    A plate read, a zone crossing and a person track were three unrelated rows about the same van arriving
    at the same gate at the same moment, and an operator had to assemble the event in their head from three
    screens. An incident is that assembly, made once and kept.
    """

    __tablename__ = "security_incident"

    incident_id: Mapped[uuid.UUID] = _uuid_pk()
    camera_id: Mapped[str | None] = mapped_column(String(64), index=True)
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("session.session_id", ondelete="CASCADE"), index=True)
    zone_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("camera_zone.zone_id", ondelete="SET NULL"))
    kind: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(16), nullable=False, default="warn")
    title: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text)
    # The window it spans, so two events a second apart are one incident rather than two.
    start_ts_ns: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    end_ts_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # Everything that evidenced it: plate read ids, object ids, track ids, frame ids.
    evidence: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    plate: Mapped[str | None] = mapped_column(String(16), index=True)
    person_identity: Mapped[str | None] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="open")  # open|ack|closed
    acknowledged_by: Mapped[str | None] = mapped_column(String(64))
    pack_id: Mapped[str] = mapped_column(String(32), nullable=False, default="sec")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 server_default=func.now(), index=True)

    __table_args__ = (Index("ix_incident_camera_time", "camera_id", "start_ts_ns"),)


class PersonIdentity(Base):
    """A person seen by more than one camera, as an appearance signature rather than a name.

    Deliberately not an identity in the ordinary sense. There is no name, no reference photograph and no
    enrolment: a signature is derived from tracks already recorded and links them to each other, so the
    system can say "the same person appeared at both gates" and can never say who they are. That boundary
    is the difference between re-identification for an authorised security deployment and building a face
    database, and it is enforced by there being nowhere to put a name.
    """

    __tablename__ = "person_identity"

    identity_id: Mapped[uuid.UUID] = _uuid_pk()
    signature: Mapped[list[float] | None] = mapped_column(ARRAY(Float))
    n_tracks: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    cameras: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    first_ts_ns: Mapped[int | None] = mapped_column(BigInteger)
    last_ts_ns: Mapped[int | None] = mapped_column(BigInteger)
    pack_id: Mapped[str] = mapped_column(String(32), nullable=False, default="sec")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PersonSighting(Base):
    """One track attributed to one signature, with how confident the match was."""

    __tablename__ = "person_sighting"

    sighting_id: Mapped[uuid.UUID] = _uuid_pk()
    identity_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("person_identity.identity_id", ondelete="CASCADE"), index=True)
    track_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), index=True)
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("session.session_id", ondelete="CASCADE"))
    camera_id: Mapped[str | None] = mapped_column(String(64), index=True)
    ts_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    similarity: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EdgeDevice(Base):
    """A device running a deployed artifact, and what it reports back.

    FORGYX gates on numbers measured on a bench. A bench does not thermally throttle after twenty minutes
    in a parked vehicle, does not share its GPU with a video encoder, and does not see the input
    distribution the field sees. So the gate has been passing artifacts on figures that are true in a room
    nobody deploys in.
    """

    __tablename__ = "edge_device"

    device_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str | None] = mapped_column(String(128))
    hardware: Mapped[str | None] = mapped_column(String(64))        # jetson_orin_nx | rpi5 | hailo8 | ...
    runtime: Mapped[str | None] = mapped_column(String(32))         # tensorrt | onnxruntime | litert
    artifact_id: Mapped[str | None] = mapped_column(String(128), index=True)
    model_version: Mapped[str | None] = mapped_column(String(64), index=True)
    fleet: Mapped[str | None] = mapped_column(String(64), index=True)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    meta: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EdgeTelemetry(Base):
    """One reporting window from one device.

    A window rather than a sample: a device that posted every inference would spend its uplink on telemetry,
    and the numbers that matter (p50, p95, the thermal ceiling reached) are properties of a window anyway.
    """

    __tablename__ = "edge_telemetry"

    telemetry_id: Mapped[uuid.UUID] = _uuid_pk()
    device_id: Mapped[str] = mapped_column(
        ForeignKey("edge_device.device_id", ondelete="CASCADE"), index=True)
    artifact_id: Mapped[str | None] = mapped_column(String(128), index=True)
    model_version: Mapped[str | None] = mapped_column(String(64))
    window_start_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    window_end_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    n_inferences: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    latency_p50_ms: Mapped[float | None] = mapped_column(Float)
    latency_p95_ms: Mapped[float | None] = mapped_column(Float)
    latency_max_ms: Mapped[float | None] = mapped_column(Float)
    fps: Mapped[float | None] = mapped_column(Float)
    temp_c_max: Mapped[float | None] = mapped_column(Float)
    throttled_fraction: Mapped[float | None] = mapped_column(Float)
    power_w_mean: Mapped[float | None] = mapped_column(Float)
    # Score distribution as a histogram, which is how field accuracy drift is detectable without labels:
    # the field has no ground truth, and a confidence distribution that has moved is the signal available.
    conf_histogram: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    detections_per_frame: Mapped[float | None] = mapped_column(Float)
    dropped_frames: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    meta: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 server_default=func.now(), index=True)

    __table_args__ = (Index("ix_edge_telemetry_artifact_time", "artifact_id", "created_at"),)
