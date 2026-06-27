"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-06-24
"""
from typing import Sequence, Union

import geoalchemy2
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")

    op.create_table(
        "ontology_version",
        sa.Column("version", sa.String(64), primary_key=True),
        sa.Column("hierarchy_levels", sa.Integer(), nullable=False),
        sa.Column("attributes", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "ontology_class",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("version", sa.String(64), sa.ForeignKey("ontology_version.version", ondelete="CASCADE")),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("l0", sa.String(32), nullable=False),
        sa.Column("l1", sa.String(32), nullable=False),
        sa.Column("india", sa.Boolean(), server_default=sa.false()),
        sa.Column("map_to", postgresql.JSONB(), server_default="{}"),
    )
    op.create_index("ix_ontology_class_name", "ontology_class", ["name"])

    op.create_table(
        "session",
        sa.Column("session_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("vehicle_id", sa.String(64), nullable=False),
        sa.Column("start_ts_ns", sa.BigInteger(), nullable=False),
        sa.Column("end_ts_ns", sa.BigInteger(), nullable=False),
        sa.Column("city", sa.String(64)),
        sa.Column("route", sa.String(128)),
        sa.Column("sensors", postgresql.JSONB(), server_default="{}"),
        sa.Column("raw_uri", sa.Text()),
        sa.Column("mcap_uri", sa.Text()),
        sa.Column("manifest_uri", sa.Text()),
        sa.Column("ontology_version", sa.String(64), nullable=False),
        sa.Column("commit_id", sa.String(128)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_session_start_ts", "session", ["start_ts_ns"])

    op.create_table(
        "frame",
        sa.Column("frame_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("session.session_id", ondelete="CASCADE")),
        sa.Column("ts_ns", sa.BigInteger(), nullable=False),
        sa.Column("cam_id", sa.String(32), nullable=False),
        sa.Column("img_uri", sa.Text(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("gnss", geoalchemy2.Geography(geometry_type="POINT", srid=4326)),
        sa.Column("ego_speed", sa.Float()),
        sa.Column("quality", sa.Float(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_frame_session_ts", "frame", ["session_id", "ts_ns"])
    op.create_index("ix_frame_ts", "frame", ["ts_ns"])

    op.create_table(
        "track",
        sa.Column("track_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("session.session_id", ondelete="CASCADE")),
        sa.Column("class_id", sa.Integer(), sa.ForeignKey("ontology_class.id")),
        sa.Column("first_ts_ns", sa.BigInteger(), nullable=False),
        sa.Column("last_ts_ns", sa.BigInteger(), nullable=False),
        sa.Column("trajectory", postgresql.JSONB()),
    )
    op.create_index("ix_track_session", "track", ["session_id"])

    op.create_table(
        "object",
        sa.Column("object_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("frame_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("frame.frame_id", ondelete="CASCADE")),
        sa.Column("track_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("track.track_id", ondelete="SET NULL")),
        sa.Column("class_id", sa.Integer(), sa.ForeignKey("ontology_class.id")),
        sa.Column("bbox", postgresql.ARRAY(sa.Float()), nullable=False),
        sa.Column("mask_uri", sa.Text()),
        sa.Column("mask_encoding", sa.String(16)),
        sa.Column("attrs", postgresql.JSONB(), server_default="{}"),
        sa.Column("conf", sa.Float(), nullable=False),
        sa.Column("source", sa.String(16), nullable=False, server_default="fused"),
        sa.Column("provenance", postgresql.JSONB(), server_default="{}"),
        sa.Column("state", sa.String(16), nullable=False, server_default="review"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_object_frame", "object", ["frame_id"])
    op.create_index("ix_object_state", "object", ["state"])
    op.create_index("ix_object_class", "object", ["class_id"])
    op.create_index("ix_object_track", "object", ["track_id"])

    op.create_table(
        "review",
        sa.Column("review_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("object_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("object.object_id", ondelete="CASCADE")),
        sa.Column("reviewer", sa.String(64), nullable=False),
        sa.Column("action", sa.String(32), nullable=False),
        sa.Column("before", postgresql.JSONB()),
        sa.Column("after", postgresql.JSONB()),
        sa.Column("time_spent_ms", sa.Integer(), server_default="0"),
        sa.Column("ts_ns", sa.BigInteger(), nullable=False),
    )
    op.create_index("ix_review_object", "review", ["object_id"])

    op.create_table(
        "dataset_commit",
        sa.Column("commit_id", sa.String(128), primary_key=True),
        sa.Column("parent_id", sa.String(128)),
        sa.Column("slice_spec", postgresql.JSONB(), server_default="{}"),
        sa.Column("object_count", sa.Integer(), server_default="0"),
        sa.Column("ontology_version", sa.String(64), nullable=False),
        sa.Column("export_uris", postgresql.JSONB(), server_default="{}"),
        sa.Column("notes", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("dataset_commit")
    op.drop_table("review")
    op.drop_table("object")
    op.drop_table("track")
    op.drop_table("frame")
    op.drop_table("session")
    op.drop_index("ix_ontology_class_name", table_name="ontology_class")
    op.drop_table("ontology_class")
    op.drop_table("ontology_version")
