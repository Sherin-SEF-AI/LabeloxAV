"""track_event: typed spans within a track

`Track.intents` is the same idea without a time extent, and it holds 0 rows across 11,406 tracks. An intent
can say a track cut in; it cannot say the 93-frame track cut in over frames 40 to 55, so nothing downstream
can crop the clip, count the exposure, or compare two events on one track.

A table rather than more JSONB because "every stopping_in_live_lane in the corpus" is a query the export and
the coverage datasheet both need, and scanning a JSONB list on every track is not that query.

The event type is not a check constraint. The vocabulary belongs to whichever domain pack the session was
captured under, and freezing the AV list into the schema would make the second domain's events invalid rows.
The router validates against `pack.track_events` instead.

Revision ID: 0102_track_event
Revises: 0101_ontology_0_2_0_gold
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0102_track_event"
down_revision = "0101_ontology_0_2_0_gold"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "track_event",
        sa.Column("event_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("track_id", UUID(as_uuid=True),
                  sa.ForeignKey("track.track_id", ondelete="CASCADE"), nullable=False),
        sa.Column("event_type", sa.String(48), nullable=False),
        sa.Column("start_frame_id", UUID(as_uuid=True),
                  sa.ForeignKey("frame.frame_id", ondelete="CASCADE"), nullable=False),
        sa.Column("end_frame_id", UUID(as_uuid=True),
                  sa.ForeignKey("frame.frame_id", ondelete="CASCADE"), nullable=False),
        sa.Column("start_ts_ns", sa.BigInteger(), nullable=False),
        sa.Column("end_ts_ns", sa.BigInteger(), nullable=False),
        sa.Column("source", sa.String(16), nullable=False, server_default="human"),
        sa.Column("state", sa.String(16), nullable=False, server_default="proposed"),
        sa.Column("confidence", sa.Float()),
        sa.Column("evidence", JSONB(), nullable=False, server_default="{}"),
        sa.Column("notes", sa.Text()),
        sa.Column("created_by", sa.String(128)),
        sa.Column("ontology_version", sa.String(64)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("end_ts_ns >= start_ts_ns", name="ck_track_event_span"),
        sa.CheckConstraint("state in ('proposed','accepted','rejected')", name="ck_track_event_state"),
        sa.CheckConstraint("source in ('human','heuristic','vlm')", name="ck_track_event_source"),
    )
    op.create_index("ix_track_event_track", "track_event", ["track_id"])
    op.create_index("ix_track_event_type_state", "track_event", ["event_type", "state"])


def downgrade() -> None:
    op.drop_index("ix_track_event_type_state", table_name="track_event")
    op.drop_index("ix_track_event_track", table_name="track_event")
    op.drop_table("track_event")
