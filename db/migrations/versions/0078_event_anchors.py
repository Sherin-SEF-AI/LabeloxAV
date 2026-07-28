"""Anchor timeline events to the thing they are about.

A timeline event has until now been a span of time on a session and nothing else. That is enough for an
inertial spike, which really is just an instant on a signal, and it is why the table has sat empty on a
corpus that carries 4,558 lane rows and 10,076 objects labelled with a signal state. The events that matter
on a road are not about a session, they are about an actor: this car changed lane, that signal went red,
this rider crossed a solid line. With nowhere to put the actor, deriving those events produced records
nobody could join back to anything, so nothing derived them.

track_id is the actor and frame_id is where a point event happened. Both nullable, because the existing
inertial and audio events are genuinely about the session alone and backfilling them with a made-up anchor
would be worse than leaving them unanchored. SET NULL rather than CASCADE on both: an event that a person
confirmed is a finding about the drive, and it should survive the deletion of the track that suggested it
rather than silently vanishing from the record.

conf exists because these events are derived by the same kind of fallible geometry as everything else in the
pipeline, and an event with no confidence cannot be gated, ranked for review, or compared against the
threshold that decides what a human sees first.

Revision ID: 0078_event_anchors
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0078_event_anchors"
down_revision = "0077_campaigns_secv2_edge"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("timeline_event",
                  sa.Column("track_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column("timeline_event",
                  sa.Column("frame_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column("timeline_event", sa.Column("conf", sa.Float(), nullable=True))

    op.create_foreign_key("fk_timeline_event_track", "timeline_event", "track",
                          ["track_id"], ["track_id"], ondelete="SET NULL")
    op.create_foreign_key("fk_timeline_event_frame", "timeline_event", "frame",
                          ["frame_id"], ["frame_id"], ondelete="SET NULL")

    # "every lane change on this track" and "every signal event in this session" are the two queries the
    # review surfaces run, and neither is served by the existing (session_id, t_start_ns) index.
    op.create_index("ix_timeline_event_track", "timeline_event", ["track_id"])
    op.create_index("ix_timeline_event_session_kind", "timeline_event", ["session_id", "kind"])


def downgrade() -> None:
    op.drop_index("ix_timeline_event_session_kind", table_name="timeline_event")
    op.drop_index("ix_timeline_event_track", table_name="timeline_event")
    op.drop_constraint("fk_timeline_event_frame", "timeline_event", type_="foreignkey")
    op.drop_constraint("fk_timeline_event_track", "timeline_event", type_="foreignkey")
    op.drop_column("timeline_event", "conf")
    op.drop_column("timeline_event", "frame_id")
    op.drop_column("timeline_event", "track_id")
