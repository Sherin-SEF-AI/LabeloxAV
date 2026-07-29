"""Named saves of a frame's annotations, so a good state can be got back.

The editor has had undo since it was written, and undo is not the same thing as a save. It is capped at a
hundred steps, it lives in one browser tab, and it dies with a refresh. An annotator who spends an hour on a
dense frame, tries a different interpretation of a cluttered junction, and wants the earlier one back has no
way to ask for it. The work is simply gone.

A checkpoint stores the frame's objects as they stood, under a name a person chose. Restoring one is the
inverse. That is deliberately a full snapshot rather than a diff: a diff chain is only as good as its weakest
link and the point of this table is to be the thing you can always get back to, which a chain that has to
replay cleanly is not.

Snapshots are stored inline rather than in the object store. A frame's objects are on the order of kilobytes
of JSON, the restore path wants them transactionally consistent with the rows it is about to replace, and
putting them behind a second system means a checkpoint can exist whose contents have gone.

Revision ID: 0080_annotation_checkpoint
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0080_annotation_checkpoint"
down_revision = "0079_lane_marking"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "annotation_checkpoint",
        sa.Column("checkpoint_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("frame_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("frame.frame_id", ondelete="CASCADE"), nullable=False),
        sa.Column("session_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("session.session_id", ondelete="CASCADE"), nullable=True),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        # The objects exactly as they stood. Full, not a diff: this is the thing you can always get back to.
        sa.Column("objects", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("object_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_by", sa.String(64), nullable=True),
        # Set on the checkpoint a restore takes automatically, so the state a restore replaced is never lost
        # and undoing a restore is itself a restore rather than a hope.
        sa.Column("auto", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    # "the checkpoints on this frame, newest first" is the only listing the editor does.
    op.create_index("ix_checkpoint_frame_created", "annotation_checkpoint",
                    ["frame_id", "created_at"])
    op.create_index("ix_checkpoint_session", "annotation_checkpoint", ["session_id"])


def downgrade() -> None:
    op.drop_index("ix_checkpoint_session", table_name="annotation_checkpoint")
    op.drop_index("ix_checkpoint_frame_created", table_name="annotation_checkpoint")
    op.drop_table("annotation_checkpoint")
