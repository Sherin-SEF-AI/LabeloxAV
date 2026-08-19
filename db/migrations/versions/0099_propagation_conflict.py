"""Where ego geometry and the tracker disagree about where a box went.

A label can be carried to the next frame two ways: by the ground homography the ego motion induces, or by
the tracker. When they agree the propagated box is trustworthy. When they disagree, writing either one is
a guess, and writing the average is a box neither method proposed.

So neither is written and the disagreement is recorded. The resulting table is more useful than the label
would have been: it is a list of the exact frames where the calibration, the ego pose or the tracker is
wrong, which is the thing worth fixing.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0099_propagation_conflict"
down_revision = "0098_clique_al"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "propagation_conflict",
        sa.Column("conflict_id", sa.UUID(), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("session_id", sa.UUID()),
        sa.Column("from_frame_id", sa.UUID(), nullable=False),
        sa.Column("to_frame_id", sa.UUID(), nullable=False),
        sa.Column("object_id", sa.UUID()),
        sa.Column("class_id", sa.Integer()),
        sa.Column("motion_model", sa.String(24)),
        sa.Column("geometry_box", postgresql.JSONB()),
        sa.Column("tracker_box", postgresql.JSONB()),
        sa.Column("iou", sa.Float()),
        sa.Column("tolerance", sa.Float()),
        sa.Column("reason", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["from_frame_id"], ["frame.frame_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["to_frame_id"], ["frame.frame_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["object_id"], ["object.object_id"], ondelete="SET NULL"),
    )
    op.create_index("ix_propagation_conflict_frames", "propagation_conflict",
                    ["from_frame_id", "to_frame_id"])
    op.create_index("ix_propagation_conflict_session", "propagation_conflict", ["session_id"])


def downgrade() -> None:
    op.drop_index("ix_propagation_conflict_session", table_name="propagation_conflict")
    op.drop_index("ix_propagation_conflict_frames", table_name="propagation_conflict")
    op.drop_table("propagation_conflict")
