"""Where the vehicle was, per frame, and how much of that is measured rather than inferred.

Nothing in this engine has ever known where the car was at a given moment. `Frame.gnss` is populated on
3 frames of 41,752 and `Frame.ego_speed` on 6, so `calyx/ego_propagate.ego_transform` refuses on every
session in the corpus and is right to: an uncompensated copy assumes the camera stood still, which is
wrong for every static object. Cuboids lifted from separate frames have no common frame to sit in, so a
track's boxes jitter with no way to tell motion from noise.

One row per (session, timestamp). The pose is a position and a quaternion in a session-local ENU frame
whose origin is the session's first fix, or its first frame when there is no fix at all: an absolute
frame would imply a georeferencing accuracy no monocular method here can deliver.

`measured` is the column that matters and it is separate from `quality` on purpose. `measured = true`
means the pose came from an instrument that observed position (GNSS, an IMU, a fused solution).
`measured = false` means it was inferred from the images, which is the only source available for almost
this entire corpus. A consumer that needs a real trajectory can select on it; one that needs relative
motion between neighbouring frames can use both and say which it used. `quality` then grades within a
source and never substitutes for it, because a confident visual estimate is still not a measurement.

The downgrade drops the table. Nothing else references it: readers take the absence of a row as "the
pose is unknown here", which is exactly the state the whole corpus is in today.

Revision ID: 0112_ego_pose
Revises: 0111_inference_class_vocab
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0112_ego_pose"
down_revision = "0111_inference_class_vocab"
branch_labels = None
depends_on = None

SOURCES = ("gnss_imu", "visual", "fused")


def upgrade() -> None:
    op.create_table(
        "ego_pose",
        sa.Column("session_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("session.session_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("ts_ns", sa.BigInteger(), primary_key=True),
        sa.Column("frame_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("frame.frame_id", ondelete="SET NULL"), nullable=True),
        sa.Column("x", sa.Float(), nullable=False),
        sa.Column("y", sa.Float(), nullable=False),
        sa.Column("z", sa.Float(), nullable=False, server_default="0"),
        sa.Column("qw", sa.Float(), nullable=False, server_default="1"),
        sa.Column("qx", sa.Float(), nullable=False, server_default="0"),
        sa.Column("qy", sa.Float(), nullable=False, server_default="0"),
        sa.Column("qz", sa.Float(), nullable=False, server_default="0"),
        sa.Column("speed_mps", sa.Float(), nullable=True),
        sa.Column("yaw_rate", sa.Float(), nullable=True),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("quality", sa.Float(), nullable=True),
        sa.Column("measured", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("run_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("agent_run.run_id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("source in ('" + "','".join(SOURCES) + "')", name="ck_ego_pose_source"),
        # A quaternion that is not a rotation is not a pose. Checked in the database because the writer
        # is not the only thing that will ever insert here.
        sa.CheckConstraint("abs(qw*qw + qx*qx + qy*qy + qz*qz - 1.0) < 0.01",
                           name="ck_ego_pose_unit_quaternion"),
        sa.CheckConstraint("quality is null or (quality >= 0 and quality <= 1)",
                           name="ck_ego_pose_quality_range"),
    )
    op.create_index("ix_ego_pose_frame", "ego_pose", ["frame_id"])
    op.create_index("ix_ego_pose_session_measured", "ego_pose", ["session_id", "measured"])


def downgrade() -> None:
    op.drop_index("ix_ego_pose_session_measured", table_name="ego_pose")
    op.drop_index("ix_ego_pose_frame", table_name="ego_pose")
    op.drop_table("ego_pose")
