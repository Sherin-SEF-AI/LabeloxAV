"""What space is occupied around the vehicle, and how that space is moving.

A cuboid says where one object is. It does not say what is between the objects, and "is there anything in
that lane" is a different question from "which cars did the detector find": a cuboid list has no answer
for the unlabelled truck, the pile of sand, or the gap the detector missed. An occupancy grid answers it
for every cubic metre, including the ones nothing was labelled in.

The fourth dimension is scene flow. A static grid cannot tell a parked car from one reversing toward the
ego at the moment the frame was taken, and that difference is the whole of planning. Each occupied voxel
carries a velocity in metres per second, taken from the 3D track that owns it; a voxel no track claims
holds zero flow, and the row records how much of the grid that was, so a grid mostly made of unclaimed
voxels reads as one nobody should plan against.

The grid lives in the object store as a packed npz rather than in a column, for the same reason point
clouds do: a 200 by 200 by 32 grid with three flow channels is megabytes, and Postgres is not a blob
store. The row carries the geometry needed to interpret it without opening it.

`ego_pose_ts` is the timestamp of the pose the grid was placed with, and it is nullable. A grid built
without a pose is in the ego frame of one instant and cannot be compared with the next one; recording
which pose was used, or that none was, is what keeps that from being discovered later.

Revision ID: 0113_occupancy
Revises: 0112_ego_pose
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0113_occupancy"
down_revision = "0112_ego_pose"
branch_labels = None
depends_on = None

SOURCES = ("pseudo", "lidar", "fused")


def upgrade() -> None:
    op.create_table(
        "occupancy_grid",
        sa.Column("grid_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("session_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("session.session_id", ondelete="CASCADE"), nullable=False),
        sa.Column("ts_ns", sa.BigInteger(), nullable=False),
        sa.Column("frame_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("frame.frame_id", ondelete="SET NULL"), nullable=True),
        # The world-frame position of voxel (0,0,0)'s corner, so a consumer can place the grid without
        # re-deriving it from the pose.
        sa.Column("origin", postgresql.ARRAY(sa.Float()), nullable=False),
        sa.Column("voxel_m", sa.Float(), nullable=False),
        sa.Column("dims", postgresql.ARRAY(sa.Integer()), nullable=False),
        sa.Column("grid_uri", sa.Text(), nullable=False),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("ego_pose_ts", sa.BigInteger(), nullable=True),
        sa.Column("occupied", sa.Integer(), nullable=False, server_default="0"),
        # How much of the occupied space has a velocity from a track rather than an assumed zero. A grid
        # that is mostly assumed-static is not a 4D grid and the number is what says so.
        sa.Column("flow_voxels", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("run_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("agent_run.run_id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("source in ('" + "','".join(SOURCES) + "')", name="ck_occupancy_source"),
        sa.CheckConstraint("voxel_m > 0", name="ck_occupancy_voxel_positive"),
        sa.CheckConstraint("array_length(dims, 1) = 3", name="ck_occupancy_dims_3d"),
        sa.CheckConstraint("array_length(origin, 1) = 3", name="ck_occupancy_origin_3d"),
        sa.CheckConstraint("flow_voxels <= occupied", name="ck_occupancy_flow_within_occupied"),
        sa.UniqueConstraint("session_id", "ts_ns", "source", name="uq_occupancy_session_ts_source"),
    )
    op.create_index("ix_occupancy_session_ts", "occupancy_grid", ["session_id", "ts_ns"])
    op.create_index("ix_occupancy_frame", "occupancy_grid", ["frame_id"])


def downgrade() -> None:
    op.drop_index("ix_occupancy_frame", table_name="occupancy_grid")
    op.drop_index("ix_occupancy_session_ts", table_name="occupancy_grid")
    op.drop_table("occupancy_grid")
