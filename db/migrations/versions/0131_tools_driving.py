"""Schema for the driving annotation tools: lane relations, labelled paths, and manual camera-LiDAR alignment.

Three tools need somewhere to put what a person decides:

- `lane_relation` (#7 lanes in a top-down view). A lane line on its own is geometry. The map needs to know
  which line continues which (predecessor / successor), which lines bound the same lane (left / right
  neighbour), and which signal or stop line governs it. Each relation is a row naming exactly one target: a
  lane, an object (a traffic signal box on the same frame), or a map element.
- `path_label` (#8 paths and intent). The ego path as a person corrected it, and a road user's future path as a
  person drew it. Stored beside the measurement, never over it: `ego_pose` keeps what the GNSS said, and an
  edit is a separate row with its own source. `source` tells a kinematic baseline accepted as-is from a path a
  person actually shaped.
- `calibration_alignment` (#9 camera-LiDAR alignment). Every alignment a person saved or an automatic
  refinement proposed, with the scores before and after, so a saved calibration can be traced to the evidence
  it was saved on.

And `camera_calibration` learns `source = 'manual'`. Until now it held one row per (session, camera). A
manual alignment must never overwrite the dataset's own calibration, so the unique index becomes two partial
ones: one row per camera among the non-manual sources (unchanged behaviour), and at most one manual row beside
it. The resolver prefers the manual row when one exists; deleting it restores the dataset calibration exactly.

Revision ID: 0131_tools_driving
Revises: 0130_tools_quality
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0131_tools_driving"
down_revision = "0130_tools_quality"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "lane_relation",
        sa.Column("relation_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", UUID(as_uuid=True), sa.ForeignKey("session.session_id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("lane_id", UUID(as_uuid=True), sa.ForeignKey("lane.lane_id", ondelete="CASCADE"), nullable=False),
        sa.Column("relation", sa.String(24), nullable=False),
        sa.Column("target_lane_id", UUID(as_uuid=True), sa.ForeignKey("lane.lane_id", ondelete="CASCADE")),
        sa.Column("target_object_id", UUID(as_uuid=True), sa.ForeignKey("object.object_id", ondelete="CASCADE")),
        sa.Column("target_element_id", UUID(as_uuid=True),
                  sa.ForeignKey("map_element.element_id", ondelete="CASCADE")),
        sa.Column("created_by", UUID(as_uuid=True), sa.ForeignKey("app_user.user_id", ondelete="SET NULL")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint("relation in ('predecessor','successor','left_neighbour','right_neighbour',"
                           "'controlled_by','stops_at')", name="ck_lane_relation_kind"),
        sa.CheckConstraint("(target_lane_id is not null)::int + (target_object_id is not null)::int"
                           " + (target_element_id is not null)::int = 1", name="ck_lane_relation_one_target"),
        sa.CheckConstraint("target_lane_id is null or target_lane_id <> lane_id", name="ck_lane_relation_not_self"),
    )
    op.create_index("ix_lane_relation_lane", "lane_relation", ["lane_id"])
    op.create_index("ix_lane_relation_target_lane", "lane_relation", ["target_lane_id"])
    op.create_index("ix_lane_relation_session", "lane_relation", ["session_id"])
    op.create_index("ux_lane_relation_edge", "lane_relation",
                    ["lane_id", "relation", sa.text("coalesce(target_lane_id, target_object_id, target_element_id)")],
                    unique=True)

    op.create_table(
        "path_label",
        sa.Column("path_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", UUID(as_uuid=True), sa.ForeignKey("session.session_id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("track_id", UUID(as_uuid=True), sa.ForeignKey("track.track_id", ondelete="CASCADE")),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("anchor_frame_id", UUID(as_uuid=True), sa.ForeignKey("frame.frame_id", ondelete="SET NULL")),
        sa.Column("anchor_ts_ns", sa.BigInteger(), nullable=False),
        sa.Column("coord_frame", sa.String(8), nullable=False),
        sa.Column("points", JSONB(), nullable=False, server_default="[]"),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("seed", JSONB(), nullable=False, server_default="{}"),
        sa.Column("notes", sa.Text()),
        sa.Column("created_by", UUID(as_uuid=True), sa.ForeignKey("app_user.user_id", ondelete="SET NULL")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint("kind in ('ego_path','future')", name="ck_path_label_kind"),
        sa.CheckConstraint("coord_frame in ('world','ego')", name="ck_path_label_frame"),
        sa.CheckConstraint("source in ('human','baseline_ctrv','smoothed')", name="ck_path_label_source"),
        sa.CheckConstraint("(kind = 'ego_path') = (track_id is null)", name="ck_path_label_subject"),
    )
    op.create_index("ix_path_label_session", "path_label", ["session_id"])
    op.create_index("ix_path_label_track", "path_label", ["track_id"])
    # One corrected ego path per session, and one future per track per anchor instant.
    op.create_index("ux_path_label_ego", "path_label", ["session_id"], unique=True,
                    postgresql_where=sa.text("kind = 'ego_path'"))
    op.create_index("ux_path_label_future", "path_label", ["track_id", "anchor_ts_ns"], unique=True,
                    postgresql_where=sa.text("kind = 'future'"))

    op.create_table(
        "calibration_alignment",
        sa.Column("alignment_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", UUID(as_uuid=True), sa.ForeignKey("session.session_id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("cam_id", sa.String(32), nullable=False),
        sa.Column("frame_ids", JSONB(), nullable=False, server_default="[]"),
        sa.Column("method", sa.String(16), nullable=False),
        sa.Column("base_source", sa.String(16), nullable=False),
        sa.Column("base_rpy_deg", JSONB(), nullable=False),
        sa.Column("base_xyz_m", JSONB(), nullable=False),
        sa.Column("rpy_deg", JSONB(), nullable=False),
        sa.Column("xyz_m", JSONB(), nullable=False),
        sa.Column("score_before", JSONB(), nullable=False, server_default="{}"),
        sa.Column("score_after", JSONB(), nullable=False, server_default="{}"),
        sa.Column("saved", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("calibration_id", UUID(as_uuid=True),
                  sa.ForeignKey("camera_calibration.id", ondelete="SET NULL")),
        sa.Column("tool_job_id", UUID(as_uuid=True)),
        sa.Column("notes", sa.Text()),
        sa.Column("created_by", UUID(as_uuid=True), sa.ForeignKey("app_user.user_id", ondelete="SET NULL")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint("method in ('manual','auto','pairs')", name="ck_calibration_alignment_method"),
    )
    op.create_index("ix_calibration_alignment_session", "calibration_alignment", ["session_id", "cam_id"])

    op.drop_index("ix_camera_calibration_session_cam", table_name="camera_calibration")
    op.create_index("ix_camera_calibration_session_cam", "camera_calibration", ["session_id", "cam_id"],
                    unique=True, postgresql_where=sa.text("source <> 'manual'"))
    op.create_index("ix_camera_calibration_session_cam_manual", "camera_calibration", ["session_id", "cam_id"],
                    unique=True, postgresql_where=sa.text("source = 'manual'"))


def downgrade() -> None:
    op.drop_index("ix_camera_calibration_session_cam_manual", table_name="camera_calibration")
    op.drop_index("ix_camera_calibration_session_cam", table_name="camera_calibration")
    # Two rows per camera cannot survive the old index; the manual one is the addition, so it goes. Where a
    # camera had only a manual row there is nothing to fall back to, and it is kept as the camera's row.
    op.execute("DELETE FROM camera_calibration m WHERE m.source = 'manual' AND EXISTS ("
               "SELECT 1 FROM camera_calibration d WHERE d.session_id = m.session_id AND d.cam_id = m.cam_id "
               "AND d.source <> 'manual')")
    op.create_index("ix_camera_calibration_session_cam", "camera_calibration", ["session_id", "cam_id"], unique=True)

    op.drop_index("ix_calibration_alignment_session", table_name="calibration_alignment")
    op.drop_table("calibration_alignment")
    op.drop_index("ux_path_label_future", table_name="path_label")
    op.drop_index("ux_path_label_ego", table_name="path_label")
    op.drop_index("ix_path_label_track", table_name="path_label")
    op.drop_index("ix_path_label_session", table_name="path_label")
    op.drop_table("path_label")
    op.drop_index("ux_lane_relation_edge", table_name="lane_relation")
    op.drop_index("ix_lane_relation_session", table_name="lane_relation")
    op.drop_index("ix_lane_relation_target_lane", table_name="lane_relation")
    op.drop_index("ix_lane_relation_lane", table_name="lane_relation")
    op.drop_table("lane_relation")
