"""Human 3D labels next to the generated ones: versioned point labels and occupancy edit layers.

Two existing tables gain what a person's edit needs, so a hand label never overwrites the machine's:

- `point_segmentation` gets `source` (auto: the projected fallback, model: a 3D segmenter such as PTv3, human: a
  person on /lidar/segment), `version` (per cloud and source, counting up), `base_seg_id` (the labels the edit
  started from) and `created_by`. A save is a new row; the projected and model rows are never rewritten.
- `occupancy_grid` accepts `source = 'human'`: a delta layer (voxels painted occupied or free) against one
  generated grid, named by `base_grid_id`. The generated grid stays exactly as the builder wrote it; the merged
  grid is base plus delta, computed on read.

Revision ID: 0128_tools_scene3d
Revises: 0127_tools_video
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "0128_tools_scene3d"
down_revision = "0127_tools_video"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("point_segmentation", sa.Column("source", sa.String(16), nullable=False, server_default="auto"))
    op.add_column("point_segmentation", sa.Column("version", sa.Integer(), nullable=False, server_default="1"))
    op.add_column("point_segmentation", sa.Column(
        "base_seg_id", UUID(as_uuid=True),
        sa.ForeignKey("point_segmentation.seg_id", ondelete="SET NULL"), nullable=True))
    op.add_column("point_segmentation", sa.Column(
        "created_by", UUID(as_uuid=True), sa.ForeignKey("app_user.user_id", ondelete="SET NULL"), nullable=True))
    # The projected fallback and anything a segmenter wrote before this column existed.
    op.execute("UPDATE point_segmentation SET source = 'model' WHERE method = 'ptv3'")
    op.create_check_constraint("ck_point_segmentation_source", "point_segmentation",
                               "source in ('auto','model','human')")
    op.create_index("ix_point_segmentation_cloud_source", "point_segmentation", ["cloud_id", "source", "version"])

    op.drop_constraint("ck_occupancy_source", "occupancy_grid", type_="check")
    op.create_check_constraint("ck_occupancy_source", "occupancy_grid",
                               "source in ('pseudo','lidar','fused','human')")
    op.add_column("occupancy_grid", sa.Column(
        "base_grid_id", UUID(as_uuid=True),
        sa.ForeignKey("occupancy_grid.grid_id", ondelete="CASCADE"), nullable=True))
    op.add_column("occupancy_grid", sa.Column(
        "edited_by", UUID(as_uuid=True), sa.ForeignKey("app_user.user_id", ondelete="SET NULL"), nullable=True))
    # A human layer always names the grid it edits, and only a human layer does.
    op.create_check_constraint("ck_occupancy_human_has_base", "occupancy_grid",
                               "(source = 'human') = (base_grid_id is not null)")
    op.create_index("ix_occupancy_base_grid", "occupancy_grid", ["base_grid_id"])
    op.create_index("ix_occupancy_edited_by", "occupancy_grid", ["edited_by"])
    op.create_index("ix_point_segmentation_base_seg", "point_segmentation", ["base_seg_id"])
    op.create_index("ix_point_segmentation_created_by", "point_segmentation", ["created_by"])


def downgrade() -> None:
    op.drop_index("ix_point_segmentation_created_by", table_name="point_segmentation")
    op.drop_index("ix_point_segmentation_base_seg", table_name="point_segmentation")
    op.drop_index("ix_occupancy_edited_by", table_name="occupancy_grid")
    op.drop_index("ix_occupancy_base_grid", table_name="occupancy_grid")
    op.drop_constraint("ck_occupancy_human_has_base", "occupancy_grid", type_="check")
    # Human edit layers cannot exist under the old source check; they go with the columns that describe them.
    op.execute("DELETE FROM occupancy_grid WHERE source = 'human'")
    op.drop_column("occupancy_grid", "edited_by")
    op.drop_column("occupancy_grid", "base_grid_id")
    op.drop_constraint("ck_occupancy_source", "occupancy_grid", type_="check")
    op.create_check_constraint("ck_occupancy_source", "occupancy_grid", "source in ('pseudo','lidar','fused')")

    op.drop_index("ix_point_segmentation_cloud_source", table_name="point_segmentation")
    op.drop_constraint("ck_point_segmentation_source", "point_segmentation", type_="check")
    op.drop_column("point_segmentation", "created_by")
    op.drop_column("point_segmentation", "base_seg_id")
    op.drop_column("point_segmentation", "version")
    op.drop_column("point_segmentation", "source")
