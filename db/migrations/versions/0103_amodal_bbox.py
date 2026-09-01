"""The whole extent of a partly hidden object, alongside the part you can see.

`Object` carried exactly one box, and that box has always meant "the visible extent". For a car half
behind a bus that is the right thing to store for a detector's recall and the wrong thing for anything
that reasons about the world: a planner wants to know the car is car-sized, a tracker wants a centre that
does not lurch when the occlusion starts and stops, and a 3D lift wants a footprint that is not a sliver.
The repo has no word for it at all - grepping for "amodal" across the tree returned nothing.

Additive, and idiomatic here: `rot_deg`, `polyline` and `keypoints` were each added to `object` the same
way, each with the invariant that `bbox` stays the axis-aligned visible box. Nothing reads the new column
until something writes it, and `bbox` keeps its meaning exactly.

Nullable on purpose and not backfilled. An amodal box is a judgement about what is hidden, and there is no
honest way to compute one for 578,436 existing objects: guessing from a class prior would fill the corpus
with confident fabrications that read identically to observations. Null means nobody has said.

Revision ID: 0103_amodal_bbox
Revises: 0102_track_event
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0103_amodal_bbox"
down_revision = "0102_track_event"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("object", sa.Column("bbox_amodal", postgresql.ARRAY(sa.Float()), nullable=True))
    # Partial: a filter on "has an amodal box" is the only query shape, and the column is null on
    # essentially every row, so indexing the whole table would be a large index over nothing.
    op.create_index("ix_object_amodal", "object", ["frame_id"],
                    postgresql_where=sa.text("bbox_amodal IS NOT NULL"))


def downgrade() -> None:
    op.drop_index("ix_object_amodal", table_name="object")
    op.drop_column("object", "bbox_amodal")
