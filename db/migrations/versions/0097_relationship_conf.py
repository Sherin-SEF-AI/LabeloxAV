"""A relationship proposal had a strength and nowhere to put it.

`ObjectRelationship` already carries `status` (proposed | confirmed) and `source` (human | geometry | vlm),
so the notion of a machine-proposed edge is real. What it lacked was a number: how strongly the proposer
believed it. Every writer put its own key inside the `evidence` JSONB (`overlap`, `lane_align`,
`centre_gap`, `ahead_by`), so no query could rank proposals, threshold them, or sort a review queue by how
likely an edge was to be right.

Relationship-aware NMS makes that worse by producing many more of them: every pair it declines to merge is
a candidate edge. Without a comparable strength they would arrive as an undifferentiated pile.

Nullable, and null for a human-drawn edge. A person did not assign a confidence and inventing 1.0 for them
would make the column mean two different things.
"""

import sqlalchemy as sa
from alembic import op

revision = "0097_relationship_conf"
down_revision = "0096_threshold_fit"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("object_relationship", sa.Column("conf", sa.Float(), nullable=True))
    # Proposals only. A confirmed human edge stays null on purpose.
    op.create_index("ix_object_relationship_proposed", "object_relationship", ["frame_id", "conf"],
                    postgresql_where=sa.text("status = 'proposed'"))


def downgrade() -> None:
    op.drop_index("ix_object_relationship_proposed", table_name="object_relationship")
    op.drop_column("object_relationship", "conf")
