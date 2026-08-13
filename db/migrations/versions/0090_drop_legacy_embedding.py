"""The legacy CLIP embedding table, which had one reader left and it was broken.

The intelligence layer moved to pgvector in migration 0015: `object_embedding` holds a DINOv3 vector for
visual similarity and, since 0073, a SigLIP2 vector for the shared image-text space. The `embedding` table
that preceded it was left in place and, over time, everything stopped writing to it.

Everything except one path. The correction dialog, the tool that offers to apply a fix to every object
sharing a mistake, still read it. On the live corpus that meant searching 39 rows against the 567,527 in
`object_embedding`, so it reported zero similar objects for every correction anyone had ever made. Its
source-vector helper also wrote a row back into this table on each use, which is where the 39 came from:
the feature was slowly filling its own search space with objects that had just been corrected and therefore
could never match the class being searched for.

With that path moved, and `scenario_embedding` and the two compute endpoints moved with it, nothing reads or
writes this table. It is dropped rather than left as a trap for the next person to find it and assume it is
the embedding store.

The downgrade recreates the table and its index. It cannot recreate the vectors, and should not pretend to:
they were CLIP, the model is no longer loaded anywhere in the tree, and the 39 rows described a corpus that
has since been relabelled. An empty table is the honest restoration of the schema.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0090_drop_legacy_embedding"
down_revision = "0089_queued_cloud_status"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_table("embedding")


def downgrade() -> None:
    op.create_table(
        "embedding",
        sa.Column("object_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("object.object_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("model", sa.String(48), nullable=False),
        sa.Column("dim", sa.Integer(), nullable=False),
        sa.Column("vec", postgresql.ARRAY(sa.Float()), nullable=False),
    )
