"""Object lifecycle: who put this label here, and who has ruled on it since.

An object at 0.34 confidence in state `accepted` looked exactly like one a reviewer had confirmed, because
the only attribution the row carried was `source`, and `source` collapses to "human" the moment anybody
touches it. The prior machine identity is then unrecoverable except from a Review row, which the agent path
and the 3D edit path never write.

`lifecycle` is the missing axis. `state` says which queue an object is in, `source` says who last wrote the
row, and `lifecycle` says how far along the machine-to-human path it has travelled. The three are related
but none of them is derivable from the others, which is why the badge could not answer the question an
annotator was asking it.

Deliberately not called `provenance`. That name is taken by the fusion audit blob in core/schemas.py, which
records which model paths proposed the object and with what confidence, and core/provenance.py documents its
walk as an audit spine that must never break. Two different things called provenance would be worse than a
slightly longer name.

`lifecycle_history` is append-only. Every transition is (state, actor, at), so a wrong badge can be traced to
the write that caused it rather than inferred from the current value.

Backfilling is a separate script with a dry run, not part of this migration. Under one percent of objects
have a derivable history, and a migration that silently defaults 99% of a corpus is the kind of thing that
gets believed later.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0081_object_lifecycle"
down_revision = "0080_annotation_checkpoint"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("object", sa.Column("lifecycle", sa.String(24), nullable=True))
    op.add_column("object", sa.Column(
        "lifecycle_history", postgresql.JSONB(astext_type=sa.Text()),
        server_default=sa.text("'[]'::jsonb"), nullable=False))
    # The queue reads "everything not yet ruled on by a person", so the column is filtered on far more often
    # than it is written.
    op.create_index("ix_object_lifecycle", "object", ["lifecycle"])


def downgrade() -> None:
    op.drop_index("ix_object_lifecycle", table_name="object")
    op.drop_column("object", "lifecycle_history")
    op.drop_column("object", "lifecycle")
