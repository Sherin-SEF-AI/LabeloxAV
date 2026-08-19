"""class_id plus conf is a hard argmax, and the argmax is not the useful part.

A prediction stores the winning class and its confidence. Two detections at conf 0.55 can mean entirely
different things: one is a scooter the model is simply unsure about, the other is a model torn evenly
between scooter and motorcycle. The first wants more examples of scooters in bad light; the second wants
a human to draw the boundary between two classes. Nothing downstream could distinguish them, because the
distribution that separates them was discarded at write time.

`prediction.class_probs` keeps the top-5 as {class_id: prob}. Top-5 rather than all 192, because over that
ontology the tail is numerically zero and storing it would multiply the largest table in the schema for no
signal. Nullable, and null on all 96,068 existing predictions: they were written before this existed and
the distribution is not recoverable from the argmax.

`clique_bandit` holds a Beta posterior per confusion clique. Splitting a labelling budget across the ways
a model is confused is not knowable in advance - it depends on which confusions labelling actually fixes -
so Thompson sampling makes it a measurement instead of a constant, and explores without an epsilon anybody
has to tune. Every clique starts at the uniform prior, which is honest: there is no labelling history yet.

No backfill. A class distribution cannot be reconstructed from the class that won.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0098_clique_al"
down_revision = "0097_relationship_conf"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("prediction", sa.Column("class_probs", postgresql.JSONB(), nullable=True))

    op.create_table(
        "clique_bandit",
        sa.Column("clique", sa.String(64), primary_key=True),
        sa.Column("pack_id", sa.String(32), primary_key=True),
        sa.Column("alpha", sa.Float(), nullable=False, server_default="1"),
        sa.Column("beta", sa.Float(), nullable=False, server_default="1"),
        sa.Column("n_pulls", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("n_rewards", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_allocated", sa.Integer()),
        sa.Column("last_reward", sa.Float()),
        sa.Column("last_recall_before", sa.Float()),
        sa.Column("last_recall_after", sa.Float()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("clique_bandit")
    op.drop_column("prediction", "class_probs")
