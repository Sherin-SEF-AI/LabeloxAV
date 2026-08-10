"""Metered usage: what was delivered, to whom, and whether it carried a quality claim.

346 dataset commits have left this system and none of them was recorded as a billable event. There is no
usage table, no invoice, no meter. If datasets are the product then every delivery is revenue that currently
exists only as a row in dataset_commit, which was built to record lineage rather than what anybody owes.

Unique on (kind, subject_id) so a delivery is metered once. That is not defensive coding, it follows from
how exports work: a commit id is content-addressed, so exporting the same slice twice legitimately returns
the same commit, and a meter that charged per call would bill twice for one artifact. Re-running an export
must be free.

`certified` and `certificate_signature` sit on the usage row rather than being looked up later, so an
invoice can distinguish what was sold with a measured quality claim from what was sold without one. That
distinction is the reason to put them here: most of this corpus cannot be certified today, and an invoice
that hides which lines those are is the misleading version of this feature.

Prices are stamped onto the record at the time it is written. Recomputing an old invoice against today's
price list would silently rewrite history, and a customer who was quoted one rate should not see another
because the config changed.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0083_usage_record"
down_revision = "0082_machine_verdict"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "usage_record",
        sa.Column("record_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("kind", sa.String(32), nullable=False),          # export | inference | judge
        sa.Column("account", sa.String(128), nullable=False, server_default="default"),
        sa.Column("user_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("app_user.user_id", ondelete="SET NULL"), nullable=True),
        # The thing delivered: a commit id for an export, a run id for inference. String rather than a
        # foreign key because the kinds point at different tables and a nullable column per kind would rot.
        sa.Column("subject_id", sa.String(128), nullable=False),
        sa.Column("quantity", sa.Float(), nullable=False, server_default="0"),
        sa.Column("unit", sa.String(32), nullable=False),
        sa.Column("unit_price_inr", sa.Float(), nullable=True),
        sa.Column("amount_inr", sa.Float(), nullable=True),
        sa.Column("certified", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("certificate_signature", sa.String(128), nullable=True),
        sa.Column("detail", postgresql.JSONB(astext_type=sa.Text()),
                  server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("ts_ns", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("kind", "subject_id", name="uq_usage_record_kind_subject"),
    )
    op.create_index("ix_usage_record_account", "usage_record", ["account"])
    op.create_index("ix_usage_record_kind", "usage_record", ["kind"])


def downgrade() -> None:
    op.drop_index("ix_usage_record_kind", table_name="usage_record")
    op.drop_index("ix_usage_record_account", table_name="usage_record")
    op.drop_table("usage_record")
