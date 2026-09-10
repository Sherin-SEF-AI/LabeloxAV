"""An epsilon budget per scope, and a log of every release spent from it.

Differential privacy composes additively: ten releases at epsilon 0.1 leak as much as one at epsilon 1.0.
A system that applies a per-query epsilon and never tracks the total is providing a guarantee it has
already spent, and there is no way to notice from any single query. The budget is the guarantee; the
per-query epsilon is only how it is spent.

`privacy_budget` holds one allowance per scope and window. `privacy_release_log` records what each release
cost and what mechanism produced it, so the total can be checked against the sum rather than trusted.

The log is the half that survives a mistake. If a scope's budget is reset by hand, the releases that were
already made are still in the log and the reset is visible next to them.

Revision ID: 0116_privacy_budget
Revises: 0115_label_economics
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0116_privacy_budget"
down_revision = "0115_label_economics"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "privacy_budget",
        sa.Column("scope", sa.String(64), primary_key=True),
        sa.Column("epsilon_total", sa.Float(), nullable=False),
        sa.Column("epsilon_spent", sa.Float(), nullable=False, server_default="0"),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("epsilon_total > 0", name="ck_privacy_budget_total_positive"),
        sa.CheckConstraint("epsilon_spent >= 0", name="ck_privacy_budget_spent_nonneg"),
    )

    op.create_table(
        "privacy_release_log",
        sa.Column("release_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("scope", sa.String(64), nullable=False),
        sa.Column("endpoint", sa.String(128), nullable=False),
        sa.Column("epsilon", sa.Float(), nullable=False),
        sa.Column("mechanism", sa.String(64), nullable=False),
        sa.Column("k", sa.Integer(), nullable=True),
        sa.Column("cells_released", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cells_suppressed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("requested_by", sa.String(64), nullable=True),
        sa.Column("at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("epsilon > 0", name="ck_privacy_release_epsilon_positive"),
    )
    op.create_index("ix_privacy_release_scope_at", "privacy_release_log", ["scope", "at"])

    # What privacy treatment an export shipped under, on the export job itself. A bundle whose datasheet
    # cannot say whether its geography was aggregated is a bundle nobody can assess. On `export_job`
    # rather than the `ExportRecord` the plan named, because that is a dataclass describing one row of a
    # bundle and not a table; the job is what durably records an export having happened.
    op.add_column("export_job", sa.Column("privacy", postgresql.JSONB(), nullable=False,
                                          server_default=sa.text("'{}'::jsonb")))


def downgrade() -> None:
    op.drop_column("export_job", "privacy")
    op.drop_index("ix_privacy_release_scope_at", table_name="privacy_release_log")
    op.drop_table("privacy_release_log")
    op.drop_table("privacy_budget")
