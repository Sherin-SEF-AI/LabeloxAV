"""What the language and vision models were asked, by whom, and what it cost.

Two cloud providers are wired in and nothing counted a token. The only budget was a call counter two
agents kept in memory. This is the ledger: one row per day, provider, model and purpose, added to as
calls happen, so "what did the judge cost last week" is a query instead of an invoice surprise.

Daily totals rather than one row per call. Auto-label makes tens of thousands of VLM calls in a session;
a row each would be a table that outgrows the objects it describes, to answer questions that are all
sums.

Revision ID: 0125_llm_usage
Revises: 0124_import_job_spec
"""

import sqlalchemy as sa
from alembic import op

revision = "0125_llm_usage"
down_revision = "0124_import_job_spec"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "llm_usage_daily",
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("provider", sa.String(32), nullable=False),
        sa.Column("model", sa.String(128), nullable=False),
        sa.Column("purpose", sa.String(48), nullable=False),
        sa.Column("calls", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("failures", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("input_tokens", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("images", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("latency_ms", sa.BigInteger(), nullable=False, server_default="0"),
        # Calls whose reply carried no token counts, so a reader can tell "no tokens" from "not reported".
        sa.Column("unmetered_calls", sa.BigInteger(), nullable=False, server_default="0"),
        # NULL when no price is configured for the model. Zero is a real answer (a local model); an
        # unpriced cloud model must not read as free.
        sa.Column("cost_usd", sa.Float(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("day", "provider", "model", "purpose", name="pk_llm_usage_daily"),
    )


def downgrade() -> None:
    op.drop_table("llm_usage_daily")
