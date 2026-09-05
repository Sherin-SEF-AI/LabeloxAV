"""Sequential acceptance on settlement lots: the rule, the cap, the log-likelihood ratio, the increments.

A settlement lot was one fixed draw (110 to 562 crops, sized to survive one defect) judged to the end
before any decision. Wald's sequential probability ratio test asks until the evidence is decisive: a
class whose sample runs clean stops early, a class that throws four or five defects fails in a handful
of verdicts instead of a hundred. This migration records what the sequential rule needs on the lot.

`rule` names the decision rule the lot was planned under: 'wilson' for every lot that exists today
(the fixed draw, decided by `acceptance_decision` alone) and 'sprt' for lots planned from now on.
`cap_n` is the fixed sample a sequential lot may grow to, the same number the fixed draw used, so a
sequential lot that reaches its cap has drawn exactly the sample the old rule would have; 0 means
"planned before the rule existed" and the engine treats it as the fixed rule. `llr` is the ratio at
the last tally; null means never computed, which is not the same as zero. `sprt` carries the test's
parameters and the trajectory of (n, defects, llr) at every tally, so a decision can be replayed;
`increments` records each draw in order, because the tally counts only completed increments (a
half-judged increment is the hardest-first prefix of itself, not a random sample).

The downgrade drops the five columns and nothing else: a judging sequential lot then tallies as a
fixed draw under the Wilson rule, which is the stricter one, so no row needs moving.

Revision ID: 0107_settlement_sprt
Revises: 0106_settlement
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0107_settlement_sprt"
down_revision = "0106_settlement"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("settlement_lot", sa.Column("rule", sa.String(16), nullable=False,
                                              server_default="wilson"))
    op.add_column("settlement_lot", sa.Column("cap_n", sa.Integer(), nullable=False,
                                              server_default="0"))
    op.add_column("settlement_lot", sa.Column("llr", sa.Float(), nullable=True))
    op.add_column("settlement_lot", sa.Column("sprt", postgresql.JSONB(), nullable=False,
                                              server_default=sa.text("'{}'::jsonb")))
    op.add_column("settlement_lot", sa.Column("increments", postgresql.JSONB(), nullable=False,
                                              server_default=sa.text("'[]'::jsonb")))


def downgrade() -> None:
    op.drop_column("settlement_lot", "increments")
    op.drop_column("settlement_lot", "sprt")
    op.drop_column("settlement_lot", "llr")
    op.drop_column("settlement_lot", "cap_n")
    op.drop_column("settlement_lot", "rule")
