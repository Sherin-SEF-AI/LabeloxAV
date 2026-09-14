"""When each control verdict landed, so the measurement can say how old it is.

`ControlSample.human_verdict` is the realized precision of the auto-accept gate - the number the 0.97
drift floor compares against - and it recorded WHAT a person ruled with no record of WHEN. A
measurement without a timestamp cannot be refused for staleness, and the autonomy work (plan phase 1)
makes staleness a first-class refusal everywhere: a consumer acting on a six-month-old precision
number silently is how a control chart goes decorative.

Nullable, no backfill: the existing verdicts' times were never recorded and inventing them from
`created_at` (the seeding time, not the ruling time) would be a confident fabrication. Null means
"ruled before 2026-09-05, exact time unknown", which is the truth.

Revision ID: 0105_control_verdict_at
Revises: 0104_promote_by_approval
"""

import sqlalchemy as sa
from alembic import op

revision = "0105_control_verdict_at"
down_revision = "0104_promote_by_approval"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("control_sample",
                  sa.Column("verdict_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("control_sample", "verdict_at")
