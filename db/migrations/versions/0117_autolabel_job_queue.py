"""Auto-label jobs a separate GPU worker can pick up: their parameters on the row, and a crash count.

Until now an auto-label job lived only as an asyncio task inside the API. The row held status and
progress, while the parameters it was started with (the frame limit, whether to skip frames that already
have labels) were arguments to that task and nowhere else. A restart lost the task, and with it any way to
run the job again: the reaper could only mark it failed.

`params` puts the request on the row, so whichever process claims the job reads what to do from the
database, and a job orphaned by a crash can be put back in the queue rather than failed. `attempts` counts
those returns, so a job that crashes its worker every time is stopped after a few tries instead of taking
the GPU worker down in a loop.

Revision ID: 0117_autolabel_job_queue
Revises: 0116_privacy_budget
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0117_autolabel_job_queue"
down_revision = "0116_privacy_budget"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("autolabel_job", sa.Column("params", postgresql.JSONB(), nullable=False,
                                             server_default=sa.text("'{}'::jsonb")))
    op.add_column("autolabel_job", sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"))


def downgrade() -> None:
    op.drop_column("autolabel_job", "attempts")
    op.drop_column("autolabel_job", "params")
