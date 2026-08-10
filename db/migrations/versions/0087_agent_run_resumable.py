"""A running job that died has to be distinguishable from one that is still going.

Background work is launched with `asyncio.create_task` inside the API process. When that process goes away
(a restart, a crash, an OOM) the task goes with it and nothing writes the row again, so the run stays
`running` forever and looks exactly like live work. The live corpus is carrying the evidence: an
`overnight_auditor` marked running 863 hours ago, a `flywheel` planned 911 hours ago, and two runs
(`redetect_all`, `relabel_all`) stranded by an API restart an hour before this migration was written.

Nothing could tell them apart, so nothing could offer to resume them, and a stuck run also blocks the guards
that refuse to start a second job while one holds the GPU.

Two columns fix that.

`heartbeat_at` is written as the job makes progress. A `running` row whose heartbeat has gone stale is a
dead job, and that is a decision a startup sweep can make mechanically instead of a person guessing from a
timestamp. Nullable because every existing row predates the mechanism, and a null heartbeat on an old
`running` row means exactly what the sweep should conclude anyway.

`progress` is the resume cursor, shaped by whatever the job wants to record: the sessions already swept, the
frames already relabelled. It is deliberately opaque to the schema, because the only thing every job shares
is the need to say "this much is done", and forcing a common shape would push each job to lie about its
unit of work.

Additive and reversible. No status value changes here; `interrupted` is written by the sweep at runtime and
the column has no CHECK constraint to widen.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0087_agent_run_resumable"
down_revision = "0086_machine_verdict_per_batch"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("agent_run", sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("agent_run", sa.Column(
        "progress", postgresql.JSONB(astext_type=sa.Text()), server_default="{}", nullable=False))
    # The startup sweep asks one question: which rows are running and stale. Without this it is a sequential
    # scan of every run ever recorded, on a path that delays the API answering its first request.
    op.create_index("ix_agent_run_status_heartbeat", "agent_run", ["status", "heartbeat_at"])


def downgrade() -> None:
    op.drop_index("ix_agent_run_status_heartbeat", table_name="agent_run")
    op.drop_column("agent_run", "progress")
    op.drop_column("agent_run", "heartbeat_at")
