"""The overnight autopilot's settings and its nights (agent F: language edits and autopilot).

`autopilot_config` is a single row (id 1) holding what a person set on /autopilot: whether it runs, the
nightly GPU-minutes budget, which steps it chains, and the kill switch. It is created on first read rather
than seeded here, so a database whose tables were emptied (the test reset) still answers with the defaults.

`autopilot_night` is one row per night the autopilot ran or was asked to plan. A dry run is a row too, with
`dry_run` set and nothing changed, so the history page shows what it would have done next to what it did.
At most one real night per calendar night: the partial unique index is the guard against a second daemon
tick, or a person pressing "run now", starting the same night twice.

The language bulk edit (#17) needs no table: it is an AssistantProposal applied as an AgentRun.

Revision ID: 0129_tools_agents
Revises: 0128_tools_scene3d
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0129_tools_agents"
down_revision = "0128_tools_scene3d"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "autopilot_config",
        sa.Column("id", sa.SmallInteger(), primary_key=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("killed", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("killed_reason", sa.Text(), nullable=True),
        sa.Column("killed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("gpu_minutes", sa.Integer(), nullable=False, server_default="240"),
        sa.Column("max_sessions", sa.Integer(), nullable=False, server_default="20"),
        sa.Column("vlm_max_checks", sa.Integer(), nullable=False, server_default="300"),
        sa.Column("earliest_start_ist", sa.String(5), nullable=False, server_default="20:00"),
        sa.Column("steps", JSONB, nullable=False, server_default=sa.text(
            """'{"autolabel": true, "vlm_check": true, "error_sweep": true, "auto_accept": true}'::jsonb""")),
        sa.Column("updated_by", UUID(as_uuid=True), sa.ForeignKey("app_user.user_id", ondelete="SET NULL"),
                  nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "autopilot_night",
        sa.Column("night_id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("night_date", sa.Date(), nullable=False),
        sa.Column("dry_run", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("status", sa.String(16), nullable=False, server_default="planned"),
        sa.Column("trigger", sa.String(16), nullable=False, server_default="scheduler"),
        sa.Column("tool_job_id", UUID(as_uuid=True), nullable=True),
        sa.Column("agent_run_id", UUID(as_uuid=True), nullable=True),
        sa.Column("settings", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("plan", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("report", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("gpu_seconds", sa.Float(), nullable=False, server_default="0"),
        sa.Column("stopped_reason", sa.Text(), nullable=True),
        sa.Column("created_by", UUID(as_uuid=True), sa.ForeignKey("app_user.user_id", ondelete="SET NULL"),
                  nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_autopilot_night_created", "autopilot_night", ["created_at"])
    op.create_index("ix_autopilot_night_created_by", "autopilot_night", ["created_by"])
    op.create_index("uq_autopilot_night_real", "autopilot_night", ["night_date"], unique=True,
                    postgresql_where=sa.text("dry_run = false"))


def downgrade() -> None:
    op.drop_index("uq_autopilot_night_real", table_name="autopilot_night")
    op.drop_index("ix_autopilot_night_created_by", table_name="autopilot_night")
    op.drop_index("ix_autopilot_night_created", table_name="autopilot_night")
    op.drop_table("autopilot_night")
    op.drop_table("autopilot_config")
