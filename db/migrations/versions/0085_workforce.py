"""An external labeling workforce, and the record of what was sent to it.

253 of 570,379 objects carry a human verdict. Every quality number downstream is starved by that one figure,
and `assign_job` takes a user id, so the only way work reaches anybody is a person picking a name from a
list. There is no way to send a batch to a team that labels for a living.

Two tables rather than a column on label_job, because a dispatch has a life of its own. A job can be sent,
returned, rejected on quality, and sent again, possibly to a different workforce; a nullable
`workforce_id` on the job would keep only the last of those and lose exactly the history that says which
workforce is worth using.

`secret` is the HMAC key for this workforce's callbacks, following services/integrations/webhooks.py rather
than inventing a second signing scheme. A returned batch is annotation data being written into the corpus by
an outside party, so it has to be attributable to a workforce and unforgeable by anyone else.

`min_honeypot_accuracy` sits on the workforce because the acceptance bar is a commercial term, negotiated
per vendor, not a global constant. The honeypot machinery it gates already exists and is good: deterministic
per job id, scored on return, failed jobs sent back.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0085_workforce"
down_revision = "0084_error_candidate_decision"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "workforce",
        sa.Column("workforce_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.String(128), nullable=False, unique=True),
        # internal teams route through the same machinery as vendors, so quality is compared on one scale
        # rather than "our people" being exempt from measurement.
        sa.Column("kind", sa.String(16), nullable=False, server_default="vendor"),  # vendor | internal
        sa.Column("endpoint", sa.Text(), nullable=True),      # where a dispatch is POSTed; null = pull only
        sa.Column("secret", sa.String(128), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        # what this workforce can label: {"classes": [...], "modalities": [...], "languages": [...]}
        sa.Column("capabilities", postgresql.JSONB(astext_type=sa.Text()),
                  server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("capacity_jobs_per_day", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("min_honeypot_accuracy", sa.Float(), nullable=False, server_default="0.9"),
        sa.Column("contact", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "workforce_assignment",
        sa.Column("assignment_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("job_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("label_job.job_id", ondelete="CASCADE"), nullable=False),
        sa.Column("workforce_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("workforce.workforce_id", ondelete="CASCADE"), nullable=False),
        # dispatched -> returned -> accepted | rejected. `expired` is a dispatch nobody sent back.
        sa.Column("state", sa.String(16), nullable=False, server_default="dispatched"),
        sa.Column("external_ref", sa.String(128), nullable=True),   # the vendor's own id for the batch
        sa.Column("dispatched_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("returned_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("honeypot_accuracy", sa.Float(), nullable=True),
        sa.Column("objects_returned", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("detail", postgresql.JSONB(astext_type=sa.Text()),
                  server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.CheckConstraint(
            "state in ('dispatched','returned','accepted','rejected','expired')",
            name="ck_workforce_assignment_state"),
    )
    # One live dispatch per job. Enforced partially rather than absolutely: a job rejected on quality must be
    # dispatchable again, so only the open states are exclusive.
    op.create_index("uq_workforce_assignment_open", "workforce_assignment", ["job_id"], unique=True,
                    postgresql_where=sa.text("state in ('dispatched','returned')"))
    op.create_index("ix_workforce_assignment_workforce", "workforce_assignment", ["workforce_id", "state"])


def downgrade() -> None:
    op.drop_index("ix_workforce_assignment_workforce", table_name="workforce_assignment")
    op.drop_index("uq_workforce_assignment_open", table_name="workforce_assignment")
    op.drop_table("workforce_assignment")
    op.drop_table("workforce")
