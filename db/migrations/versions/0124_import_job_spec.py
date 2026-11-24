"""Record everything an import was asked to do, so one a restart interrupted can be continued.

`import_job` kept the format, the source and the target vehicle, and dropped the options and the project.
A restarted API therefore could not have continued an import even in principle: it no longer knew what
had been asked for. `spec` holds the rest of the request, how many times the job has been picked back up,
and the session and time base the first run chose, which is what lets a resumed run write into the same
session and skip the frames already there instead of starting a duplicate.

Revision ID: 0124_import_job_spec
Revises: 0123_remaining_fk_indexes
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0124_import_job_spec"
down_revision = "0123_remaining_fk_indexes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("import_job", sa.Column("spec", postgresql.JSONB(), nullable=False,
                                          server_default=sa.text("'{}'::jsonb")))


def downgrade() -> None:
    op.drop_column("import_job", "spec")
