"""Which classes a project labels: a subset of the shared ontology, per project.

The ontology is one list of classes with global ids, because the detector, the exports, the metrics and every
stored object agree on those ids; splitting it per project would break all four. What differs between teams is
which of those classes they care about: a project on Indian roads wants all of them, a project on European or
American data wants the road users that exist there and none of the autorickshaw and cycle-rickshaw long tail.

`label_project.class_ids` is that subset, as a JSON list of class ids, or NULL for the whole ontology (every
project today, and the default). It narrows what the editor offers and what auto-labelling writes; objects
already stored keep their class.

Revision ID: 0119_project_classes
Revises: 0118_project_access
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0119_project_classes"
down_revision = "0118_project_access"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("label_project", sa.Column("class_ids", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("label_project", "class_ids")
