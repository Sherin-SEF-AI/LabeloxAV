"""Provenance for the frame-level context a person can set.

The obvious move was a `frame_context` table. There is no need: `Frame.scene` already carries the
frame-level facts, written at ingest by the scene classifier with a confidence per axis, and a second table
would be exactly the parallel attribute mechanism this repo forbids for objects.

What `scene` lacked is provenance. A value a person set and a value a classifier guessed are the same JSON,
so a human correction cannot survive the next classifier pass and nothing can tell the two apart afterwards.
`scene_provenance` mirrors the key set with who set each one.

It is NOT missing an index: 0062_cascade_and_scene_gin already created ix_frame_scene_gin over the whole
document. This migration first tried to create it again and failed on the duplicate, which is how that was
found.

Revision ID: 0100_frame_context
Revises: 0099_propagation_conflict
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0100_frame_context"
down_revision = "0099_propagation_conflict"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("frame", sa.Column("scene_provenance", postgresql.JSONB(astext_type=sa.Text()),
                                     nullable=True))


def downgrade() -> None:
    op.drop_column("frame", "scene_provenance")
