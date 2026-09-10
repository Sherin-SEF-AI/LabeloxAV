"""Where pixels came from: `origin` on session and frame, and a synthetic state and source for objects.

The copy-paste generator (services/synth/copy_paste.py) builds composite frames from real donors and
real backgrounds to feed starved classes. Nothing that measures a model may see a synthetic pixel:
not the gold set, not a settlement lot, not the blind audit, not an embedding, not a datasheet.
Quarantine is structural, not a predicate someone remembers to add. A composite session and its
frames carry `origin = 'synthetic'`; every label on a composite carries `state = 'synthetic'` and
`source = 'synthetic'`, so each allow-list reader (source == 'human', a state list, a cycle id)
excludes them without change, and the deny-list readers that sweep frames by time say
`Frame.origin == 'real'` (core/origin.py). `frame.source_frame_id` names the real frame a composite
or perturbed frame was built from, SET NULL on delete so erasure of the source leaves the composite
and its own erasure path intact.

The downgrade removes what the columns described rather than leaving it unlabelled: objects in the
synthetic state, then every non-real frame (its objects cascade), then every non-real session, and
only then restores the tighter CHECKs and drops the columns. That is the conservative direction: a
composite with no origin column would be indistinguishable from a real frame, which is the exact
defect the column exists to prevent.

Revision ID: 0108_origin
Revises: 0107_settlement_sprt
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0108_origin"
down_revision = "0107_settlement_sprt"
branch_labels = None
depends_on = None

_ORIGINS = "origin IN ('real', 'synthetic', 'perturbed')"
_STATE_WITH = ("state IN ('review', 'auto_accept', 'accepted', 'rejected', 'annotate', 'submitted', "
               "'settled', 'synthetic')")
_STATE_WITHOUT = ("state IN ('review', 'auto_accept', 'accepted', 'rejected', 'annotate', 'submitted', "
                  "'settled')")
_SOURCE_WITH = ("source IN ('fused', 'auto_accept', 'human', 'imported', 'relabel', 'interpolated', "
                "'propagated', 'recall', 'vlm_review', 'synthetic')")
_SOURCE_WITHOUT = ("source IN ('fused', 'auto_accept', 'human', 'imported', 'relabel', 'interpolated', "
                   "'propagated', 'recall', 'vlm_review')")


def upgrade() -> None:
    op.add_column("session", sa.Column("origin", sa.String(16), nullable=False, server_default="real"))
    op.create_check_constraint("ck_session_origin", "session", _ORIGINS)

    op.add_column("frame", sa.Column("origin", sa.String(16), nullable=False, server_default="real"))
    op.add_column("frame", sa.Column("source_frame_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key("fk_frame_source_frame", "frame", "frame", ["source_frame_id"], ["frame_id"],
                          ondelete="SET NULL")
    op.create_check_constraint("ck_frame_origin", "frame", _ORIGINS)
    op.create_index("ix_frame_origin", "frame", ["origin"], postgresql_where=sa.text("origin <> 'real'"))
    op.create_index("ix_frame_source_frame", "frame", ["source_frame_id"],
                    postgresql_where=sa.text("source_frame_id IS NOT NULL"))

    op.drop_constraint("ck_object_state", "object", type_="check")
    op.create_check_constraint("ck_object_state", "object", _STATE_WITH)
    op.drop_constraint("ck_object_source", "object", type_="check")
    op.create_check_constraint("ck_object_source", "object", _SOURCE_WITH)


def downgrade() -> None:
    # Data first, in the direction that leaves nothing unlabelled: a composite with no origin column
    # would pass for a real frame.
    op.execute(sa.text("DELETE FROM object WHERE state = 'synthetic' OR source = 'synthetic'"))
    op.execute(sa.text("DELETE FROM frame WHERE origin <> 'real'"))
    op.execute(sa.text("DELETE FROM session WHERE origin <> 'real'"))

    op.drop_constraint("ck_object_source", "object", type_="check")
    op.create_check_constraint("ck_object_source", "object", _SOURCE_WITHOUT)
    op.drop_constraint("ck_object_state", "object", type_="check")
    op.create_check_constraint("ck_object_state", "object", _STATE_WITHOUT)

    op.drop_index("ix_frame_source_frame", table_name="frame")
    op.drop_index("ix_frame_origin", table_name="frame")
    op.drop_constraint("ck_frame_origin", "frame", type_="check")
    op.drop_constraint("fk_frame_source_frame", "frame", type_="foreignkey")
    op.drop_column("frame", "source_frame_id")
    op.drop_column("frame", "origin")

    op.drop_constraint("ck_session_origin", "session", type_="check")
    op.drop_column("session", "origin")
