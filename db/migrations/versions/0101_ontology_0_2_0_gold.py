"""carry sealed gold sets forward to ontology labelox-in-0.2.0

The ontology gains classes 167-168 and eighteen attributes. Ids 1-166 are untouched, every addition is a new
leaf, and no existing label changes meaning, so a gold set sealed under 0.1.0 is still valid evidence.

Without this the bump silently disables the promotion gate. `services/govern/champion.py:209` asks
`latest_gold_id(db, onto.version)` for the yardstick both models are scored against, and `gold_eval.py:33`
filters on exact equality, so all seven sealed sets stop matching and the gate falls through to each model's
own val split. Nothing errors. Two models get compared on two different splits and the promotion decision
stops meaning anything.

Only rows still on 0.1.0 are carried, and each carries a marker so the downgrade can put back exactly those
and leave anything genuinely sealed under 0.2.0 alone.

Revision ID: 0101_ontology_0_2_0_gold
Revises: 0100_frame_context
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0101_ontology_0_2_0_gold"
down_revision = "0100_frame_context"
branch_labels = None
depends_on = None

_FROM = "labelox-in-0.1.0"
_TO = "labelox-in-0.2.0"
_MARK = "carried_from_ontology"


def upgrade() -> None:
    op.execute(sa.text("""
        UPDATE gold_set
           SET ontology_version = :to,
               spec = COALESCE(spec, '{}'::jsonb) || jsonb_build_object(:mark, :from)
         WHERE ontology_version = :from
    """).bindparams(to=_TO, **{"from": _FROM}, mark=_MARK))


def downgrade() -> None:
    op.execute(sa.text("""
        UPDATE gold_set
           SET ontology_version = spec ->> :mark,
               spec = spec - :mark
         WHERE spec ? :mark
    """).bindparams(mark=_MARK))
