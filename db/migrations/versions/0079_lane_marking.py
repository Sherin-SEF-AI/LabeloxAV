"""A lane's type becomes an observation instead of a default.

`lane.lane_type` has been the string "solid" written as a literal in every path that creates a lane. Nothing
has ever looked at the image. On the corpus that produced 4,548 solid, 9 dashed and 1 implicit, and the nine
were drawn by hand.

That is not a weak classifier, it is the absence of one, and it quietly disabled a distinction the event
layer depends on: crossing a solid line is a traffic violation and crossing a dashed one is an ordinary
manoeuvre, so with every lane typed solid every crossing derived as an offence and the severity axis carried
no information at all.

marking_conf is the reason this migration exists rather than the classifier just writing a better string. A
type inferred from paint that is worn, wet or half in shadow is a weaker claim than one read off a crisp line
in daylight, and without somewhere to put that difference the review queue cannot sort by it and the event
deriver cannot decline to call a low-confidence crossing a violation. provenance carries the evidence behind
the call, so a disputed lane type can be argued with rather than just overruled.

Revision ID: 0079_lane_marking
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0079_lane_marking"
down_revision = "0078_event_anchors"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("lane", sa.Column("marking_conf", sa.Float(), nullable=True))
    op.add_column("lane", sa.Column("provenance", postgresql.JSONB(astext_type=sa.Text()),
                                    nullable=True, server_default=sa.text("'{}'::jsonb")))
    # "which lanes in this session has nothing measured yet" is the query the backfill runs, and it is a
    # scan of the session's lanes without it.
    op.create_index("ix_lane_session_conf", "lane", ["session_id", "marking_conf"])


def downgrade() -> None:
    op.drop_index("ix_lane_session_conf", table_name="lane")
    op.drop_column("lane", "provenance")
    op.drop_column("lane", "marking_conf")
