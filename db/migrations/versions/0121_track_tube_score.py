"""A track's temporal coherence, stored on the track.

core/accel/tube_score.py scores how much a track behaves like one real object over time: it is present on
the frames it spans, its box is steady relative to its size, and it keeps one class. Measured on the
champion's gold run (services/oraclyx/joint_calibration.py::tube_by_prediction), adding it to confidence
lifts how well right and wrong detections are ranked, AUC 0.787 -> 0.874 (frame bootstrap 95% CI of the
gain 0.027-0.155). Ranking is what the review queue does, so the score is kept where a query can reach it.

NULL means not measured: a track shorter than the tube scorer's minimum, or one tracked before this column.

Revision ID: 0121_track_tube_score
Revises: 0120_assistant_proposal
"""

import sqlalchemy as sa
from alembic import op

revision = "0121_track_tube_score"
down_revision = "0120_assistant_proposal"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("track", sa.Column("tube_score", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("track", "tube_score")
