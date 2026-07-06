"""ORACLYX M14: 4D uncertainty-aware pseudo-GT. Calibrated uncertainty and expected-info-gain on pseudo labels.
Additive only.

Revision ID: 0054_oraclyx_m14
Revises: 0053_labelox_m13
Create Date: 2026-07-06
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0054_oraclyx_m14"
down_revision: str | None = "0053_labelox_m13"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("pseudo_label", sa.Column("uncertainty", sa.Float(), nullable=True))
    op.add_column("pseudo_label", sa.Column("info_gain", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("pseudo_label", "info_gain")
    op.drop_column("pseudo_label", "uncertainty")
