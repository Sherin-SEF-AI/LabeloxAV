"""Promotion defaults to propose-and-approve instead of autonomous.

`GovernanceState.auto_promote_enabled` was born True, which made champion promotion the one fully
autonomous act in the loop: a challenger that passed the gate replaced the champion with no person in
the chain. The operator's decision (2026-09-01) is the opposite default: the loop may retrain, gate,
and file a ready-to-promote notification, and a person clicks promote. The flag survives as the
documented opt-in to full closure; a human's explicit approval through POST /govern/promote bypasses
the flag (a person clicking IS the approval) and never the gate.

This is a data migration: the ORM default only shapes rows born after it, and the singleton control
row already exists on both databases with the old True. Flipping only the model would leave the
running deployment silently autonomous, which is the 0103 lesson wearing governance clothes.

Revision ID: 0104_promote_by_approval
Revises: 0103_amodal_bbox
"""

import sqlalchemy as sa
from alembic import op

revision = "0104_promote_by_approval"
down_revision = "0103_amodal_bbox"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(sa.text("UPDATE governance_state SET auto_promote_enabled = false WHERE id = 1"))


def downgrade() -> None:
    # Restores the pre-decision behaviour: the loop promotes unattended when the gate passes.
    op.execute(sa.text("UPDATE governance_state SET auto_promote_enabled = true WHERE id = 1"))
