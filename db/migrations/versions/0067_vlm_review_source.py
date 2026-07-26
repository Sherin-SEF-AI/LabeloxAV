"""Add 'vlm_review' to the object.source check constraint.

VLM-as-judge review promotes a rare-class detection the VLM confidently confirms to an accepted training
label. It is a distinct provenance from a human review or an auto-accept, and recording it honestly (rather
than laundering it as 'human') is the whole point, so the source column must allow it.

Revision ID: 0067_vlm_review_source
"""

from __future__ import annotations

from alembic import op

revision = "0067_vlm_review_source"
down_revision = "0066_webhooks_storage"
branch_labels = None
depends_on = None

_WITH = ("source IN ('fused', 'auto_accept', 'human', 'imported', 'relabel', 'interpolated', "
         "'propagated', 'recall', 'vlm_review')")
_WITHOUT = ("source IN ('fused', 'auto_accept', 'human', 'imported', 'relabel', 'interpolated', "
            "'propagated', 'recall')")


def upgrade() -> None:
    op.drop_constraint("ck_object_source", "object", type_="check")
    op.create_check_constraint("ck_object_source", "object", _WITH)


def downgrade() -> None:
    op.drop_constraint("ck_object_source", "object", type_="check")
    op.create_check_constraint("ck_object_source", "object", _WITHOUT)
