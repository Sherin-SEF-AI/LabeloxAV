"""Multi-modal spine: assets and annotations.

LabeloxAV keeps two annotation spines on purpose:

    AV spine       Session -> Frame -> Object (+ TimelineEvent)   the driving corpus, untouched here
    project spine  LabelProject -> Asset -> Annotation            audio, text, time series, documents, images

Forcing text spans and audio regions into `object` (welded to a bbox, an ontology class id and the confidence
gate) would have degraded both. An asset can reference an existing frame or session, so an AV project reuses
the corpus through the same job machinery without copying a row.

Additive only.

Revision ID: 0065_asset_annotation
Revises: 0064_labelops
Create Date: 2026-07-24
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0065_asset_annotation"
down_revision: str | None = "0064_labelops"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_UUID = postgresql.UUID(as_uuid=True)
_JSONB = postgresql.JSONB(astext_type=sa.Text())


def upgrade() -> None:
    op.create_table(
        "asset",
        sa.Column("asset_id", _UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("project_id", _UUID, sa.ForeignKey("label_project.project_id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("media_type", sa.String(length=16), nullable=False),
        sa.Column("uri", sa.Text(), nullable=True),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("external_id", sa.String(length=200), nullable=True),
        sa.Column("frame_id", _UUID, sa.ForeignKey("frame.frame_id", ondelete="CASCADE"), nullable=True),
        sa.Column("session_id", _UUID, sa.ForeignKey("session.session_id", ondelete="CASCADE"), nullable=True),
        sa.Column("meta", _JSONB, server_default="{}"),
        sa.Column("state", sa.String(length=16), nullable=False, server_default="new"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_asset_project", "asset", ["project_id", "state"])
    # Unique per project only when the caller supplied an external id, so re-importing the same source is
    # idempotent while assets without one are still allowed.
    op.create_index("ix_asset_external", "asset", ["project_id", "external_id"], unique=True,
                    postgresql_where=sa.text("external_id IS NOT NULL"))

    op.create_table(
        "annotation",
        sa.Column("annotation_id", _UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("asset_id", _UUID, sa.ForeignKey("asset.asset_id", ondelete="CASCADE"), nullable=False),
        sa.Column("kind", sa.String(length=20), nullable=False),
        sa.Column("label", sa.String(length=120), nullable=True),
        sa.Column("payload", _JSONB, server_default="{}"),
        sa.Column("fields", _JSONB, server_default="{}"),
        sa.Column("conf", sa.Float(), nullable=True),
        sa.Column("source", sa.String(length=16), nullable=False, server_default="human"),
        sa.Column("state", sa.String(length=16), nullable=False, server_default="accepted"),
        sa.Column("provenance", _JSONB, server_default="{}"),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_by", _UUID, sa.ForeignKey("app_user.user_id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_annotation_asset", "annotation", ["asset_id", "kind"])
    op.create_index("ix_annotation_label", "annotation", ["label"])


def downgrade() -> None:
    op.drop_table("annotation")
    op.drop_table("asset")
