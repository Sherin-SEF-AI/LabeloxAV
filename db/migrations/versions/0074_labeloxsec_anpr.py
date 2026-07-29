"""LabeloxSec persistence: the plate watchlist and the plate reads.

The security pack had a tested plate-format kernel, a recogniser, and watchlist matching, and nowhere to put
a result: every read was ephemeral and the watchlist was a list the caller had to supply on each call. That
is why nothing could be built on top of it.

Plate reads are personal data, which is the whole reason the AV pack blurs plates and refuses ANPR. The
session and frame FKs cascade so an erasure request takes the plate text with the rest of the session,
instead of leaving the most sensitive field in the corpus behind.

Revision ID: 0074_labeloxsec_anpr
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0074_labeloxsec_anpr"
down_revision = "0073_object_siglip_and_ocr_trgm"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "plate_watchlist",
        sa.Column("entry_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        # Unique on the normalised mark: the same plate written three ways is one entry, and a duplicate
        # would mean a hit reported two or three times.
        sa.Column("plate_normalized", sa.String(16), nullable=False, unique=True),
        sa.Column("plate_raw", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text()),
        sa.Column("severity", sa.String(16), nullable=False, server_default="warn"),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("added_by", sa.String(64)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_plate_watchlist_normalized", "plate_watchlist", ["plate_normalized"])

    op.create_table(
        "plate_read",
        sa.Column("read_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("session_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("session.session_id", ondelete="CASCADE")),
        sa.Column("frame_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("frame.frame_id", ondelete="CASCADE")),
        sa.Column("camera_id", sa.String(64)),
        sa.Column("plate_normalized", sa.String(16), nullable=False),
        sa.Column("plate_raw", sa.Text(), nullable=False),
        sa.Column("plate_type", sa.String(16), nullable=False),
        sa.Column("state_code", sa.String(4)),
        sa.Column("rto_district", sa.String(8)),
        sa.Column("valid", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("det_conf", sa.Float(), nullable=False, server_default="0"),
        # Nullable: a generative-VLM reader exposes no calibrated score, and inventing one would make a
        # confidence filter look meaningful when it is not.
        sa.Column("ocr_conf", sa.Float()),
        sa.Column("format_confidence", sa.Float(), nullable=False, server_default="0"),
        sa.Column("bbox", postgresql.ARRAY(sa.Float())),
        sa.Column("watchlist_hit", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("watchlist_severity", sa.String(16)),
        sa.Column("pack_id", sa.String(32), nullable=False, server_default="sec"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_plate_read_session", "plate_read", ["session_id"])
    op.create_index("ix_plate_read_frame", "plate_read", ["frame_id"])
    op.create_index("ix_plate_read_camera", "plate_read", ["camera_id"])
    op.create_index("ix_plate_read_normalized", "plate_read", ["plate_normalized"])
    # The two queries the console actually runs: the recent feed, and hits only.
    op.create_index("ix_plate_read_created", "plate_read", ["created_at"])
    op.create_index("ix_plate_read_hit", "plate_read", ["watchlist_hit"])


def downgrade() -> None:
    op.drop_table("plate_read")
    op.drop_table("plate_watchlist")
