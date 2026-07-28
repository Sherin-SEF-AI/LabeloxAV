"""Text-searchable object crops, and an index behind OCR search.

Two gaps in one migration because both are about finding an object by what it looks like or says.

object_embedding.siglip_vec: the crop table carried only a DINOv3 vector, and DINOv3 has no text tower, so
"cattle at night" could retrieve frames but never crops. SigLIP2 is a joint image-text space, so storing the
crop's SigLIP vector alongside makes text-to-object retrieval a single ANN query. Nullable, because it is
backfilled asynchronously by the embedding daemon and an object without one must still be searchable by
image similarity.

object.ocr_text index: OCR search ran as ILIKE '%needle%', which cannot use a b-tree and is a sequential scan
of the whole object table on every query. pg_trgm's GIN index is what makes an unanchored substring match
indexable.

Revision ID: 0073_object_siglip_and_ocr_trgm
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision = "0073_object_siglip_and_ocr_trgm"
down_revision = "0072_user_token_version"
branch_labels = None
depends_on = None

# Matches the frame-level SigLIP2 so400m width already stored on frame_embedding, so the two spaces are
# directly comparable and one text encoder serves both.
_SIGLIP_DIM = 1152


def upgrade() -> None:
    op.add_column("object_embedding", sa.Column("siglip_vec", Vector(_SIGLIP_DIM), nullable=True))
    # Same index type and opclass as the frame-level vectors, so recall behaviour is consistent across the
    # two planes rather than one being exact and the other approximate.
    op.create_index(
        "ix_object_embedding_siglip",
        "object_embedding",
        ["siglip_vec"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"siglip_vec": "vector_cosine_ops"},
    )

    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.create_index(
        "ix_object_ocr_text_trgm",
        "object",
        ["ocr_text"],
        postgresql_using="gin",
        postgresql_ops={"ocr_text": "gin_trgm_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_object_ocr_text_trgm", table_name="object")
    # The extension is left in place: other objects may depend on it, and dropping a shared extension is a
    # wider blast radius than this migration owns.
    op.drop_index("ix_object_embedding_siglip", table_name="object_embedding")
    op.drop_column("object_embedding", "siglip_vec")
