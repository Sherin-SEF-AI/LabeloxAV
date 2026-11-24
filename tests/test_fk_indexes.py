"""Every foreign key a parent delete must search has an index.

`eval_patch.object_id` had none, so deleting one object scanned 36 million rows and an overnight re-label
never finished its first drive (migration 0122). Two more on the same table survived that fix (0123). This
reads the migrated schema and fails on the next one, at the moment it is added rather than the night it is
first deleted from at scale.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

pytestmark = pytest.mark.db

# Foreign keys to ontology_class: classes are retired, never deleted, so the index would cost every insert
# into the referencing table and speed up nothing. Listed by table so a new exemption is a visible decision.
EXEMPT_PARENTS = {"ontology_class"}


@pytest.mark.asyncio
async def test_no_foreign_key_is_unindexed():
    from db.session import get_engine

    eng = get_engine()
    async with eng.connect() as c:
        rows = (await c.execute(text("""
            SELECT c.conrelid::regclass::text, a.attname, c.confrelid::regclass::text
            FROM pg_constraint c
            JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = c.conkey[1]
            WHERE c.contype = 'f' AND array_length(c.conkey, 1) = 1
              AND c.connamespace = 'public'::regnamespace
              AND NOT EXISTS (SELECT 1 FROM pg_index i
                              WHERE i.indrelid = c.conrelid AND i.indkey[0] = c.conkey[1])
        """))).all()
    await eng.dispose()
    missing = sorted(f"{t}.{col} -> {ref}" for t, col, ref in rows if ref not in EXEMPT_PARENTS)
    assert not missing, ("foreign keys with no index leading on the column; add one in a migration "
                         f"(see 0123_remaining_fk_indexes): {missing}")
