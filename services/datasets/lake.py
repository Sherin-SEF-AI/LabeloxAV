"""Labels as SQL: the annotation plane published to Iceberg, for the customer's own query engine.

Every "can we get a custom report" ask is the same ask underneath: somebody wants to join the labels against
something we have never heard of. Answering that with a BI feature means guessing which joins matter, and
guessing wrong a dozen times. Publishing the tables instead answers all of them at once, in whatever tool
the customer already uses, and stops us being the bottleneck on questions about their own data.

Iceberg rather than plain Parquet files because a directory of Parquet has no schema evolution, no snapshot
isolation and no way to say "as of last Tuesday". Iceberg has all three, and DuckDB, Trino, Spark, Snowflake
and Databricks all read it, so this is compatibility with their stack rather than a partnership with any of
them.

No new infrastructure. The catalog is a SQL catalog on the Postgres already running, and the data files land
in the object store already running, so the lake is two tables in a database that exists and some Parquet in
a bucket that exists.

Three tables, matching the three planes the system already keeps apart:

  labels      one row per Object, the annotation plane
  provenance  one row per Review, who decided what and when
  quality     one row per AnnotationQuality, what the QA layer thought

They are published, not mirrored live. A snapshot is taken on demand and is immutable once written, which is
the property that makes a number in a customer's dashboard reproducible a month later.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pyarrow as pa
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logging import get_logger
from db.models import AnnotationQuality, Frame, Object, Review
from db.models import Session as DbSession

log = get_logger("datasets.lake")

NAMESPACE = "labelox"
TABLES = ("labels", "provenance", "quality")

# Rows per Arrow batch when reading out of Postgres. The whole point is that a corpus larger than memory can
# be published, so nothing here materialises the full table.
BATCH = 20_000


def _schemas() -> dict[str, pa.Schema]:
    """Arrow schemas for the three tables.

    Written out rather than inferred, because an inferred schema changes shape when a column happens to be
    all-null in one snapshot, and a customer's query breaking because yesterday's export had no rejections
    is precisely the failure this is meant to prevent.
    """
    return {
        "labels": pa.schema([
            ("object_id", pa.string()), ("frame_id", pa.string()), ("session_id", pa.string()),
            ("track_id", pa.string()), ("class_id", pa.int32()), ("class_name", pa.string()),
            ("state", pa.string()), ("source", pa.string()), ("conf", pa.float64()),
            ("bbox_x1", pa.float64()), ("bbox_y1", pa.float64()),
            ("bbox_x2", pa.float64()), ("bbox_y2", pa.float64()),
            ("city", pa.string()), ("vehicle_id", pa.string()), ("cam_id", pa.string()),
            ("ts_ns", pa.int64()), ("version", pa.int32()),
        ]),
        "provenance": pa.schema([
            ("review_id", pa.string()), ("object_id", pa.string()), ("reviewer", pa.string()),
            ("user_id", pa.string()), ("action", pa.string()),
            ("before_state", pa.string()), ("after_state", pa.string()),
            ("before_class_id", pa.int32()), ("after_class_id", pa.int32()),
            ("time_spent_ms", pa.int32()), ("ts_ns", pa.int64()),
        ]),
        "quality": pa.schema([
            ("object_id", pa.string()), ("quality", pa.float64()), ("agreement", pa.float64()),
            ("audit_verdict", pa.string()), ("flags", pa.list_(pa.string())),
        ]),
    }


def _catalog():
    """A SQL catalog on the Postgres that is already running, with warehouse files in the object store."""
    from pyiceberg.catalog.sql import SqlCatalog

    cfg = get_settings()
    pg = cfg.postgres
    uri = f"postgresql+psycopg2://{pg.user}:{pg.password}@{pg.host}:{pg.port}/{pg.db}"
    m = cfg.minio
    return SqlCatalog("labelox", **{
        "uri": uri,
        "warehouse": f"s3://{m.bucket}/lake",
        "s3.endpoint": m.endpoint,
        "s3.access-key-id": m.access_key,
        "s3.secret-access-key": m.secret_key,
    })


async def _labels_batches(db: AsyncSession, schema: pa.Schema):
    from services.autolabel.ontology import get_ontology
    onto = get_ontology()

    stmt = (select(Object.object_id, Object.frame_id, Object.track_id, Object.class_id, Object.state,
                   Object.source, Object.conf, Object.bbox, Object.version,
                   Frame.session_id, Frame.cam_id, Frame.ts_ns, DbSession.city, DbSession.vehicle_id)
            .join(Frame, Frame.frame_id == Object.frame_id)
            .join(DbSession, DbSession.session_id == Frame.session_id)
            .execution_options(yield_per=BATCH))
    rows: list[dict] = []
    async for r in (await db.stream(stmt)).mappings():
        bb = list(r["bbox"] or [None] * 4)
        rows.append({
            "object_id": str(r["object_id"]), "frame_id": str(r["frame_id"]),
            "session_id": str(r["session_id"]),
            "track_id": (str(r["track_id"]) if r["track_id"] else None),
            "class_id": int(r["class_id"]) if r["class_id"] is not None else None,
            "class_name": (onto.by_id(r["class_id"]).name if r["class_id"] else None),
            "state": r["state"], "source": r["source"],
            "conf": (float(r["conf"]) if r["conf"] is not None else None),
            "bbox_x1": _f(bb, 0), "bbox_y1": _f(bb, 1), "bbox_x2": _f(bb, 2), "bbox_y2": _f(bb, 3),
            "city": r["city"], "vehicle_id": r["vehicle_id"], "cam_id": r["cam_id"],
            "ts_ns": int(r["ts_ns"] or 0), "version": int(r["version"] or 0),
        })
        if len(rows) >= BATCH:
            yield pa.Table.from_pylist(rows, schema=schema)
            rows = []
    if rows:
        yield pa.Table.from_pylist(rows, schema=schema)


def _f(seq, i):
    try:
        return float(seq[i])
    except (IndexError, TypeError, ValueError):
        return None


async def _provenance_batches(db: AsyncSession, schema: pa.Schema):
    rows: list[dict] = []
    async for r in (await db.stream(select(Review).execution_options(yield_per=BATCH))).scalars():
        before, after = (r.before or {}), (r.after or {})
        rows.append({
            "review_id": str(r.review_id), "object_id": str(r.object_id), "reviewer": r.reviewer,
            "user_id": (str(r.user_id) if r.user_id else None), "action": r.action,
            "before_state": before.get("state"), "after_state": after.get("state"),
            "before_class_id": before.get("class_id"), "after_class_id": after.get("class_id"),
            "time_spent_ms": int(r.time_spent_ms or 0), "ts_ns": int(r.ts_ns or 0),
        })
        if len(rows) >= BATCH:
            yield pa.Table.from_pylist(rows, schema=schema)
            rows = []
    if rows:
        yield pa.Table.from_pylist(rows, schema=schema)


async def _quality_batches(db: AsyncSession, schema: pa.Schema):
    rows: list[dict] = []
    async for r in (await db.stream(select(AnnotationQuality).execution_options(yield_per=BATCH))).scalars():
        rows.append({
            "object_id": str(r.object_id),
            "quality": (float(r.quality) if r.quality is not None else None),
            "agreement": (float(r.agreement) if r.agreement is not None else None),
            "audit_verdict": r.audit_verdict,
            "flags": [str(f) for f in (r.flags or [])],
        })
        if len(rows) >= BATCH:
            yield pa.Table.from_pylist(rows, schema=schema)
            rows = []
    if rows:
        yield pa.Table.from_pylist(rows, schema=schema)


async def publish(db: AsyncSession, *, tables: tuple[str, ...] = TABLES) -> dict:
    """Write a snapshot of the annotation plane to Iceberg. Returns what landed where."""
    from pyiceberg.exceptions import NamespaceAlreadyExistsError

    cat = _catalog()
    try:
        cat.create_namespace(NAMESPACE)
    except NamespaceAlreadyExistsError:
        pass

    schemas = _schemas()
    producers: dict[str, Any] = {
        "labels": _labels_batches, "provenance": _provenance_batches, "quality": _quality_batches,
    }

    out: dict = {"namespace": NAMESPACE, "at": datetime.now(UTC).isoformat(), "tables": {}}
    for name in tables:
        schema = schemas[name]
        ident = f"{NAMESPACE}.{name}"
        try:
            tbl = cat.load_table(ident)
        except Exception:  # noqa: BLE001  first publish creates it
            tbl = cat.create_table(ident, schema=schema)

        # Overwrite rather than append. A snapshot is a statement about the corpus as it is now, and
        # appending would make every reader deduplicate by hand. Iceberg keeps the prior snapshot, so
        # "as of last Tuesday" still answers, which is the property plain Parquet cannot give.
        n = 0
        first = True
        # Streamed, not collected. Gathering the batches into a list first would hold the whole table in
        # memory, which is the one thing this is written to avoid.
        async for batch in producers[name](db, schema):
            if first:
                tbl.overwrite(batch)
                first = False
            else:
                tbl.append(batch)
            n += batch.num_rows
        if first:
            tbl.overwrite(pa.Table.from_pylist([], schema=schema))
        out["tables"][name] = {"rows": n, "location": tbl.location()}
        log.info("lake.published", table=ident, rows=n)

    return out


def scan(table: str, *, limit: int = 100) -> list[dict]:
    """Read a published table back, the way any Iceberg reader would."""
    if table not in TABLES:
        raise ValueError(f"unknown lake table '{table}'")
    tbl = _catalog().load_table(f"{NAMESPACE}.{table}")
    return tbl.scan(limit=limit).to_arrow().to_pylist()
