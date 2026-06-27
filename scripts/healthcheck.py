"""Verify every infra dependency is reachable. Exit non-zero on any failure.

    uv run python scripts/healthcheck.py
"""

from __future__ import annotations

import asyncio
import sys

from core.config import get_settings
from core.logging import get_logger, setup_logging

log = get_logger("healthcheck")


async def check_postgres() -> bool:
    from sqlalchemy import text

    from db.session import get_engine

    try:
        async with get_engine().connect() as conn:
            await conn.execute(text("SELECT 1"))
            ext = (await conn.execute(text("SELECT extname FROM pg_extension WHERE extname='postgis'"))).first()
        return ext is not None
    except Exception as exc:  # noqa: BLE001
        log.error("postgres.fail", error=str(exc))
        return False


def check_minio() -> bool:
    from core.storage import get_object_store

    try:
        store = get_object_store()
        store.ensure_bucket()
        return store.exists(store.uri("__healthcheck__")) or True
    except Exception as exc:  # noqa: BLE001
        log.error("minio.fail", error=str(exc))
        return False


def check_redis() -> bool:
    import redis as redis_lib

    try:
        client = redis_lib.Redis.from_url(get_settings().redis.url)
        return bool(client.ping())
    except Exception as exc:  # noqa: BLE001
        log.error("redis.fail", error=str(exc))
        return False


async def check_redpanda() -> bool:
    from aiokafka.admin import AIOKafkaAdminClient

    admin = AIOKafkaAdminClient(bootstrap_servers=get_settings().redpanda.brokers)
    try:
        await admin.start()
        await admin.list_topics()
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("redpanda.fail", error=str(exc))
        return False
    finally:
        try:
            await admin.close()
        except Exception:  # noqa: BLE001
            pass


def check_pii_gate() -> bool:
    # When the DPDPA gate is enabled, the face detector weights must exist or ingestion would store
    # un-anonymized frames. Fail loud (run `make pii-models`). Plate model is optional.
    from pathlib import Path

    cfg = get_settings().pii
    if not cfg.enabled:
        return True
    ok = Path(cfg.face_weights).exists()
    if not ok:
        log.error("pii_gate.face_weights_missing", path=cfg.face_weights, hint="run make pii-models")
    return ok


async def main() -> None:
    setup_logging(get_settings().log_level)
    results = {
        "postgres": await check_postgres(),
        "minio": check_minio(),
        "redis": check_redis(),
        "redpanda": await check_redpanda(),
        "pii_gate": check_pii_gate(),
    }
    for name, ok in results.items():
        log.info("healthcheck", service=name, ok=ok)
    if not all(results.values()):
        sys.exit(1)
    log.info("healthcheck.all_ok")


if __name__ == "__main__":
    asyncio.run(main())
