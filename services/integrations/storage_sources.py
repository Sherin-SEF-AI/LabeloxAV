"""Registered external buckets a project imports from (S3, GCS, Azure, or any S3-compatible store).

No credentials are stored on the row. A source records the locator and the NAME of a server-side credential
profile; the actual keys stay in the server's environment. Persisting per-source cloud keys in the application
database would make one read of one table a breach of every connected bucket, and it is the kind of
convenience that is very hard to walk back once data exists.

Listing is read-only and bounded: this registers and previews a source so a human can confirm it points where
they think, and the existing import pipeline does the actual ingestion.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import StorageSource

log = get_logger("storage_sources")

PROVIDERS = ("s3", "gcs", "azure", "minio")


class SourceError(RuntimeError):
    pass


def _dict(s: StorageSource) -> dict:
    return {"source_id": str(s.source_id), "name": s.name, "provider": s.provider,
            "bucket": s.bucket, "prefix": s.prefix, "region": s.region,
            "endpoint_url": s.endpoint_url, "credential_profile": s.credential_profile,
            "project_id": str(s.project_id) if s.project_id else None,
            "last_object_count": s.last_object_count,
            "last_scanned_at": s.last_scanned_at.isoformat() if s.last_scanned_at else None,
            "created_at": s.created_at.isoformat() if s.created_at else None,
            "uri": uri_for(s)}


def uri_for(s: StorageSource) -> str:
    scheme = {"s3": "s3", "minio": "s3", "gcs": "gs", "azure": "az"}.get(s.provider, "s3")
    prefix = (s.prefix or "").lstrip("/")
    return f"{scheme}://{s.bucket}/{prefix}" if prefix else f"{scheme}://{s.bucket}"


async def register_source(db: AsyncSession, *, name: str, provider: str, bucket: str,
                          prefix: str | None = None, region: str | None = None,
                          endpoint_url: str | None = None, credential_profile: str | None = None,
                          project_id: str | None = None) -> dict:
    if provider not in PROVIDERS:
        raise SourceError(f"provider must be one of {PROVIDERS}")
    if not bucket.strip():
        raise SourceError("bucket required")
    s = StorageSource(name=name.strip(), provider=provider, bucket=bucket.strip(),
                      prefix=(prefix or None), region=region, endpoint_url=endpoint_url,
                      credential_profile=credential_profile,
                      project_id=UUID(project_id) if project_id else None)
    db.add(s)
    await db.commit()
    log.info("storage.source_registered", source=str(s.source_id), provider=provider, bucket=bucket)
    return _dict(s)


async def list_sources(db: AsyncSession, project_id: str | None = None) -> list[dict]:
    stmt = select(StorageSource)
    if project_id:
        stmt = stmt.where(StorageSource.project_id == UUID(project_id))
    rows = (await db.execute(stmt.order_by(StorageSource.created_at.desc()).limit(200))).scalars().all()
    return [_dict(s) for s in rows]


async def delete_source(db: AsyncSession, source_id: str) -> dict:
    s = await db.get(StorageSource, UUID(source_id))
    if s is None:
        return {"deleted": False}
    await db.delete(s)
    await db.commit()
    return {"deleted": True}


def _list_s3(bucket: str, prefix: str | None, region: str | None,
             endpoint_url: str | None, limit: int) -> list[str]:
    """List keys through boto3, reusing the timeouts and retry policy configured for the object store."""
    import boto3
    from botocore.client import Config

    cfg = Config(signature_version="s3v4", s3={"addressing_style": "path"},
                 connect_timeout=5, read_timeout=30,
                 retries={"max_attempts": 3, "mode": "standard"})
    client = boto3.client("s3", endpoint_url=endpoint_url, region_name=region or "us-east-1", config=cfg)
    out: list[str] = []
    token = None
    while len(out) < limit:
        kw = {"Bucket": bucket, "MaxKeys": min(1000, limit - len(out))}
        if prefix:
            kw["Prefix"] = prefix.lstrip("/")
        if token:
            kw["ContinuationToken"] = token
        resp = client.list_objects_v2(**kw)
        out += [c["Key"] for c in resp.get("Contents", [])]
        token = resp.get("NextContinuationToken")
        if not token:
            break
    return out[:limit]


async def preview_source(db: AsyncSession, source_id: str, limit: int = 50) -> dict:
    """List a bounded sample of keys, so a human can confirm the source points where they think before an
    import runs against it. Credentials come from the server environment, never from the row."""
    import asyncio

    s = await db.get(StorageSource, UUID(source_id))
    if s is None:
        raise SourceError("source not found")
    if s.provider not in ("s3", "minio"):
        # gcs/azure need their own SDKs; the row is still useful as a locator for a manual import.
        return {"source_id": source_id, "uri": uri_for(s), "keys": [],
                "detail": f"listing is implemented for s3-compatible stores only, not {s.provider}"}

    loop = asyncio.get_event_loop()
    try:
        keys = await loop.run_in_executor(
            None, _list_s3, s.bucket, s.prefix, s.region, s.endpoint_url, limit)
    except Exception as exc:  # noqa: BLE001
        # sanitized: a presigned or credentialed endpoint can carry a token in the message
        raise SourceError(f"could not list this source: {type(exc).__name__}") from None

    s.last_scanned_at = datetime.now(UTC)
    s.last_object_count = len(keys)
    await db.commit()
    return {"source_id": source_id, "uri": uri_for(s), "count": len(keys), "keys": keys}
