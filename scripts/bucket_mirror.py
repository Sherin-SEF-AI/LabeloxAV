"""Copy every object between a bucket and a directory: the MinIO half of scripts/backup.sh and restore.sh.

    python scripts/bucket_mirror.py download --endpoint http://minio:9000 --bucket labeloxav --dir /backup/minio
    python scripts/bucket_mirror.py upload   --endpoint http://minio:9000 --bucket labeloxav --dir /backup/minio

Credentials come from MINIO_ROOT_USER and MINIO_ROOT_PASSWORD in the environment, never the command line,
where they would sit in the process list.

It replaces `mc mirror` from MinIO's client image, which MinIO no longer publishes: Docker Hub withdrew it and
quay.io refuses anonymous pulls, so backups failed on any machine without a cached copy. It depends on boto3
alone, which the app image and the development virtualenv both carry, so it runs offline. Object keys map to
paths under the directory exactly as `mc mirror` laid them out, so backups taken before this change restore
with it. Existing files and objects are overwritten, as `mc mirror --overwrite` did.
"""

from __future__ import annotations

import argparse
import mimetypes
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from botocore.config import Config

# Enough parallel transfers to keep a local MinIO busy without flooding a small box.
WORKERS = 16


def _client(endpoint: str):
    user, secret = os.environ.get("MINIO_ROOT_USER"), os.environ.get("MINIO_ROOT_PASSWORD")
    if not user or not secret:
        raise SystemExit("set MINIO_ROOT_USER and MINIO_ROOT_PASSWORD in the environment")
    return boto3.client("s3", endpoint_url=endpoint, aws_access_key_id=user, aws_secret_access_key=secret,
                        region_name="us-east-1",
                        config=Config(s3={"addressing_style": "path"}, max_pool_connections=WORKERS,
                                      retries={"max_attempts": 5, "mode": "standard"}))


def _keys(s3, bucket: str):
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            if not obj["Key"].endswith("/"):
                yield obj["Key"]


def _safe_path(root: Path, key: str) -> Path:
    """The file for a key, refusing any key that would escape the backup directory."""
    target = (root / key).resolve()
    if root.resolve() not in target.parents:
        raise ValueError(f"object key escapes the backup directory: {key!r}")
    return target


def download(s3, bucket: str, root: Path) -> int:
    root.mkdir(parents=True, exist_ok=True)

    def one(key: str) -> None:
        dest = _safe_path(root, key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(dest))

    keys = list(_keys(s3, bucket))
    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(one, keys))
    return len(keys)


def upload(s3, bucket: str, root: Path) -> int:
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:  # noqa: BLE001 - restoring into an empty install, whose bucket does not exist yet
        s3.create_bucket(Bucket=bucket)
    files = [p for p in root.rglob("*") if p.is_file()]

    def one(path: Path) -> None:
        key = path.relative_to(root).as_posix()
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        s3.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": ctype})

    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(one, files))
    return len(files)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("direction", choices=["download", "upload"])
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--dir", required=True, type=Path)
    a = ap.parse_args()
    s3 = _client(a.endpoint)
    if a.direction == "download":
        n = download(s3, a.bucket, a.dir)
    else:
        if not a.dir.is_dir():
            raise SystemExit(f"no such directory: {a.dir}")
        n = upload(s3, a.bucket, a.dir)
    print(f"{a.direction}ed {n} objects", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
