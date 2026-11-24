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


def _objects(s3, bucket: str):
    """Every object as (key, etag, size)."""
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            if not obj["Key"].endswith("/"):
                yield obj["Key"], str(obj.get("ETag", "")).strip('"'), int(obj.get("Size", 0))


def _reuse(prev_file: Path, dest: Path) -> None:
    """Make `dest` the same file as `prev_file` without copying it when the filesystem allows.

    A hard link, so seven nightly snapshots of a corpus that changed by one drive cost one corpus plus
    one drive, not seven corpora. Safe against later snapshots: a download never writes into an existing
    file, it writes a temporary one and renames it over the name, which leaves the linked copy alone.
    Falls back to a real copy across filesystems, where a link is impossible.
    """
    import shutil

    if dest.exists():
        dest.unlink()
    try:
        os.link(prev_file, dest)
    except OSError:
        shutil.copy2(prev_file, dest)


def download(s3, bucket: str, root: Path, *, manifest: Path | None = None, link_dest: Path | None = None,
             prev_manifest: Path | None = None) -> int:
    """Mirror the bucket into `root`.

    With `link_dest` and `prev_manifest` (an earlier mirror and the manifest written beside it), an
    object whose ETag and size are unchanged is linked from that mirror instead of fetched again. The
    result is still a complete, self-contained tree: restoring it, or deleting the older snapshot, needs
    nothing else. `manifest` records what this mirror holds, for the next run to compare against.
    """
    import json

    root.mkdir(parents=True, exist_ok=True)
    prev: dict[str, list] = {}
    if link_dest is not None and prev_manifest is not None and prev_manifest.is_file():
        try:
            prev = json.loads(prev_manifest.read_text())
        except ValueError:
            prev = {}
    reused = [0]

    def one(item: tuple[str, str, int]) -> None:
        key, etag, size = item
        dest = _safe_path(root, key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if link_dest is not None and prev.get(key) == [etag, size]:
            prev_file = _safe_path(link_dest, key)
            # The size check is against the file on disk, not the manifest's word for it: a truncated
            # file in the older snapshot must not be carried forward as if it were whole.
            if prev_file.is_file() and prev_file.stat().st_size == size:
                _reuse(prev_file, dest)
                reused[0] += 1
                return
        s3.download_file(bucket, key, str(dest))

    objects = list(_objects(s3, bucket))
    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(one, objects))
    if manifest is not None:
        manifest.write_text(json.dumps({k: [e, sz] for k, e, sz in objects}))
    if link_dest is not None:
        print(f"reused {reused[0]} unchanged objects from {link_dest}", file=sys.stderr)
    return len(objects)


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
    ap.add_argument("--manifest", type=Path, help="download: write the list of mirrored objects here")
    ap.add_argument("--link-dest", type=Path, help="download: an earlier mirror to reuse unchanged objects from")
    ap.add_argument("--prev-manifest", type=Path, help="download: the manifest written with --link-dest's mirror")
    a = ap.parse_args()
    s3 = _client(a.endpoint)
    if a.direction == "download":
        n = download(s3, a.bucket, a.dir, manifest=a.manifest, link_dest=a.link_dest,
                     prev_manifest=a.prev_manifest)
    else:
        if not a.dir.is_dir():
            raise SystemExit(f"no such directory: {a.dir}")
        n = upload(s3, a.bucket, a.dir)
    print(f"{a.direction}ed {n} objects", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
