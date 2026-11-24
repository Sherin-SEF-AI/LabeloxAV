"""The MinIO half of backup and restore: a bucket copied to a directory and back, byte for byte.

scripts/bucket_mirror.py replaced `mc mirror` from MinIO's client image, which MinIO no longer publishes.
"""

from __future__ import annotations

import importlib.util
import uuid
from pathlib import Path

import pytest

from core.config import get_settings

pytestmark = pytest.mark.infra

_SPEC = importlib.util.spec_from_file_location(
    "bucket_mirror", Path(__file__).resolve().parent.parent / "scripts" / "bucket_mirror.py")
mirror = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mirror)


@pytest.fixture
def s3(monkeypatch):
    m = get_settings().minio
    monkeypatch.setenv("MINIO_ROOT_USER", m.access_key)
    monkeypatch.setenv("MINIO_ROOT_PASSWORD", m.secret_key)
    client = mirror._client(m.endpoint)
    made: list[str] = []
    yield client, made
    for b in made:
        for k in list(mirror._keys(client, b)):
            client.delete_object(Bucket=b, Key=k)
        client.delete_bucket(Bucket=b)


def test_a_bucket_survives_a_round_trip_through_a_directory(s3, tmp_path):
    client, made = s3
    src, dst = f"mirror-src-{uuid.uuid4().hex[:8]}", f"mirror-dst-{uuid.uuid4().hex[:8]}"
    made += [src, dst]
    client.create_bucket(Bucket=src)
    objects = {f"frames/s{i % 2}/cam_f/{i}.jpg": f"frame {i}".encode() for i in range(12)}
    objects["exports/a b/manifest.json"] = b'{"n": 1}'
    for k, v in objects.items():
        client.put_object(Bucket=src, Key=k, Body=v)

    assert mirror.download(client, src, tmp_path / "minio") == len(objects)
    assert (tmp_path / "minio" / "frames" / "s0" / "cam_f" / "0.jpg").read_bytes() == b"frame 0"
    assert mirror.upload(client, dst, tmp_path / "minio") == len(objects)   # creates the missing bucket

    back = {k: client.get_object(Bucket=dst, Key=k)["Body"].read() for k in mirror._keys(client, dst)}
    assert back == objects
    assert client.head_object(Bucket=dst, Key="exports/a b/manifest.json")["ContentType"] == "application/json"


def test_a_key_that_would_escape_the_backup_directory_is_refused(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        mirror._safe_path(tmp_path, "../../etc/passwd")
    assert mirror._safe_path(tmp_path, "frames/a.jpg") == (tmp_path / "frames" / "a.jpg").resolve()


def test_credentials_must_come_from_the_environment(monkeypatch):
    monkeypatch.delenv("MINIO_ROOT_USER", raising=False)
    monkeypatch.delenv("MINIO_ROOT_PASSWORD", raising=False)
    with pytest.raises(SystemExit, match="MINIO_ROOT_USER"):
        mirror._client("http://localhost:9")


def test_a_second_mirror_links_unchanged_objects_instead_of_fetching_them(s3, tmp_path):
    """Nightly backups: seven snapshots must not cost seven copies of the corpus."""
    client, made = s3
    src = f"mirror-inc-{uuid.uuid4().hex[:8]}"
    client.create_bucket(Bucket=src)
    made.append(src)
    client.put_object(Bucket=src, Key="frames/a.jpg", Body=b"A" * 2048)
    client.put_object(Bucket=src, Key="frames/b.jpg", Body=b"B" * 2048)

    first, second = tmp_path / "one", tmp_path / "two"
    mirror.download(client, src, first / "minio", manifest=first / "minio.manifest.json")

    client.put_object(Bucket=src, Key="frames/b.jpg", Body=b"CHANGED" * 300)     # rewritten in place
    client.put_object(Bucket=src, Key="frames/c.jpg", Body=b"C" * 10)            # new
    n = mirror.download(client, src, second / "minio", manifest=second / "minio.manifest.json",
                        link_dest=first / "minio", prev_manifest=first / "minio.manifest.json")
    assert n == 3

    a1, a2 = first / "minio/frames/a.jpg", second / "minio/frames/a.jpg"
    assert a1.stat().st_ino == a2.stat().st_ino, "an unchanged object must be the same file, not a copy"
    b1, b2 = first / "minio/frames/b.jpg", second / "minio/frames/b.jpg"
    assert b1.stat().st_ino != b2.stat().st_ino
    assert b1.read_bytes() == b"B" * 2048, "the older snapshot must keep what it captured"
    assert b2.read_bytes() == b"CHANGED" * 300
    assert (second / "minio/frames/c.jpg").read_bytes() == b"C" * 10


def test_an_object_deleted_from_the_bucket_is_absent_from_the_next_snapshot_only(s3, tmp_path):
    client, made = s3
    src = f"mirror-del-{uuid.uuid4().hex[:8]}"
    client.create_bucket(Bucket=src)
    made.append(src)
    client.put_object(Bucket=src, Key="k/keep.bin", Body=b"1")
    client.put_object(Bucket=src, Key="k/gone.bin", Body=b"2")
    first, second = tmp_path / "one", tmp_path / "two"
    mirror.download(client, src, first / "minio", manifest=first / "minio.manifest.json")
    client.delete_object(Bucket=src, Key="k/gone.bin")
    mirror.download(client, src, second / "minio", manifest=second / "minio.manifest.json",
                    link_dest=first / "minio", prev_manifest=first / "minio.manifest.json")
    assert (first / "minio/k/gone.bin").exists()
    assert not (second / "minio/k/gone.bin").exists()
    assert (second / "minio/k/keep.bin").read_bytes() == b"1"


def test_a_truncated_file_in_the_older_snapshot_is_fetched_again_not_carried_forward(s3, tmp_path):
    client, made = s3
    src = f"mirror-trunc-{uuid.uuid4().hex[:8]}"
    client.create_bucket(Bucket=src)
    made.append(src)
    client.put_object(Bucket=src, Key="x.bin", Body=b"0123456789")
    first, second = tmp_path / "one", tmp_path / "two"
    mirror.download(client, src, first / "minio", manifest=first / "minio.manifest.json")
    (first / "minio/x.bin").write_bytes(b"0123")                                  # damaged on disk
    mirror.download(client, src, second / "minio", manifest=second / "minio.manifest.json",
                    link_dest=first / "minio", prev_manifest=first / "minio.manifest.json")
    assert (second / "minio/x.bin").read_bytes() == b"0123456789"
