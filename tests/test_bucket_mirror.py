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
