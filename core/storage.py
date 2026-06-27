"""S3-compatible object store wrapper (the blob abstraction).

Constraint 5: the 1 TB local disk is reached only through MinIO via this client, so the same
code scales to cloud S3 by changing the endpoint. Keys are content-addressed where it matters
(Principle: raw is immutable, addressed by content hash + UTC-ns).
"""

from __future__ import annotations

import hashlib
import io
from functools import lru_cache
from pathlib import Path

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from core.config import Settings, get_settings
from core.logging import get_logger

log = get_logger(__name__)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ObjectStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        m = self.settings.minio
        self.bucket = m.bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=m.endpoint,
            aws_access_key_id=m.access_key,
            aws_secret_access_key=m.secret_key,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
            region_name="us-east-1",
        )

    def ensure_bucket(self) -> None:
        try:
            self._client.head_bucket(Bucket=self.bucket)
        except ClientError:
            self._client.create_bucket(Bucket=self.bucket)
            log.info("bucket.created", bucket=self.bucket)

    def uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{key}"

    @staticmethod
    def parse_uri(uri: str) -> tuple[str, str]:
        if not uri.startswith("s3://"):
            raise ValueError(f"not an s3 uri: {uri}")
        rest = uri[len("s3://") :]
        bucket, _, key = rest.partition("/")
        return bucket, key

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        self._client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)
        return self.uri(key)

    def put_content_addressed(
        self, prefix: str, data: bytes, suffix: str, content_type: str = "application/octet-stream"
    ) -> str:
        """Write-once by content hash. Returns the s3 uri. Idempotent for identical content."""
        digest = sha256_bytes(data)
        key = f"{prefix.rstrip('/')}/{digest}{suffix}"
        # Idempotent: skip the upload if the identical object already exists.
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
        except ClientError:
            self._client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)
        return self.uri(key)

    def put_file(self, key: str, path: str | Path, content_type: str = "application/octet-stream") -> str:
        with open(path, "rb") as fh:
            self._client.put_object(Bucket=self.bucket, Key=key, Body=fh, ContentType=content_type)
        return self.uri(key)

    def get_bytes(self, uri_or_key: str) -> bytes:
        key = self._key(uri_or_key)
        resp = self._client.get_object(Bucket=self.bucket, Key=key)
        return resp["Body"].read()

    def get_stream(self, uri_or_key: str) -> io.BytesIO:
        return io.BytesIO(self.get_bytes(uri_or_key))

    def exists(self, uri_or_key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=self._key(uri_or_key))
            return True
        except ClientError:
            return False

    def presigned_get(self, uri_or_key: str, expires: int = 3600) -> str:
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": self._key(uri_or_key)},
            ExpiresIn=expires,
        )

    # --- Direct-to-storage upload (TB-capable; bytes never transit the API process) -------------
    # The browser uploads parts straight to MinIO/S3 with presigned URLs; the API only signs. This
    # is the literal cloud seam: point the endpoint at AWS S3 and the same flow works unchanged.

    def presigned_put(self, key: str, content_type: str = "application/octet-stream", expires: int = 3600) -> str:
        """Single-shot presigned PUT for small files (no multipart overhead)."""
        return self._client.generate_presigned_url(
            "put_object",
            Params={"Bucket": self.bucket, "Key": key, "ContentType": content_type},
            ExpiresIn=expires,
        )

    def create_multipart(self, key: str, content_type: str = "application/octet-stream") -> str:
        resp = self._client.create_multipart_upload(Bucket=self.bucket, Key=key, ContentType=content_type)
        return resp["UploadId"]

    def presign_part(self, key: str, upload_id: str, part_number: int, expires: int = 3600) -> str:
        return self._client.generate_presigned_url(
            "upload_part",
            Params={"Bucket": self.bucket, "Key": key, "UploadId": upload_id, "PartNumber": part_number},
            ExpiresIn=expires,
        )

    def complete_multipart(self, key: str, upload_id: str, parts: list[dict]) -> str:
        """parts: [{"PartNumber": int, "ETag": str}] in order. Returns the s3 uri."""
        ordered = sorted(parts, key=lambda p: int(p["PartNumber"]))
        self._client.complete_multipart_upload(
            Bucket=self.bucket, Key=key, UploadId=upload_id,
            MultipartUpload={"Parts": [{"ETag": p["ETag"], "PartNumber": int(p["PartNumber"])} for p in ordered]},
        )
        return self.uri(key)

    def abort_multipart(self, key: str, upload_id: str) -> None:
        self._client.abort_multipart_upload(Bucket=self.bucket, Key=key, UploadId=upload_id)

    def set_cors(self, origins: list[str]) -> bool:
        """Allow the web origin to PUT parts directly and read the ETag back (required for the
        browser multipart client). A classic gotcha: without ExposeHeaders ETag, completion fails.

        Returns True if applied. MinIO does not implement the S3 PutBucketCors API (it allows CORS
        from all origins at the server level by default, or via MINIO_API_CORS_ALLOW_ORIGIN), so a
        NotImplemented response is treated as a benign no-op. On real AWS S3 the rule is applied."""
        try:
            self._client.put_bucket_cors(
                Bucket=self.bucket,
                CORSConfiguration={
                    "CORSRules": [{
                        "AllowedHeaders": ["*"],
                        "AllowedMethods": ["GET", "PUT", "POST", "DELETE", "HEAD"],
                        "AllowedOrigins": origins,
                        "ExposeHeaders": ["ETag"],
                        "MaxAgeSeconds": 3600,
                    }]
                },
            )
            log.info("bucket.cors_set", bucket=self.bucket, origins=origins)
            return True
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "NotImplemented":
                log.info("bucket.cors_skipped_minio", note="MinIO allows CORS server-side by default")
                return False
            raise

    def _key(self, uri_or_key: str) -> str:
        if uri_or_key.startswith("s3://"):
            _, key = self.parse_uri(uri_or_key)
            return key
        return uri_or_key


@lru_cache(maxsize=1)
def get_object_store() -> ObjectStore:
    return ObjectStore()
