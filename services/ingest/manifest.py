"""Session manifest: the record Plane 5 versions and (later) PostGIS indexes. Captures sensors,
serials/calibration hashes, time bounds, frame count, the GPS track and QA counters.
"""

from __future__ import annotations

import json
from uuid import UUID

from core.schemas import SessionManifest
from core.storage import ObjectStore


def manifest_key(session_id: UUID) -> str:
    return f"sessions/{session_id}/manifest.json"


def write_manifest(store: ObjectStore, manifest: SessionManifest) -> str:
    key = manifest_key(manifest.session_id)
    payload = manifest.model_dump(mode="json")
    return store.put_bytes(key, json.dumps(payload, indent=2).encode("utf-8"), "application/json")
