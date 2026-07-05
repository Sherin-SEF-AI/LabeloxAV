"""Release content fingerprinting (M6). A Release must be immutable and reproducible: the same annotation
content always hashes to the same id, and a mutated annotation yields a distinct id (so the original release
stays byte-stable and a rebuild has distinct lineage). The existing seal_commit_id hashes object ids only, so
a mutated bbox on the same object id would collide; this fingerprints the annotation CONTENT (class, box,
mask, state, version) so content changes are visible in the hash.

Pure and order-independent, so it is deterministic across rebuilds and unit-testable."""

from __future__ import annotations

import hashlib
import json


def object_fingerprint(obj: dict) -> str:
    """A stable fingerprint of one annotation's content. Rounds the box so floating-point noise does not
    change the hash, and includes the fields that define the label (class, geometry, state, version)."""
    box = [round(float(v), 3) for v in (obj.get("bbox") or [])]
    payload = {
        "object_id": str(obj.get("object_id")),
        "class_id": obj.get("class_id"),
        "bbox": box,
        "mask_uri": obj.get("mask_uri"),
        "state": obj.get("state"),
        "version": obj.get("version"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def content_fingerprint(objects: list[dict], spec: dict, ontology_version: str) -> str:
    """The release content hash: the sorted set of per-object fingerprints plus the slice spec and ontology
    version. Order-independent (members are sorted) and content-sensitive."""
    h = hashlib.sha256()
    h.update(json.dumps(spec, sort_keys=True).encode())
    h.update(ontology_version.encode())
    for fp in sorted(object_fingerprint(o) for o in objects):
        h.update(fp.encode())
    return f"lbx-{h.hexdigest()[:16]}"
