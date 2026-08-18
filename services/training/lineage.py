"""What a model was actually trained on, as a content hash rather than a name.

`register(..., dataset_commit=name)` stamped the human-readable build name - "loop-v1", "nightly-3". Two
builds a week apart carry the same string and are different data, so a promoted champion could not be
traced to the corpus that produced it: the strongest claim the governance layer makes ("this model beat the
champion on a sealed gold set") sat on top of a training set nobody could reconstruct.

`DatasetCommit.content_fingerprint` already existed and was already written by the export path. Training
simply never wrote one.

The fingerprint is taken over the built dataset directory rather than over the query that produced it,
which is what makes it task-agnostic: detection, segmentation, pose and lane builders all write label files
into the same layout, so one function covers every plugin and keeps covering the next one. It is also the
honest measure - it hashes the labels the trainer read, not the intent that selected them, so an object
edited between selection and build changes the hash.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from core.logging import get_logger

log = get_logger("training.lineage")

# The label trees every task plugin writes under the build directory. Images are deliberately excluded:
# they are content-addressed blobs the labels already reference, hashing gigabytes of JPEG per build would
# dominate the build time, and a frame's pixels do not change under an annotation edit - which is the thing
# this exists to detect.
_LABEL_DIRS = ("labels/train", "labels/val", "labels")


def dataset_fingerprint(build_dir: str | Path, ontology_version: str, spec: dict | None = None) -> str:
    """A content hash of the labels in a built training set.

    Order-independent (paths are sorted) and content-sensitive: one changed box changes the hash. Returns
    an `lbx-` prefixed digest, matching the shape services/release/fingerprint.py produces for exports so
    the two are recognisably the same kind of identifier.
    """
    root = Path(build_dir)
    h = hashlib.sha256()
    h.update(json.dumps(spec or {}, sort_keys=True, default=str).encode())
    h.update(ontology_version.encode())

    seen = 0
    for rel in _LABEL_DIRS:
        d = root / rel
        if not d.is_dir():
            continue
        for f in sorted(d.rglob("*")):
            if not f.is_file():
                continue
            h.update(str(f.relative_to(root)).encode())
            h.update(hashlib.sha256(f.read_bytes()).digest())
            seen += 1
    if seen == 0:
        # No labels found means the layout moved, and a fingerprint over nothing is a constant that would
        # make every build look identical - worse than no fingerprint, because it looks like lineage.
        log.warning("lineage.no_labels_found", build_dir=str(root),
                    note="fingerprint would be constant; recording it as unfingerprinted instead")
        return ""
    h.update(str(seen).encode())
    return f"lbx-{h.hexdigest()[:16]}"


async def record_dataset_commit(db, commit_id: str, *, build: dict, spec: dict, fingerprint: str) -> None:
    """Write the DatasetCommit row a trained model points at. Idempotent on commit_id."""
    from db.models import DatasetCommit

    if not commit_id:
        return
    existing = await db.get(DatasetCommit, commit_id)
    if existing is not None:
        return
    db.add(DatasetCommit(
        commit_id=commit_id,
        slice_spec=spec,
        object_count=int(build.get("n_train_objects") or 0) + int(build.get("n_val_objects") or 0),
        ontology_version=str(build.get("ontology_version") or ""),
        content_fingerprint=fingerprint or None,
        notes=(f"training build {build.get('name')}: "
               f"{build.get('n_train_images')} train / {build.get('n_val_images')} val images, "
               f"split={build.get('split')}"),
    ))
    log.info("lineage.dataset_commit", commit_id=commit_id, fingerprint=fingerprint,
             images=build.get("n_train_images"))
