"""The MinIO to pod-workspace data-movement contract, implemented rather than described.

`services/training/cloud.py` documented this contract in a docstring and raised. That was honest about not
being wired, and it left four call sites (training, autolabel, relabel, hdmap) parked in `queued-cloud`
forever with no way to move a byte.

This module is the movement. It is deliberately independent of any pod: it uploads a prepared local
directory into the object store under a workspace prefix, and it pulls results back. A pod, wherever it
runs, only needs an S3 client and the same prefix convention. That keeps the transfer testable here, with
no GPU and no provisioned machine, which is the difference between a contract that exists and one that is
asserted.

Three decisions that matter:

- **The manifest is written last.** A pod polls for it, so a partial upload never looks ready: if the
  manifest is present, every file it lists is already there. Uploading it first would let a pod start
  training on half a dataset.
- **Every file is checksummed.** A truncated transfer produces a dataset that trains and is silently wrong,
  which is far worse than one that fails to load. The pull side verifies.
- **Nothing is deleted on failure.** A partial workspace is recoverable by re-running the push; a workspace
  that cleans itself up on error destroys the evidence of what went wrong.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from core.logging import get_logger
from core.storage import get_object_store

log = get_logger("cloud_transfer")

WORKSPACE_PREFIX = "workspace"
MANIFEST_NAME = "MANIFEST.json"

# Files a training run produces that are worth bringing home. Anything else on the pod is scratch: pulling
# the whole run directory back would drag tens of gigabytes of cached datasets through the network for the
# sake of a few weight files.
RESULT_PATTERNS = ("*.pt", "*.pth", "*.onnx", "*.yaml", "*.json", "*.csv", "results.png")


class TransferError(RuntimeError):
    """A transfer failed in a way the caller must handle rather than retry blindly."""


@dataclass
class Manifest:
    job_id: str
    kind: str                       # training | autolabel | relabel | hdmap
    prefix: str
    files: dict[str, str] = field(default_factory=dict)   # relative path -> sha256
    bytes_total: int = 0
    entrypoint: str | None = None
    args: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({
            "job_id": self.job_id, "kind": self.kind, "prefix": self.prefix,
            "files": self.files, "bytes_total": self.bytes_total,
            "entrypoint": self.entrypoint, "args": self.args,
            "contract": 1,
        }, indent=2, sort_keys=True)

    @staticmethod
    def from_json(raw: bytes | str) -> Manifest:
        d = json.loads(raw)
        return Manifest(job_id=d["job_id"], kind=d["kind"], prefix=d["prefix"],
                        files=d.get("files") or {}, bytes_total=int(d.get("bytes_total") or 0),
                        entrypoint=d.get("entrypoint"), args=d.get("args") or {})


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    """Streamed, so a multi-gigabyte weights file does not have to fit in memory to be checksummed."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def workspace_prefix(kind: str, job_id: str) -> str:
    return f"{WORKSPACE_PREFIX}/{kind}/{job_id}"


def push_workspace(local_dir: Path, *, job_id: str, kind: str,
                   entrypoint: str | None = None, args: dict | None = None) -> dict:
    """Upload a prepared directory and publish the manifest that marks it ready.

    The manifest goes last, on purpose: a pod polls for it, so its presence is the signal that every file it
    names is already uploaded. Publishing it first would let a pod begin on a half-transferred dataset and
    report a number computed from part of the data.
    """
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        raise TransferError(f"{local_dir} is not a directory")

    store = get_object_store()
    store.ensure_bucket()
    prefix = workspace_prefix(kind, job_id)

    manifest = Manifest(job_id=job_id, kind=kind, prefix=prefix,
                        entrypoint=entrypoint, args=dict(args or {}))
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir).as_posix()
        if rel == MANIFEST_NAME:
            continue
        store.put_file(f"{prefix}/{rel}", path)
        manifest.files[rel] = sha256_file(path)
        manifest.bytes_total += path.stat().st_size

    if not manifest.files:
        raise TransferError(f"{local_dir} contains no files to upload")

    store.put_bytes(f"{prefix}/{MANIFEST_NAME}", manifest.to_json().encode())
    log.info("cloud.workspace_pushed", job=job_id, kind=kind, files=len(manifest.files),
             bytes=manifest.bytes_total)
    return {"prefix": prefix, "files": len(manifest.files), "bytes": manifest.bytes_total,
            "manifest_uri": f"{prefix}/{MANIFEST_NAME}"}


def read_manifest(kind: str, job_id: str) -> Manifest | None:
    """The readiness check a pod polls. None means the workspace is not ready yet."""
    store = get_object_store()
    try:
        return Manifest.from_json(store.get_bytes(f"{workspace_prefix(kind, job_id)}/{MANIFEST_NAME}"))
    except Exception:  # noqa: BLE001 - absent or unreadable both mean "not ready"
        return None


def pull_workspace(dest_dir: Path, *, job_id: str, kind: str, verify: bool = True) -> dict:
    """Download a published workspace, verifying every checksum.

    Verification is on by default because the failure it catches is the quiet one: a truncated file produces
    a dataset that loads, trains, and is silently wrong, which is much worse than one that fails outright.
    """
    manifest = read_manifest(kind, job_id)
    if manifest is None:
        raise TransferError(f"no published workspace for {kind}/{job_id}; it is not ready")

    store = get_object_store()
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    corrupt: list[str] = []
    written = 0
    for rel, digest in sorted(manifest.files.items()):
        target = dest_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(store.get_bytes(f"{manifest.prefix}/{rel}"))
        if verify and sha256_file(target) != digest:
            corrupt.append(rel)
            continue
        written += 1

    if corrupt:
        # Named, not summarised. "3 files corrupt" is not actionable; the paths are.
        raise TransferError(f"checksum mismatch on {len(corrupt)} file(s): {sorted(corrupt)[:10]}")

    log.info("cloud.workspace_pulled", job=job_id, kind=kind, files=written)
    return {"dir": str(dest_dir), "files": written, "bytes": manifest.bytes_total,
            "entrypoint": manifest.entrypoint, "args": manifest.args}


def push_results(local_dir: Path, *, job_id: str, kind: str,
                 patterns: tuple[str, ...] = RESULT_PATTERNS) -> dict:
    """Send a finished run's artifacts home, filtered to what is worth keeping.

    Pattern-filtered rather than wholesale: a run directory holds the cached dataset and every intermediate
    checkpoint, and pulling all of it back would move tens of gigabytes for the sake of a few weight files.
    """
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        raise TransferError(f"{local_dir} is not a directory")

    store = get_object_store()
    store.ensure_bucket()
    prefix = f"{workspace_prefix(kind, job_id)}/results"

    uris: dict[str, str] = {}
    total = 0
    for pattern in patterns:
        for path in sorted(local_dir.rglob(pattern)):
            if not path.is_file():
                continue
            rel = path.relative_to(local_dir).as_posix()
            if rel in uris:
                continue
            uris[rel] = store.put_file(f"{prefix}/{rel}", path)
            total += path.stat().st_size

    if not uris:
        raise TransferError(
            f"no artifacts matched {patterns} under {local_dir}; the run produced nothing to bring home")

    store.put_bytes(f"{prefix}/{MANIFEST_NAME}",
                    json.dumps({"job_id": job_id, "kind": kind, "files": sorted(uris),
                                "bytes_total": total, "contract": 1}, indent=2).encode())
    log.info("cloud.results_pushed", job=job_id, kind=kind, files=len(uris), bytes=total)
    return {"prefix": prefix, "files": len(uris), "bytes": total, "uris": uris}


def fetch_results(job_id: str, kind: str) -> dict:
    """What a finished cloud run left behind, as object-store uris the registry can point at."""
    store = get_object_store()
    prefix = f"{workspace_prefix(kind, job_id)}/results"
    try:
        listing = json.loads(store.get_bytes(f"{prefix}/{MANIFEST_NAME}"))
    except Exception:  # noqa: BLE001
        return {"ready": False, "prefix": prefix, "files": []}
    return {"ready": True, "prefix": prefix, "bytes": listing.get("bytes_total", 0),
            "files": [{"path": p, "uri": f"{prefix}/{p}"} for p in listing.get("files", [])]}


def weights_uri(job_id: str, kind: str = "training") -> str | None:
    """The trained weights from a cloud run, so it joins the same registry as a local one.

    `best.pt` is preferred over `last.pt` deliberately: last is whatever the final epoch produced, which
    after an overfitting tail is not the model that was evaluated.
    """
    results = fetch_results(job_id, kind)
    if not results["ready"]:
        return None
    paths = {f["path"]: f["uri"] for f in results["files"]}
    for want in ("weights/best.pt", "best.pt"):
        for path, uri in paths.items():
            if path.endswith(want):
                return uri
    for path, uri in paths.items():
        if path.endswith(".pt"):
            return uri
    return None
