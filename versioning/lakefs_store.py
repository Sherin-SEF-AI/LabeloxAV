"""lakeFS-backed dataset versioning (M4.3, also used by M4.2 relabel). Git-like branch, commit, and merge
over the dataset, backed by the existing MinIO object store. A curation, a relabel run, or an annotator's
work is a branch; QA approval is a merge to main; an export and a model pin a commit. The lakeFS commit id
is kept as the commit_id pin so existing export and provenance continue to pin correctly.

A "label set" on a branch is one JSON manifest per object under labels/<object_id>.json, so a diff between
two branches is exactly the set of changed labels. Thin, deterministic wrappers over the lakeFS SDK.
"""

from __future__ import annotations

import functools
import json

import lakefs
from lakefs.client import Client

from core.config import get_settings
from core.logging import get_logger

log = get_logger("lakefs_store")


@functools.lru_cache(maxsize=1)
def get_client() -> Client:
    s = get_settings().phase4.lakefs
    return Client(host=s.endpoint, username=s.access_key, password=s.secret_key)


@functools.lru_cache(maxsize=1)
def get_repo():
    s = get_settings().phase4.lakefs
    return lakefs.Repository(s.repo, client=get_client()).create(
        storage_namespace=s.storage_namespace, default_branch=s.default_branch, exist_ok=True)


def default_branch() -> str:
    return get_settings().phase4.lakefs.default_branch


def ensure_branch(name: str, source: str | None = None) -> str:
    repo = get_repo()
    b = repo.branch(name).create(source_reference=source or default_branch(), exist_ok=True)
    return b.id


def delete_branch(name: str) -> None:
    try:
        get_repo().branch(name).delete()
    except Exception as exc:  # noqa: BLE001
        log.warning("lakefs.delete_branch_failed", branch=name, error=str(exc))


def list_branches() -> list[str]:
    return [b.id for b in get_repo().branches()]


def put_label(branch: str, object_id: str, label: dict) -> None:
    get_repo().branch(branch).object(f"labels/{object_id}.json").upload(
        data=json.dumps(label, sort_keys=True).encode(), content_type="application/json")


def read_label(ref: str, object_id: str) -> dict | None:
    try:
        data = get_repo().ref(ref).object(f"labels/{object_id}.json").reader().read()
        return json.loads(data)
    except Exception:  # noqa: BLE001
        return None


def commit(branch: str, message: str, metadata: dict | None = None) -> str:
    md = {k: str(v) for k, v in (metadata or {}).items()}  # lakeFS metadata values must be strings
    ref = get_repo().branch(branch).commit(message=message, metadata=md)
    return ref.get_commit().id


def diff_branches(source: str, dest: str | None = None) -> list[dict]:
    """Changes on source relative to dest (default main): each changed label object."""
    repo = get_repo()
    dest = dest or default_branch()
    return [{"path": d.path, "type": d.type} for d in repo.ref(dest).diff(other_ref=source)]


def merge(source: str, into: str | None = None, message: str | None = None) -> str:
    repo = get_repo()
    into = into or default_branch()
    repo.branch(source).merge_into(repo.branch(into))
    return get_repo().branch(into).head.get_commit().id


def revert(branch: str, ref: str, parent_number: int = 1) -> None:
    """Revert a commit on a branch (undo a bad merge or relabel)."""
    get_repo().branch(branch).revert(reference=ref, parent_number=parent_number)


def log_commits(ref: str | None = None, amount: int = 20) -> list[dict]:
    repo = get_repo()
    ref = ref or default_branch()
    out = []
    for c in repo.ref(ref).log(max_amount=amount):
        out.append({"id": c.id, "message": c.message, "metadata": c.metadata,
                    "creation_date": c.creation_date})
    return out
