"""Installation health for administrators: the same checks `scripts/doctor.sh` runs, as JSON.

The web app's system status page reads this, so an administrator sees "the GPU worker is not running, start
it with ..." on screen instead of learning it from a job bar that never moves. Admin only: the results name
hosts, ports and model files.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from services.api.deps import require_role

router = APIRouter()


@router.get("/system/features", dependencies=[Depends(require_role("annotator"))])
async def system_features():
    """Which optional parts this installation has, so the web app shows only what exists.

    Any signed-in person may read it: it says which pages to offer, nothing about hosts or files.
    """
    from core.config import get_settings
    from packs.registry import enabled_optional, optional_pack_ids
    from services.doctor import gpu_worker_running

    enabled = enabled_optional()
    return {
        "optional_packs": {pid: pid in enabled for pid in optional_pack_ids()},
        # Whether queued GPU jobs (training, and auto-label in worker mode) will actually run.
        "gpu_worker": await gpu_worker_running(),
        "autolabel_executor": get_settings().jobs.autolabel_executor,
        # Switching identity in the account menu exists only for a dev server with auth off; with auth on,
        # who you are is your sign-in.
        "auth_enabled": get_settings().auth.enabled,
    }


@router.get("/system/checks", dependencies=[Depends(require_role("admin"))])
async def system_checks():
    from services.doctor import run_checks, summarize

    return summarize(await run_checks())


# The components whose licences reach what a team runs (docs/LICENSING.md has the reasoning).
_COPYLEFT = [
    {"name": "Ultralytics", "licence": "AGPL-3.0", "where": "app image", "why": "the YOLO detector: auto-labelling, training, evaluation"},
    {"name": "cleanlab", "licence": "AGPL-3.0", "where": "app image", "why": "confident learning on the Label errors page"},
    {"name": "MinIO", "licence": "AGPL-3.0", "where": "its own container, unmodified", "why": "object storage"},
    {"name": "PostGIS", "licence": "GPL-2.0", "where": "the database container", "why": "spatial queries"},
    {"name": "psycopg", "licence": "LGPL-3.0", "where": "app image, as a library", "why": "the PostgreSQL driver"},
    {"name": "Redis 7.4+", "licence": "RSALv2 / SSPLv1", "where": "its own container", "why": "job and cache bus"},
    {"name": "Redpanda", "licence": "BSL 1.1", "where": "its own container", "why": "event stream"},
]


@router.get("/system/about", dependencies=[Depends(require_role("annotator"))])
async def about():
    """The running version, where its source is, and the licences that apply to it."""
    from importlib import metadata

    from core.config import get_settings
    from core.version import code_sha

    cfg = get_settings().about
    try:
        version = metadata.version("labeloxav")
    except metadata.PackageNotFoundError:
        version = None
    sha = code_sha()
    base = cfg.source_url.rstrip("/")
    # An unmodified deployment links the exact commit it runs, so "the source of this version" is one click.
    exact = f"{base}/tree/{sha}" if sha and not cfg.modified and "github.com" in base else None
    return {"version": version, "commit": sha, "source_url": cfg.source_url, "source_of_this_version": exact,
            "modified": cfg.modified, "own_licence": "Apache-2.0", "combined_licence": "AGPL-3.0",
            "copyleft": _COPYLEFT, "licensing_doc": f"{base}/blob/{sha or 'main'}/docs/LICENSING.md"}
