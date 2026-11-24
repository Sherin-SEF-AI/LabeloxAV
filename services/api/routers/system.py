"""Installation health for administrators: the same checks `scripts/doctor.sh` runs, as JSON.

The web app's system status page reads this, so an administrator sees "the GPU worker is not running, start
it with ..." on screen instead of learning it from a job bar that never moves. Admin only: the results name
hosts, ports and model files.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.deps import ApiOut, db_session, require_role

router = APIRouter()


class FeaturesOut(ApiOut):
    optional_packs: dict[str, bool]             # pack id -> whether this installation has it on
    gpu_worker: bool
    autolabel_executor: str                     # api | worker
    auth_enabled: bool


class CheckOut(ApiOut):
    """One line of `scripts/doctor.sh`: services.doctor.Check, field for field."""

    name: str
    status: str                                 # ok | warn | fail | info
    detail: str
    fix: str                                    # empty when there is nothing to do
    group: str


class ChecksOut(ApiOut):
    status: str                                 # the worst status among the checks
    checks: list[CheckOut]


class CopyleftOut(ApiOut):
    name: str
    licence: str
    where: str
    why: str


class AboutOut(ApiOut):
    version: str | None                         # null when the package is not installed (a source checkout)
    commit: str | None
    source_url: str
    source_of_this_version: str | None          # only an unmodified deployment links an exact commit
    modified: bool
    own_licence: str
    combined_licence: str
    copyleft: list[CopyleftOut]
    licensing_doc: str


@router.get("/system/features", dependencies=[Depends(require_role("annotator"))],
            response_model=FeaturesOut, response_model_exclude_unset=True)
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


@router.get("/system/checks", dependencies=[Depends(require_role("admin"))],
            response_model=ChecksOut, response_model_exclude_unset=True)
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


@router.get("/system/about", dependencies=[Depends(require_role("annotator"))],
            response_model=AboutOut, response_model_exclude_unset=True)
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


@router.get("/system/models", dependencies=[Depends(require_role("admin"))])
async def loaded_models():
    """What this process has on the card: every model the shared registry holds, how big it was
    estimated to be, whether something is using it right now, and how long it has sat idle.

    This process only. The GPU worker is a separate process with its own registry, so on a Docker
    deployment this describes the API (the editor's segmenter, the describe tool), not auto-label.
    """
    from core.config import get_settings
    from core.model_manager import manager, vram_free_mb, vram_total_mb

    device = get_settings().gpu.device
    return {"device": device, "vram_free_mb": vram_free_mb(device), "vram_total_mb": vram_total_mb(device),
            "models": manager().stats()}


@router.post("/system/models/release", dependencies=[Depends(require_role("admin"))])
async def release_models(key: str | None = None):
    """Free idle models in this process: one by key, or every idle one. A model in use is left alone
    and reported, so this is safe to press while somebody is annotating."""
    import asyncio

    from core.model_manager import manager

    m = manager()
    before = {s["key"] for s in m.stats()}
    if key is not None:
        await asyncio.to_thread(m.release, key)
    else:
        await asyncio.to_thread(m.release_all)
    after = {s["key"] for s in m.stats()}
    return {"released": sorted(before - after), "still_loaded": sorted(after)}


@router.get("/system/llm-usage", dependencies=[Depends(require_role("admin"))])
async def llm_usage(days: Annotated[int, Query(ge=1, le=366)] = 30, db: AsyncSession = Depends(db_session)):
    """What the language and vision models were asked and what it cost, by day, model and purpose.

    Counters still in memory are written first, so the answer includes the calls of the last few seconds.
    `unpriced` lists cloud models that were called with no price configured: their cost is unknown, not
    zero, and the total says so rather than under-reporting quietly.
    """
    import asyncio
    from datetime import timedelta

    from sqlalchemy import select

    from core.config import get_settings
    from db.models import LlmUsageDaily
    from services.llm import ledger

    await asyncio.to_thread(ledger.flush)
    since = ledger.today() - timedelta(days=days - 1)
    rows = (await db.execute(select(LlmUsageDaily).where(LlmUsageDaily.day >= since)
                             .order_by(LlmUsageDaily.day.desc(), LlmUsageDaily.provider,
                                       LlmUsageDaily.model, LlmUsageDaily.purpose))).scalars().all()
    out, unpriced = [], set()
    totals = {"calls": 0, "failures": 0, "input_tokens": 0, "output_tokens": 0, "images": 0,
              "cost_usd": 0.0, "cloud_cost_usd": 0.0}
    for r in rows:
        cloud = r.provider not in ledger.LOCAL_PROVIDERS
        if cloud and r.cost_usd is None and (r.input_tokens or r.output_tokens or r.calls):
            unpriced.add(f"{r.provider}:{r.model}")
        out.append({"day": r.day.isoformat(), "provider": r.provider, "model": r.model, "purpose": r.purpose,
                    "calls": r.calls, "failures": r.failures, "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens, "images": r.images,
                    "avg_latency_ms": round(r.latency_ms / r.calls) if r.calls else None,
                    "unmetered_calls": r.unmetered_calls, "cost_usd": r.cost_usd, "cloud": cloud})
        for k in ("calls", "failures", "input_tokens", "output_tokens", "images"):
            totals[k] += getattr(r, k)
        totals["cost_usd"] += r.cost_usd or 0.0
        if cloud:
            totals["cloud_cost_usd"] += r.cost_usd or 0.0
    cap = float(get_settings().llm.daily_budget_usd or 0.0)
    spent_today = await asyncio.to_thread(ledger.cloud_spend_today)
    return {"since": since.isoformat(), "days": days, "rows": out,
            "totals": {**totals, "cost_usd": round(totals["cost_usd"], 6),
                       "cloud_cost_usd": round(totals["cloud_cost_usd"], 6)},
            "unpriced": sorted(unpriced),
            "budget": {"daily_usd": cap or None, "spent_today_usd": round(spent_today, 6),
                       "exhausted": bool(cap and spent_today >= cap)}}
