"""Publish the model history this system already has, without re-scoring anything.

Thirteen registered models and every evaluation ever run live only in Postgres. Re-running them to populate a
tracking server would cost hours of GPU and, worse, would produce *today's* numbers under yesterday's dates:
this corpus has had its gold set rebuilt twice and its harness reconciled since some of those evaluations,
so a re-score is a different measurement wearing the same name.

So this reads the stored rows and publishes what was actually recorded at the time. Each run is tagged
`backfilled=true` and carries the original timestamp, because a history that cannot tell "measured then" from
"measured now" is the thing being fixed rather than the fix.

    MLFLOW_TRACKING_URI=http://localhost:5500 .venv/bin/python -m scripts.backfill_mlflow --dry-run
    MLFLOW_TRACKING_URI=http://localhost:5500 .venv/bin/python -m scripts.backfill_mlflow --apply
"""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import select

from core.logging import get_logger, setup_logging
from db.models import Evaluation, ModelRegistry
from db.session import get_sessionmaker

log = get_logger("backfill_mlflow")


async def _gather() -> tuple[list[dict], list[dict]]:
    async with get_sessionmaker()() as db:
        models = (await db.execute(select(ModelRegistry))).scalars().all()
        evals = (await db.execute(select(Evaluation).order_by(Evaluation.created_at))).scalars().all()

    m_rows = [{
        "model_version": m.model_version,
        "gold_metrics": dict(m.gold_metrics or {}),
        "is_champion": bool(m.is_champion),
        "notes": m.notes or "",
        "created_at": m.created_at,
        "has_weights": bool(m.weights_uri),
    } for m in models]

    e_rows = [{
        "model_version": e.model_version,
        "gold_id": e.gold_id,
        "aggregate": dict(e.aggregate or {}),
        "verdict": e.verdict,
        "created_at": e.created_at,
        "slices": len(e.per_slice or {}),
    } for e in evals]
    return m_rows, e_rows


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    setup_logging("INFO")

    from services.integrations.mlflow_sink import enabled, log_evaluation, status

    st = status()
    print(f"  tracking uri : {st['tracking_uri'] or '(unset: nothing to publish to)'}")
    models, evals = await _gather()
    with_metrics = [m for m in models if m["gold_metrics"]]
    print(f"  models       : {len(models)} registered, {len(with_metrics)} carrying stored metrics")
    print(f"  evaluations  : {len(evals)}")

    if args.dry_run:
        for m in with_metrics[:10]:
            gm = m["gold_metrics"]
            print(f"    {m['model_version'][:40]:<42} map50={gm.get('map50')}  "
                  f"gold={gm.get('gold_id') or '(unrecorded)'}")
        for e in evals[:10]:
            print(f"    eval {e['model_version'][:34]:<36} verdict={e['verdict']:<12} "
                  f"gold={e['gold_id'] or '(none)'}  slices={e['slices']}")
        return 0

    if not enabled():
        print("  MLFLOW_TRACKING_URI is not set, so there is nowhere to publish. Nothing written.")
        return 1

    n = 0
    for m in with_metrics:
        gm = m["gold_metrics"]
        # A stored metric with no gold id is exactly the number this integration exists to stop shipping
        # naked, so it is published with the gap named rather than dropped or quietly labelled.
        gold = gm.get("gold_id") or "unrecorded"
        rid = log_evaluation(
            model_version=m["model_version"], gold_id=gold, metrics=gm,
            run_name=f"backfill:{m['model_version']}",
            tags={"backfilled": "true", "is_champion": str(m["is_champion"]),
                  "recorded_at": m["created_at"].isoformat() if m["created_at"] else "",
                  "source": "external" if m["notes"].startswith("external") else "labeloxav",
                  "gold_id_recorded": str(bool(gm.get("gold_id")))})
        n += 1 if rid else 0

    for e in evals:
        rid = log_evaluation(
            model_version=e["model_version"], gold_id=e["gold_id"] or "unrecorded",
            metrics=e["aggregate"], run_name=f"backfill-eval:{e['model_version']}",
            tags={"backfilled": "true", "verdict": e["verdict"], "slices": str(e["slices"]),
                  "recorded_at": e["created_at"].isoformat() if e["created_at"] else ""})
        n += 1 if rid else 0

    print(f"  published {n} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
