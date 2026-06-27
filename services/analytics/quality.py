"""Gate B, part (d): the quality sheet. Measures the active model against a sealed gold set and
caches the numbers (per-class P/R, mAP, Safe-mIoU, calibration ECE) onto the gold_set row. The API
serves the cached sheet (no GPU in the request path); `make m9` runs the measurement.

    python -m services.analytics.quality --gold <gold_id>
"""

from __future__ import annotations

import asyncio
import json

import click
from sqlalchemy import select

from core.config import get_settings
from core.logging import get_logger, setup_logging
from core.storage import get_object_store
from db.models import GoldSet
from db.session import get_sessionmaker
from services.training.gold import gold_data_yaml

log = get_logger("quality")


async def list_gold_sets() -> list[dict]:
    async with get_sessionmaker()() as db:
        rows = (await db.execute(select(GoldSet).order_by(GoldSet.created_at.desc()))).scalars().all()
    return [
        {"gold_id": g.gold_id, "name": g.name, "n_objects": g.n_objects, "n_frames": g.n_frames,
         "ontology_version": g.ontology_version, "measured": bool(g.metrics),
         "created_at": g.created_at.isoformat() if g.created_at else None}
        for g in rows
    ]


def _isotonic_ece() -> dict | None:
    cfg = get_settings().calibrate
    if cfg.method != "isotonic" or not cfg.isotonic_uri:
        return None
    try:
        data = json.loads(get_object_store().get_bytes(cfg.isotonic_uri).decode())
        return {"ece": data.get("ece"), "n_train": data.get("n_train"), "uri": cfg.isotonic_uri}
    except Exception:  # noqa: BLE001
        return None


async def measure_gold(gold_id: str, weights: str | None = None) -> dict:
    """Run the (GPU) eval + Safe-mIoU against the gold set and cache the result. CLI / worker only."""
    from services.training.eval import evaluate, safe_miou_report
    from services.training.gold import materialize_for_model

    settings = get_settings()
    weights = weights or settings.models.yolo.weights

    # Align the gold val split to the model's class order so index-based eval is correct; classes the
    # model does not know are dropped. Falls back to the as-sealed split if the model can't be read.
    def _model_names():
        from ultralytics import YOLO

        return dict(YOLO(weights).names)

    try:
        names = await asyncio.to_thread(_model_names)
        data_yaml = await materialize_for_model(gold_id, names)
    except Exception as exc:  # noqa: BLE001
        log.warning("quality.align_failed", error=str(exc))
        data_yaml = await gold_data_yaml(gold_id)

    ev = await asyncio.to_thread(evaluate, weights, data_yaml)
    sm = await asyncio.to_thread(safe_miou_report, weights, data_yaml)

    metrics = {
        "weights": weights,
        "map50": ev["map50"], "map": ev["map"],
        "precision": ev["precision"], "recall": ev["recall"],
        "per_class_pr": ev.get("per_class_pr", {}),
        "safe_miou": sm.get("safe_miou"),
        "safety_weight": sm.get("safety_weight"),
        "offdiag_mass": sm.get("offdiag_mass"),
        "calibration": _isotonic_ece(),
    }
    async with get_sessionmaker()() as db:
        g = await db.get(GoldSet, gold_id)
        if g is None:
            raise RuntimeError(f"gold set {gold_id} not found")
        g.metrics = metrics
        await db.commit()
        info = {"name": g.name, "n_objects": g.n_objects, "n_frames": g.n_frames,
                "ontology_version": g.ontology_version}
    log.info("quality.measured", gold_id=gold_id, map50=metrics["map50"], safe_miou=metrics["safe_miou"])
    return {"gold_id": gold_id, "measured": True, **info, "metrics": metrics}


async def quality_sheet(gold_id: str) -> dict:
    """Cached sheet for the API (no GPU). Returns measured=False until `make m9` has run."""
    async with get_sessionmaker()() as db:
        g = await db.get(GoldSet, gold_id)
    if g is None:
        return {"gold_id": gold_id, "found": False}
    return {
        "gold_id": gold_id, "found": True, "name": g.name,
        "n_objects": g.n_objects, "n_frames": g.n_frames, "ontology_version": g.ontology_version,
        "measured": bool(g.metrics), "metrics": g.metrics or {},
    }


@click.command()
@click.option("--gold", "gold_id", required=True)
@click.option("--weights", default=None, help="default: configured active Path A model")
def main(gold_id, weights) -> None:
    setup_logging(get_settings().log_level)
    click.echo(json.dumps(asyncio.run(measure_gold(gold_id, weights)), indent=2))


if __name__ == "__main__":
    main()
