"""Documentation Agent: the pure Markdown renderers, and a live datasheet generation over the corpus."""

from __future__ import annotations

import asyncio

import pytest

from core.config import get_settings


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


def _clear():
    from db.session import get_engine, get_sessionmaker

    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


def run_async(coro):
    _clear()
    try:
        return asyncio.run(coro)
    finally:
        _clear()


def test_render_datasheet_has_sections_and_gaps():
    from services.export.datasheet import render_datasheet

    md = render_datasheet(
        title="Test corpus",
        size={"sessions": 3, "frames": 100, "objects": 900, "human_touched": "12%", "ontology_version": "v0"},
        coverage={"gaps": ["night-rain thin (2%)", "no fog frames"]},
        class_dist=[{"name": "sedan", "l1": "four_wheeler", "count": 400},
                    {"name": "autorickshaw", "l1": "three_wheeler", "count": 200}],
        scene={"weather": {"clear": 80, "rain": 20}, "time_of_day": {"day": 70, "night": 30}},
        geo={"BLR": 60, "DEL": 40},
        quality={"n_objects": 500, "metrics": {"map50": 0.71, "map": 0.55, "safe_miou": 0.62}})
    assert "# Dataset datasheet: Test corpus" in md
    assert "## Composition" in md and "## Known gaps" in md
    assert "night-rain thin (2%)" in md and "autorickshaw" in md
    assert "## Measured quality" in md and "0.71" in md
    assert "—" not in md   # no em-dashes


def test_render_model_card_and_weekly():
    from services.export.datasheet import render_model_card, render_weekly_report

    card = render_model_card(model={
        "model_version": "det-2026-07-01", "task": "detection", "is_champion": True, "promoted_from": "det-old",
        "weights_uri": "s3://x/w.pt", "dataset_commit": "abc123",
        "gold_metrics": {"map50": 0.7, "precision": 0.9, "recall": 0.8,
                         "per_class_pr": {"pedestrian": {"precision": 0.95, "recall": 0.9, "ap50": 0.88}}}})
    assert "# Model card: det-2026-07-01" in card and "pedestrian" in card and "Limitations".lower() in card.lower()

    weekly = render_weekly_report(
        precision={"precision": 0.962, "reviewed": 200, "pending": 5},
        drift=[{"metric": "control_precision", "value": 0.94, "breach": True}],
        promotions=[{"subject": "det-2026-07-01", "decision": "promote"}],
        coverage_gaps=["night-rain thin"])
    assert "96.2%" in weekly and "control_precision = 0.94 (breach)" in weekly and "promote" in weekly


@requires_infra
def test_generate_datasheet_live():
    from db.session import get_sessionmaker
    from services.agent.doc_agent import generate_datasheet

    async def _flow():
        async with get_sessionmaker()() as db:
            r = await generate_datasheet(db, title="Smoke")
        assert r["uri"].startswith("s3://") and "docs/datasheet/" in r["uri"]
        assert "# Dataset datasheet: Smoke" in r["markdown"] and "## Composition" in r["markdown"]

    run_async(_flow())
