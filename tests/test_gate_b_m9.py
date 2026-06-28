"""Gate B (M9) tests. Pure-unit: Safe-mIoU safety ordering, isotonic JSON-knot reliability, and the
isotonic calibration path. Infra-gated: gold-set seal determinism."""

from __future__ import annotations

import numpy as np
import pytest

from core.config import CalibrateSettings, get_settings
from services.autolabel.ontology import get_ontology
from services.training.safe_miou import affinity_cost, safe_miou

_VRU = {"vru", "animal"}


def _find_classes():
    onto = get_ontology()
    vru = next((c for c in onto.classes if c.l1 in _VRU), None)
    nonvru = next((c for c in onto.classes if c.l1 not in _VRU and c.l1 != "fallback"), None)
    # a same-l1 pair (benign confusion)
    by_l1: dict[str, list] = {}
    for c in onto.classes:
        by_l1.setdefault(c.l1, []).append(c)
    pair = next((v for k, v in by_l1.items() if k not in _VRU and len(v) >= 2), None)
    return onto, vru, nonvru, (pair[0], pair[1]) if pair else None


def test_affinity_cost_unsafe_beats_benign():
    onto, vru, nonvru, pair = _find_classes()
    assert vru and nonvru and pair, "ontology should have VRU, non-VRU, and a same-l1 pair"
    unsafe = affinity_cost(onto, vru.id, nonvru.id)
    benign = affinity_cost(onto, pair[0].id, pair[1].id)
    assert unsafe == 1.0                 # VRU confused with non-VRU is maximally unsafe
    assert benign <= 0.3                 # same superclass is cheap
    assert unsafe > benign


def test_safe_miou_penalizes_unsafe_confusion():
    onto, vru, nonvru, pair = _find_classes()
    # identical mass shape; only the SEMANTICS of the confused pair differ
    matrix = [[90, 10], [0, 90]]
    unsafe = safe_miou(matrix, [vru.id, nonvru.id], onto, safety_weight=2.0)
    benign = safe_miou(matrix, [pair[0].id, pair[1].id], onto, safety_weight=2.0)
    assert 0.0 <= unsafe < benign <= 1.0


def test_safe_miou_perfect_is_one():
    onto = get_ontology()
    ids = [onto.classes[0].id, onto.classes[1].id]
    assert safe_miou([[50, 0], [0, 50]], ids, onto) == 1.0


def test_isotonic_reliability_roundtrip():
    from services.autolabel.isotonic import reliability_report

    # monotone-ish synthetic data; the JSON knots are just (x, y) breakpoints reconstructed via interp
    knot_x = np.array([0.0, 0.5, 1.0])
    knot_y = np.array([0.1, 0.5, 0.95])
    xs = np.linspace(0.05, 0.95, 50)
    ys = (xs > 0.5).astype(float)
    rep = reliability_report(xs, ys, knot_x, knot_y, bins=10)
    assert rep["n"] == 50
    assert rep["ece"] is not None and 0.0 <= rep["ece"] <= 1.0
    # interp reconstruction is exact at the knots
    assert abs(float(np.interp(0.5, knot_x, knot_y)) - 0.5) < 1e-9


def test_calibrate_uses_isotonic_curve(monkeypatch):
    import services.autolabel.isotonic as iso
    from services.autolabel.calibrate import calibrate_confidence

    iso.load_isotonic.cache_clear()
    monkeypatch.setattr(iso, "load_isotonic", lambda uri: ((0.0, 1.0), (0.0, 1.0)))
    cfg = CalibrateSettings(method="isotonic", isotonic_uri="s3://bucket/isotonic.json")
    # identity curve -> calibrated base equals raw_conf; no agreement adjustments
    val = calibrate_confidence(0.7, False, False, False, cfg)
    assert abs(val - 0.7) < 1e-6
    # isotonic output is the empirical P(correct); the agreement bonus must NOT be added on top, or the
    # calibration is corrupted and the gate's 0.95 stops meaning 95% precision (R1.5).
    val2 = calibrate_confidence(0.7, True, False, False, cfg)
    assert abs(val2 - 0.7) < 1e-6


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


@requires_infra
@pytest.mark.asyncio
async def test_seal_gold_is_deterministic():
    from sqlalchemy import func, select

    from db.models import GoldSet, Object
    from db.session import get_sessionmaker
    from services.training.gold import GoldSpec, seal_gold

    async with get_sessionmaker()() as db:
        n_human = (await db.execute(
            select(func.count()).select_from(Object).where(Object.source == "human", Object.state == "accepted")
        )).scalar_one()
    if n_human == 0:
        pytest.skip("no human-accepted objects to seal a gold set")

    spec = GoldSpec(name="test-gold", limit=25)
    r1 = await seal_gold(spec)
    r2 = await seal_gold(spec)
    assert r1["gold_id"] == r2["gold_id"]  # content-addressed -> identical
    async with get_sessionmaker()() as db:
        n = (await db.execute(select(func.count()).select_from(GoldSet).where(GoldSet.gold_id == r1["gold_id"]))).scalar_one()
        assert n == 1  # idempotent insert
