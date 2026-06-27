"""Gate B, part (c): isotonic calibration so a calibrated 0.95 actually means ~95% precision.

Training signal: auto-labeled objects a human then reviewed. raw_conf is the auto score; "correct" is
whether the human accepted that auto label (state accepted/auto_accept) versus rejected it. We fit a
monotone isotonic curve raw_conf -> P(correct) and persist it as JSON KNOTS (x/y breakpoints), not a
pickled sklearn estimator, so serve-time reconstruction is just np.interp and survives sklearn version
bumps. reliability_report() reports ECE so the calibration quality is itself a number.

    python -m services.autolabel.isotonic --gold <gold_id>
"""

from __future__ import annotations

import asyncio
import json
from functools import lru_cache

import click
import numpy as np
from sqlalchemy import select

from core.config import get_settings
from core.logging import get_logger, setup_logging
from core.storage import get_object_store
from db.models import Object
from db.models import Session as DbSession
from db.session import get_sessionmaker
from db.models import Frame

log = get_logger("isotonic")

_CORRECT = {"accepted", "auto_accept"}
_INCORRECT = {"rejected"}


async def _collect_pairs(session_id: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    maker = get_sessionmaker()
    async with maker() as db:
        stmt = select(Object.provenance, Object.state).where(
            Object.source.in_(["fused", "auto_accept"]),
            Object.state.in_(list(_CORRECT | _INCORRECT)),
        )
        if session_id:
            from uuid import UUID

            stmt = stmt.join(Frame, Object.frame_id == Frame.frame_id).where(Frame.session_id == UUID(session_id))
        rows = (await db.execute(stmt)).all()

    xs, ys = [], []
    for prov, state in rows:
        rc = (prov or {}).get("raw_conf")
        if rc is None:
            continue
        xs.append(float(rc))
        ys.append(1.0 if state in _CORRECT else 0.0)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def reliability_report(xs: np.ndarray, ys: np.ndarray, knot_x: np.ndarray, knot_y: np.ndarray, bins: int = 10) -> dict:
    """Expected Calibration Error of the fitted curve against the held data."""
    if len(xs) == 0:
        return {"ece": None, "n": 0, "bins": []}
    cal = np.interp(xs, knot_x, knot_y)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    out_bins = []
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        m = (cal >= lo) & (cal < hi if b < bins - 1 else cal <= hi)
        n = int(m.sum())
        if n == 0:
            out_bins.append({"lo": round(lo, 2), "hi": round(hi, 2), "n": 0, "conf": None, "acc": None})
            continue
        conf = float(cal[m].mean())
        acc = float(ys[m].mean())
        ece += (n / len(xs)) * abs(acc - conf)
        out_bins.append({"lo": round(lo, 2), "hi": round(hi, 2), "n": n, "conf": round(conf, 3), "acc": round(acc, 3)})
    return {"ece": round(ece, 4), "n": int(len(xs)), "bins": out_bins}


async def fit_isotonic(gold_id: str | None = None, session_id: str | None = None) -> dict:
    from sklearn.isotonic import IsotonicRegression

    xs, ys = await _collect_pairs(session_id)
    if len(xs) < 10:
        raise RuntimeError(f"need >=10 reviewed auto-labels to fit isotonic, found {len(xs)}")

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(xs, ys)
    knot_x = np.asarray(iso.X_thresholds_, dtype=float)
    knot_y = np.asarray(iso.y_thresholds_, dtype=float)
    report = reliability_report(xs, ys, knot_x, knot_y, get_settings().m9.ece_bins)

    payload = {
        "kind": "isotonic-knots-v1",
        "gold_id": gold_id,
        "x": [round(float(v), 6) for v in knot_x],
        "y": [round(float(v), 6) for v in knot_y],
        "n_train": int(len(xs)),
        "ece": report["ece"],
    }
    key = f"calibration/{gold_id or 'corpus'}/isotonic.json"
    uri = get_object_store().put_bytes(key, json.dumps(payload).encode(), "application/json")
    log.info("isotonic.fitted", uri=uri, n=len(xs), ece=report["ece"], knots=len(knot_x))
    return {"uri": uri, "n_train": int(len(xs)), "report": report, "knots": len(knot_x)}


@lru_cache(maxsize=8)
def load_isotonic(uri: str) -> tuple[tuple, tuple]:
    """Load (x_knots, y_knots) from a fitted curve. Cached; sync (boto3) so the live gate can use it."""
    data = json.loads(get_object_store().get_bytes(uri).decode())
    return tuple(data["x"]), tuple(data["y"])


def apply_isotonic(uri: str, raw_conf: float) -> float:
    x, y = load_isotonic(uri)
    return float(np.interp(raw_conf, np.asarray(x), np.asarray(y)))


@click.command()
@click.option("--gold", "gold_id", default=None)
@click.option("--session", "session_id", default=None)
def main(gold_id, session_id) -> None:
    setup_logging(get_settings().log_level)
    click.echo(asyncio.run(fit_isotonic(gold_id, session_id)))


if __name__ == "__main__":
    main()
