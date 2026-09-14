"""What the next label of each class is worth, per rupee, so a labelling budget can be spent.

The engine can already say which classes the champion gate is blocked on and roughly how many verdicts a
settlement lot still needs. It cannot say what any of that costs, so every allocation has been made on
counts alone: a class needing 500 verdicts looks twice as expensive as one needing 250, whatever either is
worth and however long its crops actually take a person to judge.

The calculation is deliberately three separate measured quantities rather than one score.

**Deficit** is how far below its recall floor the gate says a class is. It comes from
`gate_signals.recall_demands` on the latest blocked run, which is the same signal gate-directed labelling
already acts on, so this ranks the work that is already being created rather than a parallel notion of
importance.

**Minutes** come from `Review.time_spent_ms` on real reviews of that class, as a median. Not a mean: a
reviewer who left a tab open for an hour would otherwise set the price of the class. Below
`MIN_TIMED_REVIEWS` the class is reported unmeasured with the count, because a median of four timings is
not a rate.

**Rupees** come from the workforce's own entered rate. There is no default rate, and a class whose work
has no priced workforce is reported unmeasured rather than valued at an invented number.

Any of the three missing makes the row unmeasured with a reason. That is the whole discipline here: a
ranking that silently fills gaps with medians and defaults produces a confident order over classes nobody
has measured, and a budget spent from it is a budget spent on arithmetic.
"""

from __future__ import annotations

import statistics

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.timebase import now_ns

log = get_logger("label_value")

# Timed reviews of a class before its median is a rate rather than an anecdote.
MIN_TIMED_REVIEWS = 20
# A review recorded as taking longer than this was a tab left open, not a judgement.
MAX_PLAUSIBLE_MS = 10 * 60 * 1000
# And one under this was a misclick that landed on the shortcut key.
MIN_PLAUSIBLE_MS = 300
# How much of the deficit one more label is assumed to close, before diminishing returns. Stated as a
# constant rather than modelled because nothing in this corpus measures a learning curve per class; it
# cancels out of the ranking, which is what the ranking is for, and it is named so nobody reads the
# absolute value as a prediction.
GAIN_PER_LABEL = 0.001


async def timed_minutes_per_label(db: AsyncSession, class_id: int) -> dict:
    """The median minutes a review of this class takes, or why that cannot be said.

    Implausible timings are dropped before the median rather than after: a tab left open for an hour and
    a misclick are not slow and fast reviews, they are not reviews.
    """
    from db.models import Object, Review

    rows = (await db.execute(
        select(Review.time_spent_ms)
        .join(Object, Object.object_id == Review.object_id)
        .where(Object.class_id == class_id, Review.time_spent_ms.isnot(None),
               Review.time_spent_ms >= MIN_PLAUSIBLE_MS,
               Review.time_spent_ms <= MAX_PLAUSIBLE_MS))).scalars().all()
    n = len(rows)
    if n < MIN_TIMED_REVIEWS:
        return {"measured": False, "n": n,
                "reason": (f"only {n} plausibly timed reviews of this class; "
                           f"{MIN_TIMED_REVIEWS} is the floor for a median to be a rate")}
    return {"measured": True, "n": n,
            "minutes": round(statistics.median(rows) / 60000.0, 4)}


async def workforce_rate(db: AsyncSession) -> dict:
    """The median entered rupees per verdict across active workforces, or why there is none.

    Median across workforces rather than one chosen: a deployment with two vendors at different rates has
    no single price, and picking either would make the ranking depend on which row came back first.
    """
    from db.models import Workforce

    rows = (await db.execute(
        select(Workforce.rate_inr_per_verdict).where(
            Workforce.active.is_(True), Workforce.rate_inr_per_verdict.isnot(None),
            Workforce.rate_inr_per_verdict > 0))).scalars().all()
    if not rows:
        return {"measured": False,
                "reason": ("no active workforce has a rate entered; a label's cost cannot be computed "
                           "from an assumed rate without the result looking measured")}
    return {"measured": True, "inr_per_verdict": round(statistics.median(rows), 4),
            "workforces": len(rows)}


async def marginal_value(db: AsyncSession, *, run_id: str | None = None) -> dict:
    """Every class the gate is short on, ranked by recall gained per rupee, with the unmeasured said so.

    Returns the ranked rows and the inputs they were computed from, so a reader can see which of the
    three quantities was missing when a class comes back unranked.
    """
    from services.autolabel.ontology import get_ontology
    from services.flywheel.gate_signals import demands_for_run, latest_blocked_run

    onto = get_ontology()
    rid = run_id or await latest_blocked_run(db)
    if rid is None:
        return {"measured": False, "rows": [],
                "reason": "no training run is currently blocked, so the gate is asking for no labels"}
    try:
        diag = await demands_for_run(db, str(rid))
    except ValueError as exc:
        return {"measured": False, "rows": [], "reason": str(exc)}

    demands = diag.get("demands") or []
    if not demands:
        return {"measured": False, "rows": [], "run_id": str(rid),
                "reason": "recall is not what is blocking this run, so no class has a deficit to close"}

    rate = await workforce_rate(db)
    rows = []
    for d in demands:
        name = d.get("class_name") or d.get("slice")
        if not onto.has_name(name):
            continue
        cid = onto.by_name(name).id
        deficit = float(d.get("deficit") or 0.0)
        timing = await timed_minutes_per_label(db, cid)

        row = {"class_id": cid, "class_name": name, "deficit": round(deficit, 4),
               "n_timed_reviews": timing["n"], "minutes_per_label": timing.get("minutes"),
               "inr_per_label": None, "expected_recall_gain": None, "value_per_inr": None,
               "measured": False, "reason": None}
        if not timing["measured"]:
            row["reason"] = timing["reason"]
        elif not rate["measured"]:
            row["reason"] = rate["reason"]
        else:
            # Rupees per label from the verdict rate; the minutes are what make two classes at the same
            # rate cost differently, which is the whole reason the timing is measured per class.
            inr = float(rate["inr_per_verdict"])
            gain = min(deficit, GAIN_PER_LABEL)
            row.update({"inr_per_label": round(inr, 4),
                        "expected_recall_gain": round(gain, 6),
                        # Per rupee and per minute both matter, and the minute is the scarce one: the
                        # rate is the same for every class and the time is not, so dividing by both is
                        # what makes a slow class rank below a fast one at equal deficit.
                        "value_per_inr": round(gain / (inr * max(timing["minutes"], 1e-6)), 8),
                        "measured": True})
        rows.append(row)

    rows.sort(key=lambda r: (-(r["value_per_inr"] or -1), -r["deficit"]))
    measured = [r for r in rows if r["measured"]]
    return {"measured": bool(measured), "run_id": str(rid), "rows": rows,
            "n_measured": len(measured), "n_unmeasured": len(rows) - len(measured),
            "rate": rate,
            "reason": (None if measured else
                       "every class the gate is short on is missing a timing or a rate, so none can be "
                       "priced; the deficits are still listed in deficit order")}


async def snapshot(db: AsyncSession, *, run_id: str | None = None) -> dict:
    """Record the current ranking, so a decision made today stays explainable against today's inputs."""
    from db.models import LabelValueSnapshot

    res = await marginal_value(db, run_id=run_id)
    ts = now_ns()
    for r in res["rows"]:
        db.add(LabelValueSnapshot(
            class_id=r["class_id"], ts_ns=ts, model_run_id=res.get("run_id"),
            deficit=r["deficit"], minutes_per_label=r["minutes_per_label"],
            inr_per_label=r["inr_per_label"], expected_recall_gain=r["expected_recall_gain"],
            value_per_inr=r["value_per_inr"], measured=r["measured"],
            # The CHECK requires a reason on an unmeasured row, which is the constraint doing the same
            # job this module does: a null value with no reason is a zero waiting to be misread.
            reason=r["reason"], n_timed_reviews=r["n_timed_reviews"]))
    await db.commit()
    log.info("label_value.snapshot", rows=len(res["rows"]), measured=res.get("n_measured", 0))
    return {**res, "snapshot_ts_ns": ts}
