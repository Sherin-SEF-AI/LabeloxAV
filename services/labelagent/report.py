"""The benchmark as a report, and the loop that keeps the agent honest with it.

`build_report` runs stages against the answer key and returns the JSON the Agent Console shows; the script
(scripts/label_benchmark.py) and the nightly job both call it, so there is one definition of every number.

The loop is the part that matters once the agent is switched on. The rulings the answer key is built from
keep arriving, the probe refits on them, and memory changes with every correction, so the measurement that
justified letting the agent fix classes goes stale. Two rules close that:

- **Fixes are earned, not configured.** `fixes_allowed` reads the latest saved report, and the agent may
  change a class only while its own measured wrong-fix rate, at the 95% upper bound, is under
  `max_wrong_fix`. With no report, or a worse one, every fix becomes a question. Flags and questions never
  change a class and are not gated.
- **The report refreshes itself.** `maybe_benchmark` queues a fresh report outside office hours once enough
  new rulings have arrived (or the last one is a week old), so the gate tracks the agent as it is.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta

from core.logging import get_logger

log = get_logger("labelagent.report")

LATEST_KEY = "labelagent/benchmarks/latest.json"

# What a nightly refresh runs: everything that needs no image and no VLM.
NIGHTLY_STAGES = ("detector", "vlm_stored", "reasoner", "memory", "probe", "family_zs", "agent")

# Stages that learn nothing from the item they are graded on (memory and the probe are cross-fitted), so
# they are also scored on the whole set, which is what makes the wrong-fix bound tight enough to use.
UNTRAINED = {"detector", "vlm_stored", "reasoner", "classify_crop", "memory", "probe", "zs_all_stored",
             "family_zs", "agent", "agent_vlm"}
CROP_FLOORS = (0.0, 0.5, 0.8)
MARGIN_FLOORS = (0.0, 0.3, 0.5, 0.7)


async def _run_stage(db, items, s: str, *, vlm_max: int | None = None) -> dict:
    from services.labelagent import benchmark as B

    if s == "detector":
        return {"detector": B.stage_detector(items)}
    if s == "vlm_stored":
        return {"vlm_stored": await B.stage_vlm_stored(db, items)}
    if s == "reasoner":
        return {"reasoner": await B.stage_reasoner(db, items)}
    if s in ("memory", "probe"):
        raw = await (B.stage_memory(db, items) if s == "memory" else B.stage_probe(db, items))
        return {f"{s}@{fl:g}": B.with_floor(raw, fl) for fl in MARGIN_FLOORS}
    if s == "zs_all_stored":
        raw = await B.stage_zs_all_stored(db, items)
        return {f"zs_all@{fl:g}": B.with_floor(raw, fl) for fl in CROP_FLOORS}
    if s == "family_zs":
        raw = await B.stage_family_zs(db, items)
        return {f"family_zs@{fl:g}": B.with_floor(raw, fl) for fl in MARGIN_FLOORS}
    if s == "agent":
        return await B.stage_agent(db, items)
    if s == "agent_vlm":
        return await B.stage_agent_vlm(db, items, max_items=vlm_max)
    if s == "classify_crop":
        raw = await B.stage_classify_crop(db, items)
        return {f"classify_crop@{fl:g}": B.with_floor(raw, fl) for fl in CROP_FLOORS}
    raise ValueError(f"unknown stage {s!r}")


async def build_report(db, stages, *, split: str = "test", vlm_max: int | None = None, on_row=None) -> dict:
    """Run `stages` and return the report. `on_row(name, split, rep)` is called per scored row, for a
    caller that prints as it goes."""
    from services.autolabel.ontology import get_ontology
    from services.labelagent import benchmark as B
    from services.labelagent.truth import load_all, summary

    onto = get_ontology()
    items = await load_all(db)
    report: dict = {"created": datetime.now(UTC).isoformat(timespec="seconds"), "split": split,
                    "truth": summary(items), "truth_test": summary([i for i in items if i.split == "test"]),
                    "stages": {}}
    for s in stages:
        t0 = time.perf_counter()
        runs = await _run_stage(db, items, s, vlm_max=vlm_max)
        dt = round(time.perf_counter() - t0, 1)
        for name, preds in runs.items():
            entry: dict = {"seconds": dt}
            for sp in [split] + (["all"] if s in UNTRAINED and split != "all" else []):
                rep = B.score(items, preds, onto, split=sp)
                rep["coverage"] = B.coverage([p for p, i in zip(preds, items, strict=True)
                                              if sp == "all" or i.split == sp])
                entry[sp] = rep
                if on_row:
                    on_row(name, sp, rep)
            report["stages"][name] = entry
    return report


def save_report(report: dict) -> str:
    from core.storage import get_object_store

    store = get_object_store()
    blob = json.dumps(report, indent=1).encode()
    key = f"labelagent/benchmarks/{report['created'].replace(':', '')}.json"
    store.put_bytes(key, blob, "application/json")
    store.put_bytes(LATEST_KEY, blob, "application/json")
    _gate_cache.clear()
    return store.uri(key)


def load_latest() -> dict | None:
    from core.storage import get_object_store

    store = get_object_store()
    if not store.exists(LATEST_KEY):
        return None
    return json.loads(store.get_bytes(LATEST_KEY))


# ------------------------------------------------------------------------------------------------ the gate

_gate_cache: dict[str, tuple[float, tuple[bool, str]]] = {}


def gate_from(report: dict | None, max_wrong_fix: float) -> tuple[bool, str]:
    """Whether a report licenses fixes, and why. Pure, so the rule is testable without a store."""
    if not report:
        return False, "no benchmark has been saved, so the agent only asks"
    entry = (report.get("stages") or {}).get("agent") or {}
    rep = entry.get("all") or entry.get("test")
    upper = ((rep or {}).get("pooled") or {}).get("wrong_fix_upper95")
    if upper is None:
        return False, "the saved benchmark has no score for the agent, so it only asks"
    if upper > max_wrong_fix:
        return False, (f"the agent's measured wrong-fix rate could be as high as {upper:.1%}, over the "
                       f"{max_wrong_fix:.1%} limit, so it only asks")
    return True, f"measured wrong-fix rate at most {upper:.1%} (limit {max_wrong_fix:.1%})"


def fixes_allowed(policy) -> tuple[bool, str]:
    """`gate_from` on the latest saved report, cached for ten minutes."""
    hit = _gate_cache.get("g")
    if hit and time.monotonic() - hit[0] < 600:
        return hit[1]
    try:
        verdict = gate_from(load_latest(), policy.max_wrong_fix)
    except Exception as exc:  # noqa: BLE001 - an unreadable report licenses nothing
        log.warning("labelagent.gate_unreadable", error=str(exc))
        verdict = (False, "the benchmark could not be read, so the agent only asks")
    _gate_cache["g"] = (time.monotonic(), verdict)
    return verdict


# ------------------------------------------------------------------------------------------------ the loop

async def maybe_benchmark(db, *, now: datetime | None = None) -> dict:
    """Queue a fresh report when one is due: the agent is on, there is no report, the last one is a week
    old, or `refresh_after_rulings` new human rulings have arrived since it was made. One per day at most."""
    from sqlalchemy import func, select

    from core.config import get_settings
    from db.models import ToolJob
    from services.tools.registry import enqueue

    policy = get_settings().labelagent
    if not policy.enabled:
        return {"ran": False, "reason": "the label agent is off"}
    now = now or datetime.now(UTC)
    recent = (await db.execute(select(func.count(ToolJob.job_id)).where(
        ToolJob.kind == "labelagent_benchmark", ToolJob.created_at >= now - timedelta(hours=20)))).scalar() or 0
    if recent:
        return {"ran": False, "reason": "a report was queued in the last 20 hours"}

    report = load_latest()
    due, why = report is None, "no report yet"
    if report is not None:
        made = datetime.fromisoformat(report["created"])
        from services.labelagent.agent import _stamp

        rulings_now = (await _stamp(db))[0]
        then = int(report.get("rulings_at") or 0)
        if now - made > timedelta(days=7):
            due, why = True, "the last report is a week old"
        elif rulings_now - then >= policy.refresh_after_rulings:
            due, why = True, f"{rulings_now - then} new rulings since the last report"
    if not due:
        return {"ran": False, "reason": "the report is current"}
    job = await enqueue(db, "labelagent_benchmark", {})
    log.info("labelagent.benchmark_queued", job_id=str(job.job_id), reason=why)
    return {"ran": True, "job_id": str(job.job_id), "reason": why}
