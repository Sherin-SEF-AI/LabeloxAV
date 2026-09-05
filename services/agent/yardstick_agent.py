"""Keep the yardsticks measurable: reseal rotted gold sets, and keep unfinished blind audits visible.

Two artifacts hold the ground truth every promotion decision is compared against, and both fail in
silence. A gold set is a frozen list of object ids, so a re-import or corpus rebuild deletes its
objects out from under it and it goes on claiming a size it no longer has - this deployment carried
five in that state, one listing 171 objects of which zero survive. A blind audit is the only recall
denominator the model did not build, and it does nothing at all until a person labels its frames; the
seeded audit has waited since 2026-08 because nothing kept asking.

Neither repair invents data. Resealing runs the set's own stored spec over the CURRENT human-accepted
corpus and produces a new content-addressed gold_id; the rotted set keeps its id and its history, and
every metric already attached to it stays attached to what it was measured against. The audit hook
writes nothing but a superseding notification: the labeling is precisely the thing a machine must not
do here, so the only honest automation is refusing to let it be forgotten.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger

log = get_logger("agent.yardstick")

GOLD_KIND = "gold_repair"
AUDIT_NUDGE_KIND = "blind_audit_nudge"

# A set that has lost more than this fraction of its objects grades against a sample nobody chose;
# below it, the survivors are still the sealed population minus noise.
ROT_FRACTION = 0.1
# Resealing materializes a dataset (frame images to local disk), so a night repairs at most this many.
REPAIRS_PER_NIGHT = 2


async def maybe_repair_gold(db: AsyncSession) -> dict:
    """Off-hours hook, weekly: reseal every gold set that has rotted past the threshold."""
    from services.agent.runtime.report import launch, ran_since
    from services.analytics.quality import list_gold_sets

    if await ran_since(db, GOLD_KIND, datetime.now(UTC) - timedelta(days=7)):
        return {"ran": False, "reason": "checked within the last 7 days"}

    sets = await list_gold_sets()
    rotted = [g for g in sets
              if g["n_objects"] and (not g["usable"]
                                     or g["n_missing"] / max(1, g["n_objects"]) > ROT_FRACTION)]
    if not rotted:
        return {"ran": False, "reason": f"all {len(sets)} gold sets are within the rot threshold"}

    async def worker(run_id):
        from db.session import get_sessionmaker
        from services.agent.runtime.report import finish_run as _finish
        from services.training.gold import GoldSpec, seal_gold

        report: dict = {"rotted": [{k: g[k] for k in ("gold_id", "name", "n_objects", "n_alive")}
                                   for g in rotted],
                        "resealed": [], "refused": []}
        status = "committed"
        try:
            for g in rotted[:REPAIRS_PER_NIGHT]:
                async with get_sessionmaker()() as wdb:
                    from db.models import GoldSet

                    row = await wdb.get(GoldSet, g["gold_id"])
                    spec = dict(row.spec or {}) if row else {}
                if not spec:
                    report["refused"].append({"gold_id": g["gold_id"],
                                              "reason": "no stored spec; this set cannot be resealed "
                                                        "without a person restating what it measures"})
                    continue
                try:
                    sealed = await seal_gold(GoldSpec(**spec))
                    report["resealed"].append({"old": g["gold_id"], "new": sealed["gold_id"],
                                               "n_objects": sealed["n_objects"],
                                               "n_frames": sealed["n_frames"]})
                except Exception as exc:  # noqa: BLE001 - one refusal, recorded, not a dead night
                    report["refused"].append({"gold_id": g["gold_id"], "reason": str(exc)[:200]})
            if len(rotted) > REPAIRS_PER_NIGHT:
                report["deferred"] = [g["gold_id"] for g in rotted[REPAIRS_PER_NIGHT:]]

            if report["resealed"] or report["refused"]:
                async with get_sessionmaker()() as wdb:
                    from services.notify import notify

                    n_ok, n_no = len(report["resealed"]), len(report["refused"])
                    await notify(wdb, kind="gold_repair", severity="warn" if n_no else "info",
                                 title=f"gold repair: {n_ok} resealed, {n_no} refused",
                                 body="; ".join(f"{r['old'][:12]} -> {r['new'][:12]} "
                                                f"({r['n_objects']} objects)"
                                                for r in report["resealed"]) or None,
                                 href="/quality", subject_type="gold_repair",
                                 subject_id=datetime.now(UTC).date().isoformat(),
                                 meta=report)
        except Exception as exc:  # noqa: BLE001
            status = "error"
            report["error"] = str(exc)[:400]
            log.error("yardstick.gold_repair_failed", error=str(exc))
        await _finish(run_id, status=status, report=report)

    return {"ran": True, "rotted": len(rotted),
            **(await launch(db, GOLD_KIND, worker, created_by="scheduler"))}


async def maybe_nudge_blind_audit(db: AsyncSession) -> dict:
    """Daily hook: one superseding notification per blind audit still waiting on human labels.

    States the cost and what it buys, because a queue item without either is easy to defer forever:
    ~a minute of from-scratch labeling per frame, and the only recall denominator the model did not
    build - threshold_fit's own caveat says its FAR fits are untrustworthy until this exists.
    """
    from services.agent.runtime.report import ran_since
    from services.verdyx.blind_audit import list_audits

    day_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    if await ran_since(db, AUDIT_NUDGE_KIND, day_start):
        return {"ran": False, "reason": "already nudged today"}

    audits = [a for a in await list_audits(db)
              if a["status"] not in ("scored", "cancelled") and a["n_labeled"] < a["n_frames"]]
    if not audits:
        return {"ran": False, "reason": "no blind audit is waiting on labels"}

    from services.agent.runtime.report import finish_run, launch

    async def worker(run_id):
        from db.session import get_sessionmaker
        from services.notify import notify

        async with get_sessionmaker()() as wdb:
            for a in audits:
                remaining = a["n_frames"] - a["n_labeled"]
                await notify(
                    wdb, kind="blind_audit_pending", severity="info",
                    title=f"blind audit waiting: {remaining} of {a['n_frames']} frames unlabeled",
                    body=(f"About {remaining} minutes of from-scratch labeling. It buys the only "
                          "recall denominator the model did not build; per-class FAR fits stay "
                          "untrustworthy until it exists."),
                    href=f"/verdyx?audit={a['audit_id']}", subject_type="blind_audit",
                    subject_id=a["audit_id"],
                    meta={"n_frames": a["n_frames"], "n_labeled": a["n_labeled"],
                          "job_id": a["job_id"], "run_id": a["run_id"]})
        await finish_run(run_id, status="committed",
                         report={"audits": [{k: a[k] for k in ("audit_id", "n_frames", "n_labeled")}
                                            for a in audits]})

    return {"ran": True, "waiting": len(audits),
            **(await launch(db, AUDIT_NUDGE_KIND, worker, created_by="scheduler"))}
