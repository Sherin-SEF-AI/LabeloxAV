"""Per-class label precision, measured rather than inferred from confidence.

Confidence is the obvious proxy for "which classes are wrong" and it is a poor one. It is the detector's
opinion of its own output, so a class the detector is confidently wrong about looks healthy and a class it
is diffidently right about looks broken. `object_fallback` sits at mean confidence 0.806 while meaning
"I do not know what this is", and `traffic_signal` sits at 0.271 across 64,741 objects. Neither number tells
you whether the label is right.

So this asks a judge, per class, and reports the answer with its uncertainty. Everything about the judging
is `services/labelops/vlm_review.py` unchanged: the same "the label says X, is that right?" prompt, the same
first-class `unsure`, the same `MachineVerdict` plane that is deliberately not the human `review` table, and
the same Rogan-Gladen correction through the judge's measured sensitivity and specificity. The only thing
that is new here is which crops get judged.

The sample is **random within the class, never the highest-scoring**. A class's most confident detections are
its best case, and measuring those reports how good the detector is when it is surest, which is not what a
remediation decision needs to know. This mirrors `services/errordetect/judge_detectors.py::sample_candidates`.

Human-reviewed objects are excluded. An object a person already ruled on is not evidence about the machine's
precision, and including them inflates the rate by exactly the amount of review that has happened.

**This is a GPU job and it behaves like one.** It takes the `core/gpu_slot.py` advisory lock for the
duration, so it cannot run beside a training run, a corpus relabel or an autolabel pass - any two of those
on one card is an out-of-memory part way through a batch, which the caller counts as a failed unit rather
than as contention. It yields to a live training job rather than competing with one. And it checks free VRAM
between batches and waits rather than pushing the card into swap, because a judge sweep is worth nothing
compared to the training run it would take down.
"""

from __future__ import annotations

from sqlalchemy import Text, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Object

log = get_logger("labelops.class_precision")

# How much VRAM the judge needs free before another batch starts. The judge model is an 8B at q8, which sits
# around 9 GB resident; this is the headroom that keeps a batch from being the thing that tips the card over.
MIN_FREE_VRAM_MB = 2_000

# Crops per batch, between which the guards are re-checked. Small enough that a training job waiting on the
# card waits seconds rather than the length of a class.
BATCH = 20

# How long to wait when the card is busy before checking again.
_BACKOFF_S = 15.0

# One batch id per class, so `judged_precision` can be asked about a single class and so two classes never
# share a denominator.
BATCH_PREFIX = "class-precision"

# Sources that represent a machine's own opinion. `human` and `vlm_review` are excluded because they carry a
# ruling already; judging them measures the reviewer, not the detector.
_MACHINE_SOURCES = ("fused", "auto_accept", "imported", "interpolated", "propagated", "relabel", "recall")


def batch_id_for(class_name: str) -> str:
    return f"{BATCH_PREFIX}:{class_name}"


async def sample_class(db: AsyncSession, class_id: int, n: int, *, seed: float | None = None,
                       min_side_px: float = 0.0) -> list[Object]:
    """A pseudo-random sample of one class's machine-labelled objects, stable across runs.

    Ordered by a hash of the object id and the seed, not by `random()`. `setseed` plus `ORDER BY random()`
    looks reproducible and is not: random() is evaluated per row in scan order, so the draw survives a
    repeat only while the query plan and the table are unchanged. Both change here. A remediation sweep
    rewrites `class_id` on exactly the rows this selects, so the follow-up measurement would draw a
    different sample and the difference between the two numbers would be partly the fix and partly the
    draw - which is the one thing the before/after comparison exists to avoid. Measured: a re-run with the
    identical seed took `sedan` from 80 stored verdicts to 128, meaning 48 fresh crops.

    A hash of the id is stable against both. An object keeps its position whatever else happens to the
    table, so the second sample is the first one plus or minus only the rows that genuinely left the class.

    `min_side_px` drops crops too small to judge. A 6-pixel box is not a label the judge can rule on, and an
    `unsure` from an unjudgeable crop tells you nothing about the class while still costing a call.
    """
    q = (select(Object)
         .where(Object.class_id == class_id, Object.source.in_(_MACHINE_SOURCES)))
    if min_side_px > 0:
        q = q.where((Object.bbox[3] - Object.bbox[1]) >= min_side_px,
                    (Object.bbox[4] - Object.bbox[2]) >= min_side_px)
    key = f"{seed if seed is not None else 0.0}"
    q = q.order_by(func.md5(func.concat(func.cast(Object.object_id, Text), key))).limit(n)
    return list((await db.execute(q)).scalars().all())


async def free_vram_mb() -> float | None:
    """Free VRAM on the busiest card, or None when there is no GPU to read.

    None is not zero. A box with no nvidia-smi is a box where this check cannot apply, and treating that as
    "no memory free" would stop the sweep on every CPU-only host.
    """
    from services.hardening.resources import gpus

    try:
        cards = gpus()
    except Exception:  # noqa: BLE001 - a reading that fails must not decide the job
        return None
    # Derived from used and total: `gpus()` reports those two and no free figure, and a `memory_free_mb`
    # lookup silently returned None on a box with a working card, which made this guard a no-op that read
    # as "no GPU here".
    vals = [float(c["memory_total_mb"]) - float(c["memory_used_mb"]) for c in cards
            if c.get("memory_total_mb") is not None and c.get("memory_used_mb") is not None]
    return min(vals) if vals else None


async def wait_for_headroom(db: AsyncSession, *, holder: str, max_wait_s: float = 900.0) -> dict:
    """Block until a live training job is done and the card has room, or give up and say so.

    Returns what it waited for rather than raising, so the caller can record why a sweep stalled instead of
    a stack trace that says only that it did.
    """
    import asyncio
    import time

    from services.training.gpu_lease import training_holds_gpu

    started = time.monotonic()
    waited_for: list[str] = []
    while time.monotonic() - started < max_wait_s:
        if await training_holds_gpu(db):
            if "training" not in waited_for:
                log.info("class_precision.yielding", holder=holder, to="training")
                waited_for.append("training")
            await asyncio.sleep(_BACKOFF_S)
            continue
        free = await free_vram_mb()
        if free is not None and free < MIN_FREE_VRAM_MB:
            if "vram" not in waited_for:
                log.info("class_precision.yielding", holder=holder, free_mb=free,
                         need_mb=MIN_FREE_VRAM_MB)
                waited_for.append("vram")
            await asyncio.sleep(_BACKOFF_S)
            continue
        return {"ok": True, "waited_for": waited_for, "waited_s": round(time.monotonic() - started, 1)}
    return {"ok": False, "waited_for": waited_for, "waited_s": round(time.monotonic() - started, 1)}


async def judge_class(db: AsyncSession, class_name: str, *, n: int = 120, seed: float | None = 0.42,
                      min_side_px: float = 12.0, client=None, model_version: str | None = None,
                      batch: int = BATCH, take_slot: bool = True) -> dict:
    """Judge a random sample of one class and record the verdicts. Returns the judging summary.

    Runs in batches of `batch` crops, re-checking the card between them, so a training job that starts
    mid-class waits seconds rather than the length of the class. Verdicts upsert, so a sweep that stops
    early has still banked everything it judged and a re-run resumes rather than restarting.

    Reporting is deliberately not done here: `services/labelops/vlm_review.py::judged_precision` already
    turns verdicts into a raw Wilson interval and a corrected one, and already refuses to correct when the
    judge is unmeasured. A second precision calculation here would be a second thing to keep honest.
    """
    import contextlib

    from core.gpu_slot import gpu_slot
    from services.autolabel.ontology import get_ontology
    from services.labelops.vlm_review import judge_objects

    onto = get_ontology()
    if not onto.has_name(class_name):
        raise ValueError(f"unknown class '{class_name}'")
    cid = onto.by_name(class_name).id

    objects = await sample_class(db, cid, n, seed=seed, min_side_px=min_side_px)
    if not objects:
        return {"class_name": class_name, "class_id": cid, "judged": 0,
                "skipped_reason": "no machine-labelled objects of this class are large enough to judge"}

    holder = f"class_precision:{class_name}"
    # `take_slot=False` is for a caller already holding the slot for a whole sweep, so a fifteen-class run
    # does not release and re-acquire the card between every class and hand it to a waiting job mid-sweep.
    slot = gpu_slot(holder, timeout_s=None) if take_slot else contextlib.nullcontext()

    totals = {"judged": 0, "skipped": 0, "unreadable": 0, "failed": 0}
    counts: dict[str, int] = {}
    stalled = None
    async with slot:
        for i in range(0, len(objects), max(1, batch)):
            head = await wait_for_headroom(db, holder=holder)
            if not head["ok"]:
                stalled = f"gave up after {head['waited_s']}s waiting for {', '.join(head['waited_for'])}"
                log.warning("class_precision.stalled", holder=holder, reason=stalled)
                break
            chunk = objects[i:i + max(1, batch)]
            out = await judge_objects(db, chunk, batch_id_for(class_name), client=client,
                                      model_version=model_version)
            for k in totals:
                totals[k] += out.get(k, 0) or 0
            for k, v in (out.get("by_verdict") or {}).items():
                counts[k] = counts.get(k, 0) + v

    res = {"class_name": class_name, "class_id": cid, "sampled": len(objects),
           "by_verdict": counts, **totals}
    if stalled:
        res["stalled"] = stalled
    log.info("class_precision.judged", **{k: v for k, v in res.items() if k != "by_verdict"})
    return res


async def class_targets(db: AsyncSession, *, min_objects: int = 10_000, limit: int = 20) -> list[dict]:
    """The classes worth measuring: the biggest ones, because that is where being wrong costs most.

    Volume rather than suspicion, on purpose. Picking the classes that already look bad measures a
    hypothesis instead of testing it, and a class that is large and quietly wrong is the expensive case.
    """
    rows = (await db.execute(
        select(Object.class_id, func.count(Object.object_id).label("n"),
               func.avg(Object.conf).label("conf"))
        .where(Object.source.in_(_MACHINE_SOURCES))
        .group_by(Object.class_id)
        .having(func.count(Object.object_id) >= min_objects)
        .order_by(func.count(Object.object_id).desc())
        .limit(limit))).all()

    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    out = []
    for cid, n, conf in rows:
        try:
            name = onto.by_id(int(cid)).name
        except KeyError:
            continue
        out.append({"class_id": int(cid), "class_name": name, "n": int(n),
                    "mean_conf": round(float(conf), 3)})
    return out
