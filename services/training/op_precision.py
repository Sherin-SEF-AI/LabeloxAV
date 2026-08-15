"""How often a batch operation was right, measured rather than assumed.

Every agent operation reports volume: batch-fix says it relabelled 40 objects, fit-3D says it attached 12
boxes. None of them says how many were correct, so the tool is fast and cannot show it is right. That is the
gap this closes, and it closes it from data already on disk rather than by asking for a new labelling effort.

The measurement rests on two records that already exist. `AgentRun.changes` names every object an operation
touched, and `Review` is an append-only log of what a human afterwards decided about an object. Joining them
on object id, restricted to reviews that happened after the run, gives the operation's outcomes: a human who
confirmed what it did is a hit, one who rejected or reclassified it is a miss.

Three things this deliberately does not do.

It does not fabricate a number when the evidence is thin. Below MIN_SAMPLES the answer is None, which the
API turns into a 404 and the UI turns into an unmeasured state that forces a dry run. An operation nobody
has checked is not an operation with a good score.

It does not treat silence as success. An object a human never looked at counts toward neither side. Counting
untouched objects as correct is how an automation measures itself at 100% precision by being ignored.

It does not hide that review is a biased sample. Objects reach a human because something drew attention to
them, so this is precision over reviewed outcomes, not over the corpus. The base rate is reported alongside
so a reader can see whether the operation beat what the queue would have done anyway, which is the same
framing `services/autolabel/reasoner/attribution.py` established for the reasoner checks.
"""

from __future__ import annotations

from core.logging import get_logger

log = get_logger("op_precision")

# Human review actions that say the operation got it right, and the ones that say it got it wrong.
#
# `confirm` and `accept` are endorsements. `reject`, `reclassify` and `error_fix` are corrections: the human
# had to undo or change what the operation did. `adjust_geometry` is deliberately in neither set, because a
# nudged box is usually a right answer with an imprecise outline rather than a wrong one, and scoring it as
# a miss would make every geometry operation look worse than it is.
ENDORSING_ACTIONS = frozenset({"confirm", "accept"})
CORRECTING_ACTIONS = frozenset({"reject", "reclassify", "reclassify_track", "error_fix"})

# Below this many reviewed outcomes there is no measurement, only an anecdote. Matches the floor
# `attribution.py` uses for the reasoner checks so the two surfaces cannot disagree about what counts as
# enough evidence.
MIN_SAMPLES = 25

# The operation kinds worth measuring separately, keyed by the `AgentRun.kind` each writes.
OPERATION_KINDS = (
    "frame",          # the batch dry-run and commit over a whole frame
    "cuboid",         # fit 3D boxes
    "attribute",      # fill attributes
    "relabel",        # relabel this frame
    "copilot_batch",  # batch-fix N similar
    "propagate",      # auto-track selected object
    "crosscam",       # propagate to other cameras
    "ops_agent",      # the corpus-level operations agent
    "overnight_auditor",
)


# asyncpg refuses a statement carrying more than 32,767 bound parameters, so an IN clause over the objects a
# kind has touched cannot be sent in one go. `relabel` alone reaches that: the corpus pass left 34,067
# committed child runs, one per frame, and this endpoint began returning 500 the moment it finished. Chunked
# well under the ceiling rather than at it, because the same statement carries a handful of other parameters
# and a limit hit in production is a 500, not a slow query.
ID_CHUNK = 8_000

# How far back a score looks, in days.
#
# An operation's code changes and its old behaviour should not follow it forever. Thirty days is short enough
# that a fix can prove itself within a working month and long enough to clear MIN_SAMPLES on an operation
# anybody actually uses. It is a rolling window rather than a marker at the last code change because nothing
# in the database records when an operation's behaviour changed, and inventing that link would be a guess
# dressed as a measurement.
WINDOW_DAYS = 30


def _chunks(items: list, size: int) -> list[list]:
    """Split a list into batches of at most `size`, preserving order."""
    return [items[i:i + size] for i in range(0, len(items), size)] if items else []


async def measure_operation(db, op_kind: str, *, since_ns: int | None = None,
                            runs_since_ns: int | None = None,
                            id_chunk: int = ID_CHUNK) -> dict:
    """Precision for one operation kind, or an explicit statement that it is not measurable yet.

    `runs_since_ns` bounds which runs are scored, and it is the window that matters. An operation's score is
    meant to answer "how right is this as it now behaves", and a run is the only record of which behaviour
    produced an object. Without the bound a fault that has since been fixed is scored forever: `relabel`
    moved 1,047 buses into a bus shelter in August, the ontology guard that makes that impossible landed
    afterwards, and the 63 corrections a human made while cleaning it up sat in the denominator with no way
    for the repaired operation to ever outvote them.

    `since_ns` bounds the reviews instead, and does not solve that: the corrections to an old run arrive
    whenever somebody gets round to cleaning up, which is exactly when a review window would count them.
    """
    from sqlalchemy import select

    from db.models import AgentRun, Review

    all_runs = (await db.execute(
        select(AgentRun).where(AgentRun.kind == op_kind, AgentRun.status == "committed"))).scalars().all()

    def _started(run) -> int:
        return int((run.created_at.timestamp() if run.created_at else 0) * 1e9)

    runs = all_runs if runs_since_ns is None else [r for r in all_runs if _started(r) >= runs_since_ns]
    excluded = len(all_runs) - len(runs)

    # A reverted run is excluded entirely rather than counted as a miss. Reverting is a human saying "undo
    # all of this", which is a verdict on the run, not a per-object review, and mixing the two would
    # double-count the same judgement.
    #
    # Runs outside the window are dropped before this, so "earliest run that touched an object" means
    # earliest among the runs being scored. An object an old run and a new run both touched carries the new
    # run's state, so the human is ruling on the new one.
    touched: dict[str, int] = {}
    for run in runs:
        started = _started(run)
        for object_id in (run.changes or {}):
            # Keep the earliest run that touched an object: a later operation's outcome should not be
            # attributed to an earlier one.
            prev = touched.get(str(object_id))
            if prev is None or started < prev:
                touched[str(object_id)] = started
    if not touched:
        # "No runs" and "no runs recently" are different facts, and a reader deciding whether to trust the
        # operation needs to know which one they are looking at.
        why = ("no committed runs of this operation" if not excluded
               else f"no committed runs in the window; {excluded} older runs excluded")
        return _unmeasured(op_kind, 0, why, excluded_runs=excluded)

    import uuid as _uuid

    ids = [_uuid.UUID(o) for o in touched]
    rows: list = []
    for batch in _chunks(ids, max(1, id_chunk)):
        rows += (await db.execute(
            select(Review.object_id, Review.action, Review.ts_ns)
            .where(Review.object_id.in_(batch)))).all()
    # Ordered by review time across the whole set, not per batch. The loop below keeps the first verdict per
    # object, and "first" has to mean first overall: batching is an artefact of a driver limit and must not
    # decide which human ruling counts.
    rows.sort(key=lambda r: (r[2] or 0))

    hits = misses = 0
    seen: set[str] = set()
    for object_id, action, ts_ns in rows:
        oid = str(object_id)
        started = touched.get(oid)
        # Only reviews after the operation ran can be a verdict on it.
        if started is None or (ts_ns or 0) < started:
            continue
        if since_ns is not None and (ts_ns or 0) < since_ns:
            continue
        # One verdict per object: the first human to rule after the operation. Later edits are their own
        # story and counting them again would let one contested object dominate.
        if oid in seen:
            continue
        if action in ENDORSING_ACTIONS:
            seen.add(oid)
            hits += 1
        elif action in CORRECTING_ACTIONS:
            seen.add(oid)
            misses += 1

    n = hits + misses
    if n < MIN_SAMPLES:
        return _unmeasured(op_kind, n, f"only {n} reviewed outcomes, {MIN_SAMPLES} needed",
                           excluded_runs=excluded)

    precision = hits / n
    return {
        "op_type": op_kind,
        "measured": True,
        "precision": round(precision, 4),
        # Recall is not computable here and is reported as null rather than as a plausible-looking number.
        # It would need the objects the operation should have touched and did not, which no record holds.
        "recall": None,
        "n": n,
        "hits": hits,
        "misses": misses,
        "objects_touched": len(touched),
        "reviewed_fraction": round(n / len(touched), 4),
        "runs_scored": len(runs),
        "excluded_runs": excluded,
        "dataset_slice": ("reviewed outcomes after the operation ran"
                          if not excluded else
                          f"reviewed outcomes after the operation ran, over the {len(runs)} runs in the "
                          f"window ({excluded} older runs excluded)"),
        "caveat": ("precision over reviewed outcomes only; objects reach review because something drew "
                   "attention to them, so this is not a random sample of the operation's output"),
    }


def _unmeasured(op_kind: str, n: int, why: str, *, excluded_runs: int = 0) -> dict:
    return {"op_type": op_kind, "measured": False, "precision": None, "recall": None, "n": n,
            "reason": why, "excluded_runs": excluded_runs}


async def measure_all(db, *, window_days: int | None = WINDOW_DAYS) -> dict:
    """Every operation kind, measured or explicitly not, over the recent window.

    `window_days=None` restores the all-time view, which is the right answer for an audit of what an
    operation has ever done and the wrong one for a chip that is supposed to say whether to trust it today.
    """
    from core.timebase import now_ns

    runs_since_ns = None if window_days is None else now_ns() - int(window_days) * 86_400 * 1_000_000_000
    out = {}
    for kind in OPERATION_KINDS:
        out[kind] = await measure_operation(db, kind, runs_since_ns=runs_since_ns)
    measured = sum(1 for v in out.values() if v.get("measured"))
    log.info("op_precision.measured", kinds=len(OPERATION_KINDS), measured=measured,
             window_days=window_days)
    return out
