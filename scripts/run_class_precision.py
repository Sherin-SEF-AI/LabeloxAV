"""Measure per-class label precision with the VLM judge, so remediation targets come from evidence.

Confidence is the obvious way to pick which classes are wrong and it is the wrong way: it is the detector's
opinion of its own output. `object_fallback` sits at mean confidence 0.806 while meaning "I do not know what
this is". This asks a judge instead, per class, on a random sample, and reports the answer with its
uncertainty.

Two rates come out and they mean different things. A **refinement** is a `sedan` that should have been an
`suv`: the superclass is right and the leaf is wrong, which is taxonomy drift. A **cross-superclass** error
is a pole labelled `traffic_signal`: the wrong kind of thing entirely, which is what "wrong names" looks
like to somebody reading a frame. Rank remediation on the second.

    .venv/bin/python -m scripts.run_class_precision --n 40 --classes traffic_signal,object_fallback
    .venv/bin/python -m scripts.run_class_precision                # all classes over 10,000 objects
    .venv/bin/python -m scripts.run_class_precision --report-only  # re-print without judging again

Idempotent: verdicts upsert on (object, judge, model_version, batch_id), so an interrupted run resumes and
a re-run with the same seed re-judges the same crops rather than drawing a fresh sample.
"""

from __future__ import annotations

import argparse
import asyncio
import json

from core.logging import setup_logging
from db.session import get_sessionmaker
from services.labelops.class_precision import batch_id_for, class_targets, judge_class
from services.labelops.vlm_review import judged_precision


def _fmt(iv: dict | None) -> str:
    if not iv or iv.get("p") is None:
        return "n/a"
    return f"{iv['p']:.3f} ({iv['lo']:.2f}-{iv['hi']:.2f})"


async def _report(db, names: list[str]) -> list[dict]:
    rows = []
    for name in names:
        pr = await judged_precision(db, batch_id_for(name))
        decided = pr.get("judged") or 0        # `judged` is the DECIDED count; unsure is reported apart
        if not decided:
            continue
        rej = pr.get("rejections") or {}
        rows.append({
            "class_name": name,
            "decided": decided,
            "unsure": pr.get("unsure") or 0,
            "raw": pr.get("raw"),
            "raw_superclass": pr.get("raw_superclass"),
            "corrected": pr.get("corrected"),
            "caveat": pr.get("caveat"),
            "refinements": rej.get("refinement_within_superclass") or 0,
            "cross_superclass": rej.get("cross_superclass") or 0,
            "top_proposals": await _top_proposals(db, batch_id_for(name)),
        })
    # Worst first by strict rejection rate, NOT by the cross-superclass share.
    #
    # The superclass split is computed on l1, and l1 is too coarse to carry this ranking. Every one of
    # pole, street_light, traffic_sign, traffic_signal, hoarding, bus_shelter and tree is l1 `fixed`, so a
    # pole labelled `traffic_signal` scores as a "refinement" and the class reads as 0.0% wrong-kind while
    # the judge rejected 15 of 15. Same coarseness that made l1 `heavy` unable to distinguish a bus from a
    # tractor. The rejection rate needs no superclass theory, and the proposals column says what it is
    # actually being confused with.
    rows.sort(key=lambda r: (r["raw"] or {}).get("p") if (r["raw"] or {}).get("p") is not None else 1.0)
    return rows


async def _top_proposals(db, batch_id: str, limit: int = 3) -> list[str]:
    """What the judge said it was instead, most common first. This is the actionable half of the finding."""
    from sqlalchemy import text

    rows = (await db.execute(text("""
        select coalesce(c.name, '?') n, count(*) k
          from machine_verdict v left join ontology_class c on c.id = v.proposed_class_id
         where v.batch_id = :b and v.verdict = 'incorrect' and v.proposed_class_id is not null
         group by 1 order by 2 desc limit :l"""), {"b": batch_id, "l": limit})).all()
    return [f"{n} x{k}" for n, k in rows]


def _print(rows: list[dict]) -> None:
    if not rows:
        print("no verdicts yet - run without --report-only first")
        return
    print(f"\n{'class':<20}{'decided':>8}{'unsure':>7}{'strict precision':>20}   {'judge says it is'}")
    print("-" * 96)
    for r in rows:
        print(f"  {r['class_name']:<18}{r['decided']:>8}{r['unsure']:>7}{_fmt(r['raw']):>20}   "
              f"{', '.join(r['top_proposals']) or '-'}")
    print("\n  strict precision = the judge confirmed the exact label. Ranked worst first.")
    print("  'unsure' is not folded into either side: a judge that abstains on the hard crops and is")
    print("  scored only on the easy ones reports a precision that flatters itself.")
    caveats = {r["caveat"] for r in rows if r.get("caveat")}
    for c in sorted(caveats):
        print(f"\n  note: {c}")


async def _run(n: int, names: list[str] | None, seed: float, report_only: bool, out: str | None) -> None:
    async with get_sessionmaker()() as db:
        if names is None:
            targets = await class_targets(db)
            names = [t["class_name"] for t in targets]
            print(f"measuring {len(names)} classes over 10,000 objects: {', '.join(names)}")

        if not report_only:
            # The slot is taken once for the whole sweep rather than per class. Acquiring and releasing it
            # fifteen times would hand the card to a waiting job halfway through and leave this run stalled
            # behind it; `judge_class` still re-checks headroom every batch inside the block, so a training
            # job that starts mid-sweep waits a batch and not a sweep.
            from core.gpu_slot import gpu_slot

            async with gpu_slot(f"class_precision:{len(names)}-classes", timeout_s=None) as slot:
                if slot.get("waited_s"):
                    print(f"waited {slot['waited_s']}s for the GPU slot")
                for i, name in enumerate(names, 1):
                    print(f"[{i}/{len(names)}] judging {name} ...", flush=True)
                    res = await judge_class(db, name, n=n, seed=seed, take_slot=False)
                    if res.get("skipped_reason"):
                        print(f"    skipped: {res['skipped_reason']}")
                        continue
                    print(f"    {res.get('judged', 0)} judged, {res.get('skipped', 0)} already done, "
                          f"{res.get('unreadable', 0)} unreadable, {res.get('failed', 0)} call failures, "
                          f"{res.get('by_verdict')}")
                    if res.get("stalled"):
                        print(f"    STALLED: {res['stalled']}")

        rows = await _report(db, names)

    _print(rows)
    if out:
        with open(out, "w") as fh:
            json.dump(rows, fh, indent=2, default=str)
        print(f"\nwrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=120, help="crops sampled per class")
    ap.add_argument("--classes", default=None, help="comma-separated class names; default all over 10k")
    ap.add_argument("--seed", type=float, default=0.42,
                    help="fixes the sample so a re-measurement compares like with like")
    ap.add_argument("--report-only", action="store_true", help="re-print stored verdicts without judging")
    ap.add_argument("--out", default=None, help="write the table as JSON here")
    args = ap.parse_args()
    setup_logging("INFO")
    names = [c.strip() for c in args.classes.split(",")] if args.classes else None
    asyncio.run(_run(args.n, names, args.seed, args.report_only, args.out))


if __name__ == "__main__":
    main()
