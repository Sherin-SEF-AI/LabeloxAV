"""Estimate each error detector's precision by judging a random sample of its candidates.

298,528 candidates carry one human verdict between them, so every detector reports as unmeasured and the
active-learning selector weights all of them at a placeholder. Nobody is going to rule on 298,528 crops; the
judge can rule on a sample of each.

What comes out is a machine estimate and is labelled as one everywhere it appears. It never touches
error_candidate.status, so the human-verdict path keeps measuring humans.

    .venv/bin/python -m scripts.run_detector_judging
    .venv/bin/python -m scripts.run_detector_judging --n 30 --kinds near_dup_inconsistent,critic_flag
"""

from __future__ import annotations

import argparse
import asyncio

from core.logging import setup_logging
from db.session import get_sessionmaker
from services.errordetect.judge_detectors import judge_all_detectors


async def _run(n: int, kinds: list[str] | None) -> None:
    async with get_sessionmaker()() as db:
        res = await judge_all_detectors(db, n=n, kinds=kinds)
    print(f"{'detector':<26}{'judged':>7}{'strict':>18}{'cross-superclass':>20}")
    for kind, d in res["per_kind"].items():
        s = d["precision_strict"]
        c = d["precision_cross_superclass"]
        st = f"{s['p']:.3f} ({s['lo']:.2f}-{s['hi']:.2f})" if s and s["p"] is not None else "n/a"
        cs = f"{c['p']:.3f} ({c['lo']:.2f}-{c['hi']:.2f})" if c and c["p"] is not None else "n/a"
        flag = "" if d["usable"] else "  (too few to rank on)"
        print(f"  {kind:<24}{d['judged']:>7}{st:>18}{cs:>20}{flag}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=50, help="candidates sampled per detector")
    ap.add_argument("--kinds", default=None, help="comma-separated detector kinds; default all")
    args = ap.parse_args()
    setup_logging("INFO")
    kinds = [k.strip() for k in args.kinds.split(",")] if args.kinds else None
    asyncio.run(_run(args.n, kinds))


if __name__ == "__main__":
    main()
