"""Measure the machine judge against human rulings that already exist.

A judge whose accuracy nobody knows produces numbers nobody can use, and measuring it looked like it needed
somebody's afternoon. It does not: the corpus holds hundreds of human rulings recorded for other reasons,
and read correctly they are a labelled evaluation set for the judge that costs nothing to collect.

    .venv/bin/python -m scripts.run_judge_calibration
    .venv/bin/python -m scripts.run_judge_calibration --limit 40
"""

from __future__ import annotations

import argparse
import asyncio

from core.logging import setup_logging
from db.session import get_sessionmaker
from services.labelops.judge_calibration import calibrate_judge


async def _run(limit: int | None) -> None:
    async with get_sessionmaker()() as db:
        r = await calibrate_judge(db, limit=limit)
    print(f"judged {r['objects_judged']} objects across {r['independent_decisions']} independent decisions")
    print(f"  sensitivity {r['sensitivity']}  {r['sensitivity_interval']}")
    print(f"  specificity {r['specificity']}  {r['specificity_interval']}")
    print(f"  confusion   {r['confusion']}")
    print(f"  usable for correction: {r['usable']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="cap items per class, for a quick pass")
    args = ap.parse_args()
    setup_logging("INFO")
    asyncio.run(_run(args.limit))


if __name__ == "__main__":
    main()
