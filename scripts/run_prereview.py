"""Run the machine judge over a batch, so the verdicts exist before anybody is asked to adjudicate.

A thin entry point rather than logic: everything of substance is in services/labelops/vlm_review.py. It
exists because judging a batch is a long job (tens of minutes against a local model, real money against a
metered API) and wants to be launchable, resumable and watchable from a shell rather than from a request
that would time out.

Resumable by construction: verdicts are unique on (object, judge, model_version) and written in chunks, so
re-running skips what already landed instead of paying for it twice.

    .venv/bin/python -m scripts.run_prereview precision-221d7b86
    .venv/bin/python -m scripts.run_prereview precision-221d7b86 --limit 25
"""

from __future__ import annotations

import argparse
import asyncio

from core.logging import setup_logging
from db.session import get_sessionmaker
from services.labelops.vlm_review import judged_precision, prereview_batch


async def _run(batch_id: str, limit: int | None) -> None:
    async with get_sessionmaker()() as db:
        result = await prereview_batch(db, batch_id, limit=limit)
        print(f"judged {result['judged']} of {result['objects']} "
              f"(skipped {result['skipped']}, unreadable {result['unreadable']}) "
              f"via {result['provider']}/{result['model_version']}")
        print(f"verdicts: {result['by_verdict']}")

        summary = await judged_precision(db, batch_id, model_version=result["model_version"])
        print(f"raw agreement rate: {summary['raw']}")
        print(f"corrected precision: {summary['corrected']}")
        if summary["caveat"]:
            print(f"caveat: {summary['caveat']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("batch_id", help="the flywheel cycle_id stamped on the batch, e.g. precision-221d7b86")
    ap.add_argument("--limit", type=int, default=None, help="judge only the first N objects")
    args = ap.parse_args()
    setup_logging("INFO")
    asyncio.run(_run(args.batch_id, args.limit))


if __name__ == "__main__":
    main()
