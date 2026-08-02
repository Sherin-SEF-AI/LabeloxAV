"""Recompute stored error-candidate scores under the corrected scoring, without a full detection pass.

The near-duplicate detector used to store the frame similarity as its score and the embedding-outlier
detector its raw cosine distance. Neither is a suspicion score: the first cannot fall below the similarity
gate, so all 45,313 candidates sat within 0.986 to 1.0, and the second runs to 2.0, so candidates existed
above 1.0. Since the review queue ranks across detectors by score, the queue was ordered by which frames
looked alike, and 4,608 candidates outranked the best `confident_learning` candidate, the one detector
reporting a genuine probability.

The code is fixed, but the scores already in the table are not, and nothing rewrites them until somebody
runs a full detection pass over the corpus. So this exists to make the fix take effect on the data that is
already there.

It is not a shortcut around a real re-run. Only the scoring formula changed, not the rule deciding which
objects are candidates, so for these two kinds this produces exactly what a full pass would, and it is
reversible by running one. Both inputs it needs are already recorded: the similarity in `detail`, and the
object's confidence on the object itself.

    .venv/bin/python -m scripts.rescore_error_candidates          # report only
    .venv/bin/python -m scripts.rescore_error_candidates --apply
"""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import select, update

from core.logging import get_logger, setup_logging
from db.models import ErrorCandidate, Object
from db.session import get_sessionmaker
from services.errordetect.near_dup import DEFAULT_SIM_THRESH, _suspicion

log = get_logger("rescore_error_candidates")

# The gate the stored candidates were produced under, taken from the detector rather than repeated here: a
# candidate cannot exist below it, which is why the raw similarity carried no information, and the margin is
# measured across the band above it. Two copies of this number drifting apart would silently mis-score
# everything in the table.
SIM_THRESH = DEFAULT_SIM_THRESH


async def rescore(apply: bool = False) -> dict:
    maker = get_sessionmaker()
    changed: list[tuple] = []
    async with maker() as db:
        rows = (await db.execute(
            select(ErrorCandidate.candidate_id, ErrorCandidate.kind, ErrorCandidate.score,
                   ErrorCandidate.detail, Object.conf)
            .join(Object, Object.object_id == ErrorCandidate.object_id)
            .where(ErrorCandidate.status == "pending",
                   ErrorCandidate.kind.in_(("near_dup_inconsistent", "embedding_outlier"))))).all()

        for cid, kind, score, detail, conf in rows:
            if kind == "near_dup_inconsistent":
                sim = (detail or {}).get("similarity")
                if sim is None:
                    continue    # produced before the similarity was recorded; a full pass is the only fix
                margin = (float(sim) - SIM_THRESH) / max(1e-6, 1.0 - SIM_THRESH)
                new = _suspicion(float(conf or 0.0), margin)
            else:
                new = round(min(1.0, float(score)), 4)
            if abs(new - float(score)) > 1e-9:
                changed.append((cid, new))

        if apply:
            for cid, new in changed:
                await db.execute(update(ErrorCandidate).where(ErrorCandidate.candidate_id == cid)
                                 .values(score=new))
            await db.commit()

    out = {"examined": len(rows), "rescored": len(changed), "applied": apply}
    log.info("rescore.done", **out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="write the new scores (default: report only)")
    args = ap.parse_args()
    setup_logging("INFO")
    res = asyncio.run(rescore(apply=args.apply))
    print(f"examined {res['examined']}, rescored {res['rescored']}, "
          f"{'applied' if res['applied'] else 'dry run, pass --apply to write'}")


if __name__ == "__main__":
    main()
