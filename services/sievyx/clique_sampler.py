"""Dividing a labelling budget across the ways the model is confused, and learning which division works.

Active learning picks the frames a person should label next. The usual rule is "lowest confidence first",
which spends the whole budget wherever ambiguity is densest. On this corpus that is two-wheelers, which is
also where a confusion costs least: a scooter called a motorcycle is a rounding error to a planner, and a
pedestrian called a bollard is not.

So the selection has two stages, and the second is the interesting one.

WITHIN A CLIQUE the score is core/accel/clique_margin.py: how torn the model is between the top two
classes, weighted by what confusing that pair costs.

ACROSS CLIQUES the split is a Thompson-sampled Beta posterior per clique, because the right split is not
knowable in advance. It depends on which confusions labelling actually fixes, and that is a property of the
data and the model rather than a constant somebody can pick. Thompson sampling also explores on its own,
without an epsilon anybody has to tune, and its exploration shrinks exactly as fast as the evidence
justifies.

THE POSTERIOR STARTS AT THE PRIOR AND SAYS SO. Reward is "labelling this clique moved gold recall for its
classes", which needs completed labelling cycles, and there have been none. Every clique begins Beta(1, 1),
the allocation is therefore near-uniform with sampling noise, and `learned` is False in the report. That is
the honest state, and it is very different from a uniform allocation that had been arrived at.

Selected frames enter the existing queue as `FrameCandidate`-shaped reasons ("clique:two_wheelers"),
following services/activelearn/false_negatives.py, so a reviewer sees which boundary a batch is buying
rather than an undifferentiated pile of uncertain frames.
"""

from __future__ import annotations

from uuid import UUID

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.accel.clique_margin import score_batch
from core.logging import get_logger
from db.models import CliqueBandit, InferenceRun, Prediction

log = get_logger("clique_sampler")

# The Thompson draw is seeded per cycle so a selection can be reproduced from its cycle id. Unseeded, two
# people asking "why were these frames chosen" get two different answers.
_SEED = 20260820


async def _posteriors(db: AsyncSession, pack_id: str, names: list[str]) -> dict[str, CliqueBandit]:
    rows = (await db.execute(select(CliqueBandit).where(
        CliqueBandit.pack_id == pack_id, CliqueBandit.clique.in_(names)))).scalars().all()
    have = {r.clique: r for r in rows}
    for n in names:
        if n not in have:
            row = CliqueBandit(clique=n, pack_id=pack_id, alpha=1.0, beta=1.0)
            db.add(row)
            have[n] = row
    if len(have) != len(rows):
        await db.commit()
    return have


def _thompson(posteriors: dict[str, CliqueBandit], budget: int, seed: int) -> dict[str, int]:
    """Allocate `budget` frames across cliques by sampling each posterior once and normalising.

    One draw per clique per cycle rather than one per frame: the allocation is a single decision about how
    to spend a batch, and drawing per frame would average the posteriors away and reduce to proportional
    sampling, which is the thing Thompson sampling exists not to be.
    """
    rng = np.random.default_rng(seed)
    names = sorted(posteriors)
    draws = np.array([rng.beta(max(posteriors[n].alpha, 1e-9), max(posteriors[n].beta, 1e-9))
                      for n in names], dtype=float)
    total = float(draws.sum())
    if total <= 0:
        share = np.full(len(names), 1.0 / max(len(names), 1))
    else:
        share = draws / total
    # Largest-remainder, so the allocation sums exactly to the budget rather than to budget minus rounding.
    exact = share * budget
    alloc = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - alloc))[:budget - int(alloc.sum())]:
        alloc[i] += 1
    return {n: int(a) for n, a in zip(names, alloc, strict=True)}


async def select_frames(db: AsyncSession, *, run_id: str, budget: int = 200,
                        seed: int = _SEED) -> dict:
    """Choose `budget` frames to label next, split across confusion cliques and ranked within each.

    Frames rather than objects, because a person opens a frame: selecting objects would send a reviewer to
    forty frames for forty boxes. A frame's score is its best-scoring detection, so one genuinely
    ambiguous object is enough to earn the frame a look.
    """
    run = await db.get(InferenceRun, UUID(run_id))
    if run is None:
        return {"error": "inference run not found", "run_id": run_id}

    from packs.registry import default_pack_id, get_pack
    from services.autolabel.ontology import get_ontology

    pack_id = default_pack_id()
    spec = get_pack(pack_id).cliques
    if spec is None or not spec.cliques:
        return {"error": "this pack defines no confusion cliques, so there is nothing to allocate across",
                "pack_id": pack_id}
    onto = get_ontology()

    def _name(cid: int) -> str:
        try:
            return onto.by_id(cid).name
        except Exception:  # noqa: BLE001
            return f"class_{cid}"

    def pair_cost(a: int, b: int) -> float:
        return spec.pair_cost(_name(a), _name(b))

    def clique_of(a: int, b: int) -> str | None:
        ca, cb = spec.clique_of(_name(a)), spec.clique_of(_name(b))
        return ca.name if (ca is not None and ca is cb) else None

    # jsonb_typeof rather than IS NOT NULL. SQLAlchemy's JSON types store a Python None as the JSON value
    # `null` rather than as SQL NULL, so `is_not(None)` matches a row whose distribution is the JSON null
    # and the scorer would then be handed nothing while believing it had been given something. The
    # migration's backfilled rows are true SQL NULL and anything written through the ORM may not be; this
    # is correct for both.
    rows = (await db.execute(
        select(Prediction.frame_id, Prediction.class_probs)
        .where(Prediction.run_id == run.run_id,
               func.jsonb_typeof(Prediction.class_probs) == "object"))).all()
    n_total = (await db.execute(
        select(Prediction.prediction_id).where(Prediction.run_id == run.run_id))).scalars().all()

    if not rows:
        # Every prediction predates class_probs, or the model's runtime exposed no distribution. Refusing
        # is right: a margin cannot be reconstructed from the class that won, and falling back to
        # lowest-confidence would silently be the old behaviour under a new name.
        return {"error": "no prediction in this run carries a class distribution, so no margin can be "
                         "computed; re-run inference on a build that records class_probs",
                "run_id": run_id, "n_predictions": len(n_total)}

    scored = score_batch([r[1] for r in rows], pair_cost=pair_cost, clique_of=clique_of)
    detail = scored["detail"]

    # Best-scoring detection per frame, per clique. A frame can be the best candidate for two boundaries
    # at once and is offered to both; the dedupe at the end keeps it once.
    best: dict[str, dict[object, dict]] = {c.name: {} for c in spec.cliques}
    unassigned = 0
    for (fid, _probs), d in zip(rows, detail, strict=True):
        if not d.measured or d.score is None or d.clique is None:
            unassigned += 1
            continue
        cur = best[d.clique].get(fid)
        if cur is None or d.score > cur["score"]:
            best[d.clique][fid] = {"frame_id": fid, "score": d.score, "margin": d.margin,
                                   "pair": [_name(d.top_pair[0]), _name(d.top_pair[1])]
                                   if d.top_pair else None}

    posteriors = await _posteriors(db, pack_id, [c.name for c in spec.cliques])
    alloc = _thompson(posteriors, budget, seed)

    # Budget allocated to a clique with no candidates is budget nobody spends. Redistribute it to the
    # cliques that still have frames, highest-scoring first, rather than returning a short batch: a
    # reviewer asked for 200 frames and getting 1 is not a smaller version of the same answer. Ordered so
    # the redistribution is deterministic under the same seed.
    pools = {name: sorted(best[name].values(), key=lambda x: -x["score"]) for name in alloc}
    selected: list[dict] = []
    seen: set[object] = set()
    per_clique: dict[str, dict] = {}
    for name in sorted(alloc):
        pool = pools[name]
        take = [p for p in pool if p["frame_id"] not in seen][:alloc[name]]
        for p in take:
            seen.add(p["frame_id"])
            selected.append({**p, "frame_id": str(p["frame_id"]), "clique": name,
                             "reason": f"clique:{name}"})
        per_clique[name] = {"allocated": alloc[name], "available": len(pool), "selected": len(take),
                            "alpha": posteriors[name].alpha, "beta": posteriors[name].beta,
                            "n_pulls": posteriors[name].n_pulls or 0,
                            # Said per clique because a reader looking at one row cannot otherwise tell an
                            # allocation the bandit earned from one it drew out of the prior.
                            "from_prior": not (posteriors[name].n_pulls or 0)}

    shortfall = budget - len(selected)
    if shortfall > 0:
        leftovers = sorted(
            (p for name in sorted(alloc) for p in pools[name] if p["frame_id"] not in seen),
            key=lambda x: -x["score"])
        for p in leftovers[:shortfall]:
            seen.add(p["frame_id"])
            name = next(n for n in sorted(alloc) if any(q["frame_id"] == p["frame_id"] for q in pools[n]))
            selected.append({**p, "frame_id": str(p["frame_id"]), "clique": name,
                             "reason": f"clique:{name}", "redistributed": True})
            per_clique[name]["selected"] += 1

    learned = any((p.n_pulls or 0) > 0 for p in posteriors.values())
    log.info("clique_sampler.selected", run=run_id, budget=budget, selected=len(selected),
             cliques=len(per_clique), unassigned=unassigned, learned=learned,
             redistributed=shortfall)
    return {
        "run_id": run_id, "pack_id": pack_id, "budget": budget, "seed": seed,
        "n_selected": len(selected), "n_scored": len(rows), "n_predictions": len(n_total),
        "n_unassigned": unassigned, "learned": learned, "n_redistributed": shortfall,
        "per_clique": per_clique, "frames": selected,
        "caveat": (None if learned else
                   "every posterior is at its Beta(1,1) prior because no labelling cycle has reported a "
                   "reward yet, so this allocation is uniform with sampling noise and is not a finding"),
    }


async def record_reward(db: AsyncSession, *, clique: str, allocated: int,
                        recall_before: float | None, recall_after: float | None,
                        pack_id: str | None = None) -> dict:
    """Update one clique's posterior from what its batch actually bought.

    Reward is binary on purpose: recall for the clique's classes went up, or it did not. A continuous
    reward would need a scale nobody has calibrated, and a Beta posterior over "did this help" is exactly
    the question being asked.

    An unmeasurable outcome (either recall missing) updates nothing. Counting it as a failure would teach
    the bandit to avoid cliques whose recall happens not to have been measured, which is a property of the
    evaluation and not of the clique.
    """
    from packs.registry import default_pack_id

    pid = pack_id or default_pack_id()
    row = await db.get(CliqueBandit, (clique, pid))
    if row is None:
        row = CliqueBandit(clique=clique, pack_id=pid, alpha=1.0, beta=1.0)
        db.add(row)
    if recall_before is None or recall_after is None:
        log.warning("clique_sampler.reward_unmeasurable", clique=clique,
                    before=recall_before, after=recall_after)
        return {"clique": clique, "updated": False,
                "reason": "recall was not measured on both sides, so nothing can be attributed"}

    improved = recall_after > recall_before
    row.alpha = float(row.alpha if row.alpha is not None else 1.0) + (1.0 if improved else 0.0)
    row.beta = float(row.beta if row.beta is not None else 1.0) + (0.0 if improved else 1.0)
    # `or 0`: a row added in this call has not been flushed, so its Python-side defaults are not applied
    # and the columns read back as None rather than as zero.
    row.n_pulls = int(row.n_pulls or 0) + 1
    row.n_rewards = int(row.n_rewards or 0) + (1 if improved else 0)
    row.last_allocated = int(allocated)
    row.last_reward = 1.0 if improved else 0.0
    row.last_recall_before = float(recall_before)
    row.last_recall_after = float(recall_after)
    await db.commit()
    log.info("clique_sampler.reward", clique=clique, improved=improved, alpha=row.alpha, beta=row.beta,
             before=recall_before, after=recall_after)
    return {"clique": clique, "updated": True, "improved": improved,
            "alpha": row.alpha, "beta": row.beta, "n_pulls": row.n_pulls}


async def bandit_report(db: AsyncSession, pack_id: str | None = None) -> dict:
    from packs.registry import default_pack_id

    pid = pack_id or default_pack_id()
    rows = (await db.execute(select(CliqueBandit).where(
        CliqueBandit.pack_id == pid).order_by(CliqueBandit.clique))).scalars().all()
    return {
        "pack_id": pid, "n_cliques": len(rows),
        "learned": any((r.n_pulls or 0) > 0 for r in rows),
        "cliques": [{
            "clique": r.clique, "alpha": r.alpha, "beta": r.beta,
            "mean": round(r.alpha / (r.alpha + r.beta), 4),
            "n_pulls": r.n_pulls, "n_rewards": r.n_rewards,
            "from_prior": not (r.n_pulls or 0),
            "last_allocated": r.last_allocated, "last_reward": r.last_reward,
            "last_recall_before": r.last_recall_before, "last_recall_after": r.last_recall_after,
        } for r in rows],
    }


__all__ = ["select_frames", "record_reward", "bandit_report"]
