"""Background job kinds for the quality annotation tools (agent B: #11 consensus, #13 VLM reasons, #14 rare scenes).

  consensus_merge   CPU   score a task's submitted replica groups and merge them (services/labelops/agreement.py)
  vlm_label_check   GPU   ask the VLM about many labels, store verdict + reason (services/intelligence/vlm_qa.py).
                          GPU because Ollama holds the card while it answers, so it queues behind the other GPU
                          work; auto-labelling, so it does not start during office hours.
  rare_scene_rank   CPU   rank the corpus by cross-session embedding density (services/intelligence/rare_scenes.py)
"""

from __future__ import annotations

import uuid

from sqlalchemy import select

from db.session import get_sessionmaker
from services.tools.registry import JobContext, register

# A batch check is bounded so one job cannot walk the corpus through a 7B model.
MAX_BATCH_CHECKS = 400


@register("consensus_merge", label="Consensus merge")
async def consensus_merge(ctx: JobContext) -> dict:
    from db.models import LabelJob
    from services.labelops.agreement import SUBMITTED_STATES, AgreementError, score_and_merge

    p = ctx.params
    kw = {"iou_thresh": float(p.get("iou_thresh", 0.5)), "quorum": int(p.get("quorum", 2)),
          "include_model": bool(p.get("include_model", True)),
          "require_human": bool(p.get("require_human", True)),
          "created_by": str(ctx.created_by) if ctx.created_by else None}
    async with get_sessionmaker()() as db:
        if p.get("replica_group"):
            groups = [str(p["replica_group"])]
        else:
            jobs = (await db.execute(select(LabelJob).where(
                LabelJob.task_id == uuid.UUID(str(p["task_id"])), LabelJob.replica_group.isnot(None)))).scalars().all()
            by_group: dict = {}
            for j in jobs:
                by_group.setdefault(str(j.replica_group), []).append(j)
            groups = [g for g, js in by_group.items()
                      if len(js) >= 2 and all(j.state in SUBMITTED_STATES for j in js)]
    totals: dict[str, int] = {}
    skipped: list[dict] = []
    per_class: dict[str, dict[str, int]] = {}
    for i, gid in enumerate(groups):
        await ctx.progress(i / max(1, len(groups)), f"group {i + 1} of {len(groups)}")
        async with get_sessionmaker()() as db:
            try:
                res = await score_and_merge(db, gid, **kw)
            except AgreementError as exc:
                skipped.append({"replica_group": gid, "reason": str(exc)})
                continue
        for k, v in res["counts"].items():
            totals[k] = totals.get(k, 0) + int(v)
        for cname, c in res["per_class"].items():
            pc = per_class.setdefault(cname, {"accepted": 0, "disputed": 0, "dropped": 0})
            for k2, v2 in c.items():
                pc[k2] += v2
    return {"groups": len(groups) - len(skipped), "skipped": skipped, "counts": totals, "per_class": per_class,
            "params": {k: v for k, v in kw.items() if k != "created_by"}}


@register("vlm_label_check", gpu=True, autolabel=True, label="VLM label check")
async def vlm_label_check(ctx: JobContext) -> dict:
    from db.models import Frame, Object
    from services.intelligence.vlm_qa import check_labels

    p = ctx.params
    limit = max(1, min(MAX_BATCH_CHECKS, int(p.get("limit", 50))))
    async with get_sessionmaker()() as db:
        if p.get("object_ids"):
            ids = [str(o) for o in p["object_ids"]][:limit]
        else:
            sid = uuid.UUID(str(p.get("session_id") or ctx.session_id))
            # Least confident first, the same order vlm_qa_session uses: where a second opinion is worth most.
            q = (select(Object.object_id).join(Frame, Frame.frame_id == Object.frame_id)
                 .where(Frame.session_id == sid, Object.state != "rejected")
                 .order_by(Object.conf.asc()).limit(limit))
            ids = [str(o) for o in (await db.execute(q)).scalars().all()]
        await ctx.progress(0.02, f"checking {len(ids)} labels")

        async def prog(frac: float) -> None:
            await ctx.progress(0.02 + 0.96 * frac)

        res = await check_labels(db, ids, user_id=str(ctx.created_by) if ctx.created_by else None, progress=prog)
    verdicts: dict[str, int] = {}
    for r in res["results"].values():
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
    return {"requested": res["requested"], "checked": res["checked"], "failed": res["failed"],
            "errors": res["errors"], "verdicts": verdicts, "model": res["model"],
            "object_ids": list(res["results"].keys())[:MAX_BATCH_CHECKS]}


@register("rare_scene_rank", label="Rare-scene ranking")
async def rare_scene_rank(ctx: JobContext) -> dict:
    from services.intelligence.rare_scenes import rank_corpus

    p = ctx.params
    async with get_sessionmaker()() as db:
        return await rank_corpus(
            db, space=str(p.get("space", "dino")), k=int(p.get("k", 10)), top=int(p.get("top", 300)),
            per_session=int(p.get("per_session", 4)), per_cluster=int(p.get("per_cluster", 6)),
            session_id=p.get("session_id"), city=p.get("city") or None, progress=ctx.progress)
