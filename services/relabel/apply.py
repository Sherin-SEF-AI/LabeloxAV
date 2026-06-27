"""Selective relabel application (M4.2). Auto-apply only confident improvements that do not touch
human-verified objects; route conflicts and regressions to review; record old and new in provenance so the
change is a single reversible walk. Every run lands on its own lakeFS branch (the versioned proposal set),
so it can be reviewed, merged, or discarded, and apply is reversible from the recorded history."""

from __future__ import annotations

import uuid
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.timebase import now_ns
from db.models import Object, Review
from services.relabel.diff import classify_change, summarize
from versioning import lakefs_store as L

log = get_logger("relabel_apply")


async def apply_relabel(db: AsyncSession, proposals: list[dict], model_version: str,
                        branch: str | None = None, run_id: str | None = None) -> dict:
    branch = branch or f"relabel-{(run_id or str(uuid.uuid4()))[:8]}"
    L.ensure_branch(branch, source=L.default_branch())

    applied = routed = regressions = unchanged = conflicts = 0
    classified_all = []
    for p in proposals:
        v = classify_change(p)
        classified_all.append(v)
        # the branch records every proposed label (the reviewable, versioned proposal set)
        L.put_label(branch, p["object_id"], {
            "class_id": p["new_class_id"], "class": p["new_class"], "conf": p["new_conf"],
            "old_class_id": p["old_class_id"], "old_class": p["old_class"],
            "verdict": v["verdict"], "model_version": model_version})

        if v["verdict"] == "unchanged":
            unchanged += 1
            continue
        if v["apply"]:
            obj = await db.get(Object, UUID(p["object_id"]))
            if obj is None:
                continue
            old = {"class_id": obj.class_id, "conf": obj.conf, "source": obj.source, "state": obj.state}
            obj.class_id = p["new_class_id"]
            obj.source, obj.state = "relabel", "auto_accept"
            prov = dict(obj.provenance or {})
            prov.setdefault("relabel_history", []).append(
                {"old_class_id": old["class_id"], "new_class_id": p["new_class_id"],
                 "model_version": model_version, "branch": branch, "run_id": run_id})
            prov["relabel"] = {"from": p["old_class"], "to": p["new_class"], "model_version": model_version}
            obj.provenance = prov
            db.add(Review(object_id=obj.object_id, reviewer="relabel", action="relabel_apply",
                          before=old, after={"class_id": p["new_class_id"], "source": "relabel"},
                          time_spent_ms=0, ts_ns=now_ns()))
            applied += 1
        else:
            routed += 1
            if v["verdict"] == "regression":
                regressions += 1
            if v["verdict"] == "conflict":
                conflicts += 1  # human-verified, left entirely untouched (visible only on the branch)
            else:
                obj = await db.get(Object, UUID(p["object_id"]))
                if obj is not None and obj.source != "human":
                    obj.state = "review"  # route the non-human change to a human

    commit_id = L.commit(branch, f"relabel {model_version}: {applied} applied / {routed} routed",
                         {"model_version": model_version, "applied": applied, "routed": routed,
                          "regressions": regressions})
    await db.commit()
    out = {"branch": branch, "commit": commit_id, "proposed": len(proposals), "applied": applied,
           "routed_to_review": routed, "conflicts": conflicts, "regressions": regressions,
           "unchanged": unchanged, "verdicts": summarize(classified_all)}
    log.info("relabel.applied", **{k: out[k] for k in ("branch", "applied", "routed_to_review", "regressions")})
    return out


async def revert_run(db: AsyncSession, run_id: str) -> dict:
    """Reverse an applied relabel run: restore the previous class on every object it touched (the recorded
    history), and write a reverting correction. The lakeFS branch remains as the audit record."""
    from sqlalchemy import select

    restored = 0
    rows = (await db.execute(select(Object).where(Object.source == "relabel"))).scalars().all()
    for obj in rows:
        hist = (obj.provenance or {}).get("relabel_history", [])
        entry = next((h for h in reversed(hist) if h.get("run_id") == run_id), None)
        if entry is None:
            continue
        old = {"class_id": obj.class_id, "source": obj.source, "state": obj.state}
        obj.class_id = entry["old_class_id"]
        obj.source, obj.state = "fused", "auto_accept"
        prov = dict(obj.provenance or {})
        prov.pop("relabel", None)
        obj.provenance = prov
        db.add(Review(object_id=obj.object_id, reviewer="relabel", action="relabel_revert",
                      before=old, after={"class_id": entry["old_class_id"]}, time_spent_ms=0, ts_ns=now_ns()))
        restored += 1
    await db.commit()
    log.info("relabel.reverted", run_id=run_id, restored=restored)
    return {"run_id": run_id, "restored": restored}
