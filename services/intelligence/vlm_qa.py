"""VLM auto-QA + auto-attributes: a model-as-reviewer pass over already-labeled objects. Re-runs the
Qwen-VL verifier on each object's crop; when it disagrees with the current class ACROSS superclasses it
routes the object to the QA queue (state=submitted) with the VLM's suggestion recorded, and it fills the
ontology's typed attributes the detector never set. Closes the active-learning loop: the model flags its
own likely mistakes and pre-fills attributes so humans only adjudicate.

    python -m services.intelligence.vlm_qa --session <uuid> --limit 40
"""

from __future__ import annotations

import asyncio
import time
from uuid import UUID

import click
import cv2
import numpy as np
from sqlalchemy import func, select

from core.config import get_settings
from core.logging import get_logger, setup_logging
from core.storage import get_object_store
from db.models import Frame, Object, Review
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology

log = get_logger("vlm_qa")

# Objects worth a second opinion: machine labels that a human has not already locked in.
_QA_STATES = ("review", "annotate", "accepted", "submitted")


async def vlm_qa_session(
    session_id: UUID, limit: int = 40, fill_attrs: bool = True, flag_disagreements: bool = True
) -> dict:
    from services.autolabel.paths.path_c_vlm import VlmVerifier, make_vlm_client

    onto = get_ontology()
    verifier = VlmVerifier(make_vlm_client(), onto)
    store = get_object_store()
    maker = get_sessionmaker()

    checked = flagged = filled = agreed = 0
    async with maker() as db:
        rows = (
            await db.execute(
                select(Object, Frame.img_uri)
                .join(Frame, Frame.frame_id == Object.frame_id)
                .where(Frame.session_id == session_id, Object.state.in_(_QA_STATES))
                .order_by(Object.conf.asc())  # least-confident first = highest QA value
                .limit(limit)
            )
        ).all()

        by_frame: dict[str, list] = {}
        for obj, uri in rows:
            by_frame.setdefault(uri, []).append(obj)

        for uri, objs in by_frame.items():
            try:
                img = cv2.imdecode(np.frombuffer(store.get_bytes(uri), np.uint8), cv2.IMREAD_COLOR)
            except Exception:  # noqa: BLE001
                continue
            if img is None:
                continue
            for obj in objs:
                cur_name = onto.by_id(obj.class_id).name
                res = verifier.verify_object(img, tuple(obj.bbox), obj.class_id)
                checked += 1

                # auto-attributes: VlmVerifier already validated res.attrs against the ontology.
                if fill_attrs and res.attrs:
                    cur = dict(obj.attrs or {})
                    added = {k: v for k, v in res.attrs.items() if k not in cur}
                    if added:
                        cur.update(added)
                        obj.attrs = cur
                        filled += 1

                # auto-QA: flag cross-superclass disagreements into the QA queue.
                if res.class_name and onto.has_name(res.class_name):
                    if res.class_name == cur_name:
                        agreed += 1
                    elif (flag_disagreements and res.confident
                          and onto.by_id(obj.class_id).l1 != onto.by_name(res.class_name).l1):
                        prov = dict(obj.provenance or {})
                        prov["vlm_qa"] = {"suggested": res.class_name, "caption": res.caption,
                                          "agreement": round(res.agreement, 2)}
                        obj.provenance = prov
                        before_state = obj.state
                        obj.state = "submitted"
                        db.add(Review(object_id=obj.object_id, reviewer="vlm-qa", action="flag",
                                      before={"class": cur_name, "state": before_state},
                                      after={"class_suggested": res.class_name}, ts_ns=time.time_ns()))
                        flagged += 1
            await db.commit()

    out = {"checked": checked, "flagged": flagged, "attrs_filled": filled, "agreed": agreed,
           "model": get_settings().models.vlm.ollama_tag}
    log.info("vlm_qa.done", **out)
    return out


# ---- a reason per label (tool #13) ---------------------------------------------------------------------
#
# A check is a machine verdict, so it is stored in `machine_verdict`, the plane kept apart from human
# `review` rows on purpose (db/models.py MachineVerdict), under its own judge name and batch so it never
# mixes into the judge-calibration batches `vlm_review` measures. One row per object per model: checking
# again replaces the previous opinion of the same model, and a different model's stands beside it.

CHECK_JUDGE = "vlm_reason"
CHECK_BATCH = "label-check"
# A synchronous check from a page holds a request open while the model looks, so it is kept small; more
# than this goes through the background job.
MAX_SYNC_CHECKS = 5


def _check_row(mv, onto, current_class_id: int | None) -> dict:
    d = dict(mv.detail or {})
    alt = None
    if mv.proposed_class_id is not None:
        try:
            alt = onto.by_id(int(mv.proposed_class_id)).name
        except Exception:  # noqa: BLE001
            alt = None
    given_id = d.get("given_class_id")
    return {
        "object_id": str(mv.object_id), "verdict": mv.verdict, "reason": d.get("reason", ""),
        "alternative": alt, "given_class": d.get("given_class"),
        # The label may have changed since the model looked; a verdict about a class the object no longer
        # carries must say so rather than read as a verdict on the current one.
        "stale": current_class_id is not None and given_id is not None and int(given_id) != int(current_class_id),
        "provider": mv.provider, "model": mv.model_version, "latency_ms": d.get("latency_ms"),
        "cost_usd": d.get("cost_usd"), "checked_at": mv.created_at.isoformat() if mv.created_at else None,
        "checked_by": d.get("checked_by"),
    }


async def latest_checks(db, object_ids: list) -> dict[str, dict]:
    """The newest stored check per object, keyed by object id. Objects never checked are absent."""
    from db.models import MachineVerdict

    ids = [UUID(str(o)) for o in object_ids]
    if not ids:
        return {}
    onto = get_ontology()
    cur = {oid: cid for oid, cid in (await db.execute(
        select(Object.object_id, Object.class_id).where(Object.object_id.in_(ids)))).all()}
    rows = (await db.execute(
        select(MachineVerdict).where(MachineVerdict.object_id.in_(ids), MachineVerdict.judge == CHECK_JUDGE,
                                     MachineVerdict.batch_id == CHECK_BATCH)
        .order_by(MachineVerdict.ts_ns.desc()))).scalars().all()
    out: dict[str, dict] = {}
    for mv in rows:
        out.setdefault(str(mv.object_id), _check_row(mv, onto, cur.get(mv.object_id)))
    return out


async def check_labels(db, object_ids: list, *, client=None, user_id: str | None = None,
                       progress=None) -> dict:
    """Ask the VLM about each object's current label and store verdict, reason and alternative.

    Frames are read once each and the model is called off the event loop. A call that never reached the
    model is counted as failed and stores nothing, so an outage cannot write a column of `unsure`.
    Cost: every call is posted through `services/llm/ledger.py` (purpose `vlm_label_check`); the row keeps
    its own latency and, for a local model, its zero cost, and leaves a cloud call's cost to the ledger,
    which is the one place tokens are priced.
    """
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from db.models import MachineVerdict
    from services.autolabel.paths.path_c_vlm import VlmVerifier, make_vlm_client
    from services.llm.ledger import LOCAL_PROVIDERS

    onto = get_ontology()
    verifier = VlmVerifier(client or make_vlm_client(), onto)
    store = get_object_store()
    ids = [UUID(str(o)) for o in object_ids]
    rows = (await db.execute(
        select(Object, Frame.img_uri).join(Frame, Frame.frame_id == Object.frame_id)
        .where(Object.object_id.in_(ids)))).all()
    by_uri: dict[str, list] = {}
    for obj, uri in rows:
        by_uri.setdefault(uri, []).append(obj)

    checked = failed = 0
    errors: list[str] = []
    done = 0
    for uri, objs in by_uri.items():
        try:
            raw = await asyncio.to_thread(store.get_bytes, uri)
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        except Exception as exc:  # noqa: BLE001
            img = None
            errors.append(f"frame unreadable: {str(exc)[:80]}")
        for obj in objs:
            done += 1
            if img is None or not obj.bbox or len(obj.bbox) < 4:
                failed += 1
                continue
            res = await asyncio.to_thread(verifier.check_label, img, tuple(obj.bbox), int(obj.class_id))
            if not res.ok:
                failed += 1
                if res.error:
                    errors.append(res.error)
                continue
            alt_id = onto.by_name(res.alternative).id if res.alternative and onto.has_name(res.alternative) else None
            detail = {"reason": res.reason, "given_class": res.given_class, "given_class_id": int(obj.class_id),
                      "latency_ms": res.latency_ms,
                      "cost_usd": 0.0 if res.provider in LOCAL_PROVIDERS else None,
                      "ledger_purpose": "vlm_label_check", "checked_by": user_id}
            stmt = pg_insert(MachineVerdict).values(
                object_id=obj.object_id, judge=CHECK_JUDGE, provider=res.provider[:32] or "unknown",
                model_version=(res.model or "unknown")[:128], verdict=res.verdict, proposed_class_id=alt_id,
                confidence=None, agreement=None, detail=detail, batch_id=CHECK_BATCH, ts_ns=time.time_ns())
            stmt = stmt.on_conflict_do_update(
                constraint="uq_machine_verdict_object_judge_batch",
                set_={"verdict": stmt.excluded.verdict, "proposed_class_id": stmt.excluded.proposed_class_id,
                      "detail": stmt.excluded.detail, "provider": stmt.excluded.provider,
                      "ts_ns": stmt.excluded.ts_ns, "created_at": func.now()})
            await db.execute(stmt)
            await db.commit()
            checked += 1
            if progress is not None:
                await progress(done / max(1, len(rows)))
    results = await latest_checks(db, [o.object_id for o, _ in rows])
    out = {"requested": len(ids), "found": len(rows), "checked": checked, "failed": failed,
           "errors": sorted(set(errors))[:5], "results": results,
           "model": get_settings().models.vlm.judge_tag or get_settings().models.vlm.ollama_tag}
    log.info("vlm_qa.checked", **{k: v for k, v in out.items() if k not in ("results", "errors")})
    return out


@click.command()
@click.option("--session", "session_id", required=True)
@click.option("--limit", type=int, default=40)
@click.option("--no-attrs", is_flag=True, default=False)
def main(session_id, limit, no_attrs) -> None:
    setup_logging(get_settings().log_level)
    click.echo(asyncio.run(vlm_qa_session(UUID(session_id), limit, fill_attrs=not no_attrs)))


if __name__ == "__main__":
    main()
