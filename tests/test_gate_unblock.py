"""Phase 0 of the autonomy work: a blocked gate fires its own unblock, and the fleet reports its night.

What these tests pin down, in the order the plan states them:

- the VLM promotion lever is one revertible AgentRun: every promoted object carries the run id and its
  prior state/source, the generic revert restores them, and an object a person took over in the
  meantime is left alone;
- `maybe_unblock_gate` declines with a reason rather than re-running (once per day, once per blocked
  run), and a synthetic gate refusal produces a committed run whose report names the levers, plus a
  `gate_batch_ready` notification aimed at the blocked training run;
- campaigns on the scheduler: a require-approval campaign ticked by the daemon surfaces a decision
  (awaiting_approval) instead of silently doing nothing or running the stage anyway;
- the digest waits for the night to finish (declines while any agent run is in flight) and then sends
  exactly one superseding notification per day.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime

import numpy as np
import pytest

from core.config import get_settings
from core.storage import get_object_store
from core.timebase import now_ns, seconds_to_ns

pytestmark = pytest.mark.db


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


def _clear_db_cache():
    from db.session import get_engine, get_sessionmaker

    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


def run_async(coro):
    _clear_db_cache()
    try:
        return asyncio.run(coro)
    finally:
        _clear_db_cache()


async def _seed_review_objects(class_name: str, n: int) -> tuple[uuid.UUID, list[uuid.UUID]]:
    """One session + frame (real jpg in the store, so the crop decode is the real path) + n review objects."""
    import cv2

    from db.models import Frame, Object
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    cid = onto.by_name(class_name).id
    store = get_object_store()
    store.ensure_bucket()
    sid, fid = uuid.uuid4(), uuid.uuid4()
    start = now_ns()
    img = np.random.default_rng(7).integers(30, 220, size=(480, 640, 3), dtype=np.uint8)
    _ok, buf = cv2.imencode(".jpg", img)
    uri = store.put_bytes(f"frames/{sid}/cam_f/{start}.jpg", buf.tobytes(), "image/jpeg")
    oids = []
    async with get_sessionmaker()() as db:
        db.add(DbSession(session_id=sid, vehicle_id="UNBLOCK-01", start_ts_ns=start,
                         end_ts_ns=start + seconds_to_ns(1), city="BLR", sensors={},
                         ontology_version=onto.version))
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=start, cam_id="cam_f", img_uri=uri,
                     width=640, height=480, quality=0.9))
        for i in range(n):
            oid = uuid.uuid4()
            oids.append(oid)
            db.add(Object(object_id=oid, frame_id=fid, class_id=cid, bbox=[10 + 40 * i, 100, 40 + 40 * i, 200],
                          conf=0.9 - 0.01 * i, source="fused", state="review", provenance={}, attrs={},
                          version=1))
        await db.commit()
    return fid, oids


class _ConfirmingVerifier:
    """Says yes to everything, as the class asked about. The lever's own logic is what is under test."""

    def __init__(self, class_name: str):
        self._cn = class_name

    def verify_object(self, img, bbox, class_id):
        from types import SimpleNamespace

        return SimpleNamespace(class_name=self._cn)


@requires_infra
def test_promote_class_writes_a_revertible_run(monkeypatch):
    """The lever's contract: accepted-with-vlm_review provenance, and one revert undoes the night."""
    from db.models import AgentRun, Object
    from db.session import get_sessionmaker
    from services.agent.runs import revert_run
    from services.labelops.vlm_promote import promote_class

    _fid, oids = run_async(_seed_review_objects("cattle", 3))
    run_id = uuid.uuid4()

    async def _flow():
        async with get_sessionmaker()() as db:
            db.add(AgentRun(run_id=run_id, kind="gate_unblock", scope={}, status="running",
                            policy={}, counts={}, changes={}, critic={}, created_by="tester"))
            await db.commit()

        res = await promote_class("cattle", per_class=2, min_conf=0.1, oversample=3,
                                  agent_run_id=run_id, verifier=_ConfirmingVerifier("cattle"))
        assert res["promoted"] == 2, res
        assert len(res["changes"]) == 2
        for ch in res["changes"].values():
            assert ch == {"from_state": "review", "from_source": "fused"}

        async with get_sessionmaker()() as db:
            promoted_ids = [uuid.UUID(k) for k in res["changes"]]
            for oid in promoted_ids:
                obj = await db.get(Object, oid)
                assert obj.state == "accepted" and obj.source == "vlm_review"
                assert obj.provenance.get("agent_run_id") == str(run_id), \
                    "without the stamp, revert_run refuses the object and the night is not undoable"

            # a person takes one over before the revert: theirs must stand
            taken = await db.get(Object, promoted_ids[0])
            taken.source = "human"
            run = await db.get(AgentRun, run_id)
            run.status, run.changes = "committed", res["changes"]
            await db.commit()

        async with get_sessionmaker()() as db:
            rev = await revert_run(db, run_id)
        assert rev["reverted"] == 1 and rev["skipped"] == 1

        async with get_sessionmaker()() as db:
            restored = await db.get(Object, promoted_ids[1])
            assert restored.state == "review" and restored.source == "fused"
            kept = await db.get(Object, promoted_ids[0])
            assert kept.source == "human", "revert must never touch what a person now owns"
            untouched = [o for o in oids if o not in promoted_ids][0]
            assert (await db.get(Object, untouched)).state == "review"

    run_async(_flow())


@requires_infra
def test_unblock_runs_once_per_day():
    from db.models import AgentRun
    from db.session import get_sessionmaker
    from services.agent.gate_unblock import maybe_unblock_gate

    async def _flow():
        async with get_sessionmaker()() as db:
            db.add(AgentRun(run_id=uuid.uuid4(), kind="gate_unblock", scope={}, status="committed",
                            policy={}, counts={}, changes={}, critic={}, created_by="tester"))
            await db.commit()
            res = await maybe_unblock_gate(db)
            assert res["ran"] is False and "already ran today" in res["reason"]

    run_async(_flow())


@requires_infra
def test_a_synthetic_refusal_produces_a_run_a_batch_and_a_notification(monkeypatch):
    """The plan's 0c verification, end to end on the test DB with the expensive levers stubbed."""
    from sqlalchemy import delete, select

    from db.models import AgentRun, LabelProject, ModelRun, Notification
    from db.session import get_sessionmaker
    from services.agent.gate_unblock import maybe_unblock_gate
    from services.autolabel.ontology import get_ontology

    run_name = f"blocked-{uuid.uuid4().hex[:6]}"
    calls: dict = {}

    async def fake_promote(class_name, **kw):
        calls.setdefault("promoted", []).append(class_name)
        return {"class": class_name, "seen": 5, "confirmed": 2, "promoted": 2,
                "changes": {str(uuid.uuid4()): {"from_state": "review", "from_source": "fused"}}}

    async def fake_materialize(db, run_id, project_id, *, budget=500, jobs_of=50):
        calls["materialized"] = {"run_id": run_id, "project_id": project_id, "budget": budget}
        return {"tasks": [{"class_name": "pedestrian", "task_id": "t-1", "n_frames": 3, "n_jobs": 1}],
                "total_frames": 3, "exhausted_classes": [], "rationale": "synthetic"}

    import services.flywheel.gate_directed as gd
    import services.labelops.vlm_promote as vp

    monkeypatch.setattr(vp, "promote_class", fake_promote)
    monkeypatch.setattr(gd, "materialize_gate_batch", fake_materialize)

    async def _flow():
        async with get_sessionmaker()() as db:
            await db.execute(delete(AgentRun).where(AgentRun.kind == "gate_unblock"))
            db.add(LabelProject(name=f"unblock-{uuid.uuid4().hex[:6]}", modality="image"))
            # a run the gate refused: pedestrian recall far below the 0.50 safety floor
            db.add(ModelRun(run_id=run_name, base_weights="yolo11n", dataset_name="synthetic",
                            metrics={"per_class_recall": {"pedestrian": 0.2}},
                            promoted=False, ontology_version=get_ontology().version))
            await db.commit()

        try:
            async with get_sessionmaker()() as db:
                res = await maybe_unblock_gate(db)
            assert res["ran"] is True and res["target_run"] == run_name, res

            workers = [t for t in asyncio.all_tasks()
                       if t.get_name() == "worker" and t is not asyncio.current_task()]
            await asyncio.gather(*workers)

            assert calls["promoted"] == ["pedestrian"]
            assert calls["materialized"]["run_id"] == run_name
            assert calls["materialized"]["project_id"] is not None

            async with get_sessionmaker()() as db:
                run = (await db.execute(select(AgentRun).where(AgentRun.kind == "gate_unblock")
                                        .order_by(AgentRun.created_at.desc()))).scalars().first()
                assert run.status == "committed"
                assert run.counts["target_run"] == run_name
                assert run.counts["levers"]["pedestrian"]["promoted"] == 2
                assert run.counts["batch"]["tasks"][0]["task_id"] == "t-1"

                note = (await db.execute(select(Notification).where(
                    Notification.kind == "gate_batch_ready",
                    Notification.subject_id == run_name))).scalars().first()
                assert note is not None, "an unblock nobody hears about is the old silence again"

                # once per blocked run: even tomorrow, the same run is not re-attempted
                run.created_at = datetime(2020, 1, 1, tzinfo=UTC)
                await db.commit()
                res2 = await maybe_unblock_gate(db)
                assert res2["ran"] is False and "already attempted" in res2["reason"]
        finally:
            async with get_sessionmaker()() as db:
                await db.execute(delete(Notification).where(Notification.subject_id == run_name))
                mr = await db.get(ModelRun, run_name)
                if mr:
                    await db.delete(mr)
                await db.commit()

    run_async(_flow())


@requires_infra
def test_daemon_tick_surfaces_a_campaign_decision_instead_of_running_the_stage():
    from sqlalchemy import select

    from db.models import Campaign, CampaignStep
    from db.session import get_sessionmaker
    from services.flywheel.campaign import maybe_tick_campaigns, stop_campaign

    name = f"tick-{uuid.uuid4().hex[:6]}"

    async def _flow():
        from services.flywheel.campaign import create_campaign

        async with get_sessionmaker()() as db:
            await create_campaign(db, name=name, class_name="cattle", target_value=0.6,
                                  label_budget=100, require_approval=True)
            res = await maybe_tick_campaigns(db)
            mine = [t for t in res["ticked"] if t["campaign"] == name]
            assert mine and mine[0]["action"] == "awaiting_approval"

            step = (await db.execute(
                select(CampaignStep).join(Campaign, Campaign.campaign_id == CampaignStep.campaign_id)
                .where(Campaign.name == name))).scalars().first()
            assert step is not None and step.awaiting == "approval to run mine", \
                "the board must show a queue of decisions, not a row that apparently stopped"

            c = (await db.execute(select(Campaign).where(Campaign.name == name))).scalars().first()
            await stop_campaign(db, str(c.campaign_id), reason="test cleanup")

    run_async(_flow())


@requires_infra
def test_digest_waits_for_the_night_then_sends_once():
    from sqlalchemy import delete, select

    from db.models import AgentRun, Notification
    from db.session import get_sessionmaker
    from services.agent.digest import maybe_send_digest

    marker = uuid.uuid4()

    async def _flow():
        today = datetime.now(UTC).date().isoformat()
        async with get_sessionmaker()() as db:
            await db.execute(delete(AgentRun).where(AgentRun.kind == "nightly_digest"))
            # settle anything left running by earlier tests, then add one genuinely running row
            from sqlalchemy import update

            await db.execute(update(AgentRun).where(AgentRun.status == "running")
                             .values(status="committed"))
            db.add(AgentRun(run_id=marker, kind="overnight_auditor", scope={}, status="running",
                            policy={}, counts={"demoted": 4}, changes={}, critic={},
                            created_by="tester"))
            await db.commit()

            res = await maybe_send_digest(db)
            assert res["ran"] is False and "still running" in res["reason"], \
                "a digest sent mid-night reports a night that has not happened yet"

            run = await db.get(AgentRun, marker)
            run.status = "committed"
            await db.commit()

            res2 = await maybe_send_digest(db)
            assert res2["ran"] is True

        workers = [t for t in asyncio.all_tasks()
                   if t.get_name() == "worker" and t is not asyncio.current_task()]
        await asyncio.gather(*workers)

        async with get_sessionmaker()() as db:
            note = (await db.execute(select(Notification).where(
                Notification.kind == "nightly_digest",
                Notification.subject_id == today))).scalars().first()
            assert note is not None
            assert any(r["kind"] == "overnight_auditor" for r in note.meta["runs"])
            assert "demoted" in note.title, "the auditor's demotions are the headline, not a footnote"

            res3 = await maybe_send_digest(db)
            assert res3["ran"] is False and "already sent" in res3["reason"]

            await db.execute(delete(Notification).where(Notification.kind == "nightly_digest"))
            await db.execute(delete(AgentRun).where(AgentRun.run_id == marker))
            await db.commit()

    run_async(_flow())
