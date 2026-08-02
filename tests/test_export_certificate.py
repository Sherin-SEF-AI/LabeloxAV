"""The quality certificate: what a buyer is actually purchasing.

346 dataset commits shipped from this system with a datasheet saying what is in the release and nothing
about how good it is. These tests pin the three things that make the replacement worth signing: that per
class precision is counted on the right axis, that a class with too little gold says so instead of showing a
number, and that editing the manifest breaks the signature.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from core.config import get_settings
from services.export.certificate import (
    MIN_GOLD_PER_CLASS,
    build_certificate,
    gold_membership_fingerprint,
    render_certificate_markdown,
    verify_certificate,
)


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


def _clear():
    from db.session import get_engine, get_sessionmaker

    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


def run_async(coro):
    _clear()
    try:
        return asyncio.run(coro)
    finally:
        _clear()


# --- signing ----------------------------------------------------------------------------------


def test_editing_a_number_invalidates_the_signature():
    """The point of signing. A certificate a recipient cannot verify is a claim, not evidence."""
    manifest = {"commit_id": "c1", "overall": {"precision": {"p": 0.7}}}
    sig = __import__("hmac").new(b"k", __import__("json").dumps(
        manifest, sort_keys=True, separators=(",", ":")).encode(), __import__("hashlib").sha256).hexdigest()
    assert verify_certificate(manifest, sig, "k")

    tampered = {"commit_id": "c1", "overall": {"precision": {"p": 0.95}}}
    assert not verify_certificate(tampered, sig, "k")


def test_an_absent_signature_does_not_verify():
    assert not verify_certificate({"a": 1}, "", "k")
    assert not verify_certificate({"a": 1}, None, "k")


def test_the_gold_fingerprint_ignores_ordering_but_not_membership():
    """Order-dependent would make the check useless through false alarms; membership-blind would make it
    useless full stop."""
    a = gold_membership_fingerprint(["o1", "o2", "o3"])
    assert a == gold_membership_fingerprint(["o3", "o1", "o2"])
    assert a != gold_membership_fingerprint(["o1", "o2", "o4"])


# --- the arithmetic ---------------------------------------------------------------------------


@requires_infra
def test_precision_is_charged_to_the_predicted_class_and_recall_to_the_actual_one():
    """The classic error this is written to avoid.

    A false positive belongs to the class the model claimed, a false negative to the class that was really
    there. Using one class axis for both silently reports a different quantity than the label says: charge
    a rider-shaped false positive to 'pedestrian' and pedestrian precision improves for a mistake it did
    not make.
    """
    from sqlalchemy import delete

    from db.models import EvalPatch
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.export.certificate import per_class_precision_recall

    onto = get_ontology()
    ped = next(c.id for c in onto.classes if c.name == "pedestrian")
    rider = next(c.id for c in onto.classes if c.name == "rider")
    eval_id = uuid.uuid4()

    async def _flow():
        async with get_sessionmaker()() as db:
            # 12 pedestrians present, 10 found. One rider was called a pedestrian.
            for _ in range(10):
                db.add(EvalPatch(eval_id=eval_id, outcome="tp", gt_class_id=ped, pred_class_id=ped))
            for _ in range(2):
                db.add(EvalPatch(eval_id=eval_id, outcome="fn", gt_class_id=ped, pred_class_id=None))
            db.add(EvalPatch(eval_id=eval_id, outcome="fp", gt_class_id=None, pred_class_id=ped))
            # and 11 riders present, all missed
            for _ in range(11):
                db.add(EvalPatch(eval_id=eval_id, outcome="fn", gt_class_id=rider, pred_class_id=None))
            await db.commit()

        async with get_sessionmaker()() as db:
            out = await per_class_precision_recall(db, str(eval_id))

        p = out["pedestrian"]
        assert p["tp"] == 10 and p["fp"] == 1 and p["fn"] == 2
        assert p["precision"]["p"] == round(10 / 11, 4)   # of what it called pedestrian
        assert p["recall"]["p"] == round(10 / 12, 4)      # of the pedestrians that were there
        r = out["rider"]
        assert r["recall"]["p"] == 0.0 and r["precision"] is None  # never predicted, so precision undefined

        async with get_sessionmaker()() as db:
            await db.execute(delete(EvalPatch).where(EvalPatch.eval_id == eval_id))
            await db.commit()

    run_async(_flow())


@requires_infra
def test_a_class_with_too_little_gold_is_named_unmeasured_rather_than_scored():
    """A certificate whose weak numbers look like its strong ones is worse than one that admits the gap.

    Nine gold instances at 100% recall is not a 1.0; it is an interval from about 0.7 to 1.0 that happens
    to have a flattering midpoint.
    """
    from sqlalchemy import delete

    from db.models import EvalPatch
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.export.certificate import per_class_precision_recall

    onto = get_ontology()
    cattle = next(c.id for c in onto.classes if c.name == "cattle")
    eval_id = uuid.uuid4()
    n = MIN_GOLD_PER_CLASS - 1

    async def _flow():
        async with get_sessionmaker()() as db:
            for _ in range(n):
                db.add(EvalPatch(eval_id=eval_id, outcome="tp", gt_class_id=cattle, pred_class_id=cattle))
            await db.commit()
        async with get_sessionmaker()() as db:
            out = await per_class_precision_recall(db, str(eval_id))

        c = out["cattle"]
        assert c["measured"] is False
        assert "not measured" in c["note"] and str(n) in c["note"]
        # the rate is still computed and still carries its interval; it is the claim that is withheld
        assert c["recall"]["p"] == 1.0 and c["recall"]["lo"] < 0.8

        async with get_sessionmaker()() as db:
            await db.execute(delete(EvalPatch).where(EvalPatch.eval_id == eval_id))
            await db.commit()

    run_async(_flow())


@requires_infra
def test_a_certificate_cannot_be_issued_without_naming_what_it_measured_against():
    """An unattributable number is worse than no number, because it gets quoted anyway."""
    from db.session import get_sessionmaker

    async def _flow():
        async with get_sessionmaker()() as db:
            r = await build_certificate(db, commit_id="c1", eval_id=str(uuid.uuid4()),
                                        gold_id="does-not-exist", model_version="m1", key="k")
        assert "error" in r and "gold set not found" in r["error"]

    run_async(_flow())


@requires_infra
def test_the_certificate_leads_with_what_it_did_not_measure():
    """A quality document that buries its gaps below its headline numbers is doing the opposite of its job."""
    from sqlalchemy import delete

    from db.models import EvalPatch, GoldSet
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    ped = next(c.id for c in onto.classes if c.name == "pedestrian")
    cattle = next(c.id for c in onto.classes if c.name == "cattle")
    eval_id = uuid.uuid4()
    gold_id = f"gold-cert-{uuid.uuid4().hex[:8]}"

    async def _flow():
        async with get_sessionmaker()() as db:
            oids = [str(uuid.uuid4()) for _ in range(40)]
            db.add(GoldSet(gold_id=gold_id, name="cert test", spec={}, object_ids=oids,
                           n_objects=len(oids), n_frames=4, ontology_version=onto.version,
                           metrics={}, track_ids=[], tracks_sealed=False))
            for _ in range(30):
                db.add(EvalPatch(eval_id=eval_id, outcome="tp", gt_class_id=ped, pred_class_id=ped))
            for _ in range(4):
                db.add(EvalPatch(eval_id=eval_id, outcome="fp", gt_class_id=None, pred_class_id=ped))
            for _ in range(3):     # cattle: below the floor
                db.add(EvalPatch(eval_id=eval_id, outcome="tp", gt_class_id=cattle, pred_class_id=cattle))
            await db.commit()

        async with get_sessionmaker()() as db:
            cert = await build_certificate(db, commit_id="c-release-1", eval_id=str(eval_id),
                                           gold_id=gold_id, model_version="real-v2", key="secret")
        m = cert["manifest"]
        assert m["classes_measured"] == ["pedestrian"]
        assert m["classes_not_measured"] == ["cattle"]
        assert m["gold_objects"] == 40
        # The overall aggregates every patch, including the three cattle that are individually below the
        # floor: 33 of 37 predictions correct. Dropping a class from the total because its own interval is
        # too wide to quote would be cherry-picking, and would make the headline figure disagree with a
        # hand count of the patch rows, which is the property that makes this auditable at all.
        assert m["overall"]["precision"]["p"] == round(33 / 37, 4)
        assert m["overall"]["tp"] == 33 and m["overall"]["fp"] == 4
        assert verify_certificate(m, cert["signature"], "secret")
        assert not verify_certificate(m, cert["signature"], "other-key")

        md = render_certificate_markdown(cert)
        # the gaps appear before the headline numbers, not after them
        assert md.index("Not measured") < md.index("## Overall")
        assert "`cattle`" in md and "not measured" in md
        assert "Wilson" in md

        async with get_sessionmaker()() as db:
            await db.execute(delete(EvalPatch).where(EvalPatch.eval_id == eval_id))
            await db.execute(delete(GoldSet).where(GoldSet.gold_id == gold_id))
            await db.commit()

    run_async(_flow())
