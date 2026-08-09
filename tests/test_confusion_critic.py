"""A critic aimed at the one distinction this corpus actually gets wrong.

Counted over the live Review trail: 296 class corrections, of which 250 involve `e_auto` and 206 involve
`motorcycle`. The single pair e_auto <-> motorcycle is 195 of them, 66% of every class correction ever made
here. The next contributor is `rider` at 35.

That pair is bidirectional: 125 corrections went e_auto -> motorcycle and 70 went the other way. An
asymmetric confusion is a bias and a threshold fixes it. A symmetric one is a boundary the annotators
themselves are not applying consistently, and a critic forced to choose between the two inherits that
ambiguity while reporting it as a decision.

So the tests below are mostly about refusing to manufacture certainty: `uncertain` is a real verdict, an
unnameable disagreement is not filed, and an ambiguous crop becomes a gold-set candidate rather than a
suspected error. The confusion set itself is derived from the trail rather than hardcoded, so a test that
seeds different corrections gets a different set.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import func, select

from core.timebase import now_ns
from db.models import ErrorCandidate, Frame, Object, Review
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.agent.confusion_critic import (
    KIND,
    VERDICTS,
    confusion_neighbourhood,
    confusion_pairs,
    critic_schema,
    run_confusion_sweep,
)
from services.autolabel.ontology import get_ontology

pytestmark = pytest.mark.db

# Every Review this module writes is stamped with this reviewer so it can be removed again. The suite shares
# one database and does not reset it between runs, so a seeded correction is not scoped to a test, it is
# scoped to the database. These rows leaked into test_judge_calibration, which builds its items by scanning
# the whole Review trail, and broke it in a later run of a different file: the failure survived even after
# the code that caused it was stashed, which is what makes this class of pollution expensive to trace.
SEED_REVIEWER = "test-confusion-critic"


@pytest.fixture(autouse=True)
async def _cleanup_seeded_reviews():
    yield
    from sqlalchemy import delete

    async with get_sessionmaker()() as db:
        await db.execute(delete(Review).where(Review.reviewer == SEED_REVIEWER))
        await db.commit()


class _FakeVlm:
    """A judge that answers from a script, so the sweep's bookkeeping is what is under test."""

    provider_name = "fake"

    def __init__(self, replies: list[dict]):
        self.replies = list(replies)
        self.seen: list[dict] = []

    def chat_json(self, prompt, *, image_jpeg=None, temperature=0.0, schema=None, model=None):
        self.seen.append({"prompt": prompt, "schema": schema})
        return self.replies.pop(0) if self.replies else {"verdict": "agree"}


async def _seed_corrections(db, pairs: list[tuple[str, str, int]]) -> None:
    """Write Review rows that say `old -> new`, `n` times."""
    onto = get_ontology()
    sess = DbSession(session_id=uuid.uuid4(), vehicle_id="TEST-CC", start_ts_ns=0, end_ts_ns=1,
                     ontology_version="test")
    db.add(sess)
    frame = Frame(frame_id=uuid.uuid4(), session_id=sess.session_id, ts_ns=now_ns(), cam_id="cam_f",
                  img_uri="s3://labeloxav/t.jpg", width=1920, height=1080)
    db.add(frame)
    await db.flush()
    for old, new, n in pairs:
        for _ in range(n):
            o = Object(object_id=uuid.uuid4(), frame_id=frame.frame_id, class_id=onto.by_name(old).id,
                       bbox=[10, 10, 100, 100], conf=0.5, source="auto_accept", state="review")
            db.add(o)
            await db.flush()
            db.add(Review(review_id=uuid.uuid4(), object_id=o.object_id, reviewer=SEED_REVIEWER,
                          action="reclassify",
                          ts_ns=now_ns(),
                          before={"class_id": onto.by_name(old).id},
                          after={"class_id": onto.by_name(new).id}))
    await db.commit()


# ------------------------------------------------------------------------------- the schema

def test_the_verdict_is_constrained_to_three_answers():
    """An unparseable "probably not, though it could be" is a discarded call, not a soft one."""
    assert critic_schema(["e_auto", "motorcycle"])["properties"]["verdict"]["enum"] == list(VERDICTS)


def test_uncertain_is_one_of_them():
    """The pair is symmetric in the corpus. Forcing a binary choice produces confident noise."""
    assert "uncertain" in VERDICTS


def test_the_suggestion_is_limited_to_the_confusion_neighbourhood():
    """Offering all 186 classes invites an answer to a question nobody asked."""
    s = critic_schema(["e_auto", "motorcycle", "rider"])
    assert s["properties"]["suggested_class"]["enum"] == ["e_auto", "motorcycle", "rider"]


def test_a_suggestion_is_not_demanded():
    """It is meaningless on agree and misleading on uncertain, so requiring it would force the model to name
    a replacement it does not believe in."""
    assert "suggested_class" not in critic_schema(["e_auto"])["required"]


def test_a_reason_is_demanded():
    """Left optional, it came back empty in 28 of 28 queued rows, which is what an optional field means to a
    model scored on nothing else. A candidate whose argument is missing is one a reviewer can only judge on
    vibes, and it is useless for calibrating the critic afterwards."""
    assert "reason" in critic_schema(["e_auto"])["required"]


# ------------------------------------------------------------------------------- deriving the set

async def test_the_confusion_set_comes_from_the_corrections_not_a_constant():
    async with get_sessionmaker()() as db:
        await _seed_corrections(db, [("cattle", "truck", 5)])
        pairs = await confusion_pairs(db)
    names = [set(p["classes"]) for p in pairs]
    assert {"cattle", "truck"} in names


async def test_a_pair_corrected_both_ways_is_marked_symmetric():
    """The diagnosis, not a detail: asymmetric is a model bias, symmetric is a boundary humans disagree on."""
    async with get_sessionmaker()() as db:
        await _seed_corrections(db, [("cattle", "truck", 4), ("truck", "cattle", 3)])
        pairs = await confusion_pairs(db)
    p = next(p for p in pairs if set(p["classes"]) == {"cattle", "truck"})
    assert p["symmetric"] is True
    assert p["count"] == p["forward"] + p["reverse"]


async def test_a_one_off_correction_is_not_a_confusion():
    """With 296 corpus-wide corrections, one occurrence is an incident. Admitting it would send the critic
    after a boundary one person crossed once."""
    async with get_sessionmaker()() as db:
        await _seed_corrections(db, [("cattle", "pole", 1)])
        pairs = await confusion_pairs(db, min_count=3)
    assert not any(set(p["classes"]) == {"cattle", "pole"} for p in pairs)


async def test_the_neighbourhood_always_offers_the_current_label():
    """"The label is right" has to be sayable, or agreement is unreachable."""
    async with get_sessionmaker()() as db:
        await _seed_corrections(db, [("cattle", "truck", 5)])
        hood = await confusion_neighbourhood(db, "cattle")
    assert hood["candidates"][0] == "cattle"
    assert "truck" in hood["candidates"]


async def test_a_class_nothing_is_confused_with_says_so_rather_than_sweeping():
    async with get_sessionmaker()() as db:
        hood = await confusion_neighbourhood(db, "cattle", min_count=10_000)
    assert hood["candidates"] == ["cattle"]
    assert "nothing" in hood["detail"]


async def test_a_class_outside_the_ontology_is_refused():
    async with get_sessionmaker()() as db:
        hood = await confusion_neighbourhood(db, "not_a_real_class")
    assert hood["candidates"] == []


# ------------------------------------------------------------------------------- the sweep

async def _seed_objects(db, cls: str, n: int) -> list[uuid.UUID]:
    onto = get_ontology()
    sess = DbSession(session_id=uuid.uuid4(), vehicle_id="TEST-CC2", start_ts_ns=0, end_ts_ns=1,
                     ontology_version="test")
    db.add(sess)
    frame = Frame(frame_id=uuid.uuid4(), session_id=sess.session_id, ts_ns=now_ns(), cam_id="cam_f",
                  img_uri="s3://labeloxav/t.jpg", width=1920, height=1080)
    db.add(frame)
    await db.flush()
    ids = []
    for _ in range(n):
        o = Object(object_id=uuid.uuid4(), frame_id=frame.frame_id, class_id=onto.by_name(cls).id,
                   bbox=[10, 10, 100, 100], conf=0.5, source="auto_accept", state="review")
        db.add(o)
        ids.append(o.object_id)
    await db.commit()
    return ids


@pytest.fixture
def fake_image(monkeypatch):
    import numpy as np

    import services.recall.backends as b

    monkeypatch.setattr(b, "load_image_bgr", lambda store, uri: np.full((300, 300, 3), 128, np.uint8))
    return b


async def test_a_crop_too_small_to_settle_the_question_is_not_judged(monkeypatch):
    """The first sweep filed confident disagreements about crops of 36x25 pixels. The model answers those;
    the answer is noise wearing a verdict's clothes."""
    import numpy as np

    import services.recall.backends as b

    monkeypatch.setattr(b, "load_image_bgr", lambda store, uri: np.full((20, 20, 3), 128, np.uint8))
    async with get_sessionmaker()() as db:
        await _seed_corrections(db, [("cattle", "truck", 5)])
        await _seed_objects(db, "cattle", 2)
        out = await run_confusion_sweep(db, focus="cattle", limit=2,
                                        client=_FakeVlm([{"verdict": "disagree",
                                                          "suggested_class": "truck", "reason": "r"}]))
    assert out["counts"]["too_small"] == 2
    assert out["judged"] == 0 and out["queued_as_errors"] == 0


async def test_a_confident_disagreement_becomes_an_error_candidate(fake_image):
    """It enters the existing triage queue, which already has a verdict path, rather than a private one."""
    async with get_sessionmaker()() as db:
        await _seed_corrections(db, [("cattle", "truck", 5)])
        await _seed_objects(db, "cattle", 1)
        vlm = _FakeVlm([{"verdict": "disagree", "suggested_class": "truck", "reason": "it has a cargo bed"}])
        out = await run_confusion_sweep(db, focus="cattle", limit=1, client=vlm)

        assert out["queued_as_errors"] == 1
        row = (await db.execute(
            select(ErrorCandidate).where(
                ErrorCandidate.object_id == uuid.UUID(out["error_candidate_ids"][0])))).scalars().first()
    assert row is not None
    assert row.proposed_label["class_name"] == "truck"
    assert "cargo bed" in row.detail["reason"]


async def test_an_uncertain_verdict_is_a_gold_candidate_not_a_suspected_error(fake_image):
    """The model and the label disagreeing about a boundary humans also disagree about is not evidence of an
    error. It is evidence the boundary needs adjudicating once and writing down."""
    async with get_sessionmaker()() as db:
        await _seed_corrections(db, [("cattle", "truck", 5)])
        await _seed_objects(db, "cattle", 1)
        before = (await db.execute(
            select(func.count()).select_from(ErrorCandidate).where(ErrorCandidate.kind == KIND))).scalar()
        vlm = _FakeVlm([{"verdict": "uncertain"}])
        out = await run_confusion_sweep(db, focus="cattle", limit=1, client=vlm)
        after = (await db.execute(
            select(func.count()).select_from(ErrorCandidate).where(ErrorCandidate.kind == KIND))).scalar()

    assert len(out["gold_candidates"]) == 1
    assert out["queued_as_errors"] == 0
    assert after == before


async def test_agreement_queues_nothing(fake_image):
    async with get_sessionmaker()() as db:
        await _seed_corrections(db, [("cattle", "truck", 5)])
        await _seed_objects(db, "cattle", 1)
        out = await run_confusion_sweep(db, focus="cattle", limit=1, client=_FakeVlm([{"verdict": "agree"}]))
    assert out["counts"]["agree"] == 1 and out["queued_as_errors"] == 0


async def test_a_disagreement_that_names_nothing_is_not_filed(fake_image):
    """A row a reviewer can only dismiss is worse than no row: it costs attention and teaches nothing."""
    async with get_sessionmaker()() as db:
        await _seed_corrections(db, [("cattle", "truck", 5)])
        await _seed_objects(db, "cattle", 1)
        out = await run_confusion_sweep(db, focus="cattle", limit=1,
                                        client=_FakeVlm([{"verdict": "disagree"}]))
    assert out["queued_as_errors"] == 0
    assert out["counts"]["disagree_unnamed"] == 1


async def test_a_disagreement_naming_the_current_label_is_not_a_disagreement(fake_image):
    async with get_sessionmaker()() as db:
        await _seed_corrections(db, [("cattle", "truck", 5)])
        await _seed_objects(db, "cattle", 1)
        out = await run_confusion_sweep(
            db, focus="cattle", limit=1,
            client=_FakeVlm([{"verdict": "disagree", "suggested_class": "cattle"}]))
    assert out["queued_as_errors"] == 0


async def test_an_unparsed_reply_is_counted_not_folded_into_uncertain(fake_image):
    """Only reachable on an unconstrained backend, and folding it in would inflate the abstention rate with
    parse failures, which is the number the whole design rests on."""
    async with get_sessionmaker()() as db:
        await _seed_corrections(db, [("cattle", "truck", 5)])
        await _seed_objects(db, "cattle", 1)
        out = await run_confusion_sweep(db, focus="cattle", limit=1,
                                        client=_FakeVlm([{"verdict": "maybe?"}]))
    assert out["counts"]["unparsed"] == 1
    assert out["counts"].get("uncertain", 0) == 0


async def test_the_sweep_sends_the_schema_and_names_the_current_label(fake_image):
    async with get_sessionmaker()() as db:
        await _seed_corrections(db, [("cattle", "truck", 5)])
        await _seed_objects(db, "cattle", 1)
        vlm = _FakeVlm([{"verdict": "agree"}])
        await run_confusion_sweep(db, focus="cattle", limit=1, client=vlm)
    call = vlm.seen[0]
    assert call["schema"]["properties"]["verdict"]["enum"] == list(VERDICTS)
    assert "cattle" in call["prompt"], "the judge must be shown the claim it is checking"


async def test_one_unreadable_crop_does_not_end_the_sweep(monkeypatch):
    """A sweep is a background job over hundreds of crops. One missing image must not stop it."""
    import services.recall.backends as b

    def boom(store, uri):
        raise RuntimeError("NoSuchKey")

    monkeypatch.setattr(b, "load_image_bgr", boom)
    async with get_sessionmaker()() as db:
        await _seed_corrections(db, [("cattle", "truck", 5)])
        await _seed_objects(db, "cattle", 2)
        out = await run_confusion_sweep(db, focus="cattle", limit=2, client=_FakeVlm([]))
    assert out["counts"]["error"] == 2
    assert out["judged"] == 0


async def test_a_focus_class_with_no_confusions_does_not_sweep(fake_image):
    """Nothing to check against means nothing to ask, and asking anyway spends GPU on an empty question."""
    async with get_sessionmaker()() as db:
        await _seed_objects(db, "cattle", 3)
        out = await run_confusion_sweep(db, focus="cattle", limit=3, min_count=10_000,
                                        client=_FakeVlm([{"verdict": "agree"}]))
    assert out["judged"] == 0


async def test_a_label_that_is_wrong_in_an_unlisted_way_gets_its_own_verdict(fake_image):
    """The first good sweep returned "a large billboard or sign with text and graphics, not a vehicle of any
    kind" and had to file it as `motorcycle`, because the neighbourhood is built from vehicle confusions and
    offered nowhere else to go. Without an escape the schema turns "this label is nonsense" into "this label
    is slightly the wrong vehicle", which is a worse answer than the model actually gave."""
    async with get_sessionmaker()() as db:
        await _seed_corrections(db, [("cattle", "truck", 5)])
        await _seed_objects(db, "cattle", 1)
        out = await run_confusion_sweep(
            db, focus="cattle", limit=1,
            client=_FakeVlm([{"verdict": "other", "reason": "this is a billboard, not an animal"}]))

        assert out["counts"]["other"] == 1
        assert len(out["out_of_set"]) == 1
        # Scoped to the object this sweep actually judged. The suite shares a database and the sweep samples
        # at random, so neither an unscoped `.first()` nor the seeded id is reliable: the first returns
        # whatever an earlier test left behind, the second assumes a sample that was never guaranteed.
        row = (await db.execute(
            select(ErrorCandidate).where(
                ErrorCandidate.object_id == uuid.UUID(out["out_of_set"][0])))).scalars().first()
    assert row is not None
    assert row.proposed_label is None, "the critic cannot name a class from this set, and must not pretend to"
    assert "billboard" in row.detail["reason"]


async def test_out_of_set_is_counted_apart_from_a_boundary_disagreement(fake_image):
    """A class boundary problem is fixed by adjudicating the boundary. This is not that, and pooling them
    would hide a contamination problem inside a taxonomy one."""
    async with get_sessionmaker()() as db:
        await _seed_corrections(db, [("cattle", "truck", 5)])
        await _seed_objects(db, "cattle", 2)
        out = await run_confusion_sweep(
            db, focus="cattle", limit=2,
            client=_FakeVlm([{"verdict": "other", "reason": "a hoarding"},
                             {"verdict": "disagree", "suggested_class": "truck", "reason": "cargo bed"}]))
    assert out["counts"]["other"] == 1 and out["counts"]["disagree"] == 1
    assert len(out["out_of_set"]) == 1 and len(out["error_candidate_ids"]) == 1
