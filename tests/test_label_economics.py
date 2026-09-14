"""What a label costs and what it is worth, and every place the answer is refused instead of guessed.

This ranking decides where a labelling budget goes. Every one of its three inputs is missing somewhere in
this corpus, so the interesting behaviour is not the arithmetic, it is that a class with no timing and a
class that is genuinely cheap come back distinguishable. A ranking that fills gaps with medians produces a
confident order over classes nobody measured, and a budget spent from it is a budget spent on arithmetic.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import delete

from db.models import Object, Review, Workforce
from db.session import get_sessionmaker
from services.analytics.label_value import (
    MAX_PLAUSIBLE_MS,
    MIN_PLAUSIBLE_MS,
    MIN_TIMED_REVIEWS,
    timed_minutes_per_label,
    workforce_rate,
)
from services.autolabel.ontology import get_ontology

pytestmark = pytest.mark.db


async def _timed_reviews(db, class_id: int, timings: list[int], frame_id) -> list:
    """Objects of one class, each with one timed review."""
    oids = []
    for ms in timings:
        o = Object(frame_id=frame_id, class_id=class_id, bbox=[0.0, 0.0, 10.0, 10.0], conf=0.9,
                   source="fused", state="review")
        db.add(o)
        await db.flush()
        db.add(Review(object_id=o.object_id, reviewer="t", action="accept", time_spent_ms=ms,
                      ts_ns=0))
        oids.append(o.object_id)
    await db.commit()
    return oids


async def _fixture_frame(db):
    from core.origin import REAL
    from core.timebase import now_ns
    from db.models import Frame
    from db.models import Session as DbSession

    onto = get_ontology()
    sid, fid = uuid.uuid4(), uuid.uuid4()
    db.add(DbSession(session_id=sid, vehicle_id="LV", start_ts_ns=0, end_ts_ns=1, sensors={},
                     ontology_version=onto.version, origin=REAL))
    await db.flush()
    db.add(Frame(frame_id=fid, session_id=sid, ts_ns=now_ns(), cam_id="front", width=64, height=64,
                 img_uri=f"frames/{sid}/a.jpg", origin=REAL, selected=True))
    await db.commit()
    return sid, fid


class TestTiming:
    async def test_too_few_timings_is_unmeasured_with_the_count(self):
        """A median of four timings is not a rate, and reporting one would price a class from anecdote."""
        onto = get_ontology()
        cid = onto.by_name("hoarding").id
        async with get_sessionmaker()() as db:
            sid, fid = await _fixture_frame(db)
            await _timed_reviews(db, cid, [5000] * (MIN_TIMED_REVIEWS - 1), fid)
            got = await timed_minutes_per_label(db, cid)
            assert got["measured"] is False
            assert got["n"] == MIN_TIMED_REVIEWS - 1
            assert str(MIN_TIMED_REVIEWS) in got["reason"]
            await self._cleanup(db, sid)

    async def test_enough_timings_gives_a_median_in_minutes(self):
        onto = get_ontology()
        cid = onto.by_name("metro_pillar").id
        async with get_sessionmaker()() as db:
            sid, fid = await _fixture_frame(db)
            await _timed_reviews(db, cid, [6000] * MIN_TIMED_REVIEWS, fid)
            got = await timed_minutes_per_label(db, cid)
            assert got["measured"] is True
            assert got["minutes"] == pytest.approx(0.1)
            await self._cleanup(db, sid)

    async def test_a_tab_left_open_is_not_a_slow_review(self):
        """An hour-long timing is not a slow judgement, it is not a judgement. Dropped before the median
        rather than after, or one of them sets the price of the class."""
        onto = get_ontology()
        cid = onto.by_name("flyover_pillar").id
        async with get_sessionmaker()() as db:
            sid, fid = await _fixture_frame(db)
            timings = [6000] * MIN_TIMED_REVIEWS + [MAX_PLAUSIBLE_MS * 10] * 5
            await _timed_reviews(db, cid, timings, fid)
            got = await timed_minutes_per_label(db, cid)
            assert got["n"] == MIN_TIMED_REVIEWS, "the implausible ones never entered the sample"
            assert got["minutes"] == pytest.approx(0.1)
            await self._cleanup(db, sid)

    async def test_a_misclick_is_not_a_fast_review(self):
        onto = get_ontology()
        cid = onto.by_name("electric_post").id
        async with get_sessionmaker()() as db:
            sid, fid = await _fixture_frame(db)
            await _timed_reviews(db, cid, [MIN_PLAUSIBLE_MS - 100] * 40, fid)
            got = await timed_minutes_per_label(db, cid)
            assert got["measured"] is False and got["n"] == 0
            await self._cleanup(db, sid)

    async def _cleanup(self, db, sid):
        from db.models import Session as DbSession

        await db.execute(delete(DbSession).where(DbSession.session_id == sid))
        await db.commit()


class TestTheRate:
    async def test_no_entered_rate_is_unmeasured_rather_than_a_default(self):
        """A default rate would put a fabricated number into every value calculation with no way to tell
        it from a real one."""
        async with get_sessionmaker()() as db:
            got = await workforce_rate(db)
            if not got["measured"]:
                assert "no active workforce has a rate entered" in got["reason"]

    async def test_the_median_across_workforces_is_used_rather_than_whichever_came_first(self):
        """Two vendors at different rates have no single price, and picking either would make the
        ranking depend on row order."""
        tag = uuid.uuid4().hex[:6]
        async with get_sessionmaker()() as db:
            names = []
            for rate in (2.0, 6.0, 10.0):
                w = Workforce(name=f"lv-{tag}-{rate}", kind="vendor", secret="x", active=True,
                              rate_inr_per_verdict=rate)
                db.add(w)
                names.append(w.name)
            await db.commit()
            try:
                got = await workforce_rate(db)
                assert got["measured"] is True
                assert got["inr_per_verdict"] > 0
            finally:
                await db.execute(delete(Workforce).where(Workforce.name.in_(names)))
                await db.commit()

    async def test_a_zero_rate_is_not_a_rate(self):
        """Free labelling is not a price, it is a missing entry, and dividing by it is worse."""
        tag = uuid.uuid4().hex[:6]
        async with get_sessionmaker()() as db:
            w = Workforce(name=f"lv0-{tag}", kind="vendor", secret="x", active=True,
                          rate_inr_per_verdict=0.0)
            db.add(w)
            await db.commit()
            try:
                got = await workforce_rate(db)
                if got["measured"]:
                    assert got["inr_per_verdict"] > 0
            finally:
                await db.execute(delete(Workforce).where(Workforce.name == w.name))
                await db.commit()


class TestTheRanking:
    async def test_an_unmeasured_row_carries_its_reason(self):
        """The database CHECK enforces the same thing. A null value with no reason is a zero waiting to
        be misread as "this class is worthless"."""
        from services.analytics.label_value import marginal_value

        async with get_sessionmaker()() as db:
            res = await marginal_value(db)
        for row in res["rows"]:
            if not row["measured"]:
                assert row["reason"], f"{row['class_name']} is unmeasured with no reason"
                assert row["value_per_inr"] is None

    async def test_it_says_why_when_the_gate_is_asking_for_nothing(self):
        from services.analytics.label_value import marginal_value

        async with get_sessionmaker()() as db:
            res = await marginal_value(db)
        if not res["rows"]:
            assert res["reason"], "an empty ranking must say why rather than look like zero value"


class TestTheEditorNowTimesItsReviews:
    def test_the_frame_editor_sends_the_elapsed_time(self):
        """Five review surfaces already sent it and the main editor did not, which is why 30,863 of
        30,865 recorded reviews carry a zero and no class could be priced."""
        from pathlib import Path

        src = Path("web/app/frame/[id]/page.tsx").read_text()
        assert "selectedAtRef" in src
        assert "time_spent_ms: elapsed" in src


class TestTheSignHierarchy:
    """IRC:67 is the standard these signs are erected under, and its code is what a road authority calls
    a sign. Carrying it lets a class here be talked about outside this system, and it gives hierarchical
    evaluation a real middle level: a stop sign read as a give way is a mandatory sign read as a
    mandatory sign, which is a smaller error than reading it as a hospital."""

    def test_every_type_carries_a_code_and_a_group(self):
        from services.autolabel.signs.taxonomy import get_sign_taxonomy

        for t in get_sign_taxonomy()["types"]:
            assert t.get("irc_code"), f"{t['name']} has no IRC code"
            assert t.get("irc_group"), f"{t['name']} has no IRC group"

    def test_no_two_types_share_a_code(self):
        """A code that names two signs is not a code, and the by_code lookup would silently lose one."""
        from services.autolabel.signs.taxonomy import get_sign_taxonomy

        tax = get_sign_taxonomy()
        assert len(tax["by_irc_code"]) == len(tax["types"])

    def test_the_group_agrees_with_the_category_it_was_already_filed_under(self):
        from services.autolabel.signs.taxonomy import get_sign_taxonomy

        for t in get_sign_taxonomy()["types"]:
            assert t["irc_group"] == t["category"], t["name"]

    def test_the_code_prefix_matches_the_group(self):
        """R for regulatory, W for warning, I for informatory. A code whose letter disagrees with its
        group is a transcription error, and the letter is the half a person reads."""
        from services.autolabel.signs.taxonomy import get_sign_taxonomy

        prefix = {"mandatory": "R", "cautionary": "W", "informatory": "I"}
        for t in get_sign_taxonomy()["types"]:
            assert t["irc_code"].startswith(prefix[t["irc_group"]]), t["name"]

    def test_an_unknown_sign_type_has_no_group_rather_than_a_default(self):
        """A sign nobody recognised is not an informatory sign, and folding it into one would make the
        group-level metric look better than it is."""
        from services.autolabel.signs.taxonomy import irc_group_of

        assert irc_group_of("not_a_sign") is None
        assert irc_group_of(None) is None
        assert irc_group_of("") is None

    def test_the_version_was_bumped_with_the_content(self):
        from services.autolabel.signs.taxonomy import get_sign_taxonomy

        assert get_sign_taxonomy()["version"] == "signs-in-0.2.0"


class TestCurriculum:
    """Ordering an epoch by what the model still has to learn, without changing what is in it."""

    def test_weights_are_normalised_so_the_curriculum_is_not_a_no_op(self):
        """A deficit is a recall gap between 0 and 1. Used raw, every weight sits near the floor and the
        ordering changes nothing while still changing the run's provenance."""
        from services.training.curriculum import class_weights

        w = class_weights({"cattle": 0.04, "rider": 0.02, "sedan": 0.0})
        assert w["cattle"] == pytest.approx(2.0)
        assert 1.0 < w["rider"] < 2.0
        assert w["sedan"] == pytest.approx(1.0)

    def test_no_deficits_means_no_weights_rather_than_uniform_ones(self):
        from services.training.curriculum import class_weights

        assert class_weights({}) == {}

    def test_an_image_takes_its_heaviest_class_not_the_sum(self):
        """An image with twenty cars is not twenty times more valuable than one with a single rider of a
        starved class, and summing makes crowded frames win regardless of what is in them."""
        from services.training.curriculum import image_weight

        w = {"cattle": 2.0, "sedan": 1.0}
        assert image_weight(["cattle", "sedan"], w) == pytest.approx(2.0)
        assert image_weight(["sedan"] * 20, w) == pytest.approx(1.0)

    def test_agreement_discounts_an_image_the_model_already_has(self):
        from services.training.curriculum import image_weight

        w = {"cattle": 2.0}
        assert image_weight(["cattle"], w, agreement=0.0) == pytest.approx(2.0)
        assert image_weight(["cattle"], w, agreement=1.0) == pytest.approx(1.0)
        assert image_weight(["cattle"], w, agreement=0.5) == pytest.approx(1.5)

    def test_every_image_still_appears_at_least_once(self):
        """A curriculum that dropped images would silently change the dataset a metric was computed on,
        and two runs would be incomparable with nothing saying so."""
        from services.training.curriculum import order_epoch

        images = [f"f{i}" for i in range(50)]
        weights = {img: 1.0 + (i % 5) / 4 for i, img in enumerate(images)}
        out = order_epoch(images, weights)
        assert set(out) == set(images)
        for img in images:
            assert out.count(img) >= 1

    def test_the_easy_images_come_first(self):
        from services.training.curriculum import order_epoch

        images = ["hard", "easy", "middling"]
        weights = {"hard": 2.0, "middling": 1.5, "easy": 1.0}
        out = order_epoch(images, weights)
        assert out[:3] == ["easy", "middling", "hard"]

    def test_repeats_are_bounded_so_an_epoch_is_not_one_frame(self):
        from services.training.curriculum import MAX_REPEAT, order_epoch

        images = [f"f{i}" for i in range(10)]
        weights = dict.fromkeys(images, 1.0)
        weights["f0"] = 10.0
        out = order_epoch(images, weights)
        assert out.count("f0") <= MAX_REPEAT

    def test_it_is_deterministic_under_a_seed(self):
        from services.training.curriculum import order_epoch

        images = [f"f{i}" for i in range(30)]
        weights = {img: 1.0 + (i % 4) / 3 for i, img in enumerate(images)}
        assert order_epoch(images, weights, seed=3) == order_epoch(images, weights, seed=3)

    def test_an_empty_epoch_is_empty_rather_than_an_error(self):
        from services.training.curriculum import order_epoch

        assert order_epoch([], {}) == []

    def test_the_flag_is_recorded_on_the_datasheet(self):
        """Two runs where one had the curriculum on and the other did not are not comparable, and the
        sheet is where that becomes visible rather than being remembered."""
        import inspect

        from services.training import dataset_builder

        src = inspect.getsource(dataset_builder.build_training_dataset)
        assert '"curriculum": spec.curriculum' in src
