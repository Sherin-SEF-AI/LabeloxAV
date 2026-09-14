"""The annotation canvas additions: describing an object, judging a tube, choosing what to do next.

The pure parts are tested here because each of them can be wrong in a way that still produces output. A
contact sheet that stretches its crops changes what the object looks like; a track sampler that takes the
first eight frames asks the hardest possible version of the question; and a phrase that is not normalised
sends the annotator's capitals to a text encoder that does not want them.
"""

from __future__ import annotations

import numpy as np
import pytest

from services.autolabel.describe import MAX_PHRASE_CHARS, normalize_phrase
from services.labelops.vlm_review import (
    TUBE_JUDGE,
    TUBE_SHEET_COLS,
    TUBE_SHEET_MAX,
    build_contact_sheet,
    sample_track_objects,
)


class TestThePhrase:
    def test_it_collapses_whitespace_and_case(self):
        assert normalize_phrase("  Cycle   Rickshaw ") == "cycle rickshaw"

    def test_an_empty_phrase_is_empty_rather_than_a_prompt(self):
        for junk in ("", "   ", "\n\t", None):
            assert normalize_phrase(junk) == ""

    def test_a_paragraph_is_bounded(self):
        assert len(normalize_phrase("a " * 500)) <= MAX_PHRASE_CHARS

    def test_normalising_is_idempotent(self):
        once = normalize_phrase("  Water   TANKER  ")
        assert normalize_phrase(once) == once


class TestTheContactSheet:
    def _crop(self, w: int, h: int, value: int = 200):
        return np.full((h, w, 3), value, dtype=np.uint8)

    def test_no_crops_yields_no_sheet_rather_than_a_blank_one(self):
        assert build_contact_sheet([]) is None
        assert build_contact_sheet([None, None]) is None

    def test_one_crop_makes_a_one_cell_sheet(self):
        sheet = build_contact_sheet([self._crop(50, 50)], cols=4, cell=100)
        assert sheet.shape == (100, 100, 3)

    def test_the_grid_grows_by_rows_not_by_stretching(self):
        sheet = build_contact_sheet([self._crop(20, 20)] * 5, cols=4, cell=64)
        assert sheet.shape == (2 * 64, 4 * 64, 3)

    def test_a_wide_crop_is_letterboxed_rather_than_squashed(self):
        """An aspect ratio is evidence about what a thing is. A squashed motorcycle looks like a
        different vehicle, and the judge is being asked exactly that question."""
        sheet = build_contact_sheet([self._crop(200, 50)], cols=1, cell=100)
        # 200x50 into a 100 cell scales to 100x25, so the top and bottom quarters stay background.
        assert sheet[0, 50].tolist() == [0, 0, 0], "the top band is padding"
        assert sheet[50, 50].tolist() != [0, 0, 0], "the middle band is the crop"

    def test_unreadable_crops_are_dropped_and_the_rest_still_tile(self):
        sheet = build_contact_sheet([self._crop(30, 30), None, self._crop(30, 30)], cols=2, cell=40)
        assert sheet.shape == (40, 2 * 40, 3)

    def test_the_default_grid_holds_one_full_sheet(self):
        assert TUBE_SHEET_MAX % TUBE_SHEET_COLS == 0


class TestTrackSampling:
    def test_a_short_track_is_taken_whole(self):
        objs = list(range(5))
        assert sample_track_objects(objs, n=8) == objs

    def test_a_long_track_is_spread_rather_than_truncated(self):
        """The first frames of a track are where the object is smallest and furthest away. A sheet made
        of those asks the hardest possible version of the question."""
        objs = list(range(100))
        got = sample_track_objects(objs, n=8)
        assert len(got) == 8
        assert got[0] == 0
        assert got[-1] > 80, "the sample reaches the end of the track"
        assert got == sorted(got), "and stays in track order"

    def test_the_sample_never_repeats_or_overruns(self):
        for length in (9, 17, 33, 64, 200):
            got = sample_track_objects(list(range(length)), n=8)
            assert len(set(got)) == len(got), f"length {length} repeated a frame"
            assert max(got) < length

    def test_the_tube_judge_has_its_own_name(self):
        """It writes against the same objects as the per-crop judge. Sharing a name would make one
        overwrite the other through the uniqueness key instead of the two being comparable."""
        from services.labelops.vlm_review import JUDGE

        assert TUBE_JUDGE != JUDGE


class TestJudgedPrecisionTakesAJudge:
    def test_the_default_is_the_per_crop_judge_so_existing_callers_are_unchanged(self):
        import inspect

        from services.labelops import vlm_review

        sig = inspect.signature(vlm_review.judged_precision)
        assert sig.parameters["judge"].default == vlm_review.JUDGE

    def test_it_filters_on_the_argument_rather_than_the_constant(self):
        import inspect

        from services.labelops import vlm_review

        src = inspect.getsource(vlm_review.judged_precision)
        assert "MachineVerdict.judge == judge" in src
        assert "MachineVerdict.judge == JUDGE" not in src


@pytest.mark.db
class TestNextObject:
    async def test_it_reports_no_next_object_rather_than_an_empty_list(self):
        """A frame with nothing unconfirmed is a finished frame, and saying so is different from a
        ranking that happened to come back empty."""
        import uuid

        from sqlalchemy import delete

        from core.origin import REAL
        from core.timebase import now_ns
        from db.models import Frame
        from db.models import Session as DbSession
        from db.session import get_sessionmaker
        from services.api.routers.objects import next_object
        from services.autolabel.ontology import get_ontology

        onto = get_ontology()
        sid, fid = uuid.uuid4(), uuid.uuid4()
        async with get_sessionmaker()() as db:
            db.add(DbSession(session_id=sid, vehicle_id="NXT", start_ts_ns=0, end_ts_ns=1, sensors={},
                             ontology_version=onto.version, origin=REAL))
            await db.flush()
            db.add(Frame(frame_id=fid, session_id=sid, ts_ns=now_ns(), cam_id="front", width=64,
                         height=64, img_uri=f"frames/{sid}/a.jpg", origin=REAL, selected=True))
            await db.commit()
            try:
                res = await next_object(fid, limit=5, db=db)
                assert res["n_candidates"] == 0
                assert res["next"] == []
                assert res["reason"]
            finally:
                await db.execute(delete(DbSession).where(DbSession.session_id == sid))
                await db.commit()


class TestTheTubeJudgeGroupsByClass:
    """A track in this corpus is usually not one object.

    9,982 of 11,288 tracks carry objects of more than one class, and the largest holds 147 objects across
    18 classes. A first version took the first object's class as the track's class and stamped that
    verdict on every object; on 88% of tracks that means stamping a verdict about sedans onto riders,
    pedestrians and traffic signs.
    """

    def test_it_groups_the_track_by_class_before_judging(self):
        import inspect

        from services.labelops import vlm_review

        src = inspect.getsource(vlm_review.judge_tracks)
        assert "by_class.setdefault(int(obj.class_id)" in src
        assert "for class_id, objects in by_class.items()" in src

    def test_each_class_group_gets_its_own_batch_id(self):
        """Sharing one batch id across classes would make the uniqueness key collide, so the last group
        judged would overwrite the others and the track would report one verdict again."""
        import inspect

        from services.labelops import vlm_review

        src = inspect.getsource(vlm_review.judge_tracks)
        assert 'f"{TUBE_BATCH_PREFIX}-{tid.hex[:8]}-c{class_id}"' in src

    def test_a_mixed_track_is_counted_and_carried_on_the_verdict(self):
        """A track spanning several classes is a tracker failure worth seeing, so the count rides on the
        verdict rather than being discovered later by whoever wonders."""
        import inspect

        from services.labelops import vlm_review

        src = inspect.getsource(vlm_review.judge_tracks)
        assert '"track_classes": len(by_class)' in src
        assert 'out["mixed_class_tracks"] += 1' in src
