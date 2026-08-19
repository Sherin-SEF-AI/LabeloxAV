"""Ego-compensated propagation, and the guard that refuses it on every session in this corpus.

Copying a box to the next frame assumes the camera stood still. On a moving vehicle that is wrong for
everything, and most wrong for the roadside furniture propagation should be best at: a parked car sweeps
across the image as the ego passes it.

Correcting for that needs a rigid transform between two frames. This corpus has GNSS on 3 frames of
41,752, ego_speed on 6, no heading or IMU attitude and no per-frame 6-DOF pose table. So the acceptance
test for this feature cannot be run, and what is tested instead is the geometry against hand-checkable
fixtures plus the refusal on real data. The refusal is not a placeholder: assuming a transform that is not
there would place boxes confidently and wrongly.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from core.accel.ego_homography import ground_homography, warp_box, warp_boxes
from core.timebase import now_ns
from db.models import Frame
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology
from services.calyx.ego_propagate import coverage_report, propagate_frame

# A plain pinhole: 1000 px focal, principal point at the centre of a 1920x1080 frame.
_K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
_N = [0.0, 1.0, 0.0]        # ground normal in camera coordinates (y down)
_H_CAM = 1.5                # camera height above the road


class TestTheHomography:
    def test_no_motion_is_the_identity(self):
        """A camera that did not move must not move a box, to the last decimal.

        The failure this rules out is subtle: a homography that is nearly the identity shifts every box a
        few pixels every frame, and over a session that accumulates into a drift nobody attributes to
        propagation.
        """
        H = ground_homography(_K, np.eye(3), [0.0, 0.0, 0.0], _N, _H_CAM)
        assert H is not None
        assert np.allclose(H, np.eye(3), atol=1e-12)
        w = warp_box([100.0, 700.0, 200.0, 800.0], H, width=1920, height=1080,
                     motion_model="static_ground")
        assert w.measured is True
        assert all(abs(a - b) < 1e-6 for a, b in zip(w.box, (100.0, 700.0, 200.0, 800.0), strict=True))

    def test_driving_forward_moves_a_ground_box_down_and_outward(self):
        """Approaching a static object on the road makes it larger and lower in the image.

        That is what an annotator sees and what an uncompensated copy fails to do.
        """
        # t is the camera-centre translation in source coordinates, so forward is +z. That sign is the
        # easy thing to get backwards, and getting it backwards shrinks an approaching object instead of
        # growing it, which looks plausible in a screenshot.
        H = ground_homography(_K, np.eye(3), [0.0, 0.0, 1.0], _N, _H_CAM)
        src = [900.0, 700.0, 1000.0, 760.0]
        w = warp_box(src, H, width=1920, height=1080, motion_model="static_ground")
        assert w.measured is True, w.reason
        assert w.box[3] > src[3], "the bottom edge should descend as the object nears"
        assert (w.box[2] - w.box[0]) > (src[2] - src[0]), "the box should grow"
        # And reversing the sign reverses the effect, which is what makes this a direction test rather
        # than a magnitude one.
        back = warp_box(src, ground_homography(_K, np.eye(3), [0.0, 0.0, -1.0], _N, _H_CAM),
                        width=1920, height=1080, motion_model="static_ground")
        assert back.box[3] < src[3] and (back.box[2] - back.box[0]) < (src[2] - src[0])

    def test_a_degenerate_setup_yields_no_homography_rather_than_a_wrong_one(self):
        assert ground_homography(_K, np.eye(3), [0, 0, 0], _N, 0.0) is None            # camera on the plane
        assert ground_homography(_K, np.eye(3), [0, 0, 0], [0, 0, 0], _H_CAM) is None  # no normal
        assert ground_homography(np.zeros((3, 3)), np.eye(3), [0, 0, 0], _N, _H_CAM) is None


class TestWhatItRefusesToWarp:
    def test_an_elevated_static_object_is_refused_not_warped(self):
        """A gantry is not on the ground plane, and the ground homography moves it the wrong way.

        This is why the motion model is three-valued rather than a boolean: refusing is cheap, and
        propagating a box confidently in the opposite direction is not.
        """
        H = ground_homography(_K, np.eye(3), [0.0, 0.0, 1.0], _N, _H_CAM)
        w = warp_box([900.0, 100.0, 1000.0, 160.0], H, width=1920, height=1080,
                     motion_model="static_elevated")
        assert w.measured is False
        assert "not on the ground plane" in w.reason

    def test_a_moving_object_is_refused(self):
        H = ground_homography(_K, np.eye(3), [0.0, 0.0, 1.0], _N, _H_CAM)
        assert warp_box([100.0, 700.0, 200.0, 800.0], H, width=1920, height=1080,
                        motion_model="moving").measured is False

    def test_no_homography_at_all_is_a_refusal(self):
        w = warp_box([100.0, 700.0, 200.0, 800.0], None, width=1920, height=1080,
                     motion_model="static_ground")
        assert w.measured is False and "degenerate" in w.reason

    def test_a_box_that_leaves_the_frame_is_refused(self):
        # Driving forward past a box in the bottom corner of the frame: it sweeps out of view.
        H = ground_homography(_K, np.eye(3), [0.0, 0.0, 1.0], _N, _H_CAM)
        w = warp_box([1900.0, 1070.0, 1919.0, 1079.0], H, width=1920, height=1080,
                     motion_model="static_ground")
        assert w.measured is False and "leaves the destination frame" in w.reason


class TestBatch:
    def test_it_counts_what_it_refused_and_why(self):
        H = ground_homography(_K, np.eye(3), [0.0, 0.0, 1.0], _N, _H_CAM)
        boxes = [[900.0, 700.0, 1000.0, 760.0]] * 3
        out = warp_boxes(boxes, H, ["static_ground", "static_elevated", "moving"],
                         width=1920, height=1080)
        assert out["n_warped"] == 1 and out["n_refused"] == 2
        assert out["reasons"], "a refusal with no reason is indistinguishable from a crash"


class TestTheMotionModelComesFromThePack:
    def test_every_named_class_exists_in_the_ontology(self):
        """A motion model naming a class that does not exist silently never applies.

        The class never matches, so the spec covers less than it claims and propagation quietly refuses
        more than intended. Sixteen of my first draft's names were exactly this.
        """
        from packs.registry import default_pack_id, get_pack

        onto = get_ontology()
        spec = get_pack(default_pack_id()).motion_models
        assert spec is not None
        missing = sorted(n for n in (spec.static_ground | spec.static_elevated) if not onto.has_name(n))
        assert missing == [], missing

    def test_an_unlisted_class_is_moving_which_refuses(self):
        """Guessing static for an unlisted class would place its box by ego motion alone."""
        from packs.registry import default_pack_id, get_pack

        spec = get_pack(default_pack_id()).motion_models
        assert spec.model_for("a_class_nobody_has_defined") == "moving"
        assert spec.model_for("pedestrian") == "moving"
        assert spec.model_for("cone") == "static_ground"
        assert spec.model_for("traffic_signal") == "static_elevated"

    def test_no_class_is_both_on_and_above_the_ground(self):
        from packs.registry import default_pack_id, get_pack

        spec = get_pack(default_pack_id()).motion_models
        assert spec.static_ground & spec.static_elevated == frozenset()


class TestTheGuardOnRealData:
    pytestmark = pytest.mark.db

    async def _two_frames(self, db, same_session: bool = True):
        onto = get_ontology()
        fids = []
        sess = None
        for i in range(2):
            if sess is None or not same_session:
                sess = DbSession(session_id=uuid.uuid4(), vehicle_id="EGO", start_ts_ns=0, end_ts_ns=1,
                                 ontology_version=onto.version)
                db.add(sess)
                await db.flush()
            f = Frame(frame_id=uuid.uuid4(), session_id=sess.session_id, ts_ns=now_ns() + i,
                      cam_id="front", width=1920, height=1080, img_uri=f"s3://x/{i}.jpg")
            db.add(f)
            await db.flush()
            fids.append(str(f.frame_id))
        await db.commit()
        return fids

    async def test_it_refuses_and_names_what_the_corpus_lacks(self):
        """The acceptance test for this feature cannot run, and this records why rather than skipping.

        A skip says "not checked". This says "checked, and the corpus cannot support it", which is the
        thing to fix before the feature means anything.
        """
        async with get_sessionmaker()() as db:
            fids = await self._two_frames(db)
            res = await propagate_frame(db, from_frame_id=fids[0], to_frame_id=fids[1])
            assert res["measured"] is False
            assert res["n_propagated"] == 0
            assert "GNSS" in res["reason"]

    async def test_two_frames_from_different_sessions_have_no_ego_motion_between_them(self):
        async with get_sessionmaker()() as db:
            fids = await self._two_frames(db, same_session=False)
            res = await propagate_frame(db, from_frame_id=fids[0], to_frame_id=fids[1])
            assert res["measured"] is False and "different sessions" in res["reason"]

    async def test_the_coverage_report_states_the_blocker_rather_than_a_percentage(self):
        async with get_sessionmaker()() as db:
            rep = await coverage_report(db)
            assert rep["propagatable"] is False
            assert "attitude" in rep["reason"]
            assert rep["n_frames"] >= 0
