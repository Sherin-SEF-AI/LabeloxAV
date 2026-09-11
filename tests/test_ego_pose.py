"""Ego pose: the geometry, the scale recovery, and the distinction between measured and inferred.

The pure parts are tested exhaustively because they are where a wrong sign produces a trajectory that
looks entirely plausible and is mirrored, or a scale that is confidently ten times too large. A
trajectory nobody can check by eye is exactly the kind of output that needs its arithmetic pinned.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from services.intelligence.ego_pose import (
    GROUND_MAX_M,
    GROUND_MIN_M,
    MIN_GROUND_FEATURES,
    _integrate,
    enu_offset,
    quat_yaw,
    scale_from_depth,
    scale_from_ground,
    yaw_to_quat,
)


class Calib:
    """A nominal forward pinhole camera, the shape `resolve_calibration` returns."""

    model = "pinhole"
    fx = fy = 800.0
    cx, cy = 640.0, 480.0
    dist: list[float] = []
    rpy_deg = (0.0, 0.0, 0.0)
    xyz_m = (0.0, 0.0, 1.5)


class TestTheLocalFrame:
    def test_the_origin_is_itself(self):
        assert enu_offset(12.9, 77.6, 12.9, 77.6) == (0.0, 0.0)

    def test_north_is_positive_and_east_is_positive(self):
        e, n = enu_offset(12.9, 77.6, 12.91, 77.61)
        assert n > 0 and e > 0

    def test_a_degree_of_latitude_is_about_111_km(self):
        _e, n = enu_offset(12.9, 77.6, 13.9, 77.6)
        assert 110_000 < n < 112_000

    def test_a_degree_of_longitude_shrinks_with_latitude(self):
        near_equator, _ = enu_offset(0.0, 0.0, 0.0, 1.0)
        at_bengaluru, _ = enu_offset(12.9, 77.6, 12.9, 78.6)
        assert at_bengaluru < near_equator


class TestTheRotation:
    @pytest.mark.parametrize("yaw", [0.0, 0.5, -0.5, math.pi / 2, -math.pi / 2, 3.0])
    def test_a_yaw_survives_the_round_trip(self, yaw):
        qw, _qx, _qy, qz = yaw_to_quat(yaw)
        assert quat_yaw(qw, qz) == pytest.approx(yaw, abs=1e-9)

    def test_the_quaternion_is_a_unit_one(self):
        """The database checks this too, because the writer is not the only thing that can insert."""
        for yaw in (0.0, 1.0, -2.5, math.pi):
            q = yaw_to_quat(yaw)
            assert sum(v * v for v in q) == pytest.approx(1.0, abs=1e-12)

    def test_roll_and_pitch_are_left_at_zero(self):
        """Neither is observable from GNSS or from monocular odometry here, and a fabricated one would
        tilt every cuboid placed through this pose."""
        _qw, qx, qy, _qz = yaw_to_quat(1.2)
        assert qx == 0.0 and qy == 0.0


class TestIntegration:
    def test_driving_straight_goes_straight(self):
        steps = [{"dyaw": 0.0, "dist": 10.0} for _ in range(3)]
        out = _integrate(steps)
        assert [round(p["x"], 6) for p in out] == [10.0, 20.0, 30.0]
        assert all(round(p["y"], 9) == 0.0 for p in out)

    def test_a_quarter_turn_then_a_step_goes_sideways(self):
        out = _integrate([{"dyaw": math.pi / 2, "dist": 10.0}])
        assert out[0]["x"] == pytest.approx(0.0, abs=1e-9)
        assert out[0]["y"] == pytest.approx(10.0, abs=1e-9)

    def test_an_unscaled_step_turns_the_heading_without_moving(self):
        """The rotation is recoverable without scale and throwing it away would lose real information.
        The position simply does not advance, and the row records a null speed so nothing reads the
        stall as the vehicle having stopped."""
        out = _integrate([{"dyaw": 0.0, "dist": 10.0},
                          {"dyaw": math.pi / 2, "dist": None},
                          {"dyaw": 0.0, "dist": 10.0}])
        assert out[1]["x"] == pytest.approx(10.0) and out[1]["y"] == pytest.approx(0.0)
        assert out[2]["y"] == pytest.approx(10.0, abs=1e-9), "the turn still applied"

    def test_the_heading_accumulates_across_steps(self):
        out = _integrate([{"dyaw": 0.3, "dist": 0.0}, {"dyaw": 0.4, "dist": 0.0}])
        assert out[-1]["yaw"] == pytest.approx(0.7)


class TestScaleFromDepth:
    def test_the_baseline_is_the_forward_component_times_the_scene_depth(self):
        depth = np.full((100, 100), 20.0, dtype=np.float32)
        pts = np.tile(np.array([[50.0, 50.0]]), (MIN_GROUND_FEATURES * 4, 1))
        assert scale_from_depth(depth, pts, np.array([0.0, 0.0, 1.0])) == pytest.approx(20.0)

    def test_too_few_features_refuses_rather_than_guessing(self):
        depth = np.full((100, 100), 20.0, dtype=np.float32)
        assert scale_from_depth(depth, np.array([[50.0, 50.0]]), np.array([0.0, 0.0, 1.0])) is None

    def test_no_depth_map_refuses(self):
        pts = np.tile(np.array([[50.0, 50.0]]), (40, 1))
        assert scale_from_depth(None, pts, np.array([0.0, 0.0, 1.0])) is None

    def test_implausible_depths_are_discarded_rather_than_averaged(self):
        """Half the scene in this corpus is moving traffic and a depth of zero or 500 m is a model
        failure, not a distant object."""
        depth = np.zeros((100, 100), dtype=np.float32)
        pts = np.tile(np.array([[50.0, 50.0]]), (40, 1))
        assert scale_from_depth(depth, pts, np.array([0.0, 0.0, 1.0])) is None


class TestScaleFromGround:
    def _ground_pixel(self, forward_m: float, calib=Calib) -> tuple[float, float]:
        """The pixel a road point at this forward distance projects to, for a level pinhole camera."""
        v = calib.cy + calib.fy * calib.xyz_m[2] / forward_m
        return calib.cx, v

    def test_a_known_closing_distance_is_recovered(self):
        """Ground points at 10 m that are at 8 m one frame later mean the vehicle moved 2 m."""
        a = np.array([self._ground_pixel(10.0 + i * 0.5) for i in range(MIN_GROUND_FEATURES * 2)])
        b = np.array([self._ground_pixel(8.0 + i * 0.5) for i in range(MIN_GROUND_FEATURES * 2)])
        got = scale_from_ground(a, b, Calib, height_m=1.5)
        assert got == pytest.approx(2.0, abs=0.05)

    def test_features_above_the_horizon_are_not_ground(self):
        above = np.tile(np.array([[640.0, 100.0]]), (MIN_GROUND_FEATURES * 2, 1))
        assert scale_from_ground(above, above, Calib, height_m=1.5) is None

    def test_receding_features_measure_the_same_distance(self):
        """A rear-facing camera sees ground points recede by exactly what a forward one sees them close.

        The scale is a magnitude and the direction is the camera's business. A version of this that kept
        only closings recovered a usable scale on 3 of 1,032 pairs of the first real rig session here,
        because the busiest camera on that rig faces backwards.
        """
        near = np.array([self._ground_pixel(10.0) for _ in range(MIN_GROUND_FEATURES * 2)])
        far = np.array([self._ground_pixel(14.0) for _ in range(MIN_GROUND_FEATURES * 2)])
        closing = scale_from_ground(far, near, Calib, height_m=1.5)
        receding = scale_from_ground(near, far, Calib, height_m=1.5)
        assert closing == pytest.approx(4.0, abs=0.05)
        assert receding == pytest.approx(closing, abs=1e-6)

    def test_a_few_fast_movers_do_not_set_the_scale(self):
        """Half the features in this corpus sit on traffic that is itself moving. The median is what
        keeps a handful of them from deciding how far the vehicle went."""
        n = MIN_GROUND_FEATURES * 2
        a = [self._ground_pixel(10.0) for _ in range(n)]
        b = [self._ground_pixel(8.0) for _ in range(n)]
        # Four features on a vehicle pulling away hard.
        for i in range(4):
            b[i] = self._ground_pixel(30.0)
        got = scale_from_ground(np.array(a), np.array(b), Calib, height_m=1.5)
        assert got == pytest.approx(2.0, abs=0.1)

    def test_too_few_ground_features_refuses(self):
        a = np.array([self._ground_pixel(10.0)])
        b = np.array([self._ground_pixel(9.0)])
        assert scale_from_ground(a, b, Calib, height_m=1.5) is None

    def test_features_outside_the_usable_band_are_ignored(self):
        """Very close road texture is where the flat-plane assumption is least reliable, and very far
        texture is where a pixel of error is metres."""
        too_near = np.array([self._ground_pixel(GROUND_MIN_M - 1.0)
                             for _ in range(MIN_GROUND_FEATURES * 2)])
        near_b = np.array([self._ground_pixel(GROUND_MIN_M - 2.0)
                           for _ in range(MIN_GROUND_FEATURES * 2)])
        assert scale_from_ground(too_near, near_b, Calib, height_m=1.5) is None
        too_far = np.array([self._ground_pixel(GROUND_MAX_M + 10.0)
                            for _ in range(MIN_GROUND_FEATURES * 2)])
        far_b = np.array([self._ground_pixel(GROUND_MAX_M + 5.0)
                          for _ in range(MIN_GROUND_FEATURES * 2)])
        assert scale_from_ground(too_far, far_b, Calib, height_m=1.5) is None

    def test_a_taller_mount_reports_a_longer_baseline(self):
        """The scale is linear in the assumed camera height, which is why the height is recorded on
        every run rather than left implicit."""
        a = np.array([self._ground_pixel(10.0) for _ in range(MIN_GROUND_FEATURES * 2)])
        b = np.array([self._ground_pixel(8.0) for _ in range(MIN_GROUND_FEATURES * 2)])
        low = scale_from_ground(a, b, Calib, height_m=1.5)
        high = scale_from_ground(a, b, Calib, height_m=3.0)
        assert high == pytest.approx(2 * low, rel=1e-6)


class TestCameraChoice:
    """One camera, and the one that covers the most of the session.

    Stitching two cameras' motion into one trajectory double-counts every step, so a camera has to be
    chosen. A preference for a forward-facing one was tried and the corpus refused it: on the first real
    rig session `rear_wide` gives 1,033 poses with 42 metrically scaled steps and `front_narrow` gives
    179 with none, because the ground-plane scale needs a wide view of road surface rather than a
    forward one, and a narrow lens sees less of it than a wide rear camera does.
    """

    def _rows(self, counts: dict[str, int]):
        return [(0, 0, cam) for cam, n in counts.items() for _ in range(n)]

    def test_the_camera_covering_the_most_of_the_session_is_chosen(self):
        from services.intelligence.ego_pose import _pick_camera

        rows = self._rows({"rear_wide": 1033, "front_narrow": 179, "left_wide": 179})
        assert _pick_camera(rows, {"rear_wide", "front_narrow", "left_wide"}) == "rear_wide"

    def test_a_single_camera_session_picks_it(self):
        from services.intelligence.ego_pose import _pick_camera

        assert _pick_camera(self._rows({"cam_front": 700}), {"cam_front"}) == "cam_front"

    def test_a_rear_camera_is_used_rather_than_refused(self):
        """A rear camera sees ground features recede by exactly what a forward one sees them close, so
        it yields the same baseline. Refusing it would leave every frame of that rig without a pose."""
        from services.intelligence.ego_pose import _pick_camera

        assert _pick_camera(self._rows({"rear_wide": 400}), {"rear_wide"}) == "rear_wide"


class TestGnssRowsReadGeography:
    """`Frame.gnss` is a PostGIS geography point, and reading it as a dict silently disabled measured pose.

    `_gnss_rows` called `.get("lat")` on the selected column. Selecting a geography column hands back a
    geoalchemy element whose `__getattr__` raises `AttributeError`, so every session that actually had
    satellite fixes raised on the first row, and the only sessions that "worked" were the ones with no
    GNSS at all, where the query returned nothing and the visual fallback took over.

    That is the worst shape a bug can have: it failed exactly where the good data was, and the pass rate
    of the old test suite was unaffected because no test seeded a real fix.
    """

    @pytest.mark.db
    @pytest.mark.asyncio
    async def test_measured_poses_come_back_from_real_fixes(self):
        import uuid as _uuid

        from sqlalchemy import text

        from core.timebase import now_ns, seconds_to_ns
        from db.models import Frame, OntologyClass, OntologyVersion
        from db.models import Session as DbSession
        from db.session import get_sessionmaker
        from services.autolabel.ontology import get_ontology
        from services.intelligence.ego_pose import _gnss_rows

        t0 = now_ns()
        sid = _uuid.uuid4()
        onto = get_ontology()
        async with get_sessionmaker()() as db:
            if await db.get(OntologyVersion, onto.version) is None:
                db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
                await db.flush()
                for c in onto.classes:
                    db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0,
                                         l1=c.l1, india=c.india, map_to={}))
                await db.flush()
            db.add(DbSession(session_id=sid, vehicle_id="TEST-EGO", city="BLR",
                             route="ego-pose-gnss", start_ts_ns=t0,
                             end_ts_ns=t0 + seconds_to_ns(6), sensors={},
                             ontology_version=onto.version))
            await db.flush()
            # A short straight drive east, one fix per second, written the way ingest writes them.
            for i in range(6):
                lat, lon = 12.9716, 77.5946 + i * 0.0001
                db.add(Frame(frame_id=_uuid.uuid4(), session_id=sid, cam_id="cam_f",
                             ts_ns=t0 + seconds_to_ns(i), width=1920, height=1080, quality=0.9,
                             scene={}, img_uri=f"s3://test/ego/{i}.jpg", ego_speed=10.0,
                             gnss=f"SRID=4326;POINT({lon} {lat})"))
            await db.commit()

            try:
                rows = await _gnss_rows(db, sid)
                assert len(rows) == 6, "every fix should produce a pose"
                assert all(r["measured"] is True for r in rows), "a satellite fix is measured"
                assert all(r["source"] == "gnss_imu" for r in rows)
                # The drive went east, so easting must increase and northing must stay put.
                xs = [r["x"] for r in rows]
                assert xs == sorted(xs) and xs[-1] > xs[0] + 1.0, "eastward motion should show in x"
                assert max(abs(r["y"]) for r in rows) < 1.0, "a straight eastward drive has no northing"
            finally:
                await db.execute(text("delete from frame where session_id = :s"), {"s": sid})
                await db.execute(text("delete from session where session_id = :s"), {"s": sid})
                await db.commit()
