"""Finding a cuboid's yaw, and knowing how weak the signal for it is.

The monocular solve has existed since cuboids were added: ground-lift the box's bottom centre, size from a
class prior, then pick the yaw whose reprojection best matches the observed 2D box. It was reachable only
as a frame-wide agent run, so the editor's own cuboid tool dropped a hardcoded 1.8 x 4.2 x 1.5 box at yaw
0 on whatever it was placed on, and a bus and a scooter came out identical.

Two measurements shaped what changed.

The coarse search used four candidates 45 degrees apart, which is as close as a four-way search gets.
Adding a two-degree sweep around whichever won lifts mean reprojection IoU from 0.491 to 0.555 over 238
objects from the real corpus, with 178 improving and the best by 0.29.

Lane snapping is the other half, and the measurement there said something inconvenient. Reprojection IoU
is a weak signal for yaw and cannot be otherwise: the axis-aligned image box of a car seen head-on is very
nearly the box of the same car seen from behind. So a road direction, which is measured rather than
inferred, should be the better answer. Allowing a snap to cost up to 0.02 IoU snapped 51% of 1,050 objects
and moved mean IoU from 0.2461 to 0.2445, with 244 worse against 11 better. That does not prove the yaws
got worse, but it is the only signal available. Requiring the snap to cost nothing instead snaps 8%, with
11 better and 0 worse. The tests below pin the strict rule, because the permissive one is the tempting
mistake.
"""

import math

import pytest

from services.agent.cuboid_agent import _frange, snap_to_lane


class _Lane:
    def __init__(self, pts):
        self.control_points = pts


def test_a_yaw_is_snapped_to_a_road_running_the_same_way():
    """The ordinary case: a vehicle fitted a few degrees off the lane it is driving in."""
    yaw = math.radians(10)
    snapped = snap_to_lane(yaw, [math.radians(2)])
    assert snapped is not None
    assert math.degrees(snapped[0]) == pytest.approx(2.0, abs=0.01)


def test_a_road_running_the_other_way_is_the_same_axis():
    """A lane says which way the road runs, not which way along it a vehicle faces, so a heading and that
    heading plus pi are the same constraint. Comparing them as bearings would leave a vehicle in the
    oncoming lane unsnapped, which is exactly where a wrong yaw is worth catching."""
    yaw = math.radians(5)
    snapped = snap_to_lane(yaw, [math.radians(183)])
    assert snapped is not None
    # And the side of the axis the fit chose survives: snapping must never turn a vehicle around.
    assert abs(math.degrees(snapped[0]) - 3.0) < 0.01


def test_a_road_running_across_the_fitted_yaw_does_not_snap():
    """A vehicle crossing a junction is not aligned with the lane it is crossing, and pulling it onto that
    lane would be inventing a heading rather than measuring one."""
    assert snap_to_lane(math.radians(10), [math.radians(80)]) is None


def test_the_nearest_road_wins_when_several_are_close():
    yaw = math.radians(10)
    snapped = snap_to_lane(yaw, [math.radians(25), math.radians(12), math.radians(-8)])
    assert snapped is not None
    assert math.degrees(snapped[0]) == pytest.approx(12.0, abs=0.01)


def test_no_lanes_means_no_snap_rather_than_a_default_heading():
    """A frame with no lanes is a frame where the road direction was not measured. Falling back to zero
    would state a heading nobody observed, and axis-aligned is exactly the wrong default in a corpus of
    curving urban roads."""
    assert snap_to_lane(math.radians(10), []) is None


def test_a_lane_of_fewer_than_two_points_yields_no_heading():
    """A single control point is a position, not a direction."""
    from services.agent.cuboid_agent import lane_headings

    assert lane_headings([_Lane([[100.0, 800.0]]), _Lane([])], "cam_f", 1920, 1080) == []


def test_the_refinement_sweep_covers_the_gap_between_coarse_candidates():
    """The coarse candidates are 45 degrees apart, so a sweep of plus or minus 22.5 around the winner
    reaches every yaw. A narrower span would leave a band no search can reach."""
    from services.agent.cuboid_agent import _REFINE_HALF_SPAN, _YAW_CANDIDATES

    gap = _YAW_CANDIDATES[1] - _YAW_CANDIDATES[0]
    assert _REFINE_HALF_SPAN >= gap / 2 - 1e-9, (
        "the sweep must reach halfway to the neighbouring coarse candidate, or some yaws are unreachable")


def test_the_sweep_includes_both_ends_and_the_centre():
    vals = _frange(-1.0, 1.0, 0.5)
    assert vals[0] == pytest.approx(-1.0)
    assert vals[-1] == pytest.approx(1.0)
    assert any(abs(v) < 1e-9 for v in vals), "the coarse winner itself has to stay in the search"


@pytest.mark.db
@pytest.mark.asyncio
async def test_a_class_that_does_not_rest_on_the_road_is_refused_with_a_reason():
    """A hoarding does not have a ground contact to lift, so a cuboid fitted to one would be a number with
    no meaning. Said rather than guessed."""
    import uuid

    from core.timebase import now_ns
    from db.models import Frame, OntologyClass, OntologyVersion
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.agent.cuboid_agent import fit_cuboid_at
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        if await db.get(OntologyVersion, onto.version) is None:
            db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
            await db.flush()
            for c in onto.classes:
                db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                     india=c.india, map_to={}))
            await db.flush()
        ts, sid, fid = now_ns(), uuid.uuid4(), uuid.uuid4()
        db.add(DbSession(session_id=sid, vehicle_id="CUB-1", start_ts_ns=ts, end_ts_ns=ts + 1,
                         city="BLR", sensors={}, ontology_version=onto.version))
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri="s3://c/1.jpg",
                     width=1920, height=1080, quality=0.9, scene={}))
        await db.flush()

        res = await fit_cuboid_at(db, fid, 960.0, 900.0, "hoarding")
        assert res["cuboid"] is None
        assert "rests on the road" in res["reason"]
        await db.rollback()


@pytest.mark.db
@pytest.mark.asyncio
async def test_a_point_above_the_horizon_is_refused_rather_than_lifted():
    """Lifting it produces a negative ray parameter, which without this check becomes a cuboid behind the
    camera."""
    import uuid

    from core.timebase import now_ns
    from db.models import Frame, OntologyClass, OntologyVersion
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.agent.cuboid_agent import fit_cuboid_at
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        if await db.get(OntologyVersion, onto.version) is None:
            db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
            await db.flush()
            for c in onto.classes:
                db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                     india=c.india, map_to={}))
            await db.flush()
        ts, sid, fid = now_ns(), uuid.uuid4(), uuid.uuid4()
        db.add(DbSession(session_id=sid, vehicle_id="CUB-2", start_ts_ns=ts, end_ts_ns=ts + 1,
                         city="BLR", sensors={}, ontology_version=onto.version))
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri="s3://c/2.jpg",
                     width=1920, height=1080, quality=0.9, scene={}))
        await db.flush()

        res = await fit_cuboid_at(db, fid, 960.0, 5.0, "sedan")   # near the top of the image
        assert res["cuboid"] is None
        assert "horizon" in res["reason"] or "calibration" in res["reason"]
        await db.rollback()
