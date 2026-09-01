"""Flattening the road, and the inverse projection that makes it drawable rather than only viewable.

Lane annotation here has always been image space, which is where the paint is and also where a lane is
hardest to judge: a forward camera runs the road to a vanishing point, so the far half of a lane is a
handful of pixels, parallel lanes converge, and a curve and a lane change look alike.

The reason a bird's-eye editor did not exist is that half the mathematics was missing.
`ipm_pixel_to_vehicle` has lifted a pixel to the ground plane since the georeferencing work; the inverse,
a ground point back to the pixel that sees it, appeared nowhere in the repo. Grepping for
`vehicle_to_pixel`, `world_to_pixel` or any `*_to_pixel` across `services/` returned nothing. Without it a
warp can be looked at and not drawn on.

So the tests that matter are about the round trip. A point lifted and projected back must land where it
started, or a lane drawn on the warp is stored somewhere else on the frame and nobody finds out until the
export. And the cases with no answer must return no answer: a pixel above the horizon has no ground
intersection, and a ground point behind the camera has no pixel. Clamping either would store a coordinate
nobody drew.
"""

import math

import pytest

from services.hdmap.bev_view import BevView
from services.hdmap.georef import ipm_pixel_to_vehicle, vehicle_to_ipm_pixel

# A plausible forward camera: 1920x1440, 1.5 m up, level.
CAM = {"fx": 1200.0, "fy": 1200.0, "cx": 960.0, "cy": 720.0, "height_m": 1.5, "pitch_rad": 0.0}


@pytest.mark.parametrize("u,v", [(960, 1200), (400, 1000), (1500, 900), (960, 820), (100, 1439)])
def test_a_pixel_lifted_to_the_ground_and_projected_back_lands_where_it_started(u, v):
    """The claim the whole bird's-eye editor rests on. If this drifts, a lane drawn on the warp is stored
    somewhere else on the frame and nothing notices until the export."""
    g = ipm_pixel_to_vehicle(float(u), float(v), **CAM)
    assert g is not None, "a pixel below the horizon must have a ground intersection"
    back = vehicle_to_ipm_pixel(g[0], g[1], **CAM)
    assert back is not None
    assert math.hypot(back[0] - u, back[1] - v) < 1e-6


def test_the_round_trip_survives_a_pitched_camera():
    """Pitch is applied in the forward direction and has to be undone, not applied again, coming back. The
    sign of that rotation is the kind of thing that is invisible until a lane lands ten metres out."""
    cam = {**CAM, "pitch_rad": math.radians(6.0)}
    g = ipm_pixel_to_vehicle(700.0, 1100.0, **cam)
    assert g is not None
    back = vehicle_to_ipm_pixel(g[0], g[1], **cam)
    assert back is not None
    assert math.hypot(back[0] - 700.0, back[1] - 1100.0) < 1e-6


def test_a_point_behind_the_camera_has_no_pixel():
    """None rather than a clamped coordinate. A clamped point is a position nobody drew, stored as though
    they had."""
    assert vehicle_to_ipm_pixel(-4.0, 0.0, **CAM) is None
    assert vehicle_to_ipm_pixel(0.0, 3.0, **CAM) is None


def test_a_pixel_above_the_horizon_has_no_ground_point():
    """The other half of the same rule, and the reason a lane's control points can be dropped when it is
    shown on the warp: a point up there was never on the road plane."""
    assert ipm_pixel_to_vehicle(960.0, 100.0, **CAM) is None


def test_further_along_the_road_is_higher_up_the_image():
    """A sanity check on the direction of the projection. Getting this backwards produces a view that is
    upside down and, worse, a lane drawn near the camera stored as one far away."""
    near = vehicle_to_ipm_pixel(6.0, 0.0, **CAM)
    far = vehicle_to_ipm_pixel(40.0, 0.0, **CAM)
    assert near is not None and far is not None
    assert far[1] < near[1], "a more distant point projects nearer the horizon, which is a smaller v"


def test_the_bev_grid_maps_both_ways_exactly():
    """The metric-to-raster conversions are separate arithmetic from the projection and are just as easy
    to get subtly wrong: an off-by-one in the row direction produces a picture that looks right and is a
    metre out everywhere."""
    view = BevView(near_m=2.0, far_m=42.0, half_width_m=10.0, px_per_m=20.0)
    assert view.width == 400 and view.height == 800
    for forward, lateral in [(2.0, -10.0), (42.0, 10.0), (20.0, 0.0), (7.5, -3.25)]:
        bx, by = view.to_pixel(forward, lateral)
        f2, l2 = view.to_metric(bx, by)
        assert f2 == pytest.approx(forward) and l2 == pytest.approx(lateral)


def test_the_far_edge_of_the_view_is_the_top_row():
    """The view reads like a map, with the camera at the bottom. A renderer that assumed the opposite
    would draw a road running away from the viewer as one running towards them."""
    view = BevView(near_m=2.0, far_m=42.0, half_width_m=10.0, px_per_m=20.0)
    assert view.to_metric(0.0, 0.0)[0] == pytest.approx(42.0)
    assert view.to_metric(0.0, float(view.height))[0] == pytest.approx(2.0)
    # And the lateral axis runs left to right through zero at the centre column.
    assert view.to_metric(view.width / 2, 0.0)[1] == pytest.approx(0.0)


def test_the_view_is_bounded_by_what_the_lift_can_support():
    """The far edge comes from `ipm_max_range_m`, the codebase's own limit on where a flat-road lift stops
    meaning anything, rather than from a number chosen to fill the picture. A short-focal-length camera
    mounted low gets a shorter view, and should."""
    from services.hdmap.bev_view import MAX_FAR_M, view_for

    class _Cal:
        model = "pinhole"
        dist: list = []
        cam_id = "cam_f"
        quality = 0.5
        source = "nominal"

        def __init__(self, fy, h):
            self.fx = self.fy = fy
            self.cx, self.cy = 960.0, 720.0
            self.rpy_deg = (0.0, 0.0, 0.0)
            self.xyz_m = (0.0, 0.0, h)

    wide = view_for(_Cal(2800.0, 1.5))
    narrow = view_for(_Cal(600.0, 0.8))
    assert narrow.far_m < wide.far_m
    assert wide.far_m <= MAX_FAR_M


def test_points_that_do_not_reach_the_road_are_dropped_from_a_projection_not_clamped():
    """`bev_to_image` is what turns a drawn line into stored control points. A point it cannot project
    must vanish rather than be pinned to an edge, because a pinned point becomes a lane vertex nobody
    placed."""
    from services.hdmap.bev_view import bev_to_image

    class _Cal:
        model = "pinhole"
        dist: list = []
        fx = fy = 1200.0
        cx, cy = 960.0, 720.0
        rpy_deg = (0.0, 0.0, 0.0)
        xyz_m = (0.0, 0.0, 1.5)

    view = BevView(near_m=-10.0, far_m=30.0, half_width_m=10.0, px_per_m=10.0)
    # The bottom rows of this deliberately odd view sit behind the camera and have no image.
    pts = [[100.0, 0.0], [100.0, float(view.height) - 1]]
    out = bev_to_image(pts, _Cal(), view)
    assert len(out) == 1, f"the point behind the camera should have been dropped, got {out}"
