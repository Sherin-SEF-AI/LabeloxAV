"""4D occupancy: what is voxelised, whose velocity a voxel carries, and what round-trips.

The flow field is the part that can be quietly wrong. A voxel assigned the velocity of a cuboid it is not
inside produces a grid that looks like a measurement and describes motion that never happened, and
nothing downstream can tell, because a flow field has no ground truth to check against.
"""

from __future__ import annotations

import numpy as np
import pytest

from services.lidar.occupancy4d import (
    MIN_POINTS,
    FlowSource,
    assign_flow,
    pack_grid,
    unpack_grid,
    voxel_indices,
)

ORIGIN = np.array([0.0, 0.0, 0.0])
DIMS = (10, 10, 10)
VOXEL = 1.0


class TestVoxelisation:
    def test_an_empty_cloud_occupies_nothing(self):
        assert voxel_indices(np.zeros((0, 3)), ORIGIN, DIMS, VOXEL).shape == (0, 3)

    def test_points_in_one_cell_are_one_voxel(self):
        pts = np.array([[1.1, 1.2, 1.3], [1.4, 1.1, 1.9], [1.05, 1.95, 1.05]])
        got = voxel_indices(pts, ORIGIN, DIMS, VOXEL, min_points=3)
        assert got.tolist() == [[1, 1, 1]]

    def test_a_lone_point_is_noise_rather_than_an_object(self):
        """One point in a cell is a single bad pixel back-projected metres from anything, which is what a
        monocular depth cloud produces constantly."""
        pts = np.array([[1.5, 1.5, 1.5]])
        assert len(voxel_indices(pts, ORIGIN, DIMS, VOXEL, min_points=MIN_POINTS)) == 0

    def test_points_outside_the_window_are_dropped_not_clamped(self):
        """Clamping would pile the whole horizon onto the window's edge voxels and read as a wall."""
        pts = np.repeat(np.array([[999.0, 999.0, 999.0]]), 10, axis=0)
        assert len(voxel_indices(pts, ORIGIN, DIMS, VOXEL, min_points=1)) == 0

    def test_negative_coordinates_relative_to_the_origin_are_dropped(self):
        pts = np.repeat(np.array([[-5.0, 1.0, 1.0]]), 10, axis=0)
        assert len(voxel_indices(pts, ORIGIN, DIMS, VOXEL, min_points=1)) == 0

    def test_the_origin_shifts_the_grid_rather_than_the_points(self):
        pts = np.repeat(np.array([[11.5, 1.5, 1.5]]), 5, axis=0)
        shifted = voxel_indices(pts, np.array([10.0, 0.0, 0.0]), DIMS, VOXEL, min_points=1)
        assert shifted.tolist() == [[1, 1, 1]]


class TestSceneFlow:
    def _box_at(self, x: float, vel=(0.0, 0.0, 0.0)):
        return FlowSource(center=(x, 5.0, 1.0), dims=(2.0, 2.0, 2.0), yaw=0.0, velocity=vel)

    def test_no_sources_leaves_every_voxel_at_zero(self):
        voxels = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.int32)
        flow, claimed = assign_flow(voxels, ORIGIN, VOXEL, [])
        assert claimed == 0
        assert not flow.any()

    def test_a_voxel_inside_a_moving_cuboid_takes_its_velocity(self):
        # Voxel (5,5,1) has its centre at (5.5, 5.5, 1.5); a 2m box at (5,5,1) spans 4 to 6 in x and y.
        voxels = np.array([[5, 5, 1]], dtype=np.int32)
        flow, claimed = assign_flow(voxels, ORIGIN, VOXEL, [self._box_at(5.0, vel=(3.0, 0.0, 0.0))])
        assert claimed == 1
        assert flow[0].tolist() == pytest.approx([3.0, 0.0, 0.0])

    def test_a_voxel_outside_every_cuboid_stays_at_an_assumed_zero(self):
        """Zero is right: most of a road scene is static, and inventing motion for unclaimed space would
        be worse. But the count is what separates an assumed zero from a measured one."""
        voxels = np.array([[0, 0, 0]], dtype=np.int32)
        flow, claimed = assign_flow(voxels, ORIGIN, VOXEL, [self._box_at(5.0, vel=(3.0, 0.0, 0.0))])
        assert claimed == 0
        assert flow[0].tolist() == [0.0, 0.0, 0.0]

    def test_a_static_cuboid_claims_its_voxels_at_zero_velocity(self):
        """A parked car and empty road both read zero in the field, and only the claim count tells them
        apart. That is exactly why the count exists rather than counting non-zero flow."""
        voxels = np.array([[5, 5, 1]], dtype=np.int32)
        flow, claimed = assign_flow(voxels, ORIGIN, VOXEL, [self._box_at(5.0, vel=(0.0, 0.0, 0.0))])
        assert claimed == 1 and not flow.any()

    def test_rotation_is_honoured_rather_than_treated_as_axis_aligned(self):
        """A long thin box turned 90 degrees covers a different set of voxels. Ignoring yaw would claim
        voxels beside the vehicle for one in front of it."""
        long_box = FlowSource(center=(5.0, 5.0, 1.0), dims=(8.0, 1.0, 2.0), yaw=0.0,
                              velocity=(1.0, 0.0, 0.0))
        turned = FlowSource(center=(5.0, 5.0, 1.0), dims=(8.0, 1.0, 2.0), yaw=np.pi / 2,
                            velocity=(1.0, 0.0, 0.0))
        along_x = np.array([[8, 5, 1]], dtype=np.int32)
        _f1, c1 = assign_flow(along_x, ORIGIN, VOXEL, [long_box])
        _f2, c2 = assign_flow(along_x, ORIGIN, VOXEL, [turned])
        assert c1 == 1, "the unturned long box reaches along x"
        assert c2 == 0, "the turned one does not"

    def test_claimed_never_exceeds_the_voxel_count(self):
        # The database checks the same thing; a flow count above the occupied count is not a grid.
        voxels = np.array([[5, 5, 1], [5, 5, 1]], dtype=np.int32)
        _flow, claimed = assign_flow(voxels, ORIGIN, VOXEL,
                                     [self._box_at(5.0, vel=(1.0, 0.0, 0.0))] * 3)
        assert claimed <= len(voxels)


class TestPacking:
    def test_a_grid_round_trips(self):
        voxels = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        flow = np.array([[1.5, 0.0, -2.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        got = unpack_grid(pack_grid(voxels, flow, DIMS))
        assert got["voxels"].tolist() == voxels.tolist()
        assert got["dims"] == DIMS
        assert got["flow"] == pytest.approx(flow, abs=0.01)

    def test_an_empty_grid_round_trips_as_empty(self):
        got = unpack_grid(pack_grid(np.zeros((0, 3), np.int32), np.zeros((0, 3), np.float32), DIMS))
        assert len(got["voxels"]) == 0 and got["dims"] == DIMS

    def test_sparse_packing_is_far_smaller_than_dense_would_be(self):
        """A 160x120x12 grid is 230,400 cells and a road scene fills a few thousand. Dense would be
        almost entirely zeros, three times over for the flow channels."""
        n = 3000
        rng = np.random.default_rng(7)
        voxels = rng.integers(0, 100, size=(n, 3)).astype(np.int32)
        flow = rng.normal(size=(n, 3)).astype(np.float32)
        packed = len(pack_grid(voxels, flow, (160, 120, 12)))
        dense = 160 * 120 * 12 * (1 + 3 * 2)
        assert packed < dense / 10


class TestCuboidProjectionIsTierTwo:
    """Projecting a cuboid is exact geometry; deriving a box from where it touches the road is not.

    The ground path infers a 3D position from the box's contact pixel, which fails for anything not
    standing on the road and carries the flat-road assumption into every target view. A cuboid is already
    a 3D extent.
    """

    def test_the_cuboid_branch_runs_before_the_ground_lift(self):
        import inspect

        from services.multicam import propagate

        src = inspect.getsource(propagate.propagate_object)
        cuboid_at = src.index("_cuboid_for(db, object_id)")
        ground_at = src.index("_lift_ground(*base_px, src_calib)")
        assert cuboid_at < ground_at, "the exact path must be tried before the inferred one"

    def test_a_rejected_cuboid_is_never_projected(self):
        """A rejected 3D box is one somebody looked at and said was wrong. Projecting it into four other
        views multiplies that mistake by four."""
        import inspect

        from services.multicam import propagate

        assert 'Object3D.state != "rejected"' in inspect.getsource(propagate._cuboid_for)

    def test_corners_behind_the_camera_are_dropped_rather_than_projected(self):
        """A point behind the lens maps to a plausible-looking pixel on the wrong side of the image, and
        a hull that includes one is a box in the wrong place with nothing marking it wrong."""
        import inspect

        from services.multicam import propagate

        src = inspect.getsource(propagate._propagate_cuboid)
        assert 'zip(proj["corners_uv"], proj["in_front"]' in src

    def test_the_projected_box_carries_the_cuboid_it_came_from(self):
        import inspect

        from services.multicam import propagate

        src = inspect.getsource(propagate._propagate_cuboid)
        assert '"method": "cuboid_project"' in src
        assert '"object_3d_id"' in src
        assert '"cuboid_conf"' in src


class TestTheProjectionGeometry:
    """The hull of a projected cuboid, checked against a camera whose geometry is known."""

    def test_a_cuboid_in_front_projects_to_a_box_around_the_image_centre(self):
        from services.calibration.resolve import nominal_calibration
        from services.lidar.boxes import project_cuboid

        calib = nominal_calibration("cam_f", 1280, 960)
        proj = project_cuboid([20.0, 0.0, 0.0], [4.0, 1.8, 1.5], 0.0, "cam_f", 1280, 960, calib=calib)
        uv = [p for p, f in zip(proj["corners_uv"], proj["in_front"], strict=False) if f]
        assert len(uv) == 8, "every corner of a box 20 m ahead is in front of a forward camera"
        xs = [p[0] for p in uv]
        assert min(xs) < 640 < max(xs), "the box straddles the optical axis"

    def test_a_cuboid_behind_the_camera_has_no_corners_in_front(self):
        from services.calibration.resolve import nominal_calibration
        from services.lidar.boxes import project_cuboid

        calib = nominal_calibration("cam_f", 1280, 960)
        proj = project_cuboid([-20.0, 0.0, 0.0], [4.0, 1.8, 1.5], 0.0, "cam_f", 1280, 960, calib=calib)
        assert not any(proj["in_front"])

    def test_a_further_cuboid_projects_smaller(self):
        from services.calibration.resolve import nominal_calibration
        from services.lidar.boxes import project_cuboid

        calib = nominal_calibration("cam_f", 1280, 960)

        def width_at(x):
            p = project_cuboid([x, 0.0, 0.0], [4.0, 1.8, 1.5], 0.0, "cam_f", 1280, 960, calib=calib)
            uv = [q for q, f in zip(p["corners_uv"], p["in_front"], strict=False) if f]
            return max(q[0] for q in uv) - min(q[0] for q in uv)

        assert width_at(40.0) < width_at(20.0) < width_at(10.0)
