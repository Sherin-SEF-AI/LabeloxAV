"""The ML-side gaps: three missing training plugins, tracking that could never be scored, the frames active
learning could not see, experiment comparison, and the recall priors nobody ever measured.

Each of these was the same shape: the corpus already held the supervision and the system had no way to
consume it. Keypoints could be drawn and never trained on until PoseTask; lanes and drivable surfaces still
could not; crops carried every reviewer correction and nothing learned from them; MOTA existed and had no
ground truth to score against; active learning could only ever propose reviewing what the detector already
found.
"""

from __future__ import annotations

import math
import uuid

import numpy as np
import pytest

pytestmark = pytest.mark.db


# ---------------------------------------------------------------- the task registry

def test_every_task_type_is_registered_with_head_appropriate_weights():
    """A classifier started from a detection checkpoint discards the pretrained head, so the base weights
    are part of whether the plugin is correct rather than a configuration detail."""
    from services.training.tasks import get_task, list_tasks

    types = {t["task_type"] for t in list_tasks()}
    assert types == {"detection", "segmentation", "pose", "classification", "lane", "detect3d"}

    assert "cls" in get_task("classification").default_base_weights()
    assert "seg" in get_task("segmentation").default_base_weights()
    assert "pose" in get_task("pose").default_base_weights()
    # A lane model is a segmentation model over ribbon masks, so it inherits that head.
    assert "seg" in get_task("lane").default_base_weights()


def test_the_3d_task_refuses_to_train_rather_than_faking_a_number():
    """A plugin that quietly trained a 2D model on 3D labels and reported an mAP would be worse than one
    that refuses: the number would look like progress."""
    from services.training.tasks.detect3d import Detection3dTask, NativeTrainingUnavailable

    task = Detection3dTask()
    with pytest.raises(NativeTrainingUnavailable) as exc:
        task.train("/tmp/ds", "w.pt", {"name": "x"}, lambda _: None)
    # The refusal names what is missing, so it is actionable rather than a wall.
    assert "OpenPCDet" in str(exc.value) or "MMDetection3D" in str(exc.value)

    with pytest.raises(NativeTrainingUnavailable):
        task.evaluate("w.pt", "/tmp/ds", 640)


# ---------------------------------------------------------------- lane rasterisation

def test_a_lane_polyline_rasterises_to_a_ribbon_of_the_requested_width():
    """A polyline has no width, so an IoU against it is degenerate. This is why lane training goes through
    a segmentation head over ribbons rather than scoring lines directly."""
    from services.training.tasks.lane import rasterize_lane

    mask = rasterize_lane([[10, 50], [90, 50]], width=100, height=100, line_width_px=10)
    assert mask is not None
    column = mask[:, 50]
    on = int((column > 127).sum())
    # Anti-aliased, so exact equality would be wrong; the ribbon is about the width asked for.
    assert 8 <= on <= 13


def test_a_degenerate_lane_produces_nothing_rather_than_an_empty_label():
    from services.training.tasks.lane import rasterize_lane

    assert rasterize_lane([], 100, 100, 8) is None
    assert rasterize_lane([[5, 5]], 100, 100, 8) is None
    # Two identical points are one point: a line of zero length is not a lane.
    assert rasterize_lane([[5, 5], [5, 5]], 100, 100, 8) is None


def test_mask_polygons_are_normalized_and_drop_specks():
    from services.training.tasks.lane import mask_to_polygons, rasterize_lane

    mask = rasterize_lane([[10, 10], [80, 80]], 100, 100, 12)
    polys = mask_to_polygons(mask, 100, 100)
    assert polys
    for poly in polys:
        assert len(poly) % 2 == 0 and len(poly) >= 6
        assert all(0.0 <= v <= 1.0 for v in poly)

    speck = np.zeros((100, 100), dtype=np.uint8)
    speck[50:52, 50:52] = 255
    assert mask_to_polygons(speck, 100, 100, min_area_px=100) == []


# ---------------------------------------------------------------- KITTI 3D format

def test_a_kitti_label_line_follows_kitti_conventions_not_convenient_ones():
    """Every reader assumes them: height width length in that order, the BOTTOM centre as the location,
    and a separate observation angle. Getting any of these wrong produces a file that parses and means
    something else."""
    from services.training.tasks.detect3d import kitti_label_line

    line = kitti_label_line("car", center=[1.0, 0.0, 10.0], size=[1.8, 4.2, 1.5], yaw=0.0)
    parts = line.split()
    assert len(parts) == 15
    assert parts[0] == "car"
    h, w, length = float(parts[8]), float(parts[9]), float(parts[10])
    assert (h, w, length) == (1.5, 1.8, 4.2)
    # y is the bottom, which is the centre plus half the height.
    assert math.isclose(float(parts[12]), 0.0 + 1.5 / 2, abs_tol=1e-6)


def test_the_observation_angle_removes_the_viewing_direction():
    """A box at the edge of the image with the same world yaw looks quite different from one straight
    ahead, which is why KITTI carries alpha as well as rotation_y."""
    from services.training.tasks.detect3d import _alpha_from_yaw

    assert math.isclose(_alpha_from_yaw(0.0, x=0.0, z=10.0), 0.0, abs_tol=1e-6)
    off_axis = _alpha_from_yaw(0.0, x=10.0, z=10.0)
    assert math.isclose(off_axis, -math.pi / 4, abs_tol=1e-6)
    assert -math.pi <= off_axis <= math.pi


def test_a_missing_calibration_writes_identity_and_is_reported():
    """A wrong calibration silently ruins every 3D box projected through it and a reader cannot tell, so
    an absent one is written as identity and counted rather than invented."""
    from services.training.tasks.detect3d import kitti_calib_text

    text = kitti_calib_text(None)
    assert "P2:" in text and "Tr_velo_to_cam:" in text
    real = kitti_calib_text({"fx": 1000.0, "fy": 1000.0, "cx": 640.0, "cy": 360.0})
    assert "1.000000e+03" in real


# ---------------------------------------------------------------- tracking against gold

async def _seed_tracked_session(n_frames: int = 4, with_tracks: bool = True) -> dict:
    """A session whose objects share a track across frames, which is what a tracking metric needs."""
    from db.models import Frame, Object, Track
    from db.models import Session as DbSession
    from db.session import get_sessionmaker

    async with get_sessionmaker()() as db:
        s = DbSession(vehicle_id="veh-track", city="BLR", start_ts_ns=0, end_ts_ns=10**9,
                      sensors={}, ontology_version="labelox-in-0.1.0")
        db.add(s)
        await db.flush()
        # Object.track_id is a real foreign key, so the track has to exist. That constraint is precisely
        # what makes a sealed identity trustworthy: an id on an object always names a track that is there.
        track = None
        if with_tracks:
            t = Track(session_id=s.session_id, class_id=1, first_ts_ns=0,
                      last_ts_ns=n_frames * 10**8)
            db.add(t)
            await db.flush()
            track = t.track_id
        object_ids = []
        for i in range(n_frames):
            f = Frame(session_id=s.session_id, ts_ns=i * 10**8, cam_id="cam_f",
                      img_uri="s3://none", width=640, height=480, quality=0.9)
            db.add(f)
            await db.flush()
            o = Object(frame_id=f.frame_id, class_id=1, bbox=[10.0 + i, 10.0, 60.0 + i, 60.0],
                       conf=0.9, state="accepted", source="human",
                       track_id=track)
            db.add(o)
            await db.flush()
            object_ids.append(str(o.object_id))
        await db.commit()
        return {"session_id": str(s.session_id), "object_ids": object_ids,
                "track_id": str(track) if track else None}


async def test_a_gold_set_without_track_identities_is_refused_by_name():
    """The point of the whole change. Objects without an identity read as a new track in every frame, so
    the resulting MOTA measures the gap in the labels rather than the tracker."""
    from db.models import GoldSet
    from db.session import get_sessionmaker
    from services.verdyx.track_eval import TrackGroundTruthUnavailable, gold_track_detections

    seeded = await _seed_tracked_session(with_tracks=False)
    gid = f"gold-untracked-{uuid.uuid4().hex[:8]}"
    async with get_sessionmaker()() as db:
        db.add(GoldSet(gold_id=gid, name=gid, spec={}, object_ids=seeded["object_ids"],
                       n_objects=len(seeded["object_ids"]), n_frames=4,
                       ontology_version="labelox-in-0.1.0",
                       track_ids=["" for _ in seeded["object_ids"]], tracks_sealed=False))
        await db.commit()

        with pytest.raises(TrackGroundTruthUnavailable) as exc:
            await gold_track_detections(db, gid)
    assert "track identities" in str(exc.value)


async def test_a_sealed_set_with_identities_yields_ordered_ground_truth():
    """Frames are indexed by timestamp, not by uuid: MOTA counts switches between consecutive frames, and
    an arbitrary ordering would invent switches that never happened."""
    from db.models import GoldSet
    from db.session import get_sessionmaker
    from services.verdyx.track_eval import gold_track_detections

    seeded = await _seed_tracked_session(n_frames=5)
    gid = f"gold-tracked-{uuid.uuid4().hex[:8]}"
    async with get_sessionmaker()() as db:
        db.add(GoldSet(gold_id=gid, name=gid, spec={}, object_ids=seeded["object_ids"],
                       n_objects=len(seeded["object_ids"]), n_frames=5,
                       ontology_version="labelox-in-0.1.0",
                       track_ids=[seeded["track_id"]] * len(seeded["object_ids"]),
                       tracks_sealed=True))
        await db.commit()
        dets, meta = await gold_track_detections(db, gid)

    assert meta["frames"] == 5 and meta["tracks"] == 1
    assert [d.frame for d in sorted(dets, key=lambda d: d.frame)] == [0, 1, 2, 3, 4]


async def test_scoring_a_detection_run_as_a_tracker_is_refused():
    """An honest zero would be indistinguishable from a tracker that produced nothing, which is a different
    situation with a different fix."""
    from db.models import GoldSet
    from db.session import get_sessionmaker
    from services.verdyx.track_eval import TrackGroundTruthUnavailable, score_tracker

    seeded = await _seed_tracked_session(n_frames=3)
    gid = f"gold-notrackrun-{uuid.uuid4().hex[:8]}"
    async with get_sessionmaker()() as db:
        db.add(GoldSet(gold_id=gid, name=gid, spec={}, object_ids=seeded["object_ids"],
                       n_objects=3, n_frames=3, ontology_version="labelox-in-0.1.0",
                       track_ids=[seeded["track_id"]] * 3, tracks_sealed=True))
        await db.commit()
        with pytest.raises(TrackGroundTruthUnavailable) as exc:
            await score_tracker(db, gold_id=gid, run_id=str(uuid.uuid4()))
    assert "detector, not a tracker" in str(exc.value)


# ---------------------------------------------------------------- false-negative mining

def test_sparsity_only_fires_where_the_neighbours_are_populated():
    """Otherwise every genuinely empty road becomes a candidate and the queue is worthless."""
    from services.activelearn.false_negatives import _sparsity_signal

    busy = [{"frame_id": f"f{i}", "session_id": "s", "ts_ns": i, "n_objects": 10 if i != 4 else 0}
            for i in range(9)]
    quiet = [{"frame_id": f"q{i}", "session_id": "s2", "ts_ns": i, "n_objects": 0} for i in range(9)]

    assert _sparsity_signal(busy)["f4"] == 1.0
    assert _sparsity_signal(busy)["f3"] == 0.0
    assert all(v == 0.0 for v in _sparsity_signal(quiet).values())


def test_residue_peaks_just_below_the_accept_threshold():
    """A frame at 0.49 nearly produced something. One at 0.05 is a frame the model is confidently
    uninterested in, and one at 0.9 is a frame it already got."""
    from services.activelearn.false_negatives import _residue_signal

    frames = [{"frame_id": "near", "max_conf": 0.49}, {"frame_id": "far", "max_conf": 0.05},
              {"frame_id": "over", "max_conf": 0.90}, {"frame_id": "none", "max_conf": None}]
    sig = _residue_signal(frames, accept_threshold=0.5)
    assert sig["near"] > sig["far"] > 0.0
    assert sig["over"] == 0.0
    # No detections at all is sparsity's and novelty's job; scoring it here would double-count.
    assert sig["none"] == 0.0


async def test_mining_ranks_frames_and_reports_what_it_truncated():
    """A caller that asked for 5 and got 5 cannot otherwise tell whether the queue was exhausted or cut."""
    from db.session import get_sessionmaker
    from services.activelearn.false_negatives import mine_false_negatives

    seeded = await _seed_tracked_session(n_frames=6)
    async with get_sessionmaker()() as db:
        out = await mine_false_negatives(db, session_id=seeded["session_id"], top_k=2)

    assert out["considered"] == 6
    assert len(out["candidates"]) <= 2
    assert "truncated" in out
    for c in out["candidates"]:
        assert set(c["signals"]) == {"sparsity", "residue", "discontinuity", "novelty"}
        assert c["reasons"]


# ---------------------------------------------------------------- experiments

async def test_an_experiment_ranks_its_runs_and_names_what_varied():
    """A diff of forty identical columns is not a finding. "lr changed and nothing else" is."""
    from db.models import Experiment, ExperimentRun
    from db.session import get_sessionmaker
    from services.training.experiments import create_experiment, experiment_detail

    name = f"exp-{uuid.uuid4().hex[:8]}"
    async with get_sessionmaker()() as db:
        created = await create_experiment(db, name=name, hypothesis="does lr matter")
        eid = uuid.UUID(created["experiment_id"])
        for label, lr, score in [("a", 0.01, 0.42), ("b", 0.001, 0.55), ("c", 0.1, 0.31)]:
            db.add(ExperimentRun(experiment_id=eid, label=label,
                                 hparams={"lr": lr, "epochs": 20, "imgsz": 640},
                                 metrics={"map50": score}, status="done"))
        # A crashed run, which must be excluded from the ranking rather than scored zero.
        db.add(ExperimentRun(experiment_id=eid, label="crashed", hparams={"lr": 0.01, "epochs": 20,
                                                                          "imgsz": 640},
                             metrics={}, status="failed"))
        await db.commit()

        detail = await experiment_detail(db, name)

    assert detail["ranking"][0]["label"] == "b"
    assert detail["unscored_runs"] == 1
    # epochs and imgsz were held fixed, so only lr is reported as varied.
    assert detail["varied"] == ["lr"]
    assert detail["hypothesis"] == "does lr matter"

    async with get_sessionmaker()() as db:
        assert await db.get(Experiment, eid) is not None


async def test_ranking_respects_the_direction_a_metric_improves_in():
    """A loss and an mAP both move, and ranking them the same way would promote the worst run in the set."""
    from db.models import ExperimentRun
    from db.session import get_sessionmaker
    from services.training.experiments import create_experiment, experiment_detail

    name = f"exp-loss-{uuid.uuid4().hex[:8]}"
    async with get_sessionmaker()() as db:
        eid = uuid.UUID((await create_experiment(db, name=name))["experiment_id"])
        db.add(ExperimentRun(experiment_id=eid, label="high", metrics={"loss": 0.9}, status="done"))
        db.add(ExperimentRun(experiment_id=eid, label="low", metrics={"loss": 0.1}, status="done"))
        await db.commit()
        detail = await experiment_detail(db, name, metric="loss")

    assert detail["ranking"][0]["label"] == "low"


async def test_comparing_two_runs_states_which_is_better_per_metric():
    from db.models import ExperimentRun
    from db.session import get_sessionmaker
    from services.training.experiments import compare_runs, create_experiment

    name = f"exp-cmp-{uuid.uuid4().hex[:8]}"
    async with get_sessionmaker()() as db:
        eid = uuid.UUID((await create_experiment(db, name=name))["experiment_id"])
        a = ExperimentRun(experiment_id=eid, label="a", hparams={"lr": 0.01, "epochs": 10},
                          metrics={"map50": 0.40, "loss": 0.9}, gold_id="g1", status="done")
        b = ExperimentRun(experiment_id=eid, label="b", hparams={"lr": 0.001, "epochs": 10},
                          metrics={"map50": 0.55, "loss": 0.4}, gold_id="g1", status="done")
        db.add_all([a, b])
        await db.commit()
        await db.refresh(a)
        await db.refresh(b)
        cmp = await compare_runs(db, str(a.run_id), str(b.run_id))

    assert cmp["hparam_diff"] == {"lr": {"a": 0.01, "b": 0.001}}
    assert cmp["metric_diff"]["map50"]["better"] == "b"
    # Lower is better for a loss, so b wins that one too, by the opposite sign.
    assert cmp["metric_diff"]["loss"]["better"] == "b"
    assert cmp["same_gold"] is True


# ---------------------------------------------------------------- recall reliability

async def test_channel_reliability_is_measured_but_not_applied_on_thin_evidence():
    """Three confirmations out of three is not a reliability of 1.0; it is "we have seen three". Promoting
    it over a considered prior would make the queue swing on a handful of reviews."""
    from db.session import get_sessionmaker
    from services.recall.recover import channel_reliability, fit_channel_reliability

    async with get_sessionmaker()() as db:
        out = await fit_channel_reliability(db, min_verdicts=10_000, apply=True)

    for m in out["channels"].values():
        assert m["applied"] is False
    # The hand-set prior stays in effect while the evidence is thin.
    assert channel_reliability("trackgap") == 0.85


def test_laplace_smoothing_pulls_a_small_sample_toward_the_middle():
    """A raw ratio would score 4/4 as 1.0 and dominate the ranking on almost no evidence."""
    confirmed, rejected = 4, 0
    rate = (confirmed + 1) / (confirmed + rejected + 2)
    assert 0.8 < rate < 0.9      # not 1.0
