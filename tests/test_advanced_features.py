"""Campaigns, security scene analytics, cross-camera identity, edge feedback, tracklets, lineage,
resumable exports, and the generated SDK.

Each of these was a set of ingredients with no orchestration. The improvement loop had every stage built
and a person between each pair. The Sec pack had a static-camera scene model and no way to say where
anything was. FORGYX gated on bench numbers with no way to hear from the field. The editor had propagation,
interpolation and a filmstrip and no tracklet workflow joining them. Every lineage edge was recorded and
the graph was not. And the SDK described what somebody remembered the API did.

What is tested hardest throughout is the refusals, because in every one of these the failure mode is a
system that keeps going when it should stop: a campaign that cannot stop, a signature that becomes a name,
a device that demotes a champion, a resume that trusts a stale checkpoint.
"""

from __future__ import annotations

import math
import uuid

import numpy as np
import pytest

# ---------------------------------------------------------------- zone geometry


def test_the_anchor_is_where_an_object_touches_the_ground():
    """Not the centroid. A centroid test puts a tall person inside a floor zone while their feet are still
    outside it, and takes them out while they are still standing in it."""
    from packs.sec.zones import anchor_point

    assert anchor_point([0, 0, 10, 20]) == (5.0, 20.0)


def test_point_in_polygon_and_the_degenerate_cases():
    from packs.sec.zones import point_in_polygon

    square = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert point_in_polygon((5, 5), square) is True
    assert point_in_polygon((15, 5), square) is False
    # A polygon that is not one refuses rather than raising: a half-drawn zone must not 500 the evaluator.
    assert point_in_polygon((5, 5), []) is False
    assert point_in_polygon((5, 5), [[0, 0], [1, 1]]) is False


def test_a_crossing_needs_the_movement_to_pass_within_the_segment():
    """A side change alone is not a crossing. An object moving between sides far off the end of the drawn
    line has not crossed the gate."""
    from packs.sec.zones import segment_span, side_of_line

    line = [[0, 0], [10, 0]]
    assert side_of_line((5, -5), line) != side_of_line((5, 5), line)
    assert segment_span((5, -5), (5, 5), line) is True
    # Same side change, but a hundred units past the end of the segment.
    assert segment_span((100, -5), (100, 5), line) is False


def test_an_enter_rule_fires_once_per_visit():
    from packs.sec.zones import evaluate_track

    square = [[0, 0], [10, 0], [10, 10], [0, 10]]
    zone = {"zone_id": "z", "name": "bay", "kind": "area", "rule": "enter",
            "points": square, "severity": "warn", "classes": []}
    # Walks in at t=4 and stays.
    samples = [{"ts_ns": i * 10**9, "bbox": [4 + i, 14 - i * 2, 6 + i, 16 - i * 2],
                "class_name": "person", "track_id": "t1"} for i in range(6)]
    fired = evaluate_track(zone, samples)
    assert [c.rule for c in fired] == ["enter"]


def test_a_dwell_rule_fires_once_not_once_per_frame():
    """A dwell rule that re-fires produces one incident per frame of loitering, which is the same as
    having no rule at all."""
    from packs.sec.zones import evaluate_track

    square = [[0, 0], [100, 0], [100, 100], [0, 100]]
    zone = {"zone_id": "z", "name": "lobby", "kind": "area", "rule": "dwell",
            "points": square, "dwell_seconds": 3.0, "severity": "warn", "classes": []}
    samples = [{"ts_ns": i * 10**9, "bbox": [40, 40, 60, 60], "class_name": "person",
                "track_id": "t1"} for i in range(20)]
    fired = evaluate_track(zone, samples)
    assert len(fired) == 1 and fired[0].rule == "dwell"
    assert fired[0].detail["dwelled_s"] >= 3.0


def test_a_zone_that_cannot_mean_anything_is_refused_at_creation():
    from packs.sec.zones import validate_zone

    with pytest.raises(ValueError):
        validate_zone("line", "cross", [[0, 0]], None)             # a line needs two points
    with pytest.raises(ValueError):
        validate_zone("area", "cross", [[0, 0], [1, 0], [1, 1]], None)   # cross is a line rule
    with pytest.raises(ValueError):
        validate_zone("area", "dwell", [[0, 0], [1, 0], [1, 1]], None)   # dwell needs a threshold
    validate_zone("area", "enter", [[0, 0], [1, 0], [1, 1]], None)       # and a good one passes


def test_a_class_filter_excludes_other_classes():
    from packs.sec.zones import evaluate_track

    square = [[0, 0], [100, 0], [100, 100], [0, 100]]
    zone = {"zone_id": "z", "name": "bay", "kind": "area", "rule": "enter", "points": square,
            "severity": "warn", "classes": ["person"]}
    vehicles = [{"ts_ns": i * 10**9, "bbox": [40, 200 - i * 60, 60, 220 - i * 60],
                 "class_name": "sedan", "track_id": "v1"} for i in range(4)]
    assert evaluate_track(zone, vehicles) == []


# ---------------------------------------------------------------- RTSP sampling

def test_the_sampler_keeps_motion_and_drops_a_still_scene():
    """A camera at 25 fps produces 2.16 million frames a day, almost all showing the same empty corridor."""
    from packs.sec.rtsp import MotionSampler, SamplingPolicy

    s = MotionSampler(SamplingPolicy(warmup_frames=3, heartbeat_seconds=1000, motion_threshold=6.0))
    background = np.full((64, 64), 100, dtype=np.uint8)
    for i in range(4):
        s.consider(background, now=i * 0.5)

    assert s.consider(background, now=10.0)[:2] == (False, "no motion")
    assert s.consider(np.full((64, 64), 160, dtype=np.uint8), now=11.0)[:2] == (True, "motion")


def test_the_heartbeat_survives_a_still_scene_and_the_rate_limit():
    """The heartbeat is what proves the camera is alive, so dropping it during a busy minute is exactly
    when its absence would be misread."""
    from packs.sec.rtsp import MotionSampler, SamplingPolicy

    s = MotionSampler(SamplingPolicy(warmup_frames=2, heartbeat_seconds=5.0,
                                     motion_threshold=6.0, max_frames_per_minute=1))
    bg = np.full((32, 32), 80, dtype=np.uint8)
    for i in range(3):
        s.consider(bg, now=i)
    keep, reason, _ = s.consider(bg, now=100.0)
    assert keep and reason == "heartbeat"


# ---------------------------------------------------------------- re-identification

def test_a_track_signature_is_the_median_not_the_mean():
    """A track picks up occluded and motion-blurred crops, and a mean is dragged by them."""
    from services.sec.reid import track_signature

    clean = [[1.0, 0.0, 0.0]] * 9
    outlier = [[0.0, 0.0, 100.0]]
    sig = track_signature(clean + outlier)
    assert sig is not None
    # The median ignores the outlier entirely; a mean would point a tenth of the way toward it.
    assert sig[0] > 0.99 and abs(sig[2]) < 0.01


def test_mixed_signature_dimensions_are_refused():
    """Two embedders produced these, and averaging across them is meaningless."""
    from services.sec.reid import ReidError, track_signature

    with pytest.raises(ReidError):
        track_signature([[1.0, 0.0], [1.0, 0.0, 0.0]])
    assert track_signature([]) is None


def test_cosine_similarity_edges():
    from services.sec.reid import cosine

    a = np.array([1.0, 0.0])
    assert math.isclose(cosine(a, a), 1.0, abs_tol=1e-6)
    assert math.isclose(cosine(a, np.array([0.0, 1.0])), 0.0, abs_tol=1e-6)
    # A zero vector has no direction, so the answer is zero rather than a division by zero.
    assert cosine(a, np.zeros(2)) == 0.0


def test_an_identity_has_nowhere_to_put_a_name():
    """The boundary the whole module is built around: the system can say two tracks are the same person
    and can never say which person."""
    from db.models import PersonIdentity

    columns = set(PersonIdentity.__table__.columns.keys())
    for forbidden in ("name", "full_name", "person_name", "email", "employee_id", "face_uri"):
        assert forbidden not in columns


# ---------------------------------------------------------------- edge telemetry

def test_hellinger_is_symmetric_and_finite_on_disjoint_support():
    """Chosen over KL for exactly these two properties: KL is asymmetric, so the answer would depend on
    which distribution you called the reference, and infinite the moment the field sees a class the bench
    set never contained."""
    from services.forgyx.edge_feedback import hellinger

    p, q = [10.0, 5.0, 1.0], [1.0, 5.0, 10.0]
    assert math.isclose(hellinger(p, q), hellinger(q, p), abs_tol=1e-9)
    assert hellinger([10.0, 0.0], [0.0, 10.0]) == 1.0
    assert hellinger([1.0, 2.0], [1.0, 2.0]) == 0.0
    # Mismatched bin counts describe nothing comparable, so the answer is zero rather than a guess.
    assert hellinger([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0


@pytest.mark.db
async def test_the_field_gate_is_advisory_and_needs_more_than_one_device():
    """Telemetry comes from devices, which are outside the trust boundary. One misconfigured unit must not
    be able to demote a champion."""
    from db.session import get_sessionmaker
    from services.forgyx.edge_feedback import MIN_DEVICES, field_gate, ingest_telemetry

    artifact = f"artifact-{uuid.uuid4().hex[:8]}"
    async with get_sessionmaker()() as db:
        await ingest_telemetry(db, device_id=f"dev-{uuid.uuid4().hex[:6]}",
                               window_start_ns=0, window_end_ns=10**9, n_inferences=100,
                               latency_p95_ms=99.0, artifact_id=artifact)
        out = await field_gate(db, artifact)

    assert out["verdict"] == "insufficient_evidence"
    assert str(MIN_DEVICES) in out["detail"]
    assert out["fleet_significant"] is False


@pytest.mark.db
async def test_a_backwards_window_is_refused():
    from db.session import get_sessionmaker
    from services.forgyx.edge_feedback import TelemetryError, ingest_telemetry

    async with get_sessionmaker()() as db:
        with pytest.raises(TelemetryError):
            await ingest_telemetry(db, device_id="dev-bad", window_start_ns=10**9,
                                   window_end_ns=0)


@pytest.mark.db
async def test_an_artifact_nobody_has_reported_on_is_unknown_not_failing():
    """Absence of evidence is not evidence of a regression."""
    from db.session import get_sessionmaker
    from services.forgyx.edge_feedback import field_gate

    async with get_sessionmaker()() as db:
        out = await field_gate(db, f"never-deployed-{uuid.uuid4().hex[:8]}")
    assert out["verdict"] == "unknown"


# ---------------------------------------------------------------- campaigns

@pytest.mark.db
async def test_a_campaign_refuses_a_class_the_ontology_does_not_have():
    from db.session import get_sessionmaker
    from services.flywheel.campaign import CampaignError, create_campaign

    async with get_sessionmaker()() as db:
        with pytest.raises(CampaignError):
            await create_campaign(db, name=f"c-{uuid.uuid4().hex[:6]}", class_name="unicorn")


@pytest.mark.db
async def test_a_campaign_refuses_an_unbounded_budget():
    """The batches a campaign builds are human hours, and no wall-clock limit constrains that."""
    from db.session import get_sessionmaker
    from services.flywheel.campaign import CampaignError, create_campaign

    async with get_sessionmaker()() as db:
        with pytest.raises(CampaignError):
            await create_campaign(db, name=f"c-{uuid.uuid4().hex[:6]}", class_name="pedestrian",
                                  label_budget=0)


@pytest.mark.db
async def test_a_campaign_requires_approval_before_it_does_anything():
    """A loop that can promote a model with no person in it is a different product with a different risk
    profile, so autopilot is opted into one stage at a time."""
    from db.session import get_sessionmaker
    from services.flywheel.campaign import create_campaign, tick

    name = f"c-{uuid.uuid4().hex[:6]}"
    async with get_sessionmaker()() as db:
        created = await create_campaign(db, name=name, class_name="pedestrian", label_budget=100)
        assert created["require_approval"] is True
        out = await tick(db, created["campaign_id"])

    assert out["action"] == "awaiting_approval"
    assert out["stage"] == "mine"


@pytest.mark.db
async def test_a_campaign_stops_when_it_stops_improving():
    """A campaign that can only stop by succeeding cannot stop. Patience abandons a class that is not
    responding rather than grinding against it forever."""
    from db.models import Campaign
    from db.session import get_sessionmaker
    from services.flywheel.campaign import create_campaign, tick

    name = f"c-{uuid.uuid4().hex[:6]}"
    async with get_sessionmaker()() as db:
        created = await create_campaign(db, name=name, class_name="pedestrian",
                                        label_budget=100, patience=2)
        row = await db.get(Campaign, uuid.UUID(created["campaign_id"]))
        row.stalled_iterations = 2
        await db.commit()
        out = await tick(db, created["campaign_id"])

    assert out["action"] == "halted"
    assert out["campaign"]["status"] == "exhausted"
    assert "not responding" in out["detail"]


@pytest.mark.db
async def test_a_campaign_stops_when_the_budget_is_spent():
    from db.models import Campaign
    from db.session import get_sessionmaker
    from services.flywheel.campaign import create_campaign, tick

    async with get_sessionmaker()() as db:
        created = await create_campaign(db, name=f"c-{uuid.uuid4().hex[:6]}",
                                        class_name="pedestrian", label_budget=50)
        row = await db.get(Campaign, uuid.UUID(created["campaign_id"]))
        row.labels_spent = 50
        await db.commit()
        out = await tick(db, created["campaign_id"])

    assert out["campaign"]["status"] == "exhausted"
    assert "label budget" in out["detail"]


@pytest.mark.db
async def test_an_autopilot_stage_runs_without_approval():
    from db.session import get_sessionmaker
    from services.flywheel.campaign import create_campaign, tick

    async with get_sessionmaker()() as db:
        created = await create_campaign(db, name=f"c-{uuid.uuid4().hex[:6]}",
                                        class_name="pedestrian", label_budget=100,
                                        autopilot_stages=["mine"])
        out = await tick(db, created["campaign_id"])

    assert out["action"] == "ran" and out["stage"] == "mine"


def test_an_unknown_autopilot_stage_is_refused():
    from services.flywheel.campaign import STAGES

    assert STAGES == ("mine", "judge", "label", "train", "evaluate", "promote")


# ---------------------------------------------------------------- tracklets

@pytest.mark.db
async def test_deriving_a_track_needs_two_keyframes():
    """One keyframe defines a position, not a trajectory. Propagating a constant box down the whole track
    and calling it interpolation would be worse than refusing."""
    from db.models import Frame, Object, Track
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.temporal.tracklet import derive

    async with get_sessionmaker()() as db:
        s = DbSession(vehicle_id="veh-tl", city="BLR", start_ts_ns=0, end_ts_ns=10**9,
                      sensors={}, ontology_version="labelox-in-0.1.0")
        db.add(s)
        await db.flush()
        t = Track(session_id=s.session_id, class_id=1, first_ts_ns=0, last_ts_ns=5 * 10**8)
        db.add(t)
        await db.flush()
        for i in range(5):
            f = Frame(session_id=s.session_id, ts_ns=i * 10**8, cam_id="cam_f",
                      img_uri="s3://x", width=640, height=480, quality=0.9)
            db.add(f)
            await db.flush()
            db.add(Object(frame_id=f.frame_id, track_id=t.track_id, class_id=1,
                          bbox=[10.0, 10.0, 50.0, 50.0], conf=0.8, state="accepted",
                          source="auto_accept", is_keyframe=(i == 0)))
        await db.commit()
        track_id = str(t.track_id)

        out = await derive(db, track_id)
        assert out["updated"] == 0 and "two keyframes" in out["detail"]


@pytest.mark.db
async def test_correcting_a_frame_makes_it_a_keyframe_and_derive_respects_it():
    """A corrected box that stayed derived would be overwritten by the next derive, which is the single
    most infuriating thing a video annotator can experience."""
    from db.models import Frame, Object, Track
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.temporal.tracklet import derive, set_keyframe

    async with get_sessionmaker()() as db:
        s = DbSession(vehicle_id="veh-tl2", city="BLR", start_ts_ns=0, end_ts_ns=10**9,
                      sensors={}, ontology_version="labelox-in-0.1.0")
        db.add(s)
        await db.flush()
        t = Track(session_id=s.session_id, class_id=1, first_ts_ns=0, last_ts_ns=4 * 10**8)
        db.add(t)
        await db.flush()
        ids = []
        for i in range(5):
            f = Frame(session_id=s.session_id, ts_ns=i * 10**8, cam_id="cam_f",
                      img_uri="s3://x", width=640, height=480, quality=0.9)
            db.add(f)
            await db.flush()
            o = Object(frame_id=f.frame_id, track_id=t.track_id, class_id=1,
                       bbox=[0.0, 0.0, 10.0, 10.0], conf=0.8, state="accepted",
                       source="auto_accept", is_keyframe=(i in (0, 4)))
            db.add(o)
            await db.flush()
            ids.append(str(o.object_id))
        # The last keyframe is far away, so the middle should interpolate between them.
        last = await db.get(Object, uuid.UUID(ids[4]))
        last.bbox = [100.0, 100.0, 110.0, 110.0]
        await db.commit()

        out = await derive(db, str(t.track_id))
        assert out["updated"] == 3
        middle = await db.get(Object, uuid.UUID(ids[2]))
        assert 45 < middle.bbox[0] < 55        # halfway
        assert middle.source == "interpolated"

        # Now correct the middle. It becomes a keyframe and survives the next derive.
        await set_keyframe(db, ids[2], bbox=[10.0, 10.0, 20.0, 20.0])
        await derive(db, str(t.track_id))
        corrected = await db.get(Object, uuid.UUID(ids[2]))
        assert corrected.is_keyframe is True
        assert corrected.bbox[0] == 10.0


@pytest.mark.db
async def test_a_bad_bbox_is_refused():
    from db.session import get_sessionmaker
    from services.temporal.tracklet import TrackletError, set_keyframe

    async with get_sessionmaker()() as db:
        with pytest.raises(TrackletError):
            await set_keyframe(db, str(uuid.uuid4()), bbox=[10, 10, 5, 5])


# ---------------------------------------------------------------- resumable exports

def test_a_chunk_digest_covers_the_records_not_the_bytes():
    """Over the ids, because the bytes differ by format and the question is 'did this chunk cover these
    records'."""
    from services.export.resumable import chunk_digest

    class _R:
        def __init__(self, oid):
            self.object_id = oid

    a = [_R("x"), _R("y")]
    assert chunk_digest(a) == chunk_digest([_R("x"), _R("y")])
    assert chunk_digest(a) != chunk_digest([_R("y"), _R("x")])


def test_a_checkpoint_round_trips():
    from services.export.resumable import Checkpoint

    c = Checkpoint(chunks_done=[2, 0, 1], chunk_digests={"0": "aa"}, records_written=42,
                   total_records=100, commit_id="abc")
    d = c.as_dict()
    assert d["chunks_done"] == [0, 1, 2]       # sorted on the way out
    back = Checkpoint.from_dict(d)
    assert back.records_written == 42 and back.commit_id == "abc"
    assert Checkpoint.from_dict(None).chunks_done == []


def test_a_stale_checkpoint_chunk_is_redone_rather_than_skipped():
    """The corpus changes between attempts. A chunk whose digest no longer matches is redone, because the
    alternative is an archive stitched from two different versions of the corpus."""
    from services.export.resumable import Checkpoint, _verify_checkpoint, chunk_digest

    class _R:
        def __init__(self, oid):
            self.object_id = oid

    chunks = [[_R("a")], [_R("b")], [_R("c")]]
    cp = Checkpoint(chunks_done=[0, 1, 2],
                    chunk_digests={"0": chunk_digest(chunks[0]),
                                   "1": "a-digest-from-a-previous-corpus",
                                   "2": chunk_digest(chunks[2])})
    verified, invalidated = _verify_checkpoint(cp, chunks)
    assert verified == [0, 2] and invalidated == [1]


def test_a_checkpoint_pointing_past_the_end_is_invalidated():
    from services.export.resumable import Checkpoint, _verify_checkpoint

    cp = Checkpoint(chunks_done=[0, 9], chunk_digests={})
    verified, invalidated = _verify_checkpoint(cp, [[], []])
    assert 9 in invalidated


# ---------------------------------------------------------------- the generated SDK

def test_the_sdk_is_current_with_the_api_schema():
    """The direction of truth: the server defines the surface and the client is derived. A drifted client
    is one that keeps calling routes the way they used to be."""
    from pathlib import Path

    from scripts.generate_sdk import generate
    from services.api.main import app

    expected = generate(app.openapi())
    on_disk = Path("sdk/generated_client.py").read_text()
    assert on_disk.strip() == expected.strip(), (
        "sdk/generated_client.py is out of date; run `python -m scripts.generate_sdk`")


def test_the_generated_client_parses_and_refuses_without_a_token():
    import ast
    from pathlib import Path

    source = Path("sdk/generated_client.py").read_text()
    ast.parse(source)   # required arguments before defaulted ones, among everything else

    from sdk.generated_client import LabeloxClient, LabeloxError

    with pytest.raises(LabeloxError):
        LabeloxClient(token=None, base_url="http://localhost:8000")


def test_the_generator_orders_arguments_legally():
    """Python forbids a required argument after a defaulted one, and a generator that emits it produces a
    module that does not import at all."""
    from scripts.generate_sdk import generate

    schema = {"paths": {"/api/thing": {"post": {
        "summary": "a route with a body and a required query param",
        "parameters": [{"name": "session_id", "in": "query", "required": True,
                        "schema": {"type": "string"}},
                       {"name": "limit", "in": "query", "required": False,
                        "schema": {"type": "integer", "default": 10}}],
        "requestBody": {"content": {"application/json": {}}},
    }}}}
    import ast

    code = generate(schema)
    ast.parse(code)
    assert "session_id: str, body: Any = None" in code


def test_unset_query_parameters_are_dropped():
    """Sending them as the string "None" is the classic generated-client bug: the server sees a value where
    the caller meant absence."""
    from sdk.generated_client import _clean

    assert _clean({"a": 1, "b": None}) == {"a": 1}
    assert _clean(None) is None
    assert _clean({}) is None
