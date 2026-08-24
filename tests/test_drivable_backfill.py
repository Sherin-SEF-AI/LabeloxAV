"""Drivable on every frame, and the three ways it used to produce data nobody could read.

Drivable area existed on 1,643 of 41,752 frames because `segment_drivable` had exactly one caller - the
editor's button - so a mask existed only where somebody had opened a frame and clicked. The main dashcam
corpus, 37,711 frames, had seventy-nine.

The tests that matter are not that a backfill loops. They are the three failure modes that would have made
forty thousand new masks worthless on arrival: a model that silently returns a geometric trapezoid when it
cannot reach the GPU, a training reader that cannot parse what every writer writes, and an export that
copies JSON into files named `.png`.
"""

from __future__ import annotations

import json
import uuid

import cv2
import numpy as np
import pytest

from core.timebase import now_ns
from db.models import AgentRun, DrivableMask, Frame
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology


class TestTheModelNoLongerLiesUnderPressure:
    def test_a_failure_raises_instead_of_returning_a_trapezoid(self, monkeypatch):
        """The trapezoid is a fixed shape with no appearance information: it cannot match a road.

        It used to be returned on ANY exception, including out-of-memory, with one warning line. Over a
        corpus backfill that is a mechanism for manufacturing tens of thousands of confident-looking masks
        that are geometry, and only `model_version` would have said so.
        """
        from services.autolabel import drivable as mod

        monkeypatch.setattr(mod, "_segment_seg", lambda img: (_ for _ in ()).throw(RuntimeError("CUDA OOM")))
        with pytest.raises(mod.DrivableUnavailable, match="CUDA OOM"):
            mod.segment_drivable(np.zeros((64, 64, 3), np.uint8))

    def test_the_trapezoid_is_still_available_to_a_caller_that_asks(self, monkeypatch):
        """It is a legitimate starting region for a human to refine. What it is not is a silent default."""
        from services.autolabel import drivable as mod

        monkeypatch.setattr(mod, "_segment_seg", lambda img: (_ for _ in ()).throw(RuntimeError("no gpu")))
        out = mod.segment_drivable(np.zeros((200, 200, 3), np.uint8), allow_trapezoid=True)
        assert out["model"] == "trapezoid:local"
        # And it still says which class it could not estimate, rather than reporting a bare zero.
        assert out["unestimated_classes"] == ["fallback"]

    def test_the_pod_backend_names_where_the_pod_actually_lives(self):
        """The setting's better-looking value is the one that breaks.

        `backend="pod"` raises: the pod path runs through services/perception/cloud.py, which never reads
        this setting. The message used to tell you to set the value that causes the raise.
        """
        from core.config import get_settings
        from services.autolabel import drivable as mod

        cfg = get_settings().models.drivable
        assert cfg.backend == "local"
        object.__setattr__(cfg, "backend", "pod")
        try:
            with pytest.raises(NotImplementedError, match="cloud-perception"):
                mod.segment_drivable(np.zeros((8, 8, 3), np.uint8))
        finally:
            object.__setattr__(cfg, "backend", "local")


class TestTheTrainingReaderCanReadWhatIsWritten:
    """`cv2.imdecode` on a JSON blob returns None, so the drivable source contributed zero labels to every
    dataset ever built from it - and `data.yaml` went on declaring three surface classes behind them."""

    _IDX = {"surface_drivable": 0, "surface_non_drivable": 1, "surface_fallback": 2}

    class _Store:
        def __init__(self, payload: bytes):
            self.payload = payload

        def get_bytes(self, uri: str) -> bytes:
            return self.payload

    def _blob(self, **classes) -> bytes:
        base = {"drivable": [], "non_drivable": [], "fallback": []}
        base.update(classes)
        return json.dumps({"classes": base, "width": 200, "height": 100}).encode()

    def test_polygons_become_label_lines(self):
        from services.training.tasks.lane import _drivable_lines

        store = self._Store(self._blob(drivable=[[0, 0, 100, 0, 100, 50, 0, 50]]))
        lines = _drivable_lines(store, "u", 200, 100, self._IDX)
        assert len(lines) == 1
        parts = lines[0].split()
        assert parts[0] == "0"
        # 100/200 and 50/100 both normalise to 0.5, which is the whole of the coordinate work.
        assert [float(v) for v in parts[1:]] == [0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5]

    def test_the_stored_size_drives_normalisation_not_the_export_size(self):
        """A mask written at capture resolution must land correctly in an export resized to anything else.

        Normalising against the target size would move every polygon whenever the export dimensions
        differed from the capture, which is a silent misalignment rather than a visible failure.
        """
        from services.training.tasks.lane import _drivable_lines

        store = self._Store(self._blob(drivable=[[0, 0, 100, 0, 100, 50, 0, 50]]))
        at_capture = _drivable_lines(store, "u", 200, 100, self._IDX)
        at_double = _drivable_lines(store, "u", 400, 200, self._IDX)
        assert at_capture == at_double

    def test_every_surface_class_reaches_its_own_index(self):
        from services.training.tasks.lane import _drivable_lines

        sq = [0, 0, 10, 0, 10, 10, 0, 10]
        store = self._Store(self._blob(drivable=[sq], non_drivable=[sq], fallback=[sq]))
        got = sorted(ln.split()[0] for ln in _drivable_lines(store, "u", 200, 100, self._IDX))
        assert got == ["0", "1", "2"]

    def test_a_degenerate_polygon_is_dropped_not_emitted(self):
        # Two points is a line. A label line for it would train the model on a shape with no area.
        from services.training.tasks.lane import _drivable_lines

        store = self._Store(self._blob(drivable=[[0, 0, 10, 10]]))
        assert _drivable_lines(store, "u", 200, 100, self._IDX) == []

    def test_an_unreadable_blob_costs_one_frame_not_the_dataset(self):
        from services.training.tasks.lane import _drivable_lines

        assert _drivable_lines(self._Store(b"not json at all"), "u", 200, 100, self._IDX) == []

    def test_the_old_image_reader_would_have_returned_nothing(self):
        """The regression stated as the thing it was: imdecode on this payload is None, every time."""
        payload = self._blob(drivable=[[0, 0, 100, 0, 100, 50, 0, 50]])
        assert cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_GRAYSCALE) is None


class TestTheExportWritesAnImage:
    def test_the_bytes_are_a_png_that_decodes(self):
        """It used to copy the JSON through byte for byte into `masks/{frame}.png`.

        The manifest declared a ternary pixel encoding the whole time, and the existing test only asserted
        that the writer was registered, so nothing ever opened the file.
        """
        from services.export.adapter_scene import DRIVABLE_VALUES, _rasterise_drivable

        blob = {"classes": {"drivable": [[0, 0, 50, 0, 50, 100, 0, 100]],
                            "non_drivable": [], "fallback": []},
                "width": 100, "height": 100}
        png = _rasterise_drivable(blob)
        assert png[:4] == b"\x89PNG"
        back = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
        assert back.shape == (100, 100)
        # Half the frame painted drivable, and the value is the one the manifest declares.
        assert abs(float((back == DRIVABLE_VALUES["drivable"]).sum()) / back.size - 0.5) < 0.02

    def test_overlap_resolves_toward_road(self):
        """A pixel some class calls road must not export as kerb because another polygon covered it."""
        from services.export.adapter_scene import DRIVABLE_VALUES, _rasterise_drivable

        whole = [0, 0, 100, 0, 100, 100, 0, 100]
        png = _rasterise_drivable({"classes": {"drivable": [whole], "non_drivable": [whole],
                                               "fallback": [whole]},
                                   "width": 100, "height": 100})
        back = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
        assert (back == DRIVABLE_VALUES["drivable"]).all()

    def test_an_unusable_blob_is_none_rather_than_a_blank_image(self):
        # A blank PNG would export as "this frame has no drivable area", which is a claim.
        from services.export.adapter_scene import _rasterise_drivable

        assert _rasterise_drivable({"classes": {}, "width": 0, "height": 0}) is None
        assert _rasterise_drivable({"width": 10, "height": 10}) is None


class TestTheBackfill:
    pytestmark = pytest.mark.db

    async def _frames(self, db, n: int, *, with_mask: int = 0):
        onto = get_ontology()
        sess = DbSession(session_id=uuid.uuid4(), vehicle_id="DRV", start_ts_ns=0, end_ts_ns=1,
                         ontology_version=onto.version)
        db.add(sess)
        await db.flush()
        ids = []
        for i in range(n):
            f = Frame(frame_id=uuid.uuid4(), session_id=sess.session_id, ts_ns=now_ns() + i,
                      cam_id="front", width=64, height=64, img_uri=f"s3://x/{i}.jpg")
            db.add(f)
            await db.flush()
            ids.append(f.frame_id)
            if i < with_mask:
                db.add(DrivableMask(frame_id=f.frame_id, mask_uri="s3://x/m.json",
                                    coverage={"drivable": 0.0, "non_drivable": 0.0, "fallback": 0.0},
                                    source="proposed", model_version="prior"))
        await db.commit()
        return sess, ids

    async def test_coverage_breaks_down_by_capture_size(self):
        """The headline hides the shape: the buckets that look finished are small older imports."""
        from services.perception.backfill import coverage

        async with get_sessionmaker()() as db:
            await self._frames(db, 4, with_mask=1)
            cov = await coverage(db)
            assert cov["frames"] >= 4 and cov["covered"] >= 1
            assert any(b["dims"] == "64x64" for b in cov["by_size"])

    async def test_a_zero_coverage_row_counts_as_done_not_as_unstarted(self):
        """"Segmented, and there is no road here" is a finished frame.

        Treating it as pending would make the backfill re-segment every empty frame on every run, and
        coverage could never reach 100% because the frames it had checked would keep coming back.
        """
        from services.perception.backfill import _pending

        async with get_sessionmaker()() as db:
            sess, ids = await self._frames(db, 3, with_mask=2)
            pending = await _pending(db, session_id=str(sess.session_id), limit=100, redo=False)
            pending_ids = {f.frame_id for f in pending}
            assert ids[0] not in pending_ids and ids[1] not in pending_ids
            assert ids[2] in pending_ids

    async def test_redo_reconsiders_frames_that_already_have_one(self):
        # Scoped to this test's own session: the suite truncates once per session by design, so a global
        # query here would pick up every other test's frames and assert about them instead.
        from services.perception.backfill import _pending

        async with get_sessionmaker()() as db:
            sess, _ids = await self._frames(db, 3, with_mask=3)
            sid = str(sess.session_id)
            assert await _pending(db, session_id=sid, limit=100, redo=False) == []
            assert len(await _pending(db, session_id=sid, limit=100, redo=True)) == 3

    async def test_an_unreadable_frame_is_counted_and_left_without_a_row(self):
        """No row, so it stays visible as work still to do.

        Writing a zero-coverage mask for it would say "checked, no road here" about a frame nothing could
        read, which is the one thing the empty-result convention must not be used for.
        """
        from services.perception.backfill import run_drivable_backfill

        async with get_sessionmaker()() as db:
            sess, ids = await self._frames(db, 2)
            run = AgentRun(kind="drivable_backfill", status="running", scope={}, policy={})
            db.add(run)
            await db.commit()
            rid = run.run_id

        # s3://x/*.jpg does not exist in the object store, so every frame fails to load.
        await run_drivable_backfill(rid, session_id=str(sess.session_id), max_frames=10)

        async with get_sessionmaker()() as db:
            r = await db.get(AgentRun, rid)
            assert r.counts["unreadable"] == 2
            assert r.counts["masks"] == 0
            for fid in ids:
                assert await db.get(DrivableMask, fid) is None

    async def test_it_resumes_from_its_cursor_rather_than_restarting(self):
        from services.agent.resume import beat
        from services.perception.backfill import run_drivable_backfill

        async with get_sessionmaker()() as db:
            sess, ids = await self._frames(db, 3)
            run = AgentRun(kind="drivable_backfill", status="running", scope={}, policy={})
            db.add(run)
            await db.commit()
            rid = run.run_id
            # Two frames already finished by an earlier, interrupted pass.
            await beat(db, rid, progress={"done": [str(ids[0]), str(ids[1])], "total": 3},
                       counts={"frames": 2, "masks": 2, "empty": 0, "unreadable": 0, "refused": 0})

        await run_drivable_backfill(rid, session_id=str(sess.session_id), max_frames=10)

        async with get_sessionmaker()() as db:
            r = await db.get(AgentRun, rid)
            # Three seen in total, not five: the first two were not re-read.
            assert r.counts["frames"] == 3
