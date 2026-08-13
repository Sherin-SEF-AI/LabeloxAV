"""Plates were not redacted because the detector never saw them at a size it could read.

A frame from the operational corpus carries three legible Indian registrations. The gate recorded
`n_plates: 0` for it, and re-running the same model reproduced that: zero detections at conf 0.35, 0.20,
0.10 and 0.05. The obvious conclusion was that the model cannot read Indian plates, and the obvious remedy
was to find a better one. Both were wrong.

`services/anonymize/detectors.py` called `predict` without `imgsz`, so ultralytics used its default of 640.
On a 1920x1080 dashcam frame that is a 3x downsample, and a plate roughly 150x40 pixels arrives at the model
as 50x13. At native resolution the same nano model finds two of them, scoring 0.73 and 0.37. The larger
models in that family, downloaded and tested, behave identically at every resolution, which is what rules
capacity out.

The face detector beside it has always used the frame's own dimensions (`setInputSize`), which is why faces
were redacted throughout and plates were not.

Scale, measured over 255 frames the gate recorded as plate-free: at the gate's own threshold the detector
now finds a plate on 51.4% of them, against 2.4% before.
"""

from __future__ import annotations

import numpy as np
import pytest

from services.anonymize.detectors import PlateDetector, _detect_size


class TestTheInferenceSize:
    def test_a_dashcam_frame_is_read_at_its_own_resolution(self):
        assert _detect_size(1080, 1920, 1920) == 1920

    def test_it_is_never_the_library_default_for_a_full_hd_frame(self):
        # 640 is the setting that made a 150x40 plate into 50x13 and hid every small plate in the corpus.
        assert _detect_size(1080, 1920, 1920) != 640

    def test_a_smaller_source_is_not_upscaled_for_nothing(self):
        assert _detect_size(720, 1280, 1920) == 1280

    def test_a_4k_frame_is_capped_rather_than_run_at_full_size(self):
        """The cap is memory, not policy: a 3840-wide inference will not fit alongside the rest of the
        pipeline on this box."""
        assert _detect_size(2160, 3840, 1920) == 1920

    def test_a_small_frame_keeps_the_library_floor(self):
        # Below 640 there is nothing to gain and the model was trained around that scale.
        assert _detect_size(480, 640, 1920) == 640

    def test_the_size_is_always_a_multiple_of_the_stride(self):
        # A size off the stride is silently rounded by the library, so it is rounded here where it is visible.
        for h, w in ((1080, 1920), (721, 1281), (999, 1777), (100, 100)):
            assert _detect_size(h, w, 1920) % 32 == 0

    def test_the_cap_is_honoured_even_when_it_is_lowered(self):
        assert _detect_size(1080, 1920, 960) == 960

    def test_portrait_and_landscape_are_treated_the_same(self):
        """The long side decides, so a rotated camera is not quietly given a third of the resolution."""
        assert _detect_size(1920, 1080, 1920) == _detect_size(1080, 1920, 1920)


class TestTheDetectorUsesIt:
    def test_the_frame_size_reaches_the_model(self, monkeypatch):
        """The regression that matters: an `imgsz` that never leaves this function is the bug restored."""
        seen: dict = {}

        class _Model:
            def predict(self, img, **kw):
                seen.update(kw)
                return []

        det = PlateDetector.__new__(PlateDetector)
        det._model = _Model()
        det.conf, det.device, det.imgsz_cap = 0.35, "cpu", 1920
        det.detect(np.zeros((1080, 1920, 3), dtype=np.uint8))

        assert seen.get("imgsz") == 1920, "the detector fell back to the library default"
        assert seen.get("conf") == 0.35

    def test_a_missing_model_still_returns_nothing_rather_than_raising(self):
        det = PlateDetector.__new__(PlateDetector)
        det._model = None
        det.conf, det.device, det.imgsz_cap = 0.35, "cpu", 1920
        assert det.detect(np.zeros((10, 10, 3), dtype=np.uint8)) == []


class TestTheSettingIsReachable:
    def test_the_cap_is_configurable_rather_than_hardcoded(self):
        from core.config import get_settings

        assert get_settings().pii.plate_imgsz_cap >= 640

    def test_the_anonymizer_passes_the_configured_cap(self):
        """Wiring a setting nothing reads is the same defect in a different place."""
        import inspect

        from services.anonymize import anonymizer

        src = inspect.getsource(anonymizer)
        assert "plate_imgsz_cap" in src, "the anonymizer builds its detector without the cap"


@pytest.mark.skipif(
    not __import__("pathlib").Path(".scratch/models/pii/plate_yolov8.pt").exists(),
    reason="plate weights not fetched in this environment (make pii-models)")
class TestAgainstTheRealModel:
    def test_a_plate_sized_region_survives_the_resize(self):
        """Not a detection test, which would need a real frame. It pins the arithmetic that broke: at 640 a
        150x40 plate in a 1920-wide frame is under 14 pixels tall, and at native size it keeps its own."""
        plate_h, frame_w = 40, 1920
        assert plate_h * 640 / frame_w < 14
        assert plate_h * _detect_size(1080, frame_w, 1920) / frame_w == pytest.approx(plate_h)
