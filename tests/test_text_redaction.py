"""Redact then verify: blurring a plate and checking afterwards that it is gone.

Both OCR paths in this tree take their regions from existing annotations, so there was no text-region
detector anywhere in it. That is a gap on the privacy plane rather than the reading one: a licence plate
the plate detector missed is still a plate, still legible, and the release attestation says the frame was
redacted.

The tests below use a stub detector so what is under test is the redact-verify-escalate-refuse loop rather
than OpenCV's model. The important cases are the last two: a region that survives escalation must block
the frame, and an absent detector must refuse rather than pass.
"""

from __future__ import annotations

import numpy as np

from services.anonymize.text_regions import (
    RedactionOutcome,
    TextDetector,
    redact_and_verify,
)


class _Stub(TextDetector):
    """A detector that returns a scripted sequence of results, one per detect() call.

    Scripted rather than random because the loop calls detect up to three times and each call means
    something different: what was there, what survived the first blur, what survived escalation.
    """

    def __init__(self, script: list[list[tuple[float, float, float, float, float]]]):
        self.script = list(script)
        self.calls = 0
        self._net = object()          # pretend the model loaded
        self._error = None

    @property
    def available(self) -> bool:
        return True

    def detect(self, image_bgr):
        out = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return out


def _img():
    return np.full((400, 600, 3), 128, dtype=np.uint8)


_PLATE = (100.0, 200.0, 180.0, 230.0)
_VEHICLE = [(50.0, 120.0, 300.0, 300.0)]


class TestTheVehiclePrior:
    def test_text_outside_a_vehicle_is_left_alone(self):
        """An Indian street scene is full of shop signs, hoardings and bus boards.

        None of it is personal data, and redacting it destroys the frame's value while protecting nobody.
        """
        far = (500.0, 20.0, 580.0, 60.0)
        det = _Stub([[(*far, 0.9)], []])
        out = redact_and_verify(_img(), _VEHICLE, detector=det)
        assert out.n_redacted == 0
        assert out.released is True
        assert out.detected[0].in_vehicle is False

    def test_no_vehicle_boxes_is_a_documented_refusal_not_a_whole_frame_blur(self):
        det = _Stub([[(*_PLATE, 0.9)]])
        out = redact_and_verify(_img(), [], detector=det)
        assert out.n_redacted == 0 and out.released is True
        assert "no vehicle detections" in out.reason
        assert det.calls == 0, "it must not even look when there is no prior to constrain it"

    def test_text_inside_a_vehicle_is_redacted(self):
        det = _Stub([[(*_PLATE, 0.9)], []])
        out = redact_and_verify(_img(), _VEHICLE, detector=det)
        assert out.n_redacted == 1 and out.n_surviving == 0 and out.released is True


class TestTheVerifyLoop:
    def test_it_looks_again_after_blurring(self):
        """The whole reason this exists. A kernel sized for a large plate leaves a small one legible.

        Nothing in the previous design ever checked, so the failure was silent and the attestation said
        the frame was clean.
        """
        det = _Stub([[(*_PLATE, 0.9)], []])
        redact_and_verify(_img(), _VEHICLE, detector=det)
        assert det.calls >= 2, "it blurred and never looked again"

    def test_a_survivor_is_escalated_once(self):
        # Still there after the first blur, gone after the stronger one.
        det = _Stub([[(*_PLATE, 0.9)], [(*_PLATE, 0.8)], []])
        out = redact_and_verify(_img(), _VEHICLE, detector=det)
        assert out.n_escalated == 1
        assert out.n_surviving == 0 and out.released is True
        assert det.calls == 3

    def test_a_region_that_survives_escalation_blocks_the_frame(self):
        """Not a warning. A frame whose text could not be destroyed is a frame that leaks.

        There is no third kernel: a region that survives both is a failure to record, not a kernel to
        grow, and the only safe handling is refusing to release it.
        """
        det = _Stub([[(*_PLATE, 0.9)], [(*_PLATE, 0.8)], [(*_PLATE, 0.75)]])
        out = redact_and_verify(_img(), _VEHICLE, detector=det)
        assert out.n_escalated == 1
        assert out.n_surviving == 1
        assert out.released is False
        assert "blocked from storage and serving" in out.reason

    def test_the_image_is_actually_modified(self):
        """A loop that reports success without touching pixels is the worst possible outcome here."""
        img = _img()
        before = img.copy()
        det = _Stub([[(*_PLATE, 0.9)], []])
        redact_and_verify(img, _VEHICLE, detector=det)
        # A uniform image blurs to itself, so give the region something to destroy first.
        img2 = _img()
        img2[200:230, 100:180] = np.random.default_rng(1).integers(0, 255, (30, 80, 3), dtype=np.uint8)
        before2 = img2.copy()
        redact_and_verify(img2, _VEHICLE, detector=_Stub([[(*_PLATE, 0.9)], []]))
        assert not np.array_equal(img2[200:230, 100:180], before2[200:230, 100:180])
        assert np.array_equal(img[:100], before[:100]), "it touched pixels outside the region"


class TestAnAbsentDetector:
    def test_missing_weights_refuse_the_frame_rather_than_passing_it(self):
        """A privacy detector that quietly does nothing is worse than one that is not installed.

        The pipeline reports success either way, so the absence has to be loud.
        """
        det = TextDetector("")
        assert det.available is False
        out = redact_and_verify(_img(), _VEHICLE, detector=det)
        assert out.released is False
        assert out.detector_available is False
        assert "run `make pii-models`" in out.reason

    def test_a_nonexistent_path_is_the_same_refusal(self):
        det = TextDetector("/nonexistent/db_text.onnx")
        assert det.available is False and det.error


class TestItIsRedactionNotReading:
    def test_the_result_carries_no_characters_anywhere(self):
        """Reading a plate creates a record that has to be governed; blurring one removes the need.

        services/anpr/ refuses by capability gate and must keep refusing, and nothing on this path may
        become a way around it.
        """
        det = _Stub([[(*_PLATE, 0.9)], []])
        out = redact_and_verify(_img(), _VEHICLE, detector=det)
        assert isinstance(out, RedactionOutcome)
        for r in out.detected:
            assert set(r.__dataclass_fields__) == {"bbox", "score", "in_vehicle"}

    def test_the_module_never_imports_anpr(self):
        from pathlib import Path

        src = Path("services/anonymize/text_regions.py").read_text(encoding="utf-8")
        assert "services.anpr" not in src.replace("services/anpr/", "")


def test_plate_redaction_is_mandatory_on_a_real_deployment():
    """One environment variable could make every release attestation false.

    The proof asserts each frame passed the PII gate; with plate_mandatory off the gate passes frames it
    never redacted. Checked in config rather than at the call site, for the same reason the auth flags
    are: a control that only fails when somebody exercises the path is not a control until they do.
    """
    import pytest

    from core.config import Settings

    with pytest.raises(ValueError) as exc:
        Settings(env="production", pii={"plate_mandatory": False},
                 auth={"enabled": True, "accept_legacy_tokens": False},
                 postgres={"host": "db.internal", "password": "x" * 24},
                 minio={"secret_key": "y" * 24, "access_key": "z" * 12})
    assert "plate redaction is not mandatory" in str(exc.value)
