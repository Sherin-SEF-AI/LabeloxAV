"""The redaction re-check: looking again where the annotations say the people and vehicles are.

The whole-frame pass misses PII that is below the detector's floor at frame resolution. Measured on this
corpus, 82.4% of frames holding an annotated person have zero faces redacted, and a probe over 14
already-redacted frames found 0 faces whole-frame and 7 inside crops of the people in those same frames.

These cover the parts that decide whether the re-check is safe to run unattended: that it only ever adds
regions the frame does not already have, that running it twice adds nothing the second time, and that a
detection which is really just the crop it came from is thrown away rather than blurred forever.
"""

from __future__ import annotations

import numpy as np

from services.anonymize.recheck import (
    Candidate,
    _crop_window,
    _dedup,
    _detect_in_windows,
    _upscaled,
    new_regions,
)


class _Stub:
    """A detector that reports fixed boxes in whatever image it is handed."""

    available = True

    def __init__(self, boxes):
        self.boxes = boxes
        self.seen: list[tuple[int, int]] = []

    def detect(self, img):
        self.seen.append((img.shape[1], img.shape[0]))
        return list(self.boxes)


class _Anon:
    def __init__(self, face=None, plate=None):
        self.face = face or _Stub([])
        self.plate = plate or _Stub([])


_TARGETS = {"face": "face", "plate": "plate"}


class TestTheCropItLooksIn:
    def test_the_window_is_wider_than_the_annotation(self):
        """A face sits at the top of a person's box and a plate straddles a tightly drawn vehicle box, so a
        crop at the annotation's exact bounds cuts off the thing being looked for."""
        window = _crop_window([100, 100, 200, 300], 1920, 1080)
        assert window is not None
        x1, y1, x2, y2 = window
        assert x1 < 100 and y1 < 100 and x2 > 200 and y2 > 300

    def test_the_window_never_leaves_the_image(self):
        x1, y1, x2, y2 = _crop_window([0, 0, 100, 100], 640, 480)
        assert (x1, y1) == (0, 0)
        assert x2 <= 640 and y2 <= 480

    def test_a_speck_is_not_worth_reading(self):
        """Below a handful of pixels there is nothing for an upscale to recover, and what comes back is
        noise. Noise here becomes permanent blur."""
        assert _crop_window([10, 10, 14, 14], 1920, 1080) is None

    def test_a_small_crop_is_enlarged_past_the_detectors_floor(self):
        """This is the entire difference between the second look and the first. Left at its own size, the
        crop is exactly the pixels the whole-frame pass already failed on."""
        small = np.zeros((40, 30, 3), dtype=np.uint8)
        out, scale = _upscaled(small)
        assert scale > 1.0
        assert min(out.shape[0], out.shape[1]) >= 320

    def test_a_large_crop_is_left_alone(self):
        big = np.zeros((600, 800, 3), dtype=np.uint8)
        out, scale = _upscaled(big)
        assert scale == 1.0 and out.shape == big.shape


class TestMappingBackToTheFrame:
    def test_a_detection_in_a_crop_lands_where_it_really_is(self):
        """The blur is applied to the full frame, so a coordinate left in crop space would blur an unrelated
        part of the image and leave the face untouched."""
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # A 200x200 window at (500, 400). The stub reports a box at the crop's own origin.
        face = _Stub([(0.0, 0.0, 20.0, 20.0, 0.9)])
        found = _detect_in_windows(_Anon(face=face), img, [((500, 400, 700, 600), frozenset({"face"}))],
                                   _TARGETS)
        assert len(found) == 1
        x1, y1, _x2, _y2 = found[0].bbox
        assert x1 == 500 and y1 == 400

    def test_the_upscale_is_undone(self):
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        window = (100, 100, 180, 180)          # 80x80, upscaled by 4 to reach the 320 floor
        face = _Stub([(0.0, 0.0, 40.0, 40.0, 0.9)])
        found = _detect_in_windows(_Anon(face=face), img, [(window, frozenset({"face"}))], _TARGETS)
        assert face.seen == [(320, 320)], "the crop was not enlarged before detection"
        # 40 px in the 4x image is 10 px in the frame, not 40.
        assert found[0].bbox == [100.0, 100.0, 110.0, 110.0]

    def test_a_detection_the_size_of_its_own_crop_is_discarded(self):
        """A 'face' filling the whole person box is the detector agreeing with the crop rather than finding
        something in it. A blur cannot be undone, so this one is thrown away."""
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        window = (500, 400, 900, 800)          # 400x400, already past the upscale floor
        face = _Stub([(0.0, 0.0, 400.0, 400.0, 0.9)])
        assert _detect_in_windows(_Anon(face=face), img, [(window, frozenset({"face"}))], _TARGETS) == []

        # A face of a plausible size in the same crop is kept, so the guard is a size filter and not a
        # blanket refusal of anything found inside a crop.
        real = _Stub([(10.0, 10.0, 90.0, 90.0, 0.9)])
        assert len(_detect_in_windows(_Anon(face=real), img, [(window, frozenset({"face"}))], _TARGETS)) == 1

    def test_a_low_confidence_face_in_a_crop_is_not_blurred(self):
        """Sampled at the configured 0.50, a third of the face candidates from crops were an autorickshaw
        body, grass, a taillight and foliage. At 0.70 they were faces. The crop is interpolated pixels the
        detector was not calibrated on, and this blur is permanent."""
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        window = (500, 400, 900, 800)
        weak = _Stub([(10.0, 10.0, 90.0, 90.0, 0.55)])
        strong = _Stub([(10.0, 10.0, 90.0, 90.0, 0.85)])
        assert _detect_in_windows(_Anon(face=weak), img, [(window, frozenset({"face"}))], _TARGETS) == []
        assert len(_detect_in_windows(_Anon(face=strong), img, [(window, frozenset({"face"}))], _TARGETS)) == 1

    def test_a_plate_keeps_its_configured_floor(self):
        """Sampled the same way, plate boxes sat on real registration plates from 0.36 upward, one of them
        legible on a frame whose whole-frame pass found no plate at all. Holding plates to the face's bar
        would discard the half of this the corpus most needs."""
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        plate = _Stub([(10.0, 10.0, 90.0, 40.0, 0.40)])
        found = _detect_in_windows(_Anon(plate=plate), img, [((500, 400, 900, 800), frozenset({"plate"}))],
                                   _TARGETS)
        assert len(found) == 1

    def test_an_unavailable_detector_contributes_nothing(self):
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        plate = _Stub([(0.0, 0.0, 20.0, 20.0, 0.9)])
        plate.available = False
        found = _detect_in_windows(_Anon(plate=plate), img, [((0, 0, 200, 200), frozenset({"plate"}))],
                                   _TARGETS)
        assert found == [] and plate.seen == []


class TestOnlyWhatIsNew:
    def test_a_region_already_blurred_is_not_blurred_again(self):
        """Re-blurring an already blurred region darkens it a little further on every pass, and the audit
        would claim a face was found that was found the first time."""
        existing = [{"type": "face", "bbox": [100, 100, 140, 140], "score": 0.9}]
        c = Candidate(type="face", bbox=[102, 101, 141, 139], score=0.8, where="roi")
        assert new_regions([c], existing) == []

    def test_a_genuinely_new_region_survives(self):
        existing = [{"type": "face", "bbox": [100, 100, 140, 140], "score": 0.9}]
        c = Candidate(type="face", bbox=[900, 500, 940, 540], score=0.8, where="roi")
        assert new_regions([c], existing) == [c]

    def test_a_plate_is_not_covered_by_a_face_in_the_same_place(self):
        """The two are different targets and the audit records them separately, so an overlap between them
        says nothing about whether the plate was blurred."""
        existing = [{"type": "face", "bbox": [100, 100, 140, 140], "score": 0.9}]
        c = Candidate(type="plate", bbox=[100, 100, 140, 140], score=0.8, where="roi")
        assert new_regions([c], existing) == [c]

    def test_an_audit_with_no_regions_covers_nothing(self):
        c = Candidate(type="face", bbox=[10, 10, 20, 20], score=0.8, where="roi")
        assert new_regions([c], []) == [c]

    def test_a_malformed_stored_region_does_not_swallow_a_finding(self):
        """Some audits predate the region shape. A row that cannot be read is not evidence of coverage."""
        existing = [{"type": "face", "bbox": []}, {"type": "face"}]
        c = Candidate(type="face", bbox=[10, 10, 20, 20], score=0.8, where="roi")
        assert new_regions([c], existing) == [c]

    def test_running_it_twice_finds_nothing_the_second_time(self):
        """Idempotence is what makes this safe to schedule: the first run's additions are in the audit, so
        the second matches them here."""
        first = [Candidate(type="face", bbox=[10, 10, 50, 50], score=0.9, where="roi")]
        stored = [{"type": c.type, "bbox": c.bbox, "score": c.score} for c in first]
        assert new_regions(first, stored) == []


class TestTheSameRegionSeenTwice:
    def test_overlapping_windows_report_one_face_once(self):
        """A pedestrian beside a motorcycle falls inside two crops, and the audit is what a reviewer reads."""
        a = Candidate(type="face", bbox=[100, 100, 140, 140], score=0.7, where="roi")
        b = Candidate(type="face", bbox=[101, 99, 139, 141], score=0.9, where="roi")
        out = _dedup([a, b])
        assert len(out) == 1
        assert out[0].score == 0.9, "the weaker duplicate was kept over the stronger one"

    def test_two_real_faces_both_survive(self):
        a = Candidate(type="face", bbox=[100, 100, 140, 140], score=0.9, where="roi")
        b = Candidate(type="face", bbox=[800, 300, 840, 340], score=0.8, where="roi")
        assert len(_dedup([a, b])) == 2

    def test_the_whole_frame_and_crop_passes_are_unioned_not_replaced(self):
        """Measured on 14 frames: whole-frame found 4 plates and the crops found 3, and they were not the
        same 3. Either pass alone loses regions the other one caught."""
        whole = Candidate(type="plate", bbox=[10, 10, 60, 30], score=0.6, where="frame")
        roi = Candidate(type="plate", bbox=[900, 500, 950, 520], score=0.5, where="roi")
        out = _dedup([whole, roi])
        assert {c.where for c in out} == {"frame", "roi"}
