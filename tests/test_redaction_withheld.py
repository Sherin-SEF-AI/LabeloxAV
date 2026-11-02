"""A frame whose plate text survived both blur passes is never stored.

services/anonymize/text_regions.py has always said so ("blocked from storage and serving"), and returned
released=False to say it, but nothing read the flag: the frame was stored and served like any other. A
missing text detector is a different case: that frame was not verified, which the system checks report, but
withholding every frame for want of a model file would stop ingest outright.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from services.anonymize.anonymizer import PiiResult

pytestmark = pytest.mark.db


def test_only_text_that_survived_its_blur_withholds_a_frame():
    assert PiiResult(0, 0, [], "m", released=False, text_detector_available=True).withheld
    assert not PiiResult(0, 0, [], "m", released=False, text_detector_available=False).withheld
    assert not PiiResult(0, 0, [], "m", released=True).withheld


class _Alternating:
    """Withholds every other frame, the way a surviving plate would."""

    method_version = "test-anon"

    def __init__(self):
        self.n = 0

    def anonymize(self, img):
        self.n += 1
        return PiiResult(0, 0, [], self.method_version, released=self.n % 2 == 1,
                         redaction_reason=None if self.n % 2 else "plate text survived the blur",
                         text_detector_available=True)


async def test_ingest_stores_the_released_frames_and_none_of_the_withheld_ones(monkeypatch):
    from sqlalchemy import func, select

    from core.timebase import now_ns, seconds_to_ns
    from db.models import Frame
    from db.session import get_sessionmaker
    from services.ingest import run as ingest_run
    from services.ingest.quality import QualityResult
    from services.ingest.types import RawFrame

    monkeypatch.setattr(ingest_run, "score_frame",
                        lambda img, cfg: QualityResult(blur=100.0, mean_luma=120.0, clip_fraction=0.0,
                                                       score=0.9, accepted=True, reasons=[]))
    rng = np.random.default_rng(1)
    start = now_ns()
    frames = [RawFrame(ts_ns=start + seconds_to_ns(i), cam_id="cam_f",
                       image_bgr=rng.integers(30, 220, (240, 320, 3), dtype=np.uint8)) for i in range(4)]
    result = await ingest_run.ingest(frame_iter=iter(frames), vehicle="WITHHOLD", city="BLR", route="withheld-test",
                                     raw_uri=None, mcap_uri=None, source_streams=["cam_f"], anonymizer=_Alternating())
    sid = uuid.UUID(result["session_id"])
    async with get_sessionmaker()() as db:
        n = (await db.execute(select(func.count()).select_from(Frame).where(Frame.session_id == sid))).scalar_one()
    assert n == 2, f"{n} frames stored; the two withheld ones must not be"
