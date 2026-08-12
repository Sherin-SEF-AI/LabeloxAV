"""A jobs page where every import row read the same thing.

The unified jobs list labels each import with its format, so `video`. That is fine for one import and
useless the moment there is more than one: a folder of 186 clips produced 186 rows all reading "video",
identical in the one column meant to tell them apart, on the page whose entire purpose is telling one run
from another.

The filename was already in `source_uri` the whole time.
"""

from __future__ import annotations

from services.api.routers.jobs import _import_label


def test_an_import_is_named_after_its_file():
    assert _import_label(
        "s3://labeloxav/uploads/f0864d73-2e0e-4b82-9f2d-b4c62b6b6dd4/20260531152309_043703F.MP4",
        "video") == "20260531152309_043703F.MP4"


def test_two_clips_from_one_batch_are_distinguishable():
    """The actual complaint: the column exists to separate rows and separated nothing."""
    a = _import_label("s3://b/u/1/drive_01.mp4", "video")
    b = _import_label("s3://b/u/2/drive_02.mp4", "video")
    assert a != b


def test_the_format_is_still_the_fallback():
    """An import with no file behind it (a dataset pull, a re-run) keeps the old label rather than a blank."""
    assert _import_label(None, "video") == "video"
    assert _import_label("", "coco") == "coco"


def test_a_trailing_slash_does_not_produce_an_empty_label():
    assert _import_label("s3://bucket/folder/", "images") == "folder"


def test_a_uri_with_no_path_falls_back_rather_than_returning_a_scheme():
    assert _import_label("s3://", "mcap") == "mcap"


def test_nothing_at_all_still_names_the_row():
    # A blank cell in a list is worse than a generic word: it reads as a rendering failure.
    assert _import_label(None, None) == "import"
