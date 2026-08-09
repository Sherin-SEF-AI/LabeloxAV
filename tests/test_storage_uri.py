"""An object URI with no key, and the batch job it killed.

Fifteen frames carry `img_uri = 's3://x.jpg'`. That parses to bucket "x.jpg" and an empty key, which boto3
rejects with:

  Parameter validation failed: Invalid length for parameter Key, value: 0, valid min length: 1

The message names neither the URI nor the frame, so it takes a while to trace back to a row. Worse, those
rows are unfetchable permanently, so a job that walks the corpus retries them on every pass. The corpus
relabel stopped at batch 30 with 3,707 frames still to go, because twenty of these in a row convinced its
consecutive-failure guard that something systemic had broken. The guard was right to stop; it was being fed
work that could never succeed.
"""

from __future__ import annotations

import pytest

from core.storage import ObjectStore, is_fetchable_uri


def test_a_uri_with_a_bucket_and_no_key_is_refused_by_name():
    """Better than boto3's "Invalid length for parameter Key, value: 0", which names nothing."""
    with pytest.raises(ValueError, match="no object key"):
        ObjectStore.parse_uri("s3://x.jpg")


def test_the_error_quotes_the_uri_that_caused_it():
    with pytest.raises(ValueError) as e:
        ObjectStore.parse_uri("s3://x.jpg")
    assert "s3://x.jpg" in str(e.value)


def test_a_bucket_with_a_trailing_slash_has_no_key_either():
    with pytest.raises(ValueError, match="no object key"):
        ObjectStore.parse_uri("s3://labeloxav/")


def test_a_normal_uri_still_parses():
    assert ObjectStore.parse_uri("s3://labeloxav/frames/a/b.jpg") == ("labeloxav", "frames/a/b.jpg")


def test_a_non_s3_uri_is_still_rejected_the_way_it_was():
    with pytest.raises(ValueError, match="not an s3 uri"):
        ObjectStore.parse_uri("https://example.com/a.jpg")


@pytest.mark.parametrize("uri,ok", [
    ("s3://labeloxav/frames/a.jpg", True),
    ("s3://x.jpg", False),
    ("s3://labeloxav/", False),
    ("s3://", False),
    ("https://example.com/a.jpg", False),
    ("", False),
    (None, False),
])
def test_fetchability_agrees_with_what_the_parser_will_accept(uri, ok):
    """The batch jobs filter on this before doing work, so it must not disagree with the parser: a URI this
    calls fetchable and the parser then rejects is the original bug with an extra step."""
    assert is_fetchable_uri(uri) is ok
