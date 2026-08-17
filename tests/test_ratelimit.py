"""The API had one 429 in it, on login, and no budget anywhere else.

The media routes are the ones that matter: frames of Indian roads under DPDPA, two of which take no user
dependency at all, limited only by how fast the disk can read. Presigned-URL minting is worse in kind, since
each call hands out a credential that outlives the request.

A fixed window would have been easier and wrong: it lets a caller spend a whole window's budget in its last
second and the next window's in its first, which is twice the advertised rate at exactly the moment a
scraper is going fastest. These tests pin the bucket behaviour that avoids that, and the eviction rule that
keeps a limiter meant to bound memory from becoming the thing that exhausts it.
"""

from __future__ import annotations

import pytest

from services.api.ratelimit import (
    ANONYMOUS,
    DEFAULT,
    MEDIA,
    PRESIGN,
    Budget,
    MemoryLimiter,
    classify,
    retry_after_seconds,
)


class TestTheBucket:
    def test_a_burst_is_allowed_and_then_it_is_not(self):
        """An editor opening a frame pulls its crops at once. A loop pulling the same route does not stop."""
        lim = MemoryLimiter()
        b = Budget(rate_per_s=1.0, burst=5.0)
        for i in range(5):
            allowed, _ = lim.check("caller", "media", b, 1.0, now=100.0)
            assert allowed, f"burst request {i} was refused"
        allowed, wait = lim.check("caller", "media", b, 1.0, now=100.0)
        assert not allowed
        assert wait > 0

    def test_it_refills_continuously_rather_than_in_windows(self):
        """The whole reason for a bucket: a window lets a caller spend two budgets back to back across the
        boundary, at double the advertised rate."""
        lim = MemoryLimiter()
        b = Budget(rate_per_s=2.0, burst=2.0)
        assert lim.check("c", "api", b, 2.0, now=0.0)[0]
        assert not lim.check("c", "api", b, 1.0, now=0.0)[0]
        # Half a second later exactly one token exists, not a whole fresh window.
        assert lim.check("c", "api", b, 1.0, now=0.5)[0]
        assert not lim.check("c", "api", b, 1.0, now=0.5)[0]

    def test_it_never_refills_past_its_burst(self):
        """An idle caller banks a burst, not a day's worth of requests."""
        lim = MemoryLimiter()
        b = Budget(rate_per_s=10.0, burst=5.0)
        lim.check("c", "api", b, 5.0, now=0.0)
        for _ in range(5):
            assert lim.check("c", "api", b, 1.0, now=1000.0)[0]
        assert not lim.check("c", "api", b, 1.0, now=1000.0)[0]

    def test_the_wait_it_reports_is_long_enough_to_be_worth_taking(self):
        lim = MemoryLimiter()
        b = Budget(rate_per_s=2.0, burst=1.0)
        lim.check("c", "api", b, 1.0, now=0.0)
        allowed, wait = lim.check("c", "api", b, 1.0, now=0.0)
        assert not allowed
        assert wait == pytest.approx(0.5, abs=0.01)
        # Retry-After: 0 invites an immediate retry, which is the opposite of what a limit is for.
        assert retry_after_seconds(wait) >= 1

    def test_callers_do_not_share_a_bucket(self):
        lim = MemoryLimiter()
        b = Budget(rate_per_s=1.0, burst=1.0)
        assert lim.check("asha", "api", b, 1.0, now=0.0)[0]
        assert lim.check("ravi", "api", b, 1.0, now=0.0)[0], "one caller exhausted another's budget"

    def test_route_classes_do_not_share_a_bucket(self):
        """Pulling media must not exhaust the budget for the rest of the application, or a throttled
        scraper takes the editor down with it."""
        lim = MemoryLimiter()
        b = Budget(rate_per_s=1.0, burst=1.0)
        assert lim.check("c", "media", b, 1.0, now=0.0)[0]
        assert lim.check("c", "api", b, 1.0, now=0.0)[0]

    def test_a_budget_that_allows_nothing_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="allow something"):
            Budget(rate_per_s=0.0, burst=10.0)


class TestMemoryBounds:
    def test_it_does_not_grow_without_limit(self):
        """A limiter keyed by client address, unbounded, is a memory leak with a rate limit written on it."""
        lim = MemoryLimiter(max_buckets=100)
        b = Budget(rate_per_s=1000.0, burst=1000.0)
        for i in range(500):
            lim.check(f"caller-{i}", "api", b, 1.0, now=float(i))
        assert len(lim.buckets) <= 100

    def test_evicting_a_full_bucket_costs_the_caller_nothing(self):
        """They come back to a fresh full bucket, which is what they would have had anyway."""
        lim = MemoryLimiter(max_buckets=2)
        b = Budget(rate_per_s=100.0, burst=10.0)
        lim.check("a", "api", b, 1.0, now=0.0)
        lim.check("b", "api", b, 1.0, now=0.0)
        lim.check("c", "api", b, 1.0, now=100.0)     # forces an eviction
        allowed, _ = lim.check("a", "api", b, 1.0, now=100.0)
        assert allowed

    def test_a_caller_mid_spend_is_not_evicted_in_favour_of_an_idle_one(self):
        lim = MemoryLimiter(max_buckets=2)
        b = Budget(rate_per_s=0.01, burst=5.0)
        lim.check("spender", "api", b, 5.0, now=0.0)      # empty bucket, actively limited
        lim.check("idle", "api", b, 0.0, now=0.0)         # full bucket
        lim.check("new", "api", b, 1.0, now=1.0)
        assert ("spender", "api") in lim.buckets, "the limited caller lost their limit by eviction"


class TestClassification:
    def test_a_frame_image_draws_on_the_media_budget(self):
        klass, budget, _cost = classify("/api/frames/abc/image", "GET")
        assert klass == "media" and budget is MEDIA

    def test_an_object_crop_does_too(self):
        # This route takes no user dependency at all, which is why the limit matters more here.
        klass, _b, _c = classify("/api/objects/abc/crop", "GET")
        assert klass == "media"

    def test_minting_a_presigned_url_is_its_own_tight_budget(self):
        """Each call hands out a credential that outlives the request."""
        klass, budget, _c = classify("/api/upload/presign-put", "POST")
        assert klass == "presign" and budget is PRESIGN
        assert budget.rate_per_s < MEDIA.rate_per_s

    def test_everything_else_is_the_ordinary_budget(self):
        klass, budget, _c = classify("/api/frames/abc/objects", "GET")
        assert klass == "api" and budget is DEFAULT

    def test_a_path_no_route_matches_is_still_classified(self):
        """The limiter runs before routing, and a request for a path that matches nothing is exactly what a
        scanner sends."""
        klass, _b, _c = classify("/api/../../etc/passwd", "GET")
        assert klass == "api"

    def test_an_anonymous_caller_gets_less_than_an_authenticated_one(self):
        assert ANONYMOUS.rate_per_s < DEFAULT.rate_per_s
        assert ANONYMOUS.burst < DEFAULT.burst
