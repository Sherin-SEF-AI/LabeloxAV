"""The operational endpoints do not cost more, or do more, than their callers are allowed.

/metrics ran a GROUP BY over the whole object table on every scrape, for anyone who could reach the port.
/api/health, an unauthenticated probe, called ensure_bucket(), which creates the bucket when it is absent.
The dataset artifact route confined itself with a substring test.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from core.config import get_settings
from services.api import metrics as metrics_mod
from services.api.routers.dataset_query import _is_dataset_artifact

# The health and metrics routes open a database session.
pytestmark = pytest.mark.db


@pytest.fixture(autouse=True)
def _fresh_cache():
    metrics_mod._gauge_cache = None
    yield
    metrics_mod._gauge_cache = None


@pytest.mark.asyncio
async def test_corpus_gauges_are_sampled_once_per_ttl(monkeypatch):
    calls = {"n": 0}

    async def fake():
        calls["n"] += 1
        return {"lbx_review_queue_depth": float(calls["n"])}

    monkeypatch.setattr(metrics_mod, "_sample_db_gauges", fake)
    monkeypatch.setattr(get_settings().metrics, "db_gauge_ttl_s", 60.0)
    a = await metrics_mod.sample_db_gauges()
    b = await metrics_mod.sample_db_gauges()
    assert calls["n"] == 1 and a == b == {"lbx_review_queue_depth": 1.0}


@pytest.mark.asyncio
async def test_concurrent_scrapes_share_one_sample(monkeypatch):
    import asyncio

    calls = {"n": 0}

    async def slow():
        calls["n"] += 1
        await asyncio.sleep(0.05)
        return {"x": 1.0}

    monkeypatch.setattr(metrics_mod, "_sample_db_gauges", slow)
    monkeypatch.setattr(get_settings().metrics, "db_gauge_ttl_s", 60.0)
    await asyncio.gather(*(metrics_mod.sample_db_gauges() for _ in range(8)))
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_a_zero_ttl_samples_every_time(monkeypatch):
    calls = {"n": 0}

    async def fake():
        calls["n"] += 1
        return {}

    monkeypatch.setattr(metrics_mod, "_sample_db_gauges", fake)
    monkeypatch.setattr(get_settings().metrics, "db_gauge_ttl_s", 0.0)
    await metrics_mod.sample_db_gauges()
    await metrics_mod.sample_db_gauges()
    assert calls["n"] == 2


def test_metrics_demand_the_token_once_one_is_set(monkeypatch):
    from services.api.main import app

    async def fake():
        return {}

    monkeypatch.setattr(metrics_mod, "_sample_db_gauges", fake)
    c = TestClient(app)
    monkeypatch.setattr(get_settings().metrics, "token", "")
    assert c.get("/metrics").status_code == 200
    monkeypatch.setattr(get_settings().metrics, "token", "s3cret-scrape")
    assert c.get("/metrics").status_code == 401
    assert c.get("/metrics", headers={"Authorization": "Bearer wrong"}).status_code == 401
    ok = c.get("/metrics", headers={"Authorization": "Bearer s3cret-scrape"})
    assert ok.status_code == 200 and "lbx_http_requests_total" in ok.text


def test_the_health_probe_never_creates_the_bucket(monkeypatch):
    from core import storage
    from services.api.main import app

    store = storage.get_object_store()
    created = []
    monkeypatch.setattr(store, "ensure_bucket", lambda: created.append(1))
    monkeypatch.setattr(store._client, "create_bucket", lambda **k: created.append(2))
    r = TestClient(app).get("/api/health")
    assert r.status_code == 200 and "minio" in r.json()["deps"]
    assert created == [], "a liveness probe must not write"


@pytest.mark.parametrize("uri,ok", [
    ("datasets/demo/shard-00000.tar", True),
    ("datasets/demo/index.parquet", True),
    ("frames/abc/datasets/x.jpg", False),                 # contains the word, is not under the prefix
    ("datasets/../frames/abc/cam/1.jpg", False),          # parent-directory escape
    ("datasets//x", False),
    ("http://evil.example/datasets/x", False),
    ("s3://some-other-bucket/datasets/demo/shard-00000.tar", False),
    ("masks/drivable/s/f.json", False),
])
def test_only_objects_under_the_datasets_prefix_are_artifacts(uri, ok):
    assert _is_dataset_artifact(uri) is ok


def test_an_artifact_uri_in_our_own_bucket_is_accepted():
    from core.storage import get_object_store

    store = get_object_store()
    assert _is_dataset_artifact(store.uri("datasets/demo/shard-00000.tar"))
    assert not _is_dataset_artifact(store.uri("frames/s/cam/1.jpg"))
