"""Settings > About: the running version, where its source is, and the licences that reach it.

The AGPL components (docs/LICENSING.md) require that people using a modified deployment over a network can get
its source, so the link is part of the product: an unmodified deployment links the exact commit it runs, and a
modified one links wherever its administrators publish their changes.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.db


def _about(monkeypatch, **env) -> dict:
    from _authutil import auth_headers

    from core.config import get_settings
    from services.api.main import app

    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("LBX_CODE_SHA", "0123456789abcdef0123456789abcdef01234567")
    get_settings.cache_clear()
    from core import version

    version.code_sha.cache_clear()
    try:
        with TestClient(app) as c:
            r = c.get("/api/system/about", headers=auth_headers("annotator"))
            assert r.status_code == 200, r.text
            return r.json()
    finally:
        get_settings.cache_clear()
        version.code_sha.cache_clear()


def test_an_unmodified_deployment_links_the_exact_commit(monkeypatch):
    out = _about(monkeypatch)
    assert out["commit"].startswith("0123456789abcdef")
    assert out["source_of_this_version"].endswith("/tree/0123456789abcdef0123456789abcdef01234567")
    names = {c["name"]: c["licence"] for c in out["copyleft"]}
    assert names["Ultralytics"] == "AGPL-3.0" and names["cleanlab"] == "AGPL-3.0"
    assert out["own_licence"] == "Apache-2.0" and out["combined_licence"] == "AGPL-3.0"


def test_a_modified_deployment_points_at_its_own_source(monkeypatch):
    out = _about(monkeypatch, LBX_ABOUT__SOURCE_URL="https://git.example.lab/av/labeloxav", LBX_ABOUT__MODIFIED="true")
    assert out["modified"] is True and out["source_url"] == "https://git.example.lab/av/labeloxav"
    assert out["source_of_this_version"] is None
