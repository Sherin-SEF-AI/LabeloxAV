"""LabeloxSec is an optional package: off in a default install, on only when a deployment chooses it.

It reads registration plates and follows people between cameras. That is lawful for an authorised security
deployment and exactly what an AV research install must never do with dashcam footage. Capability gates
already refused those calls under the AV pack; this makes the whole pack absent unless enabled: not imported,
no routes, nothing in the web app. The suite runs with it enabled (tests/conftest.py) so its own tests run;
these check the default.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI

from core.config import PacksSettings, get_settings
from packs import registry

pytestmark = pytest.mark.db


@pytest.fixture
def default_install(monkeypatch):
    monkeypatch.setenv("LBX_PACKS__OPTIONAL", "[]")
    get_settings.cache_clear()
    registry.load_packs.cache_clear()
    yield
    get_settings.cache_clear()
    registry.load_packs.cache_clear()


@pytest.fixture
def security_install(monkeypatch):
    monkeypatch.setenv("LBX_PACKS__OPTIONAL", '["sec"]')
    get_settings.cache_clear()
    registry.load_packs.cache_clear()
    yield
    get_settings.cache_clear()
    registry.load_packs.cache_clear()


def test_it_is_off_unless_chosen():
    assert PacksSettings().optional == []


def test_the_security_pack_declares_itself_optional_and_the_av_pack_does_not():
    assert registry.optional_pack_ids() == ["sec"]


def test_a_default_install_does_not_load_it(default_install):
    assert registry.pack_ids() == ["av"]


def test_asking_for_it_says_how_to_enable_it(default_install):
    with pytest.raises(KeyError, match="optional package.*LBX_PACKS__OPTIONAL"):
        registry.get_pack("sec")


def test_an_unknown_pack_is_still_an_unknown_pack(default_install):
    with pytest.raises(KeyError, match="no domain pack 'nope'"):
        registry.get_pack("nope")


def test_enabling_it_loads_it(security_install):
    assert registry.pack_ids() == ["av", "sec"]
    assert "anpr" in registry.get_pack("sec").manifest.capabilities


def _paths(app: FastAPI) -> set[str]:
    return {getattr(r, "path", "") for r in app.routes}


def test_its_routes_exist_only_when_it_is_enabled():
    from services.api.main import mount_optional_routers

    off = FastAPI()
    assert mount_optional_routers(off, set()) == []
    assert not any(p.startswith("/api/security") for p in _paths(off))

    on = FastAPI()
    assert mount_optional_routers(on, {"sec"}) == ["sec"]
    assert "/api/security/pack" in _paths(on)


def test_the_web_app_is_told_what_is_installed(security_install):
    from _authutil import auth_headers
    from fastapi.testclient import TestClient

    from services.api.main import app

    with TestClient(app) as c:
        r = c.get("/api/system/features", headers=auth_headers("annotator"))
    assert r.status_code == 200
    assert r.json()["optional_packs"] == {"sec": True}
    # No worker runs during the suite.
    assert r.json()["gpu_worker"] is False


def test_a_default_install_reports_it_off(default_install):
    from _authutil import auth_headers
    from fastapi.testclient import TestClient

    from services.api.main import app

    with TestClient(app) as c:
        r = c.get("/api/system/features", headers=auth_headers("annotator"))
    assert r.json()["optional_packs"] == {"sec": False}
