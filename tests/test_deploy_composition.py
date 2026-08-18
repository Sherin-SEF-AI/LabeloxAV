"""The loop only runs itself if something starts it.

The engine's central claim is that it improves its own models in a closed loop. `controller.tick` is real
logic - it scans drift, gates a registered challenger, schedules an off-hours retrain - but its only
callers were a manual HTTP endpoint and a Makefile target nothing supervised. A `make` target is not a
deployment: nobody restarts it, nothing notices when it dies, and a fresh install does not have it running
at all. The same was true of the PII weights, without which the anonymizer refuses to construct and the
first ingest on a new box fails.

These are source-tree assertions over the compose overlay rather than behavioural tests, in the shape of
tests/test_route_auth.py and tests/test_db_markers.py: cheap, and they fail the moment the deployment stops
containing the thing the product description depends on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml", reason="pyyaml parses the compose overlay")

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_COMPOSE = REPO_ROOT / "docker-compose.app.yml"


def _services() -> dict:
    return yaml.safe_load(APP_COMPOSE.read_text())["services"]


def test_the_overlay_still_parses():
    services = _services()
    # Non-triviality floor: an empty or renamed file must not make every assertion below vacuous.
    assert len(services) >= 5, f"only {len(services)} services found; the overlay has moved"


def test_the_api_and_web_are_deployed():
    assert {"api", "web", "migrate"} <= set(_services())


class TestTheLoopHasADriver:
    def test_the_governance_daemon_is_a_service(self):
        services = _services()
        assert "govern-daemon" in services, (
            "the closed loop has no deployed driver again; controller.tick would only run when a human "
            "calls the manual endpoint")

    def test_it_restarts_on_its_own(self):
        # It holds a Postgres advisory lock, so a second copy exits cleanly and restarting is safe.
        assert _services()["govern-daemon"]["restart"] == "unless-stopped"

    def test_it_waits_for_the_schema(self):
        depends = _services()["govern-daemon"]["depends_on"]
        assert depends["migrate"]["condition"] == "service_completed_successfully"


class TestAFreshInstallCanIngest:
    def test_the_pii_weights_are_fetched_by_the_deployment(self):
        assert "pii-models" in _services(), (
            "nothing fetches the PII detector weights, so the anonymizer refuses to construct and the "
            "first ingest on a fresh install fails")

    def test_the_api_will_not_start_until_they_are_there(self):
        # Ordering is the whole point: an API that starts before the weights exist is an API that accepts
        # an ingest it cannot redact.
        depends = _services()["api"]["depends_on"]
        assert depends["pii-models"]["condition"] == "service_completed_successfully"

    def test_the_weights_land_on_the_volume_the_api_reads(self):
        def scratch_mounts(name):
            return [v for v in _services()[name].get("volumes", []) if "/app/.scratch" in v]

        assert scratch_mounts("pii-models"), "the weights would be written into a discarded container layer"
        assert scratch_mounts("api"), "the API would not see the weights that were fetched for it"


class TestTheWorkersAreProfiledNotAbsent:
    """A CPU-only host must not be asked to pull the CUDA image, but the workers still have to exist."""

    def test_the_training_worker_is_defined_behind_the_gpu_profile(self):
        svc = _services()["train-worker"]
        assert svc["profiles"] == ["gpu"]
        assert svc["build"]["target"] == "gpu"

    def test_the_embedding_worker_is_defined_behind_the_workers_profile(self):
        assert _services()["embed-worker"]["profiles"] == ["workers"]

    def test_neither_is_in_the_default_set(self):
        # `docker compose up` on a laptop must stay CPU-sized.
        for name in ("train-worker", "embed-worker"):
            assert _services()[name].get("profiles"), f"{name} would start by default"


class TestCorsIsConfigurable:
    """It was hardcoded to the two dev origins in services/api/main.py.

    A deployment behind any other hostname had every browser call blocked until somebody edited that line
    on the host - a source change that no longer matches the repo and is invisible to anyone reading it.
    """

    def test_the_default_is_still_the_dev_origins(self):
        from core.config import get_settings

        assert get_settings().cors.origin_list() == ["http://localhost:3000", "http://127.0.0.1:3000"]

    def test_a_deployment_can_set_its_own(self, monkeypatch):
        from core.config import CorsSettings

        assert CorsSettings(origins="https://a.example.com,https://b.example.com").origin_list() == [
            "https://a.example.com", "https://b.example.com"]

    def test_whitespace_and_empties_are_tolerated(self):
        from core.config import CorsSettings

        assert CorsSettings(origins=" https://a.example.com , ,https://b.example.com ").origin_list() == [
            "https://a.example.com", "https://b.example.com"]

    def test_allow_all_turns_credentials_off(self):
        """A browser refuses to send credentials to an allow-all origin, and so should the server.

        Combining allow_credentials with "*" is the shape that gets a browser talked into making
        authenticated requests on somebody else's behalf; Starlette refuses it outright, so an allow-all
        deployment would otherwise get a silently broken CORS policy rather than a non-credentialed one.
        """
        import re

        from core.config import CorsSettings

        # Read the decision out of the middleware wiring rather than restating it here, so this fails if
        # somebody sets allow_credentials=True unconditionally.
        main_src = (REPO_ROOT / "services/api/main.py").read_text()
        assert re.search(r'allow_credentials\s*=\s*"\*" not in _cors_origins', main_src), (
            "CORS credentials are no longer conditioned on the origin list being specific")
        assert "*" in CorsSettings(origins="*").origin_list()
        assert "*" not in CorsSettings(origins="https://a.example.com").origin_list()
