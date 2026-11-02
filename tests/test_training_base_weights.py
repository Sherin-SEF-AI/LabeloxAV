"""Starting a training run from a registered model.

A registered model's weights are an object-store address, and the trainer opens local files only, so a run
started from the champion failed on its first line with "No such file or directory: 's3:/...'". The address
is fetched to a verified local copy first; a bare model name is left for the trainer to resolve.
"""

from __future__ import annotations

import asyncio

import pytest

from services.training import jobs


def test_a_model_name_or_a_local_path_is_passed_through(tmp_path):
    local = tmp_path / "best.pt"
    local.write_bytes(b"x")
    assert asyncio.run(jobs.local_base_weights("yolo11l.pt")) == "yolo11l.pt"
    assert asyncio.run(jobs.local_base_weights(str(local))) == str(local)
    assert asyncio.run(jobs.local_base_weights(None)) is None


def test_a_registry_address_is_fetched_to_a_local_copy(monkeypatch, tmp_path):
    import services.autolabel.runner as runner

    fetched = tmp_path / "abc.pt"
    seen = []

    def fake(uri):
        seen.append(uri)
        return str(fetched)

    monkeypatch.setattr(runner, "_local_champion_weights", fake)
    uri = "s3://labeloxav/models/idd-yolo11l-local/best.pt"
    assert asyncio.run(jobs.local_base_weights(uri)) == str(fetched)
    assert seen == [uri]


def test_an_address_that_cannot_be_fetched_says_so(monkeypatch):
    import services.autolabel.runner as runner

    monkeypatch.setattr(runner, "_local_champion_weights", lambda uri: None)
    with pytest.raises(RuntimeError, match="could not be fetched"):
        asyncio.run(jobs.local_base_weights("s3://labeloxav/models/gone/best.pt"))
