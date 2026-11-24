"""The gold calibration fitter can be run, and will not switch the gate to a curve that failed its check.

docs/ENGINES.md names services/autolabel/gold_calibrate.py as the way to fit the curve the gate serves,
and nothing could run it: only its tests imported it. It has a command line now. The part worth pinning is
`activate`, because it changes every auto-accept decision in the product.
"""

from __future__ import annotations

import json

from click.testing import CliRunner

from services.autolabel import gold_calibrate as gc


class _Store:
    def __init__(self, payload: dict | None):
        self.payload = payload

    def get_bytes(self, _uri: str) -> bytes:
        if self.payload is None:
            raise FileNotFoundError("no such object")
        return json.dumps(self.payload).encode()


def _run(monkeypatch, payload, *args):
    activated: list[str] = []
    monkeypatch.setattr(gc, "get_object_store", lambda: _Store(payload))
    monkeypatch.setattr(gc, "activate_calibration", lambda uri: activated.append(uri) or {"activated": True, "uri": uri})
    result = CliRunner().invoke(gc._cli(), ["activate", "--uri", "s3://b/calibration/x.json", *args])
    return result, activated


def test_the_commands_exist():
    out = CliRunner().invoke(gc._cli(), ["--help"]).output
    assert "fit-gold" in out and "fit-vlm" in out and "activate" in out


def test_a_trustworthy_curve_is_activated(monkeypatch):
    result, activated = _run(monkeypatch, {"kind": "isotonic-knots-v1", "x": [0, 1], "y": [0, 1],
                                           "trustworthy": True})
    assert result.exit_code == 0, result.output
    assert activated == ["s3://b/calibration/x.json"]


def test_a_curve_that_failed_its_held_out_check_is_refused(monkeypatch):
    result, activated = _run(monkeypatch, {"x": [0, 1], "y": [0.97, 0.97], "trustworthy": False})
    assert result.exit_code != 0 and activated == []
    assert "did not report trustworthy" in result.output


def test_a_curve_with_no_verdict_recorded_is_refused_too(monkeypatch):
    # Curves stored before the verdict travelled with them. Unknown is not trustworthy.
    result, activated = _run(monkeypatch, {"x": [0, 1], "y": [0, 1]})
    assert result.exit_code != 0 and activated == []


def test_the_override_is_explicit(monkeypatch):
    result, activated = _run(monkeypatch, {"x": [0, 1], "y": [0, 1], "trustworthy": False},
                             "--even-if-untrustworthy")
    assert result.exit_code == 0 and len(activated) == 1


def test_a_missing_curve_is_an_error_not_an_activation(monkeypatch):
    result, activated = _run(monkeypatch, None)
    assert result.exit_code != 0 and activated == []
    assert "could not read a fitted curve" in result.output
