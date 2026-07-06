"""M9 acceptance (progression logic): the flywheel stops a session at the failing gate, so bad data never
reaches curation or the label queue; a clean session proceeds through every plane in loop order."""

from orchestration.dag import STAGES, plan


def test_quarantined_session_stops_at_sanyx():
    p = plan(sanyx_decision="quarantine", calyx_severity=None)
    assert p["proceed"] is False and p["stopped_at"] == "sanyx"
    assert p["stages_run"] == ["sanyx"]   # never reaches SIEVYX or the label queue


def test_calibration_blocked_session_stops_at_calyx():
    p = plan(sanyx_decision="pass", calyx_severity="block")
    assert p["proceed"] is False and p["stopped_at"] == "calyx"
    assert "sievyx" not in p["stages_run"]


def test_clean_session_runs_full_loop_in_order():
    p = plan(sanyx_decision="pass", calyx_severity="ok")
    assert p["proceed"] is True and p["stopped_at"] is None
    # the loop order is sanyx -> calyx -> sievyx -> labelox -> oraclyx -> verdyx -> forgyx
    assert p["stages_run"] == ["sanyx", "calyx", "sievyx", "labelox", "oraclyx", "verdyx", "forgyx"]
    assert p["stages_run"] == STAGES


def test_degraded_session_still_proceeds():
    # degraded is a warning, not a block: it proceeds with the warning attached upstream
    assert plan(sanyx_decision="degraded", calyx_severity="drift_detected")["proceed"] is True
