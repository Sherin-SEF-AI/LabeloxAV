"""M4 integration test: the gate-integrated label queue excludes samples from SANYX-quarantined or
CALYX-blocked sessions, and keeps the rest ranked by SIEVYX priority. Bad data never costs a label."""

from services.labelox.queue import apply_gates


def _it(oid, session, value):
    return {"object_id": oid, "frame_id": f"f{oid}", "session_id": session, "value": value}


def test_excludes_blocked_sessions_and_keeps_priority_order():
    items = [
        _it("a", "good", 0.9),
        _it("b", "quarantined", 0.95),   # highest value, but gated out
        _it("c", "good", 0.4),
        _it("d", "blocked", 0.8),        # gated out
    ]
    kept = apply_gates(items, blocked_sessions={"quarantined", "blocked"})
    assert [it["object_id"] for it in kept] == ["a", "c"]   # priority order, gated ones removed


def test_no_gates_keeps_all_ranked():
    items = [_it("a", "s1", 0.2), _it("b", "s1", 0.9), _it("c", "s2", 0.5)]
    kept = apply_gates(items, blocked_sessions=set())
    assert [it["object_id"] for it in kept] == ["b", "c", "a"]


def test_all_blocked_yields_empty_queue():
    items = [_it("a", "bad", 0.9), _it("b", "bad", 0.5)]
    assert apply_gates(items, blocked_sessions={"bad"}) == []
