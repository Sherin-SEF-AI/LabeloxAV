"""`GET /api/collaborate/branches` returned a bare 500 whenever lakeFS was not running.

Found by calling every GET route against the live system rather than by a test: it was the one route of
352 that answered with a server error. lakeFS is optional here. The assignment, task and merge-request
plane is all in Postgres and works without it, and only the per-annotator branch listing needs it, so
lakeFS being down is an ordinary state of this deployment rather than a fault.

Everything else in this codebase that depends on an optional service refuses with the reason. This route
silently did not, and a 500 tells an operator to go looking for a bug in the route.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException


def test_an_unreachable_lakefs_refuses_with_the_reason(monkeypatch):
    import asyncio

    from services.api.routers import collaborate

    def refuse() -> list[str]:
        raise ConnectionRefusedError(111, "Connection refused")

    monkeypatch.setattr(collaborate.L, "list_branches", refuse)
    with pytest.raises(HTTPException) as ei:
        asyncio.run(collaborate.branches())

    assert ei.value.status_code == 503, "not a 500: the service is absent, the route is not broken"
    detail = str(ei.value.detail)
    assert "lakeFS" in detail
    # The operator has to be told what still works, or they will assume collaboration is down.
    assert "Assignments, tasks and merge requests" in detail
    assert "ConnectionRefusedError" in detail


def test_branches_are_returned_when_lakefs_answers(monkeypatch):
    import asyncio

    from services.api.routers import collaborate

    monkeypatch.setattr(collaborate.L, "list_branches", lambda: ["main", "asha-1"])
    assert asyncio.run(collaborate.branches()) == {"branches": ["main", "asha-1"]}
