"""Notifications, the activity feed, and the PII access log.

Three things the system knew and never said. A blocked promotion, an assigned job, a raised issue, a drift
breach, and a kill switch were all silent unless somebody happened to be looking at the right page. Reviews
and jobs each kept their own history and none of them was a timeline. And `pii_audit` recorded what the
redactor found while nothing recorded who then looked at it, which is the half a DPDPA enquiry turns on.

The properties worth defending, and what would break without them:

- **A repeated condition supersedes rather than piles up.** Drift and SLO checks re-evaluate on a schedule,
  so an unresolved breach would otherwise add a line every cycle until the bell was noise nobody reads.
- **Emission never raises.** A notification is a message about work, not the work; failing a promotion
  because its announcement failed would be a far worse bug than a missing line in a list.
- **Role-addressed read state is per user.** A duty-queue item marked read by one reviewer must still be
  unread for the next one, or the first person to look makes it vanish for the team.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.db


def _client() -> TestClient:
    from _authutil import _clear_db_cache

    from services.api.main import app

    _clear_db_cache()
    return TestClient(app)


async def _user(role: str = "reviewer"):
    from db.models import User
    from db.session import get_sessionmaker

    async with get_sessionmaker()() as db:
        u = User(name=f"inbox-{role}-{uuid.uuid4().hex[:8]}", role=role)
        db.add(u)
        await db.commit()
        return u.user_id, u.name


# ---------------------------------------------------------------- notifications

async def test_a_personal_notification_reaches_only_its_addressee():
    from db.models import User
    from db.session import get_sessionmaker
    from services.notify import list_for, notify

    mine, _ = await _user("reviewer")
    theirs, _ = await _user("reviewer")
    async with get_sessionmaker()() as db:
        await notify(db, kind="job_assigned", user_id=mine, title="a job for you")
        me = await db.get(User, mine)
        them = await db.get(User, theirs)
        assert any(n["title"] == "a job for you" for n in (await list_for(db, me))["notifications"])
        assert not any(n["title"] == "a job for you"
                       for n in (await list_for(db, them))["notifications"])


async def test_a_role_notification_reaches_everyone_holding_that_role():
    from db.models import User
    from db.session import get_sessionmaker
    from services.notify import list_for, notify

    a, _ = await _user("reviewer")
    b, _ = await _user("reviewer")
    c, _ = await _user("annotator")
    async with get_sessionmaker()() as db:
        title = f"duty-{uuid.uuid4().hex[:6]}"
        await notify(db, kind="promotion_blocked", role="reviewer", title=title)
        for uid, expected in ((a, True), (b, True), (c, False)):
            user = await db.get(User, uid)
            seen = any(n["title"] == title for n in (await list_for(db, user))["notifications"])
            assert seen is expected


async def test_a_repeated_condition_supersedes_rather_than_piling_up():
    """Drift re-evaluates on a schedule. Without this the bell fills with the same unresolved breach."""
    from db.models import User
    from db.session import get_sessionmaker
    from services.notify import list_for, notify

    uid, _ = await _user("reviewer")
    subject = f"drift-{uuid.uuid4().hex[:6]}"
    async with get_sessionmaker()() as db:
        for i in range(4):
            await notify(db, kind="drift_breach", user_id=uid, title=f"drift breach {i}",
                         subject_type="drift", subject_id=subject)
        user = await db.get(User, uid)
        rows = [n for n in (await list_for(db, user))["notifications"]
                if n["subject_id"] == subject]
    assert len(rows) == 1
    assert rows[0]["title"] == "drift breach 3"     # the latest survives, not the first


async def test_a_distinct_subject_is_not_superseded():
    from db.models import User
    from db.session import get_sessionmaker
    from services.notify import list_for, notify

    uid, _ = await _user("reviewer")
    tag = uuid.uuid4().hex[:6]
    async with get_sessionmaker()() as db:
        await notify(db, kind="drift_breach", user_id=uid, title="a", subject_type="drift",
                     subject_id=f"x-{tag}")
        await notify(db, kind="drift_breach", user_id=uid, title="b", subject_type="drift",
                     subject_id=f"y-{tag}")
        user = await db.get(User, uid)
        rows = [n for n in (await list_for(db, user))["notifications"]
                if str(n["subject_id"]).endswith(tag)]
    assert len(rows) == 2


async def test_marking_a_role_notification_read_is_per_user():
    """The duty queue. If one reviewer reading it cleared it for everyone, the second reviewer would never
    learn the thing happened."""
    from db.models import User
    from db.session import get_sessionmaker
    from services.notify import list_for, mark_read, notify

    a, _ = await _user("reviewer")
    b, _ = await _user("reviewer")
    async with get_sessionmaker()() as db:
        nid = await notify(db, kind="issue_opened", role="reviewer",
                           title=f"issue-{uuid.uuid4().hex[:6]}", supersede=False)
        ua, ub = await db.get(User, a), await db.get(User, b)
        await mark_read(db, ua, nid)

        def _find(listing):
            return next(n for n in listing["notifications"] if n["notification_id"] == nid)

        assert _find(await list_for(db, ua))["read"] is True
        assert _find(await list_for(db, ub))["read"] is False


async def test_the_unread_count_counts_both_kinds():
    from db.models import User
    from db.session import get_sessionmaker
    from services.notify import mark_all_read, notify, unread_count

    uid, _ = await _user("reviewer")
    async with get_sessionmaker()() as db:
        user = await db.get(User, uid)
        before = await unread_count(db, user)
        await notify(db, kind="job_assigned", user_id=uid, title="personal", supersede=False)
        await notify(db, kind="issue_opened", role="reviewer", title="duty", supersede=False)
        assert await unread_count(db, user) == before + 2

        await mark_all_read(db, user)
        assert await unread_count(db, user) == 0


async def test_emission_never_raises_on_a_bad_call():
    """A notification is a message about work, not the work. Raising here would let an announcement failure
    roll back a completed promotion."""
    from db.session import get_sessionmaker
    from services.notify import notify

    async with get_sessionmaker()() as db:
        assert await notify(db, kind="job_assigned", user_id="not-a-uuid", title="x") is None


async def test_an_unknown_severity_falls_back_rather_than_failing():
    from db.models import User
    from db.session import get_sessionmaker
    from services.notify import list_for, notify

    uid, _ = await _user("reviewer")
    async with get_sessionmaker()() as db:
        title = f"sev-{uuid.uuid4().hex[:6]}"
        await notify(db, kind="job_assigned", user_id=uid, title=title, severity="apocalyptic")
        user = await db.get(User, uid)
        row = next(n for n in (await list_for(db, user))["notifications"] if n["title"] == title)
    assert row["severity"] == "info"


def test_the_notification_routes_need_a_token():
    with _client() as c:
        assert c.get("/api/notifications").status_code == 401
        assert c.get("/api/notifications/count").status_code == 401


def test_the_bell_endpoint_answers_for_a_signed_in_user():
    from _authutil import auth_headers

    h = auth_headers("reviewer")
    with _client() as c:
        r = c.get("/api/notifications/count", headers=h)
        assert r.status_code == 200 and "unread" in r.json()
        assert c.get("/api/notifications", headers=h).status_code == 200


# ---------------------------------------------------------------- activity

async def test_activity_records_and_summarises():
    from db.models import User
    from db.session import get_sessionmaker
    from services.activity import activity_summary, list_activity, record_activity

    uid, name = await _user("annotator")
    async with get_sessionmaker()() as db:
        user = await db.get(User, uid)
        for verb in ("confirmed", "confirmed", "rejected"):
            await record_activity(db, user=user, verb=verb, subject_type="object",
                                  subject_id=str(uuid.uuid4()))
        feed = await list_activity(db, user_id=str(uid))
        summary = await activity_summary(db, user_id=str(uid), hours=1)

    assert feed["total"] == 3
    assert {e["verb"] for e in feed["events"]} == {"confirmed", "rejected"}
    assert summary["by_verb"] == {"confirmed": 2, "rejected": 1}
    # Every verb carries a human label, so the feed is not a wall of raw identifiers.
    assert summary["labels"]["confirmed"] == "confirmed an object"


async def test_activity_recording_never_raises():
    from db.session import get_sessionmaker
    from services.activity import record_activity

    async with get_sessionmaker()() as db:
        await record_activity(db, user_id="not-a-uuid", verb="confirmed")   # must not raise


def test_the_activity_route_scopes_to_the_caller_when_asked():
    from _authutil import auth_headers

    with _client() as c:
        r = c.get("/api/activity?mine=true", headers=auth_headers("reviewer"))
        assert r.status_code == 200
        assert c.get("/api/activity/summary", headers=auth_headers("reviewer")).status_code == 200
        assert c.get("/api/activity").status_code == 401


# ---------------------------------------------------------------- PII access log

async def test_the_access_log_records_who_looked_and_whether_it_was_redacted():
    from db.session import get_sessionmaker
    from services.govern.pii_access import access_summary, list_access, record_access

    uid, name = await _user("reviewer")
    subject = str(uuid.uuid4())
    async with get_sessionmaker()() as db:
        await record_access(db, subject_type="frame", subject_id=subject, action="view",
                            user_id=str(uid), user_name=name, pii_kinds=["face"], redacted=True)
        await record_access(db, subject_type="plate_read", subject_id=subject, action="read_plate",
                            user_id=str(uid), user_name=name, pii_kinds=["plate"], redacted=False)
        rows = await list_access(db, subject_id=subject)
        summary = await access_summary(db, hours=1)

    assert rows["total"] == 2
    assert {r["action"] for r in rows["accesses"]} == {"view", "read_plate"}
    # Unredacted access is the number a policy is actually written about, so it is broken out.
    assert summary["unredacted"] >= 1
    assert summary["by_user"].get(name) == 2


async def test_recording_an_access_never_raises():
    """Evidence about a request must not be able to fail the request, or operators learn to route around
    the logging."""
    from db.session import get_sessionmaker
    from services.govern.pii_access import record_access

    async with get_sessionmaker()() as db:
        await record_access(db, subject_type="frame", subject_id="x", user_id="not-a-uuid")


async def test_the_access_log_does_not_copy_the_data_it_tracks():
    """A compliance log that duplicates the personal data it measures has doubled the exposure."""
    from db.models import PiiAccessLog

    columns = set(PiiAccessLog.__table__.columns.keys())
    for leaked in ("plate_text", "image", "pixels", "transcript", "plate_raw"):
        assert leaked not in columns


def test_the_access_log_is_admin_only():
    """It is a record about employees as much as about subjects, so a reviewer browsing colleagues' viewing
    history would be surveillance rather than compliance."""
    from _authutil import auth_headers

    with _client() as c:
        assert c.get("/api/govern/pii-access").status_code == 401
        assert c.get("/api/govern/pii-access",
                     headers=auth_headers("reviewer")).status_code == 403
        assert c.get("/api/govern/pii-access",
                     headers=auth_headers("admin")).status_code == 200
