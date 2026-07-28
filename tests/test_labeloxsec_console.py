"""LabeloxSec's own surface: the ANPR store, per-session pack routing, and the HTTP console endpoints.

The Sec pack shipped with an ontology, a static-camera scene model, a tested India plate-format kernel and a
recogniser, and no endpoint exposed any of it, so nothing outside the Python package could reach the second
domain. These tests cover the layer that closes that gap.

The point being defended throughout is that ANPR is capability-gated rather than role-gated alone: an admin
token is not authority to read a registration mark under the AV pack, because under DPDPA a plate is personal
data that the AV privacy plane blurs and never reads. A role check alone would let the same binary do in one
domain what it must refuse in the other.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.db


def _client() -> TestClient:
    """Used as a context manager by every HTTP test here. Entering it holds one portal open for the life of
    the block, so successive requests share an event loop; constructed per-request instead, the second call
    runs against an engine cached on the first call's now-closed loop."""
    from _authutil import _clear_db_cache

    from services.api.main import app

    _clear_db_cache()
    return TestClient(app)


async def _seed_session(pack_id: str = "sec") -> str:
    """A minimal session in the given domain. Reads hang off a session so erasure cascades to the plate text."""
    from db.models import Session as DbSession
    from db.session import get_sessionmaker

    async with get_sessionmaker()() as db:
        s = DbSession(vehicle_id=None if pack_id == "sec" else "veh_test", city="BLR",
                      start_ts_ns=1_000_000, end_ts_ns=2_000_000,
                      pack_id=pack_id, ontology_version="labelox-sec-0.1.0")
        db.add(s)
        await db.commit()
        return str(s.session_id)


# ---------------------------------------------------------------- the capability gate

async def test_recording_a_read_is_refused_under_the_av_pack():
    """The gate that makes two domains in one binary safe. An AV session must not accumulate plate text."""
    from db.session import get_sessionmaker
    from services.anpr.recognize import AnprNotAuthorised
    from services.anpr.store import record_reads

    async with get_sessionmaker()() as db:
        with pytest.raises(AnprNotAuthorised):
            await record_reads(db, [], pack_id="av")


async def test_recording_a_read_is_permitted_under_the_sec_pack():
    from db.session import get_sessionmaker
    from services.anpr.store import record_reads

    async with get_sessionmaker()() as db:
        out = await record_reads(db, [], pack_id="sec")
    assert out["recorded"] == 0


# ---------------------------------------------------------------- the watchlist

async def _make_read(plate: str, ocr_conf: float | None = 0.8):
    """A PlateRead as the recogniser produces it, so the store is exercised on its real input shape."""
    from services.anpr.india_format import parse_plate
    from services.anpr.recognize import PlateRead

    return PlateRead(bbox=(1.0, 2.0, 30.0, 12.0), det_conf=0.9, ocr_text=plate,
                     ocr_conf=ocr_conf, parse=parse_plate(plate))


async def test_the_same_mark_written_three_ways_is_one_watchlist_entry():
    """Normalisation is what makes matching work at all. Three rows would report one vehicle as three hits."""
    from db.session import get_sessionmaker
    from services.anpr.store import add_watchlist_entry, list_watchlist

    async with get_sessionmaker()() as db:
        a = await add_watchlist_entry(db, plate="KA 01 AB 1234", reason="first")
        b = await add_watchlist_entry(db, plate="ka-01-ab-1234", severity="critical")
        c = await add_watchlist_entry(db, plate="KA01AB1234")

        assert a["entry_id"] == b["entry_id"] == c["entry_id"]
        # Re-adding reactivates and re-annotates rather than failing: an operator re-adding a plate means
        # "watch this", and a duplicate-key error is a worse answer than doing what they asked.
        assert c["severity"] == "warn"
        assert c["reason"] == "first"

        marks = [e["plate"] for e in await list_watchlist(db)]
        assert marks.count("KA01AB1234") == 1


async def test_removing_an_entry_deactivates_rather_than_deletes():
    """A hit already recorded refers to why it was watched. Dropping the row makes that history unexplainable."""
    from db.session import get_sessionmaker
    from services.anpr.store import add_watchlist_entry, list_watchlist, remove_watchlist_entry

    async with get_sessionmaker()() as db:
        e = await add_watchlist_entry(db, plate="MH12XY9911", reason="stolen")
        assert (await remove_watchlist_entry(db, e["entry_id"]))["removed"] is True

        assert e["plate"] not in [x["plate"] for x in await list_watchlist(db, active_only=True)]
        gone = [x for x in await list_watchlist(db, active_only=False) if x["plate"] == e["plate"]]
        assert len(gone) == 1 and gone[0]["active"] is False and gone[0]["reason"] == "stolen"


async def test_an_unparseable_mark_is_rejected_rather_than_stored():
    from db.session import get_sessionmaker
    from services.anpr.store import add_watchlist_entry

    async with get_sessionmaker()() as db:
        with pytest.raises(ValueError):
            await add_watchlist_entry(db, plate="!!!")
        with pytest.raises(ValueError):
            await add_watchlist_entry(db, plate="KA01AB1234", severity="urgent")


# ---------------------------------------------------------------- reads and matching

async def test_a_read_matching_an_active_entry_is_flagged_with_its_severity():
    from db.session import get_sessionmaker
    from services.anpr.store import add_watchlist_entry, record_reads

    session_id = await _seed_session("sec")
    async with get_sessionmaker()() as db:
        await add_watchlist_entry(db, plate="KA 05 MZ 4321", severity="critical", reason="wanted")
        out = await record_reads(db, [await _make_read("KA05MZ4321"), await _make_read("TN10BC5678")],
                                 session_id=session_id, camera_id="cam_gate", pack_id="sec")

    assert out["recorded"] == 2
    assert out["watchlist_hits"] == 1
    hit = [r for r in out["reads"] if r["watchlist_hit"]][0]
    assert hit["plate"] == "KA05MZ4321"
    assert hit["watchlist_severity"] == "critical"
    assert [r for r in out["reads"] if not r["watchlist_hit"]][0]["watchlist_severity"] is None


async def test_a_deactivated_entry_stops_producing_hits():
    """The reason deactivation is not cosmetic: it has to actually stop matching."""
    from db.session import get_sessionmaker
    from services.anpr.store import add_watchlist_entry, record_reads, remove_watchlist_entry

    session_id = await _seed_session("sec")
    async with get_sessionmaker()() as db:
        e = await add_watchlist_entry(db, plate="DL8CAF5031")
        await remove_watchlist_entry(db, e["entry_id"])
        out = await record_reads(db, [await _make_read("DL8CAF5031")],
                                 session_id=session_id, pack_id="sec")

    assert out["watchlist_hits"] == 0


async def test_an_unmeasured_ocr_confidence_stays_null_rather_than_becoming_zero():
    """A stand-in number would make a confidence filter in the console look meaningful when it is not."""
    from db.session import get_sessionmaker
    from services.anpr.store import record_reads

    session_id = await _seed_session("sec")
    async with get_sessionmaker()() as db:
        out = await record_reads(db, [await _make_read("KL07AT1234", ocr_conf=None)],
                                 session_id=session_id, pack_id="sec")
        assert out["reads"][0]["ocr_conf"] is None

        from services.anpr.store import read_stats
        assert (await read_stats(db))["unscored_reads"] >= 1


async def test_reads_filter_on_the_normalised_mark_however_it_is_typed():
    from db.session import get_sessionmaker
    from services.anpr.store import list_reads, record_reads

    session_id = await _seed_session("sec")
    async with get_sessionmaker()() as db:
        await record_reads(db, [await _make_read("GJ01RR7788")], session_id=session_id,
                           camera_id="cam_lot", pack_id="sec")
        assert (await list_reads(db, plate="gj 01 rr 7788"))["total"] >= 1
        assert (await list_reads(db, session_id=session_id, camera_id="cam_lot"))["total"] >= 1
        assert (await list_reads(db, session_id=session_id, camera_id="nonexistent"))["total"] == 0


async def test_erasing_a_session_takes_its_plate_reads_with_it():
    """Plate text is the most sensitive field in the corpus. Leaving it behind after an erasure request is the
    exact failure the FK cascade exists to prevent."""
    from sqlalchemy import func, select

    from db.models import PlateRead
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.anpr.store import record_reads

    session_id = await _seed_session("sec")
    async with get_sessionmaker()() as db:
        await record_reads(db, [await _make_read("KA51HH2222")], session_id=session_id, pack_id="sec")
        sid = uuid.UUID(session_id)
        assert (await db.execute(select(func.count()).select_from(PlateRead)
                                 .where(PlateRead.session_id == sid))).scalar_one() == 1

        await db.delete(await db.get(DbSession, sid))
        await db.commit()
        assert (await db.execute(select(func.count()).select_from(PlateRead)
                                 .where(PlateRead.session_id == sid))).scalar_one() == 0


# ---------------------------------------------------------------- per-session pack routing

async def test_a_session_resolves_to_its_own_pack():
    """Which domain a capture belongs to is a property of the capture, not of the request that reads it."""
    from db.session import get_sessionmaker
    from services.domain import pack_id_for_session

    sec_id, av_id = await _seed_session("sec"), await _seed_session("av")
    async with get_sessionmaker()() as db:
        assert await pack_id_for_session(db, sec_id) == "sec"
        assert await pack_id_for_session(db, av_id) == "av"


async def test_an_unknown_session_falls_back_to_the_default_pack():
    """Falling back beats raising: a missing session is the caller's problem to report, and a routing helper
    that throws would turn a 404 into a 500."""
    from db.session import get_sessionmaker
    from services.domain import default_pack_id, pack_id_for_session

    async with get_sessionmaker()() as db:
        assert await pack_id_for_session(db, str(uuid.uuid4())) == default_pack_id()


# ---------------------------------------------------------------- the HTTP console

def test_the_console_endpoints_require_a_reviewer():
    """Plate reads and the watchlist are personal data and operational security state. An annotator drawing
    boxes has no reason to enumerate who drove past a camera."""
    from _authutil import auth_headers

    anon, annotator, reviewer = {}, auth_headers("annotator"), auth_headers("reviewer")
    with _client() as c:
        assert c.get("/api/security/reads", headers=anon).status_code == 401
        assert c.get("/api/security/reads", headers=annotator).status_code == 403
        assert c.get("/api/security/reads", headers=reviewer).status_code == 200


def test_the_pack_endpoint_tells_the_console_what_is_authorised():
    """The console renders off this: an AV deployment is told ANPR is refused rather than being handed a plate
    console whose every button 403s."""
    from _authutil import auth_headers

    h = auth_headers("reviewer")
    with _client() as c:
        sec = c.get("/api/security/pack?pack_id=sec", headers=h)
        assert sec.status_code == 200
        body = sec.json()
        assert body["anpr_authorised"] is True
        assert body["static_camera"] is True
        assert "sec" in body["available_packs"] and "av" in body["available_packs"]

        av = c.get("/api/security/pack?pack_id=av", headers=h).json()
        assert av["anpr_authorised"] is False

        assert c.get("/api/security/pack?pack_id=nope", headers=h).status_code == 404


def test_the_watchlist_round_trips_over_http():
    from _authutil import auth_headers

    h = auth_headers("reviewer")
    plate = "UP32GG7001"
    with _client() as c:
        created = c.post("/api/security/watchlist",
                         json={"plate": plate, "reason": "test", "severity": "warn"}, headers=h)
        assert created.status_code == 200
        entry = created.json()
        assert entry["plate"] == plate

        assert plate in [e["plate"] for e in c.get("/api/security/watchlist", headers=h).json()["entries"]]

        assert c.delete(f"/api/security/watchlist/{entry['entry_id']}", headers=h).json()["removed"] is True
        assert plate not in [e["plate"] for e in c.get("/api/security/watchlist", headers=h).json()["entries"]]

        # A malformed mark is a 400 from the router rather than an unhandled ValueError.
        assert c.post("/api/security/watchlist", json={"plate": "###"}, headers=h).status_code == 400


def test_the_stats_endpoint_reports_the_counts_the_console_renders():
    from _authutil import auth_headers

    h = auth_headers("reviewer")
    with _client() as c:
        body = c.get("/api/security/stats", headers=h).json()
    for key in ("reads", "watchlist_hits", "valid_format", "watchlist_size", "unscored_reads", "top_states"):
        assert key in body


def test_recognizing_over_a_missing_frame_is_a_404_not_a_crash():
    from _authutil import auth_headers

    h = auth_headers("reviewer")
    with _client() as c:
        r = c.post("/api/security/recognize",
                   json={"frame_id": str(uuid.uuid4()), "regions": [[0, 0, 10, 10, 0.9]]}, headers=h)
    assert r.status_code == 404


# ---------------------------------------------------------------- erasure evidence

async def test_the_erasure_certificate_states_how_many_plate_reads_it_removed():
    """A certificate that omits the plate text it removed cannot answer the request it exists to answer, and
    because the count sits inside the digested body it cannot later be edited to claim none was held."""
    import hashlib
    import json

    from db.session import get_sessionmaker
    from services.anpr.store import record_reads
    from services.govern.retention import erase_session

    session_id = await _seed_session("sec")
    async with get_sessionmaker()() as db:
        await record_reads(db, [await _make_read("KA09PP1010"), await _make_read("KA09PP2020")],
                           session_id=session_id, pack_id="sec")

        dry = await erase_session(db, uuid.UUID(session_id), reason="test", dry_run=True)
        assert dry["plate_reads"] == 2

        cert = (await erase_session(db, uuid.UUID(session_id), reason="test"))["certificate"]

    assert cert["plate_reads"] == 2
    body = {k: v for k, v in cert.items() if k != "digest"}
    assert hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() == cert["digest"]
