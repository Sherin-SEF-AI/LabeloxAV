"""SEC-M8: security-domain events on the outbound webhook infra.

A downstream consumer (e.g. Sentigon) subscribes to anpr.read / anpr.watchlist_hit via the existing HMAC-signed
webhook mechanism; here we test that those events are subscribable and that the ANPR emission helpers fan out
to a matching subscription, plus watchlist matching. We build the consumer nowhere: only the generic emission.
The webhook URL is intentionally unroutable so delivery never blocks or raises (fire-and-forget).
"""

from __future__ import annotations

import pytest

from db.session import get_sessionmaker
from services.anpr.events import emit_anpr_read, process_security_reads
from services.anpr.india_format import parse_plate
from services.anpr.recognize import PlateRead
from services.anpr.watchlist import match, normalize_watchlist
from services.integrations.webhooks import EVENTS, create_webhook, delete_webhook

pytestmark = pytest.mark.db


def _read(text: str = "KA01AB1234") -> PlateRead:
    return PlateRead(bbox=(10.0, 20.0, 110.0, 60.0), det_conf=0.9, ocr_text=text, ocr_conf=0.95,
                     parse=parse_plate(text))


# ---- watchlist (pure) -----------------------------------------------------------------------------------

def test_watchlist_matches_regardless_of_formatting():
    r = _read("KA01AB1234")
    assert match(r, ["ka 01-ab.1234"]) == "KA01AB1234"     # different formatting, same mark
    assert match(r, ["MH12DE1433"]) is None
    assert match(_read("!!!"), ["KA01AB1234"]) is None      # unreadable plate never matches


def test_normalize_watchlist_dedups_and_strips():
    wl = normalize_watchlist(["KA 01 AB 1234", "ka-01-ab-1234", "", "MH12DE1433"])
    assert wl == {"KA01AB1234", "MH12DE1433"}


# ---- the events are subscribable --------------------------------------------------------------------------

def test_security_events_are_registered():
    assert {"anpr.read", "anpr.watchlist_hit", "security.event"} <= set(EVENTS)


async def test_create_webhook_accepts_security_events():
    async with get_sessionmaker()() as db:
        wh = await create_webhook(db, url="https://consumer.example/hook",
                                  events=["anpr.read", "anpr.watchlist_hit"])
    assert wh["events"] == ["anpr.read", "anpr.watchlist_hit"]
    async with get_sessionmaker()() as db:
        await delete_webhook(db, wh["webhook_id"])


# ---- the ANPR helpers fan out to a matching subscription -------------------------------------------------

async def test_emit_anpr_read_delivers_to_a_subscriber():
    # Unroutable on purpose: emit is fire-and-forget and must not block or raise on a dead endpoint.
    async with get_sessionmaker()() as db:
        wh = await create_webhook(db, url="http://127.0.0.1:9/hook", events=["anpr.read"])
    try:
        n = await emit_anpr_read(_read(), camera_id="cam_gate", session_id="s1")
        assert n >= 1, "a subscriber to anpr.read is delivered to"
    finally:
        async with get_sessionmaker()() as db:
            await delete_webhook(db, wh["webhook_id"])


async def test_process_security_reads_emits_reads_and_watchlist_hits():
    async with get_sessionmaker()() as db:
        wh = await create_webhook(db, url="http://127.0.0.1:9/hook",
                                  events=["anpr.read", "anpr.watchlist_hit"])
    try:
        reads = [_read("KA01AB1234"), _read("MH12DE1433")]
        summary = await process_security_reads(reads, watchlist=["ka 01 ab 1234"], camera_id="cam_gate")
        assert summary["reads"] == 2
        assert summary["watchlist_hits"] == 1          # only the KA plate is on the list
        assert summary["read_deliveries"] >= 2          # one anpr.read delivery per read
        assert summary["hit_deliveries"] >= 1           # one anpr.watchlist_hit for the match
    finally:
        async with get_sessionmaker()() as db:
            await delete_webhook(db, wh["webhook_id"])
