"""Scenes for the feature tour: the features a team uses day to day, and how to use each one.

The hands-on demo (`india_scenes.py`) goes deep on the editor, tool by tool. This film goes wide and stays
practical: set up a project, bring footage in, let the models label it, check their work, find data, ship
it, train on it, ask about it. Every scene is the action done in the running product, narrated as the
steps a new user would take, and the footage is real: the operator's Bengaluru dashcam frames (plates
pixelated before upload, see india_scenes.py) are uploaded, auto-labelled and reviewed on camera.

    LBX_DEMO_OUT=.scratch/demo/features/tour LBX_TOUR_TTS=kokoro:af_heart \\
      LBX_TOUR_WEB=http://127.0.0.1:3011 LBX_TOUR_API=http://127.0.0.1:8020 \\
      .venv/bin/python scripts/demo_tour/annotate.py record --script feature_scenes

Scenes that change data clean up after earlier takes in their `before` step (run before the cut point, so
never filmed): the upload erases the previous take's sessions, the project scene removes the project it
creates, and the track scene puts the track back the way the models left it.
"""

from __future__ import annotations

import asyncio
import re

import india_scenes as I
from annotate import ROOT, Demo, Scene
from sqlalchemy import text

from db.session import get_engine, get_sessionmaker

# The India demo keeps the originals of every object its review scenes judged; the track scene here reuses
# its restore, so it must read the same file, not one beside this film's recordings.
I.ORIGINALS = ROOT / ".scratch/demo/india/tour/review_originals.json"

VEHICLE = "BLR-TOUR"
UPLOADS = [ROOT / ".scratch/demo/features/upload/bengaluru-drive-1.zip",
           ROOT / ".scratch/demo/features/upload/bengaluru-drive-2.zip"]
PROJECT = "Bengaluru pilot"
FRAME_INDEX = 4  # the same instant as the hands-on demo: the bus, the auto-rickshaw, the motorcycle, the jeep
EDITOR = I.EDITOR

CHAPTERS = ["Getting started", "Setting up", "Bringing data in", "Labelling", "Reviewing",
            "Finding data", "Delivering and training", "Getting answers"]


async def _one(sql: str, **kw):
    async with get_engine().connect() as c:
        return (await c.execute(text(sql), kw)).scalar()


async def facts() -> dict:
    sessions = await _one("select count(*) from session")
    frames = await _one("select count(*) from frame")
    return {"sessions": f"{sessions:,}", "frames": f"{frames:,}"}


async def ids() -> dict:
    """Re-read before every scene: the tour's own sessions only exist once the upload scene has run."""
    tour = [r for r in (await _rows(
        "select session_id::text from session where vehicle_id = :v order by created_at", v=VEHICLE))]
    a = tour[0] if tour else None
    frame = None
    if a:
        frame = await _one("select frame_id::text from frame where session_id = cast(:s as uuid) "
                           "order by ts_ns offset :o limit 1", s=a, o=FRAME_INDEX)
    india = await I.ids()
    return {"tour_a": a, "tour_b": tour[1] if len(tour) > 1 else None, "tour_frame": frame,
            "track": india["track"], "delhi": india["delhi"]}


async def _rows(sql: str, **kw) -> list:
    async with get_engine().connect() as c:
        return [r[0] for r in (await c.execute(text(sql), kw)).all()]


# ------------------------------------------------------------------ small helpers

async def point(d: Demo, locator, *, ms: int = 600, linger: int = 1200, dx: float = 0.5, dy: float = 0.5) -> None:
    """Move the pointer onto an element and rest there, the way a presenter points at something."""
    loc = locator.first
    try:
        await loc.wait_for(state="visible", timeout=8000)
    except Exception:  # noqa: BLE001  a pointer that has nothing to point at simply stays put
        return
    box = await loc.bounding_box()
    if box:
        await d.move(box["x"] + box["width"] * dx, box["y"] + box["height"] * dy, ms=ms)
        await d.wait(linger)


async def text_at(d: Demo, s: str, **kw) -> None:
    await point(d, d.page.get_by_text(s, exact=False), **kw)


async def fill(d: Demo, locator, value: str, *, delay: int = 55) -> None:
    await d.click(locator, ms=500, after=200)
    await d.page.keyboard.press("Control+a")
    await d.type(value, delay=delay)
    await d.wait(300)


async def pick_option(d: Demo, select, starts: str, hint: str = "") -> None:
    """Choose the option of a <select> whose visible text starts with `starts`."""
    sel = select.first
    value = await sel.evaluate("(el, s) => { const o = [...el.options].find(o => o.text.trim().startsWith(s));"
                               " return o ? o.value : null; }", starts)
    if value is None:
        raise RuntimeError(f"no option starting {starts!r}")
    # Pointed at, not clicked: a click opens the browser's own option list, which then stays drawn over
    # the page in a headless recording after the choice is made.
    await point(d, sel, ms=500, linger=250)
    await sel.select_option(value)
    await d.page.evaluate("([l, h]) => window.__demoKey && window.__demoKey(l, h)", [starts, hint])
    await d.wait(500)


# ------------------------------------------------------------------ getting started

async def home(d: Demo, i: dict) -> None:
    await text_at(d, "Waiting for a person", linger=2200)
    await text_at(d, "Waiting for review", linger=1600)
    await text_at(d, "Quick actions", linger=1400)
    await text_at(d, "Upload footage", linger=1000)
    await d.key("Control+k", "go to anything", label="Ctrl K", after=700)
    await d.type("datasets", delay=90)
    await d.wait(1600)
    await d.key("Escape", "close", label="Esc", after=500)


async def status(d: Demo, i: dict) -> None:
    await text_at(d, "Everything is working", linger=1500)
    await text_at(d, "Privacy blur models", linger=1300)
    await text_at(d, "Auto-label models", linger=1300)
    await d.click(d.page.get_by_role("button", name=re.compile("Check again")), after=2200)


# ------------------------------------------------------------------ setting up

async def drop_project(d: Demo, i: dict) -> str:
    async with get_engine().begin() as c:
        n = (await c.execute(text("delete from label_project where name = :n"), {"n": PROJECT})).rowcount
    return f"removed {n} project from earlier takes"


async def project(d: Demo, i: dict) -> None:
    await fill(d, d.page.get_by_placeholder("e.g. Pune pilot"), PROJECT)
    await pick_option(d, d.page.locator("select").filter(has=d.page.locator("option[value='members']")),
                      "Members only", "visible to")
    await d.click(d.page.get_by_role("button", name=re.compile(r"^\s*Create\s*$")), after=1500)
    await d.click(d.page.get_by_text(PROJECT, exact=True), after=1500)
    await text_at(d, "Who sees its sessions", linger=900)
    await text_at(d, "Generic driving", linger=1200)
    await text_at(d, "Indian roads", linger=1200)
    await point(d, d.page.get_by_text("Choose a teammate", exact=False), linger=1200)


async def verify_project(d: Demo, i: dict) -> str:
    vis = await _one("select visibility from label_project where name = :n", n=PROJECT)
    return f"project '{PROJECT}' saved, visible to {vis}"


async def team(d: Demo, i: dict) -> None:
    await fill(d, d.page.get_by_placeholder("e.g. priya"), "priya", delay=110)
    await pick_option(d, d.page.locator("select").first, "Reviewer", "role")
    await point(d, d.page.get_by_role("button", name=re.compile("Invite")), linger=1500)
    await text_at(d, "Sign out everywhere", linger=1200)


# ------------------------------------------------------------------ bringing data in

async def erase_tour(d: Demo, i: dict) -> str:
    """Erase the sessions an earlier take uploaded, so the tour always uploads into a clean slate."""
    from services.govern.retention import erase_session

    sids = await _rows("select session_id from session where vehicle_id = :v", v=VEHICLE)
    async with get_sessionmaker()() as db:
        for sid in sids:
            await erase_session(db, sid, reason="feature tour retake")
        await db.commit()
    return f"erased {len(sids)} sessions from earlier takes"


async def upload(d: Demo, i: dict) -> None:
    page = d.page
    await fill(d, page.locator("label:has-text('vehicle') input"), VEHICLE, delay=90)
    await text_at(d, "drag clips", linger=700)
    # What a drop does: hand the browser's file list to the page. The files are the two zips of frames.
    await page.locator("input[type=file][accept]").set_input_files([str(p) for p in UPLOADS])
    await d.wait(1500)
    await d.click(page.get_by_role("button", name=re.compile(r"^Import 2 files")), after=800)
    btn = page.get_by_role("button", name=re.compile(r"^autolabel \d+ session"))
    await btn.wait_for(state="visible", timeout=240_000)
    await d.wait(1200)
    await d.click(btn, after=2500)


async def verify_upload(d: Demo, i: dict) -> str:
    # The upload page labels a batch one session after another, and only while its tab is open; the
    # scene's tab closes after the first has started. The rest are started the same way the page would.
    for sid in await _rows("""select s.session_id::text from session s where s.vehicle_id = :v and not exists
                              (select 1 from autolabel_job j where j.session_id = s.session_id)""", v=VEHICLE):
        await d.page.evaluate(
            """async ([s, t]) => { await fetch('/api/autolabel/start', {method: 'POST',
                 headers: {authorization: 'Bearer ' + t, 'content-type': 'application/json'},
                 body: JSON.stringify({session_id: s, compute_target: 'local', only_unlabelled: false})}); }""",
            [sid, d.user["token"]])
    n = await _one("""select count(*) from frame f join session s using(session_id) where s.vehicle_id = :v""",
                   v=VEHICLE)
    jobs = await _one("""select count(*) from autolabel_job j join session s using(session_id)
                         where s.vehicle_id = :v""", v=VEHICLE)
    return f"{n} frames imported under {VEHICLE}, {jobs} auto-label jobs queued"


async def import_page(d: Demo, i: dict) -> None:
    await text_at(d, "drag .zip", linger=1200)
    await pick_option(d, d.page.locator("select").first, "yolo", "format")
    await d.wait(400)
    await text_at(d, "PII (faces/plates)", linger=1200)
    await text_at(d, "Migrate from another tool", linger=1600)


async def sessions(d: Demo, i: dict) -> None:
    cards = d.page.get_by_text(VEHICLE, exact=True)
    await point(d, cards, dx=0.5, linger=1500)
    await text_at(d, "reviewed", linger=1200)
    await point(d, d.page.get_by_text("Review", exact=True), linger=1300)
    await point(d, d.page.get_by_role("button", name="resume queue"), linger=1300)
    await point(d, d.page.get_by_role("button", name="open"), linger=1200)


async def jobs(d: Demo, i: dict) -> None:
    await d.click(d.page.get_by_role("button", name="autolabel", exact=True), after=1500)
    await point(d, d.page.get_by_text("Autolabel", exact=True), dx=0.5, linger=1500)
    await point(d, d.page.get_by_role("button", name="open"), linger=1500)
    await d.click(d.page.get_by_role("button", name="all", exact=True), after=1500)


# ------------------------------------------------------------------ labelling

async def wait_labelled(d: Demo, i: dict) -> str:
    """Before the editor scenes: wait until both auto-label jobs have finished. Never filmed."""
    for _ in range(360):
        rows = await _rows("""select j.status from autolabel_job j join session s using(session_id)
                              where s.vehicle_id = :v""", v=VEHICLE)
        if rows and all(r in ("done", "error") for r in rows):
            break
        await asyncio.sleep(5)
    n = await _one("select count(*) from object where frame_id = cast(:f as uuid)", f=i["tour_frame"])
    return f"auto-label jobs {rows}; {n} proposals on the frame"


async def _proposals(frame_id: str, *, everything: bool = False) -> list[dict]:
    async with get_engine().connect() as c:
        rows = (await c.execute(text("""
            select o.object_id::text, cl.name, o.bbox, o.state from object o
            join ontology_class cl on cl.id = o.class_id and cl.version = (select max(version) from ontology_class)
            where o.frame_id = cast(:f as uuid) and (:all or o.source <> 'human')"""),
            {"f": frame_id, "all": everything})).all()
    return [{"id": r[0], "cls": r[1], "box": list(r[2]), "state": r[3]} for r in rows]


def _w(o):
    return o["box"][2] - o["box"][0]


JUDGED: list[tuple[str, str, str, str | None]] = []


def _verdicts(props: list[dict]) -> list[tuple[dict, str, str | None]]:
    """Which proposals get which verdict, by rule, each rule checked by eye against this frame.

    The bus ahead is a bus. The "sedan" whose box runs off the bottom of the picture is the bonnet of the
    car the camera is in. The small "sedan" in the middle of the road is the olive jeep, an SUV.
    """
    out = []
    bus = [o for o in props if o["cls"] == "bus" and _w(o) > 150]
    if bus:
        out.append((max(bus, key=_w), "accept", None))
    bonnet = [o for o in props if o["box"][3] >= 1070 and _w(o) > 600]
    if bonnet:
        out.append((max(bonnet, key=_w), "reject", None))
    jeep = [o for o in props if o["cls"] in ("sedan", "hatchback", "car") and 880 < o["box"][0] < 940
            and o["box"][2] < 1000]
    if jeep:
        out.append((jeep[0], "accept", "suv"))
    return out


def _inside(o: dict) -> tuple[float, float]:
    x1, y1, x2, y2 = o["box"]
    if y2 >= 1070:                       # the bonnet: click its shiny middle, clear of every other box
        return ((x1 + x2) / 2, 900)
    return ((x1 + x2) / 2, y1 + (y2 - y1) * 0.6)


async def proposals(d: Demo, i: dict) -> None:
    await d.key("f", "fit to view", label="F", after=900)
    plan = _verdicts(await _proposals(i["tour_frame"]))
    JUDGED[:] = [(o["id"], o["cls"], verdict, cls) for o, verdict, cls in plan]
    if len(plan) < 3:
        raise RuntimeError(f"expected three proposals to judge, found {[(o['cls'], v) for o, v, _ in plan]}")
    for o, verdict, cls in plan:
        x, y = _inside(o)
        await d.key("v", "select", label="V", after=150)
        await d.click_img(x, y, ms=500)
        await d.wait(500)
        picked = await I.selected_id(d)
        if picked != o["id"][:8]:
            raise RuntimeError(f"expected to select {o['id'][:8]} ({o['cls']}), the panel shows {picked}")
        cx, cy = await d.img(x, y)
        if cls:
            await d.page.mouse.click(cx, cy, button="right")
            await d.click(d.page.get_by_role("menuitem", name=re.compile(r"^change class")), ms=300, after=300)
            await d.type(cls)
            await d.key("Enter", f"class: {cls}", label="Enter", after=500)
            await I.dismiss_fix_similar(d)
            await I.save(d)
        await d.page.mouse.click(cx, cy, button="right")
        await d.wait(500)
        await d.click(d.page.get_by_role("menuitem", name=re.compile(rf"^{verdict}\b")), ms=300, after=900)
    await I.save(d)


async def verify_proposals(d: Demo, i: dict) -> str:
    now = {o["id"]: o for o in await _proposals(i["tour_frame"], everything=True)}
    want = {"accept": "accepted", "reject": "rejected"}
    out = []
    for oid, was, verdict, cls in JUDGED:
        o = now.get(oid)
        ok = bool(o) and o["state"] == want[verdict] and o["cls"] == (cls or was)
        out.append(f"{was}->{o['cls'] if o else '?'} {o['state'] if o else '?'} {'ok' if ok else 'WRONG'}")
    return "; ".join(out)


async def draw(d: Demo, i: dict) -> None:
    # A street light the models left out, boxed by hand.
    await d.pick_class("light_pole")
    await I.draw_box(d, "light_pole", (990, 238, 1074, 568), ms=1000)
    await I.save(d)
    # The hedge on the median, from one click with AI assist.
    await d.pick_class("vegetation")
    await d.key("s", "AI assist: click", label="S", after=300)
    await d.click_img(1350, 612, ms=600)
    await d.wait(2600)
    before = (await I.panel(d))["count"]
    await d.key("Enter", "accept mask", label="Enter", after=300)
    await I.created(d, before, "the AI assist mask")
    await I.finish(d, "vegetation", (1350, 612))
    await I.save(d)
    await d.wait(600)
    await d.click(d.page.get_by_role("button", name="Confirm & next"), after=2500)


async def verify_draw(d: Demo, i: dict) -> str:
    rows = await I.human_objects(i["tour_frame"])
    return "hand-made on the frame: " + ", ".join(r[0] for r in rows)


async def clear_hand(d: Demo, i: dict) -> str:
    """Remove the hand-made objects of an earlier take from the tour frame (only that frame)."""
    # Only the classes this scene draws: a proposal a person accepted is stored as human-sourced too, and
    # those are the previous scene's verdicts, not this scene's leftovers.
    objs = await _rows("""select o.object_id::text from object o join ontology_class c on c.id = o.class_id
                          where o.frame_id = cast(:f as uuid) and o.source = 'human'
                          and c.name in ('light_pole', 'vegetation')""", f=i["tour_frame"])
    for oid in objs:
        await d.page.evaluate(
            "async ([p, t]) => { await fetch(p, {method: 'DELETE', headers: {authorization: 'Bearer ' + t}}); }",
            [f"/api/objects/{oid}", d.user["token"]])
    return f"cleared {len(objs)} hand-made objects from earlier takes"


async def track(d: Demo, i: dict) -> None:
    strip = d.page.get_by_text("timeline (", exact=False).first
    sb = await strip.bounding_box()
    if sb:
        await d.move(sb["x"] + 700, sb["y"] + 80, ms=900)
        await d.wait(1600)
    await d.click(d.page.get_by_placeholder("search class..."), ms=600, after=200)
    await d.type("cattle")
    await d.wait(500)
    relabel = d.page.locator("section", has_text="relabel entire track")
    await d.click(relabel.get_by_role("button", name=re.compile(r"^cattle\b")), ms=500, after=2500)
    await d.click(d.page.get_by_role("button", name=re.compile(r"^accept all \d+")), ms=600, after=2500)


async def verify_track(d: Demo, i: dict) -> str:
    async with get_engine().connect() as c:
        rows = (await c.execute(text("""
            select cl.name, o.state, count(*) from object o
            join ontology_class cl on cl.id = o.class_id and cl.version = (select max(version) from ontology_class)
            where o.track_id = cast(:t as uuid) group by 1, 2"""), {"t": i["track"]})).all()
    return "track now " + ", ".join(f"{n} {name} {state}" for name, state, n in rows)


# ------------------------------------------------------------------ reviewing

async def queue(d: Demo, i: dict) -> None:
    await point(d, d.page.get_by_text("Why this order?"), linger=1500)
    await point(d, d.page.get_by_text("Outline off box").first, linger=1200)
    sel = d.page.locator("select").filter(has=d.page.locator(f"option[value='{i['tour_a']}']"))
    await point(d, sel, ms=500, linger=250)
    await sel.first.select_option(i["tour_a"])
    await d.page.evaluate("([l, h]) => window.__demoKey && window.__demoKey(l, h)", ["bengaluru-drive-1", "session"])
    await d.wait(1500)
    await d.move(900, 420, ms=500)
    for _ in range(3):
        await d.key("j", "next", label="J", after=650)
    await d.key("k", "back", label="K", after=650)
    await point(d, d.page.get_by_role("button", name=re.compile("Review next")), linger=1400)


def _grid_judgement(o: dict) -> str:
    """The uncertain "bus" crops of the first drive, judged by eye on the frames themselves.

    The slivers just right of the city bus are a second bus further ahead: a bus. The box on the olive jeep,
    and the boxes along the bottom of the picture, which are the bus reflected in our own bonnet, are not.
    Anything else is left for a person to decide.
    """
    x1, y1, x2, _y2 = o["box"]
    if y1 >= 750 or (870 <= x1 <= 945 and x2 <= 1030):
        return "reject"
    if 760 <= x1 <= 840 and x2 <= 840:
        return "accept"
    return "leave"


async def grid(d: Demo, i: dict) -> None:
    queue = await d.api(f"/api/triage?session_id={i['tour_a']}&klass=bus&limit=1000")
    order = [r["object_id"] for r in queue]
    frames = await _rows("select frame_id::text from frame where session_id = cast(:s as uuid)", s=i["tour_a"])
    boxes = {o["id"]: o for f in frames for o in await _proposals(f)}
    tiles = d.page.locator('button[title^="bus"]')
    await tiles.first.wait_for(state="visible", timeout=15000)
    await d.wait(800)
    for verdict, key in (("reject", "r"), ("accept", "a")):
        group = [oid for oid in order if oid in boxes and _grid_judgement(boxes[oid]) == verdict]
        if not group:
            continue
        GRID_JUDGED.extend((oid, verdict) for oid in group)
        for k in sorted(order.index(oid) for oid in group):
            await d.click(tiles.nth(k), ms=320, after=200)
        await d.wait(700)
        await d.key(key, f"{verdict} {len(group)} at once", label=key.upper(), after=2000)
        order = [oid for oid in order if oid not in group]


GRID_JUDGED: list[tuple[str, str]] = []


async def verify_grid(d: Demo, i: dict) -> str:
    want = {"accept": "accepted", "reject": "rejected"}
    async with get_engine().connect() as c:
        now = dict((await c.execute(text("select object_id::text, state from object where object_id = any(cast(:i as uuid[]))"),
                                    {"i": [o for o, _ in GRID_JUDGED]})).all())
    ok = sum(1 for o, v in GRID_JUDGED if now.get(o) == want[v])
    rej = sum(1 for _, v in GRID_JUDGED if v == "reject")
    return f"{ok} of {len(GRID_JUDGED)} grid verdicts saved ({rej} rejects)"


async def errors(d: Demo, i: dict) -> None:
    await text_at(d, "Do the detectors find real errors?", linger=1400)
    await point(d, d.page.get_by_role("button", name=re.compile("^reanalyze")), linger=1200)
    await d.move(370, 420, ms=600)
    await d.wait(800)
    for _ in range(3):
        await d.key("ArrowRight", "next", label="→", after=500)
    await text_at(d, "why (recorded", linger=1200)


# ------------------------------------------------------------------ finding data

async def search(d: Demo, i: dict) -> None:
    box = d.page.get_by_placeholder(re.compile("^natural language"))
    await fill(d, box, "autorickshaw in heavy traffic", delay=60)
    await d.click(d.page.get_by_role("button", name="search", exact=True), after=500)
    await d.wait(9000)                                  # the text model and the ranking, on this machine
    await d.move(900, 600, ms=700)
    await d.page.mouse.wheel(0, 500)
    await d.wait(1800)


async def dataset_query(d: Demo, i: dict) -> None:
    box = d.page.locator("input, textarea").first
    await fill(d, box, "rain AND two_wheeler", delay=80)
    await d.wait(2500)
    await text_at(d, "FRAMES", linger=1200)
    await text_at(d, "class mix", linger=1400)
    await point(d, d.page.get_by_role("button", name="build shards"), linger=1400)
    await text_at(d, "labelox.load", linger=1400)


# ------------------------------------------------------------------ delivering and training

async def export(d: Demo, i: dict) -> None:
    inputs = d.page.locator("input")
    await fill(d, inputs.nth(0), "bengaluru-pilot-v1", delay=60)
    for fmt in ("yolo", "openlabel", "nuscenes"):
        await point(d, d.page.get_by_text(fmt, exact=True), linger=350)
    await d.click(d.page.get_by_text("yolo", exact=True), after=500)
    val = inputs.nth(2)
    await fill(d, val, "10", delay=120)
    await fill(d, inputs.nth(3), "10", delay=120)
    await point(d, d.page.get_by_role("button", name="export", exact=True), linger=1400)
    await text_at(d, "Sealed datasets", linger=1000)


async def training(d: Demo, i: dict) -> None:
    await fill(d, d.page.get_by_placeholder(re.compile("^a name for this model line")), "bengaluru-vehicles")
    await fill(d, d.page.get_by_placeholder(re.compile("^blank trains on every class")),
               "bus, autorickshaw, motorcycle", delay=45)
    await point(d, d.page.get_by_text("promote if gate passes"), linger=1200)
    await point(d, d.page.get_by_role("button", name="Queue training job"), linger=1500)
    await text_at(d, "JOBS (", linger=800)
    await d.move(1100, 700, ms=700)
    await d.wait(1200)


# ------------------------------------------------------------------ getting answers

async def assistant(d: Demo, i: dict) -> None:
    await d.click(d.page.get_by_role("button", name="Which drives do we have from Delhi?"), after=500)
    try:
        await d.page.get_by_text("INDIA-DEL-01", exact=False).first.wait_for(state="visible", timeout=40000)
    except Exception:  # noqa: BLE001  a slow model is filmed as slow; the scene notes what it got
        pass
    await d.wait(2500)
    await text_at(d, "Proposals", linger=1500)


S = Scene

SCENES = [
    # ---------------------------------------------------------------- getting started
    S("home", "Getting started", "Home", "/",
      "This is LabeloxAV, a data engine for driving data, and this is a tour of the features you will use "
      "every day, and how to use each one. Everything here is the live product. "
      "Home is where the day starts: what is waiting for a person, the whole corpus at a glance, and quick "
      "actions. To jump anywhere, press Control K and type the page you want.",
      run=home, settle_ms=2500),
    S("status", "Getting started", "System check", "/settings",
      "Settings opens on a system check: the database, storage, the GPU, the privacy blur models and the "
      "auto labelling models. If something is not right, it says how to fix it. Check again runs it on "
      "demand.",
      run=status, settle_ms=2500),

    # ---------------------------------------------------------------- setting up
    S("project", "Setting up", "Create a project", "/settings/projects",
      "Work is organised in projects. Give the project a name, choose who can see it, the whole team or "
      "members only, and press Create. Then pick its class set: every Indian road class, a generic driving "
      "set, or a custom list. Add teammates as members from the same page.",
      run=project, verify=verify_project, before=drop_project, settle_ms=2000),
    S("team", "Setting up", "Invite the team", "/settings/team",
      "To add a person, open Team, type the name they will sign in with and choose a role: annotator, "
      "reviewer or administrator. Invite gives you a link to send them, and they choose their own password. "
      "Anyone can be signed out everywhere with one click.",
      run=team, settle_ms=2000),

    # ---------------------------------------------------------------- bringing data in
    S("upload", "Bringing data in", "Upload footage", "/annotate/new",
      "Now some real footage: two short drives through Bengaluru, as folders of dashcam frames. "
      "Open Upload, name the vehicle, and drop the files in. Each file becomes its own session. "
      "Faces and number plates are blurred on every frame before anything is stored. "
      "When the imports finish, one click starts the models labelling both drives.",
      run=upload, verify=verify_upload, before=erase_tour, settle_ms=2000, hold_s=1.5),
    S("import", "Bringing data in", "Import a labelled dataset", "/import",
      "If you already have labels, use Import instead. Choose the format, COCO, YOLO, KITTI, nuScenes and "
      "more, and drop the archive in. Labels from tools like CVAT or Labelbox come in through Migrate, which "
      "maps their class names onto yours first.",
      run=import_page, settle_ms=2000),
    S("sessions", "Bringing data in", "Sessions", "/annotations",
      "Every drive is a session, and the two new ones are at the top. Each card shows its frames and objects, "
      "how much a person has reviewed, and where its labels stand. Open goes to the editor, and resume queue "
      "carries on reviewing where you stopped.",
      run=sessions, settle_ms=2500),
    S("jobs", "Bringing data in", "Jobs", "/jobs",
      "Auto labelling runs as a job. Jobs lists every import, labelling run, export and training run, with "
      "its progress, and a job that fails says why.",
      run=jobs, settle_ms=2000),

    # ---------------------------------------------------------------- labelling
    S("proposals", "Labelling", "Check the proposals", "/frame/{tour_frame}",
      "Open a frame and the models' proposals are already on it. Your job is to check them. "
      "Right click a proposal to accept it, reject it, or change its class. The bus is right. The sedan "
      "along the bottom is the bonnet of our own car, so it is rejected. And the sedan in the middle of the "
      "road is really a jeep, so its class is changed before it is accepted.",
      ready=EDITOR, run=proposals, verify=verify_proposals, before=wait_labelled, settle_ms=2000),
    S("draw", "Labelling", "Add what was missed", "/frame/{tour_frame}",
      "Add what the models missed. Press B and drag a box, here around a street light. "
      "For an irregular shape, press S for AI assist and click once: the hedge on the median becomes a mask. "
      "When the frame is right, Confirm and next records it and moves on.",
      ready=EDITOR, run=draw, verify=verify_draw, before=clear_hand, settle_ms=2000),
    S("track", "Labelling", "Fix a whole track", "/track/{track}",
      "Objects in video are linked into tracks. This cow in Delhi was followed through twenty eight seconds, "
      "and in one frame the model called it a goods vehicle. Relabel the track once and every frame is "
      "fixed, then accept all of it in one click.",
      ready="text=relabel entire track", run=track, verify=verify_track, before=I.restore_for("track"),
      settle_ms=2000),

    # ---------------------------------------------------------------- reviewing
    S("queue", "Reviewing", "My queue", "/queue",
      "My queue holds the objects waiting for a decision, likeliest wrong first, and each row says why it "
      "needs you. Filter it to one session, move with J and K, and accept with A or reject with R. "
      "Review next opens the next item.",
      run=queue, settle_ms=3000),
    S("grid", "Reviewing", "Review many at once", "/review/grid?session={tour_a}&class=bus",
      "For speed, the crop grid shows many objects together: here, every bus the models were unsure of in "
      "the first drive. Select crops with a click and judge them with one key. The olive jeep and the bus's "
      "reflection on our own bonnet are rejected together with R, and the second bus further ahead is "
      "accepted with A.",
      ready='button[title^="bus"]', run=grid, verify=verify_grid, settle_ms=2500),
    S("errors", "Reviewing", "Label errors", "/quality/errors",
      "Label errors collects labels the system suspects are wrong, from several detectors. Move with the "
      "arrow keys and confirm or dismiss each one with a key, and the page measures which detectors are "
      "actually right.",
      run=errors, settle_ms=3000),

    # ---------------------------------------------------------------- finding data
    S("search", "Finding data", "Search in plain words", "/search",
      "To find data, describe it. Search takes plain words, like autorickshaw in heavy traffic, and returns "
      "the frames that look most like it. You can also search with an example image.",
      run=search, settle_ms=2000),
    S("dataset_query", "Finding data", "Describe a dataset", "/datasets/query",
      "Dataset query describes the data you want, like rain and two wheelers, and counts it against the "
      "corpus as you type: frames, objects and the class mix. The same query loads from Python.",
      run=dataset_query, settle_ms=2500),

    # ---------------------------------------------------------------- delivering and training
    S("export", "Delivering and training", "Export a dataset", "/datasets",
      "To deliver, open Datasets. Name the version, choose which label states to include, pick the formats, "
      "and set a validation and test split. Export seals it, so a version never changes after it ships.",
      run=export, settle_ms=2000),
    S("training", "Delivering and training", "Train a model", "/training",
      "To train, name the model line, choose the classes, and queue the job on the GPU. A trained model "
      "only replaces the current one if it passes the promotion gate, and every run keeps its history.",
      run=training, settle_ms=2000),

    # ---------------------------------------------------------------- getting answers
    S("assistant", "Getting answers", "Ask the assistant", "/assistant",
      "Finally, the assistant answers questions about your data in plain words. It runs on a model on this "
      "machine, so nothing leaves it. Anything that would change labels becomes a proposal a reviewer "
      "approves. That is LabeloxAV, from raw footage to a model you can trust.",
      run=assistant, settle_ms=2500, hold_s=2.0),
]

__all__ = ["CHAPTERS", "SCENES", "facts", "ids"]
