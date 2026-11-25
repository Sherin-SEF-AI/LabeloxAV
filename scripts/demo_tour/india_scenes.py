"""Scenes for the hands-on demo: real Indian roads seen from the driver's seat, labelled in the product.

The footage:
  Bengaluru: the operator's own dashcam drive, 24 consecutive frames at three a second, following a city
    bus past an auto-rickshaw and a motorcycle. Two copies are imported: BLR-DEMO-01, which no model has
    touched and is annotated by hand, and BLR-DEMO-02, which the auto-labeller is run on. The number plates
    in both copies were blurred before import (they were 25 pixels wide and the redaction pass missed them).
  Delhi: "Stray Cattle in Lutyens Delhi", by User:Fowler&fowler, Wikimedia Commons, CC BY-SA 3.0, 2017.
    Filmed from a car at a roundabout while a herd of cattle crosses in front of it.

An earlier cut of this film annotated an elevated view of a junction in Cuttack. A driving product is
better shown from where a vehicle's camera sits, so the hand-annotation scenes moved to the dashcam drive.

Each scene is a route, a sentence or two of narration, and the mouse and keyboard work done on that page.
Image coordinates are in the frame's own pixels (1920 wide) and are converted to the screen at the moment
of each click, so they stay right when the editor zooms.
"""

from __future__ import annotations

import json
import re
import urllib.request

from annotate import OUT, Demo, Scene
from sqlalchemy import text

from db.session import get_engine

ROUTE = "india-demo-2026"
CUTTACK = "INDIA-CTC-01"
DELHI = "INDIA-DEL-01"
# The dashcam frames, twice: a copy no model has run on, annotated by hand, and a copy the auto-labeller
# is run on, shown as what the models make of the same instant.
MANUAL = "BLR-DEMO-01"
MACHINE = "BLR-DEMO-02"
# Which of the 24 frames: the bus, the auto-rickshaw, a motorcycle and a jeep are all well inside the frame.
MANUAL_INDEX, SHARED_INDEX = 4, 9
# The second person in the two-people scene: a reviewer who takes the frame over through the same API the
# editor uses. Created on first use; she is a row in app_user and nothing else.
COLLEAGUE = "meera"
EDITOR = "div.konvajs-content canvas"

CHAPTERS = ["Real Indian roads", "Manual annotation", "AI assist", "Beyond boxes", "Checking the machine",
            "Across frames", "Working together", "Delivering the data"]


async def _one(sql: str, **kw):
    async with get_engine().connect() as c:
        return (await c.execute(text(sql), kw)).scalar()


async def facts() -> dict:
    """Every number the narration quotes, read from the database now."""
    blr = await _one("""select count(*) from frame f join session s using(session_id) where s.vehicle_id=:v""", v=MANUAL)
    dl = await _one("""select count(*) from frame f join session s using(session_id) where s.vehicle_id=:v""", v=DELHI)
    faces = await _one("""select coalesce(sum(p.n_faces), 0) from pii_audit p join session s using(session_id)
                           where s.vehicle_id=:v""", v=DELHI)
    plates = await _one("""select coalesce(sum(p.n_plates), 0) from pii_audit p join session s using(session_id)
                            where s.vehicle_id=:v""", v=DELHI)
    twin = await _one("""
        select f.frame_id from frame f join session s using(session_id)
        where s.vehicle_id=:v order by f.ts_ns offset :o limit 1""", v=MACHINE, o=MANUAL_INDEX)
    proposals = await _one("select count(*) from object where frame_id=:f and source <> 'human'", f=twin)
    machine_objects = await _one("""select count(*) from object o join frame f using(frame_id)
                                    join session s using(session_id) where s.vehicle_id=:v""", v=MACHINE)
    return {"blr_frames": f"{blr:,}", "delhi_frames": f"{dl:,}", "faces": f"{faces:,}", "plates": f"{plates:,}",
            "proposals": f"{proposals:,}", "machine_objects": f"{machine_objects:,}"}


async def ids() -> dict:
    """Rows the scenes open, chosen by rule rather than pasted in, so a rebuilt corpus still resolves."""
    # The 198th frame of the Cuttack clip: a full city bus crosses the junction, two auto-rickshaws queue in
    # the foreground, and it lies beyond the forty frames the auto-labeller was run on, so it starts empty.
    manual = await _one("""
        select f.frame_id::text from frame f join session s using(session_id)
        where s.vehicle_id=:v order by f.ts_ns offset :o limit 1""", v=MANUAL, o=MANUAL_INDEX)
    # The frame after it, which the copied annotations are pasted onto, and one further on that two people
    # open at once. Both are in the unlabelled copy, so they start empty too.
    manual_next = await _one("""
        select f.frame_id::text from frame f join session s using(session_id)
        where s.vehicle_id=:v order by f.ts_ns offset :o limit 1""", v=MANUAL, o=MANUAL_INDEX + 1)
    shared = await _one("""
        select f.frame_id::text from frame f join session s using(session_id)
        where s.vehicle_id=:v order by f.ts_ns offset :o limit 1""", v=MANUAL, o=SHARED_INDEX)
    # The same instant in the copy the models labelled.
    machine = await _one("""
        select f.frame_id::text from frame f join session s using(session_id)
        where s.vehicle_id=:v order by f.ts_ns offset :o limit 1""", v=MACHINE, o=MANUAL_INDEX)
    # The 42nd frame of the Delhi clip: seven cattle proposals, and among the other proposals several that
    # are wrong in ways worth showing, a reflection in the car window read as a pedestrian among them.
    review = await _one("""
        select f.frame_id::text from frame f join session s using(session_id)
        where s.vehicle_id=:v order by f.ts_ns offset 41 limit 1""", v=DELHI)
    # The longest cattle track in the Delhi clip: one white cow followed through 28 seconds.
    track = await _one("""
        select o.track_id::text from object o join frame f using(frame_id) join session s using(session_id)
        join ontology_class c on c.id = o.class_id and c.version = (select max(version) from ontology_class)
        where s.vehicle_id=:v and c.name='cattle' and o.track_id is not null
        group by o.track_id order by count(*) desc, o.track_id limit 1""", v=DELHI)
    return {"manual_frame": manual, "manual_next": manual_next, "shared_frame": shared,
            "machine_frame": machine, "review_frame": review, "track": track,
            "cuttack": await _one("select session_id::text from session where vehicle_id=:v", v=CUTTACK),
            "delhi": await _one("select session_id::text from session where vehicle_id=:v", v=DELHI)}


async def human_objects(frame_id: str) -> list[tuple[str, str, list[float]]]:
    """(class, state, bbox) of every hand-drawn object on a frame, read from the database."""
    async with get_engine().connect() as c:
        rows = (await c.execute(text("""
            select c.name, o.state, o.bbox from object o
            join ontology_class c on c.id = o.class_id
             and c.version = (select max(version) from ontology_class)
            where o.frame_id = cast(:f as uuid) and o.source = 'human' order by o.created_at"""),
            {"f": frame_id})).all()
    return [(r[0], r[1], list(r[2])) for r in rows]


def _count_human(cls: str | None = None):
    async def verify(d: Demo, i: dict) -> str:
        rows = await human_objects(i["manual_frame"])
        hits = [r for r in rows if cls is None or r[0] == cls]
        return f"{len(hits)} hand-drawn {cls or 'objects'} saved on the frame"
    return verify


async def reset(d: Demo, i: dict) -> str:
    """Remove earlier takes' hand-made annotations from the demo frame, so every take starts empty.

    Deleted through the API, the way the editor deletes, so the audit trail records it. Only this one frame
    is touched: its hand-drawn objects and its hand-drawn adverse regions. The machine's labels elsewhere
    are never changed.
    """
    frames = [i["manual_frame"], i["manual_next"], i["shared_frame"]]
    async with get_engine().connect() as c:
        objs = [r[0] for r in (await c.execute(text(
            "select object_id::text from object where frame_id = any(cast(:f as uuid[])) and source = 'human'"),
            {"f": frames})).all()]
        regions = [r[0] for r in (await c.execute(text(
            "select region_id::text from adverse_region where frame_id = cast(:f as uuid) and source = 'human'"),
            {"f": i["manual_frame"]})).all()]
    for path in [f"/api/objects/{o}" for o in objs] + [f"/api/adverse/{r}" for r in regions]:
        await d.page.evaluate(
            "async ([p, t]) => { await fetch(p, {method: 'DELETE', headers: {authorization: 'Bearer ' + t}}); }",
            [path, d.user["token"]])
    return f"cleared {len(objs)} objects and {len(regions)} adverse regions from earlier takes"


async def _adverse_saved(d: Demo, i: dict) -> str:
    n = await _one("select count(*) from adverse_region where frame_id = cast(:f as uuid) and condition = 'shadow'",
                   f=i["manual_frame"])
    return f"{n} shadow region saved on the frame"


PANEL = """() => {
  const b = document.querySelector('button[aria-label="collapse panel"]');
  if (!b) return null;
  const spans = b.parentElement.querySelectorAll(':scope > span');
  return {name: (spans[0] && spans[0].textContent || '').trim(), count: parseInt(spans[1] && spans[1].textContent || '0', 10)};
}"""


# "saved" once the panel shows no unsaved edits and no save is in flight anywhere on the page.
SAVE_STATE = """() => {
  const b = document.querySelector('button[aria-label="collapse panel"]');
  const spans = b ? b.parentElement.querySelectorAll(':scope > span') : [];
  const panel = spans[2] ? spans[2].textContent.trim() : '';
  const busy = /saving/.test(document.body.innerText);
  return busy ? 'saving' : panel;
}"""


async def panel(d: Demo) -> dict:
    """The selected object's class (or "Properties" when nothing is selected) and the frame's object count."""
    return await d.page.evaluate(PANEL) or {"name": "", "count": 0}


async def settle(d: Demo, *, quiet_ms: int = 700, limit_ms: int = 4000) -> dict:
    """Wait until the panel stops changing, which is when the classifier's answer has landed."""
    last, stable, waited = await panel(d), 0, 0
    while waited < limit_ms and stable < quiet_ms:
        await d.wait(100)
        waited += 100
        now = await panel(d)
        stable = stable + 100 if now == last else 0
        last = now
    return last


async def set_class(d: Demo, cls: str) -> None:
    """Relabel the selected object, only if it is wrong, and close the fix-similar offer that follows."""
    if (await panel(d))["name"] == cls:
        return
    await d.pick_class(cls)
    await dismiss_fix_similar(d)


async def finish(d: Demo, cls: str, at: tuple[float, float]) -> None:
    """Make sure the object just created is saved as `cls`, then deselect it.

    The editor runs a classifier on every new object and replaces the class being painted with its own guess
    when that guess is at least 15% confident. It is often right; when it is wrong the fix is the one a person
    makes: select the object, pick the class. The editor leaves a new box selected but not a new mask, so the
    object is selected first when nothing is. Deselecting afterwards matters because the class picker
    relabels the selection as well as setting the class for new objects.
    """
    now = await settle(d)
    if now["name"] == "Properties":
        await d.key("v", "select", label="V", after=200)
        await d.click_img(*at, ms=350)
        await settle(d, limit_ms=1500)
    await set_class(d, cls)
    await deselect(d)


async def deselect(d: Demo) -> None:
    """Clear the selection, and check that it cleared.

    Escape clears an in-progress item before it clears the selection, so one press is not always enough,
    and a selection that survives gets relabelled by the next class pick. The object list's own "none"
    button is the last resort.
    """
    for _ in range(2):
        await d.key("Escape", "deselect", label="Esc", after=250)
        if (await panel(d))["name"] == "Properties":
            return
    await d.click(d.page.get_by_role("button", name="none", exact=True), ms=350, after=300)
    if (await panel(d))["name"] != "Properties":
        raise RuntimeError("could not clear the selection")


async def created(d: Demo, before: int, what: str) -> None:
    after = await settle(d)
    if after["count"] != before + 1:
        raise RuntimeError(f"{what} did not add an object ({before} -> {after['count']})")


async def draw_box(d: Demo, cls: str, box: tuple[float, float, float, float], *, ms: int = 900) -> None:
    """Drag a box, check it was created, and make sure it is saved as `cls`."""
    before = (await panel(d))["count"]
    x1, y1, x2, y2 = box
    await d.key("b", "box", label="B", after=200)
    await d.drag_img(x1, y1, x2, y2, ms=ms)
    await created(d, before, f"drawing {cls}")
    await finish(d, cls, ((x1 + x2) / 2, (y1 + y2) / 2))


async def dismiss_fix_similar(d: Demo, *, linger: int = 0) -> bool:
    """Close the fix-similar dialog a class correction opens, after `linger` ms on screen.

    Correcting a class offers to apply the same correction to visually similar objects across the corpus.
    It opens over the canvas and swallows the next drag until it is closed. It can take several seconds to
    appear, because it first searches the embeddings for similar objects.
    """
    dialog = d.page.get_by_text("you corrected", exact=False)
    try:
        await dialog.first.wait_for(state="visible", timeout=8000)
    except Exception:  # noqa: BLE001  the correction found nothing to offer, so no dialog opened
        return False
    if linger:
        await d.wait(linger)
    # Its cancel button, not Escape: the dialog does not take focus, so Escape goes to the editor and the
    # dialog stays up until the next click lands on its backdrop, which is how the first takes lost a box.
    await d.click(d.page.get_by_role("button", name="cancel", exact=True), ms=350, after=400)
    return True


async def save(d: Demo, *, timeout_ms: int = 20000) -> None:
    """Ctrl S, then wait until the editor says the frame is saved.

    Not a fixed pause. While the browser is also encoding video a save takes several times longer than in a
    rehearsal, and the first recorded take closed the page on "saving 1 object": the box was on screen and
    never reached the database.
    """
    await d.key("Control+s", "save", label="Ctrl S", after=400)
    waited = 0
    while waited < timeout_ms:
        status = await d.page.evaluate(SAVE_STATE)
        if status == "saved":
            await d.wait(300)
            return
        await d.wait(200)
        waited += 200
    raise RuntimeError(f"the frame did not finish saving within {timeout_ms} ms ({status})")


def _has(kind: str):
    """Verify that a hand-drawn object carrying `kind` geometry (keypoints, polyline, amodal) was saved."""
    col = {"keypoints": "keypoints", "polyline": "polyline", "amodal": "bbox_amodal"}[kind]

    async def verify(d: Demo, i: dict) -> str:
        # JSON columns can hold a JSON null as well as an SQL null, so both are excluded.
        n = await _one(f"select count(*) from object where frame_id = cast(:f as uuid) and source = 'human' "
                       f"and {col} is not null and {col}::text <> 'null'", f=i["manual_frame"])
        return f"{n} hand-drawn objects with {kind} saved on the frame"
    return verify


# ------------------------------------------------------------------ actions

async def open_editor(d: Demo, i: dict) -> None:
    await d.key("f", "fit to view", label="F")
    await d.wait(800)


async def box_bus(d: Demo, i: dict) -> None:
    await d.pick_class("bus")
    await draw_box(d, "bus", (621, 430, 816, 647), ms=1100)
    await save(d)


async def box_autos(d: Demo, i: dict) -> None:
    await d.pick_class("autorickshaw")
    await draw_box(d, "autorickshaw", (156, 546, 386, 803), ms=1000)
    await save(d)


async def fix_class(d: Demo, cls: str, at: tuple[float, float]) -> None:
    """Select the object under an image point and set its class, closing the fix-similar offer if it opens."""
    await d.key("v", "select", label="V", after=200)
    await d.click_img(*at, ms=350)
    await settle(d, limit_ms=1500)
    await set_class(d, cls)
    await deselect(d)


async def box_more(d: Demo, i: dict) -> None:
    await d.pick_class("suv")
    # Zoom about the jeep: it is 76 pixels wide, and a box drawn at fit-to-view is a guess at its edges.
    await d.zoom_at(945, 572, 8)
    await draw_box(d, "suv", (907, 535, 983, 609), ms=800)
    await d.key("f", "fit to view", label="F", after=500)
    await d.pick_class("sedan")
    await draw_box(d, "sedan", (1619, 554, 1834, 673))
    await save(d)


async def polygon(d: Demo, i: dict) -> None:
    await d.pick_class("median")
    await d.key("g", "polygon", label="G", after=300)
    # The planted median, clockwise from its near end on the road side.
    pts = [(1036, 568), (1330, 588), (1640, 612), (1760, 640), (1750, 700), (1560, 690), (1300, 640), (1050, 596)]
    for x, y in pts:
        await d.click_img(x, y, ms=320, pause=120)
    await d.dblclick_img(1036, 584)
    await d.wait(900)
    await fix_class(d, "median", (1400, 630))
    await save(d)


async def edit(d: Demo, i: dict) -> None:
    """Resize from a handle and undo it, then copy, paste and delete the copy."""
    await d.key("v", "select", label="V", after=200)
    await d.click_img(718, 500, ms=400)
    await d.wait(400)
    await d.drag_img(816, 647, 838, 662, ms=600)
    await d.wait(700)
    await d.key("Control+z", "undo", label="Ctrl Z", after=900)
    await d.click_img(718, 500, ms=300)
    await d.key("Control+c", "copy", label="Ctrl C", after=400)
    await d.key("Control+v", "paste", label="Ctrl V", after=900)
    await d.key("Delete", "delete the copy", label="Del", after=900)
    await save(d)
    bus = [o for o in await human_objects(i["manual_frame"]) if o[0] == "bus"]
    if bus and abs(bus[0][2][2] - 816) > 3:
        raise RuntimeError(f"the undo did not restore the bus box: {bus[0][2]}")


async def sam_point(d: Demo, i: dict) -> None:
    await d.pick_class("motorcycle")
    await d.zoom_at(546, 640, 6)
    await d.key("s", "SAM point", label="S", after=300)
    await d.click_img(546, 662, ms=500)
    await d.wait(2500)                                    # the mask comes back from the GPU model
    before = (await panel(d))["count"]
    await d.key("Enter", "accept mask", label="Enter", after=300)
    await created(d, before, "the SAM point mask")
    await finish(d, "motorcycle", (546, 662))
    await d.key("f", "fit to view", label="F", after=400)
    await save(d)


async def sam_box(d: Demo, i: dict) -> None:
    await d.pick_class("truck")
    await d.key("m", "SAM box", label="M", after=300)
    await d.drag_img(3, 505, 118, 733, ms=800)
    await d.wait(2500)
    before = (await panel(d))["count"]
    await d.key("Enter", "accept mask", label="Enter", after=300)
    await created(d, before, "the SAM box mask")
    await finish(d, "truck", (55, 620))
    await save(d)


async def wand(d: Demo, i: dict) -> None:
    await d.pick_class("rider")
    await d.zoom_at(546, 620, 6)
    await d.key("w", "magic wand", label="W", after=300)
    before = (await panel(d))["count"]
    await d.click_img(547, 592, ms=500)
    await d.wait(2500)
    await created(d, before, "the wand")
    await finish(d, "rider", (547, 590))
    await d.key("f", "fit to view", label="F", after=400)
    await save(d)


async def brush(d: Demo, i: dict) -> None:
    await d.key("v", "select", label="V", after=200)
    await d.zoom_at(546, 630, 7)
    await d.click_img(547, 590, ms=400)
    await settle(d, limit_ms=1500)
    if (await panel(d))["name"] != "rider":
        raise RuntimeError("the brush scene did not select the rider")
    await d.key("p", "brush", label="P", after=300)
    await d.stroke_img([(534, 600), (540, 612), (548, 620), (556, 612), (561, 600)], ms_per_seg=220)
    await d.wait(1200)
    await d.key("e", "eraser", label="E", after=300)
    await d.stroke_img([(572, 584), (576, 596), (578, 608)], ms_per_seg=220)
    await d.wait(1200)
    await d.key("f", "fit to view", label="F", after=400)
    await save(d)


async def amodal(d: Demo, i: dict) -> None:
    await d.key("v", "select", label="V", after=200)
    await d.click_img(1690, 600, ms=400)
    await settle(d, limit_ms=1500)
    if (await panel(d))["name"] != "sedan":
        raise RuntimeError("the amodal scene did not select the car behind the hedge")
    await d.key("k", "whole extent", label="K", after=300)
    await d.drag_img(1604, 548, 1906, 730, ms=900)
    await d.wait(700)
    await save(d)


async def polyline(d: Demo, i: dict) -> None:
    await d.pick_class("curb")
    await d.key("l", "polyline", label="L", after=300)
    for x, y in [(1000, 600), (1210, 655), (1440, 716)]:
        await d.click_img(x, y, ms=380, pause=150)
    await d.dblclick_img(1700, 790)
    await d.wait(900)
    await save(d)


async def keypoints(d: Demo, i: dict) -> None:
    await d.key("Shift+5", "pose mode", label="Shift 5", after=700)
    await d.pick_class("rider")
    await d.zoom_at(546, 630, 11)
    await d.key("k", "keypoint", label="K", after=300)
    # The rider, seen from behind: head, shoulders, arms forward on the bars, hips, legs on the pegs.
    skeleton = [(547, 578), (550, 576), (544, 576), (553, 579), (541, 579), (563, 597), (531, 597),
                (568, 617), (526, 617), (566, 634), (528, 634), (557, 641), (537, 641), (562, 662),
                (532, 662), (560, 683), (534, 683)]
    # The seventeenth point completes the skeleton and creates the object, so no Enter is needed.
    for x, y in skeleton:
        await d.click_img(x, y, ms=220, pause=90)
    await d.wait(900)
    await d.key("f", "fit to view", label="F", after=400)
    await save(d)


async def attributes(d: Demo, i: dict) -> None:
    await d.key("v", "select", label="V", after=200)
    await d.click_img(1690, 600, ms=400)
    await d.wait(800)
    await d.select_option(d.page.locator('label:has(> span:text-is("occlusion")) select'), "50", "occlusion 50%")
    await d.wait(600)
    await save(d)


async def measure(d: Demo, i: dict) -> None:
    await d.key("r", "measure", label="R", after=300)
    await d.drag_img(386, 720, 655, 650, ms=1300)
    await d.wait(1800)


async def adverse(d: Demo, i: dict) -> None:
    await d.key("d", "adverse region", label="D", after=400)
    await d.select_option(d.page.get_by_title("adverse condition to tag"), "shadow", "condition")
    # The carriageway under the trees on the left, clockwise, so the outline never crosses itself.
    for x, y in [(400, 690), (610, 650), (640, 720), (470, 800)]:
        await d.click_img(x, y, ms=380, pause=150)
    await d.dblclick_img(380, 790)
    await d.wait(900)
    await save(d)


async def confirm(d: Demo, i: dict) -> None:
    await d.key("v", "select", label="V", after=200)
    await d.key("Control+a", "select all", label="Ctrl A", after=900)
    await d.click(d.page.get_by_role("button", name="Confirm & next"), after=2500)


# ------------------------------------------------------------------ reviewing the machine
#
# Every verdict below was decided by looking at the object first, not by the script. Each is keyed by
# object id, and the scene checks the object on screen is the one that was judged before it presses a key,
# so a reordered queue can never turn a considered verdict into a blind one.

FRAME_VERDICTS = [
    # (object id, a point to select it by, a point to right-click it by, verdict, new class). Both points lie
    # inside the object's outline and inside no smaller object. They differ only for the small "hoarding",
    # where the selected object's label chip covers the middle and a click on the chip is not a canvas click.
    ("3003e9c3-ebef-44f8-b9bb-efc9e3afc9e1", (1100, 615), (1100, 615), "accept", None),  # calf crossing
    ("9e7a4eb2-3842-40b3-be77-3a539e459a72", (850, 590), (850, 590), "accept", None),    # white cow at the kerb
    ("d01300e3-7087-4b1d-bcc0-b7464440f506", (1760, 700), (1760, 700), "reject", None),  # legs reflected in glass
    ("ce139573-9ac3-47ef-8621-610d58ac8602", (1150, 540), (1150, 540), "reject", None),  # "pedestrian" on the calf
    ("98e632d3-5e26-46ff-bd83-4bde0952ac9b", (1446, 500), (1438, 514), "reject", None),  # "hoarding", the car window
    ("b319aba4-2846-4566-baa1-028a340f995e", (1450, 570), (1450, 570), "accept", "mpv"), # an MPV proposed as a sedan
    ("e0056534-dd80-4ece-ac88-b48d8ace33a9", (150, 950), (150, 950), "accept", None),   # the sedan alongside
]

RAPID_VERDICTS = {
    "a872e72d-99cc-4d23-b88b-0a1d5e4adc2b": ("reclassify", "motorcycle"),  # two people on a motorcycle
    "61e491e5-81ae-44b2-9401-cf742e4b4c1e": ("accept", None),              # a cow among the plants
    "13020409-1d96-46b3-b503-9f886f291f2f": ("accept", None),              # cows at the kerb
    "af3fe8db-e5b4-4093-b9ae-0a22277ee17e": ("accept", None),              # a cow's head
    "6f4569a0-f9f9-4782-93b6-c3dda9efc4b4": ("accept", None),                                          # a cow behind the plants
    "8242fa6f-87a4-4116-959d-89348f8ea0de": ("accept", None),                                          # a cow's head
    "8748db05-2d19-4837-bc95-29622e9b960a": ("skip", None),                                            # one box over two cows
    "acb8e6a6-0a33-4810-b710-2e256afc1f8e": ("skip", None),                                            # too small to tell
    "a71db0b8-3858-4e67-979e-eda0e55f62b3": ("skip", None),                # too small to tell
    # The herd reflected in the car's side window, found again and again as cattle.
    "3703a6ba-da23-4cbb-8161-e81b499f7344": ("reject", None),
    "919beb00-58a2-4b00-84cc-2501fe24d75f": ("reject", None),
    "09715908-7cfe-4042-8010-6954a9a1f688": ("reject", None),
    "12e8fed3-c1d6-42ef-8b60-a2f656291bd1": ("reject", None),
    "23deaee3-e511-4d27-b255-4ab9b9a8bdc7": ("reject", None),
    "45d8c6f3-f8c1-48f5-82ad-65a986a85c3f": ("reject", None),
    "5339f090-6af2-45e5-8dfe-9859db9c2b21": ("reject", None),
    "8d3f63da-d930-4a68-8d2e-e5d4ea1d571a": ("reject", None),
    "9e8fe80c-fbc9-4011-97cd-f78f5a3cae98": ("reject", None),
    "a2fa20df-20f0-4a40-bc85-e340da49d40e": ("reject", None),
}
RAPID_STEPS = 10

# The Delhi clip's "pedestrian" proposals, judged by eye before any take. Eight are the red and white
# bollards on the traffic island, and two are the one man who walks past the herd.
GRID_REJECT = [
    "40f392c0-2055-411e-bfa1-9e69de7e01e9",
    "3cc7c6ec-af9b-4f8c-bcb0-e1688cc5a41f",
    "555d7ef1-2658-473d-a635-3ffa4428d4ae",
    "81efaa38-79ed-4ca6-9553-886e4a7f66c3",
    "34b3e27d-d676-42d9-a9e4-4ae7084ed386",
    "2b71170c-92dc-4cc8-ae7c-d5a4621d4a5b",
    "742475d2-008f-42ff-8949-028af5a66fcd",
    "95c87081-b19c-4394-88ed-85d532283b13",
]
GRID_ACCEPT = [
    "20031d90-e0f8-4b9f-bd49-748ce485cbfc",
    "1dae3745-e768-4bba-b7c0-c81af76fccb4",
]

ORIGINALS = OUT / "review_originals.json"


async def _originals() -> dict:
    """Each judged object's state and class before any take touched it, captured once and kept.

    Captured incrementally: an object added to a scene later is recorded the first time it is seen, which is
    before any take has reviewed it, and objects already recorded keep their first capture.
    """
    known = json.loads(ORIGINALS.read_text()) if ORIGINALS.exists() else {}
    ids_ = [v[0] for v in FRAME_VERDICTS] + list(RAPID_VERDICTS) + GRID_REJECT + GRID_ACCEPT
    async with get_engine().connect() as c:
        track = (await ids())["track"]
        ids_ += [r[0] for r in (await c.execute(text(
            "select object_id::text from object where track_id = cast(:t as uuid)"), {"t": track})).all()]
        missing = [x for x in dict.fromkeys(ids_) if x not in known]
        if missing:
            rows = (await c.execute(text("""
                select o.object_id::text, o.state, cl.name from object o
                join ontology_class cl on cl.id = o.class_id
                 and cl.version = (select max(version) from ontology_class)
                where o.object_id = any(cast(:ids as uuid[]))"""), {"ids": missing})).all()
            known.update({r[0]: {"state": r[1], "class_name": r[2]} for r in rows})
            ORIGINALS.parent.mkdir(parents=True, exist_ok=True)
            ORIGINALS.write_text(json.dumps(known, indent=1))
    return known


async def _scope_ids(scope: str, i: dict) -> list[str]:
    if scope == "frame":
        return [v[0] for v in FRAME_VERDICTS]
    if scope == "rapid":
        return list(RAPID_VERDICTS)
    if scope == "grid":
        return GRID_REJECT + GRID_ACCEPT
    async with get_engine().connect() as c:
        return [r[0] for r in (await c.execute(text(
            "select object_id::text from object where track_id = cast(:t as uuid)"), {"t": i["track"]})).all()]


def restore_for(scope: str):
    """A before-step that takes back earlier takes' verdicts on this scene's objects only.

    Scoped per scene because a shared restore undid the verdicts every earlier scene of the same take had
    just recorded, leaving the database disagreeing with the film. Goes through the review pages' own
    revert, which restores the source and confidence the withdrawn verdict overwrote.
    """
    async def restore(d: Demo, i: dict) -> str:
        orig = await _originals()
        ids_ = [x for x in await _scope_ids(scope, i) if x in orig]
        for oid in ids_:
            o = orig[oid]
            body = {"reviewer": d.user.get("name", "demo"), "action": "revert", "state": o["state"],
                    "class_name": o["class_name"], "time_spent_ms": 0}
            await d.page.evaluate(
                """async ([p, t, b]) => { await fetch(p, {method: 'POST', body: JSON.stringify(b),
                     headers: {authorization: 'Bearer ' + t, 'content-type': 'application/json'}}); }""",
                [f"/api/objects/{oid}/review", d.user["token"], body])
        return f"restored {len(ids_)} {scope} objects to their state before the demo"
    return restore


async def _states(ids_: list[str]) -> dict[str, tuple[str, str]]:
    async with get_engine().connect() as c:
        rows = (await c.execute(text("""
            select o.object_id::text, o.state, cl.name from object o
            join ontology_class cl on cl.id = o.class_id and cl.version = (select max(version) from ontology_class)
            where o.object_id = any(cast(:ids as uuid[]))"""), {"ids": ids_})).all()
    return {r[0]: (r[1], r[2]) for r in rows}


def _expect(verdict: str, cls: str | None, orig_cls: str) -> tuple[str, str]:
    state = {"accept": "accepted", "reject": "rejected", "reclassify": "accepted"}[verdict]
    return state, cls or orig_cls


async def verify_frame_review(d: Demo, i: dict) -> str:
    orig = await _originals()
    now = await _states([v[0] for v in FRAME_VERDICTS])
    ok = sum(1 for oid, _pt, _mpt, v, cls in FRAME_VERDICTS
             if now.get(oid) == _expect(v, cls, orig[oid]["class_name"]))
    return f"{ok} of {len(FRAME_VERDICTS)} verdicts saved as given"


async def verify_rapid(d: Demo, i: dict) -> str:
    """How many judged crops reached on camera were saved as judged, and whether any was saved differently."""
    orig = await _originals()
    now = await _states(list(RAPID_VERDICTS))
    judged = {k: v for k, v in RAPID_VERDICTS.items() if v[0] != "skip"}
    reached = {k: v for k, v in judged.items()
               if now.get(k) != (orig[k]["state"], orig[k]["class_name"])}
    wrong = [k[:8] for k, (v, cls) in reached.items() if now.get(k) != _expect(v, cls, orig[k]["class_name"])]
    return f"{len(reached) - len(wrong)} rapid verdicts saved as judged, {len(wrong)} saved differently {wrong}"


ON_OBJECT = """() => {
  const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  while (w.nextNode()) {
    const m = /on object ([0-9a-f]{8})/.exec(w.currentNode.textContent || '');
    if (m) return m[1];
  }
  return null;
}"""


async def selected_id(d: Demo) -> str | None:
    """The first eight characters of the selected object's id, as the issues panel prints it."""
    for _ in range(15):
        got = await d.page.evaluate(ON_OBJECT)
        if got:
            return got
        await d.wait(100)
    return None


async def frame_review(d: Demo, i: dict) -> None:
    """Right-click each proposal and give its verdict from the canvas menu, staying on this frame."""
    for oid, (x, y), (mx, my), verdict, cls in FRAME_VERDICTS:
        await d.key("v", "select", label="V", after=150)
        await d.click_img(x, y, ms=450)
        await d.wait(300)
        picked = await selected_id(d)
        if picked != oid[:8]:
            raise RuntimeError(f"expected to select {oid[:8]}, the panel shows {picked}")
        cx, cy = await d.img(mx, my)
        if cls:
            await d.page.mouse.click(cx, cy, button="right")
            await d.click(d.page.get_by_role("menuitem", name=re.compile(r"^change class")), ms=300, after=300)
            await d.type(cls)
            await d.key("Enter", f"class: {cls}", label="Enter", after=500)
            await dismiss_fix_similar(d)
            await save(d)
        await d.page.mouse.click(cx, cy, button="right")
        # A menu item's accessible name carries its shortcut hint after the label ("accept A").
        await d.click(d.page.get_by_role("menuitem", name=re.compile(rf"^{verdict}\b")), ms=300, after=900)
    await save(d)


async def rapid(d: Demo, i: dict) -> None:
    """Judge the queue one crop at a time, checking each crop is an object that was judged beforehand.

    The queue order is not fixed from one take to the next, so the verdicts are keyed by object id rather
    than by position, and a crop that was not judged in advance is skipped on camera rather than guessed.
    """
    keys = {"accept": ("a", "A"), "reject": ("r", "R"), "skip": ("s", "S"), "reclassify": ("c", "C")}
    for _ in range(RAPID_STEPS):
        # The crop on screen is the object on screen: its image is fetched from /api/objects/<id>/crop.
        current = await d.page.evaluate("""() => {
            const img = [...document.images].find(i => /\\/api\\/objects\\/[0-9a-f-]+\\/crop/.test(i.src));
            return img ? img.src.split('/api/objects/')[1].split('/')[0] : null; }""")
        if current is None:
            raise RuntimeError("no crop on screen")
        verdict, cls = RAPID_VERDICTS.get(current, ("skip", None))
        key, label = keys[verdict]
        await d.wait(900)                                  # time to look at the crop
        await d.key(key, verdict, label=label, after=300)
        if verdict == "reclassify":
            await d.type(cls or "")
            await d.key("Enter", f"class: {cls}", label="Enter", after=300)
        await d.wait(700)

async def grid(d: Demo, i: dict) -> None:
    """Select every bollard the model called a pedestrian and reject them together, then accept the man."""
    queue = await d.api(f"/api/triage?session_id={i['delhi']}&klass=pedestrian&limit=1000")
    order = [r["object_id"] for r in queue]
    tiles = d.page.locator('button[title^="pedestrian"]')
    await tiles.first.wait_for(state="visible", timeout=15000)
    for group, (key, label) in ((GRID_REJECT, ("r", "R")), (GRID_ACCEPT, ("a", "A"))):
        picks = sorted(order.index(o) for o in group if o in order)
        if len(picks) != len(group):
            raise RuntimeError(f"{len(group) - len(picks)} judged tiles are not in the grid")
        for k in picks:
            await d.click(tiles.nth(k), ms=380, after=250)
        await d.wait(600)
        verdict = "reject" if key == "r" else "accept"
        await d.key(key, f"{verdict} {len(picks)} at once", label=label, after=1800)
        # Removed tiles shift the rest up, so the next group is located against the queue as it now is.
        order = [o for o in order if o not in group]


async def verify_grid(d: Demo, i: dict) -> str:
    now = await _states(GRID_REJECT + GRID_ACCEPT)
    rej = sum(1 for o in GRID_REJECT if now.get(o, ("",))[0] == "rejected")
    acc = sum(1 for o in GRID_ACCEPT if now.get(o, ("",))[0] == "accepted")
    return f"{rej} of {len(GRID_REJECT)} bollards rejected, {acc} of {len(GRID_ACCEPT)} pedestrians accepted"


# ------------------------------------------------------------------ opening, tracks and delivery

async def sessions(d: Demo, i: dict) -> None:
    for name in (MANUAL, MACHINE, DELHI):
        card = d.page.get_by_text(name, exact=True).first
        box = await card.bounding_box()
        if box:
            await d.move(box["x"] + 120, box["y"] + 40, ms=700)
            await d.wait(1300)


async def import_page(d: Demo, i: dict) -> None:
    zone = d.page.get_by_text("drag .zip", exact=False).first
    box = await zone.bounding_box()
    if box:
        await d.move(box["x"] + box["width"] / 2, box["y"] + 10, ms=700)
        await d.wait(1500)
    fmt = d.page.locator("select").first
    fb = await fmt.bounding_box()
    if fb:
        await d.move(fb["x"] + fb["width"] / 2, fb["y"] + fb["height"] / 2, ms=600)
        await d.wait(1200)


async def track_relabel(d: Demo, i: dict) -> None:
    await d.page.mouse.wheel(0, 0)
    strip = d.page.get_by_text("timeline (", exact=False).first
    sb = await strip.bounding_box()
    if sb:
        await d.move(sb["x"] + 700, sb["y"] + 70, ms=900)
        await d.wait(1500)
    await d.click(d.page.get_by_placeholder("search class..."), ms=600, after=200)
    await d.type("cattle")
    await d.wait(500)
    # Scoped to the relabel panel, because every frame tile above is also a button whose name starts with
    # the class. India-specific classes carry a "*" after the name, so the name is matched by its start.
    relabel = d.page.locator("section", has_text="relabel entire track")
    await d.click(relabel.get_by_role("button", name=re.compile(r"^cattle\b")), ms=500, after=2500)


async def verify_track(d: Demo, i: dict) -> str:
    async with get_engine().connect() as c:
        rows = (await c.execute(text("""
            select cl.name, count(*) from object o
            join ontology_class cl on cl.id = o.class_id and cl.version = (select max(version) from ontology_class)
            where o.track_id = cast(:t as uuid) group by 1"""), {"t": i["track"]})).all()
    return "track now " + ", ".join(f"{n} {name}" for name, n in rows)


async def datasets(d: Demo, i: dict) -> None:
    for fmt in ("coco", "yolo", "parquet", "openlabel", "nuscenes"):
        chip = d.page.get_by_text(fmt, exact=True).first
        box = await chip.bounding_box()
        if box:
            await d.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, ms=420)
            await d.wait(350)
    await d.wait(800)


# ------------------------------------------------------------------ keyboard editing, and work across frames
async def nudge(d: Demo, i: dict) -> None:
    """Select the jeep and walk it with the arrow keys, then walk it back: one pixel, then ten."""
    await d.key("v", "select", label="V", after=200)
    await d.zoom_at(945, 572, 6)
    await d.click_img(945, 572, ms=450)
    await d.wait(500)
    for _ in range(4):
        await d.key("ArrowRight", "nudge 1 px", label="→", after=170)
    await d.key("Shift+ArrowUp", "nudge 10 px", label="Shift ↑", after=700)
    await d.key("Shift+ArrowDown", "and back", label="Shift ↓", after=350)
    for _ in range(4):
        await d.key("ArrowLeft", "nudge 1 px", label="←", after=170)
    await d.wait(400)
    await deselect(d)
    await d.key("f", "fit to view", label="F", after=400)
    await save(d)


async def shortcuts(d: Demo, i: dict) -> None:
    """Open the shortcut sheet, point at a binding that can be changed, close it."""
    await d.key("Shift+Slash", "shortcuts", label="?", after=900)
    hint = d.page.get_by_text("click a key to change it").first
    await hint.wait_for(state="visible", timeout=8000)
    box = await hint.bounding_box()
    if box:
        await d.move(box["x"] + 60, box["y"] + 8, ms=700)
        await d.wait(1400)
    reset_btn = d.page.get_by_role("button", name="Reset to defaults").first
    rb = await reset_btn.bounding_box()
    if rb:
        await d.move(rb["x"] + rb["width"] / 2, rb["y"] + rb["height"] / 2, ms=700)
        await d.wait(1200)
    await d.key("Escape", "close", label="Esc", after=500)


async def machine_frame(d: Demo, i: dict) -> None:
    """The same instant, labelled by the models: fit, then lean in on the bus and the auto-rickshaw."""
    await d.key("f", "fit to view", label="F", after=900)
    await d.move(*(await d.img(718, 540)), ms=800)
    await d.wait(1300)
    await d.zoom_at(420, 660, 4)
    await d.wait(1500)
    await d.move(*(await d.img(270, 680)), ms=700)
    await d.wait(1200)
    await d.move(*(await d.img(546, 640)), ms=700)
    await d.wait(1200)
    await d.key("f", "fit to view", label="F", after=900)


async def paste_next(d: Demo, i: dict) -> None:
    """Copy every object on this frame, step to the next frame, paste, and nudge the bus to where it now is."""
    await d.key("v", "select", label="V", after=200)
    await d.key("Control+a", "select all", label="Ctrl A", after=800)
    await d.key("Control+c", "copy all", label="Ctrl C", after=700)
    await d.key("BracketRight", "next frame", label="]", after=400)
    await d.page.wait_for_url(f"**/frame/{i['manual_next']}", timeout=20_000)
    await d.page.wait_for_selector(EDITOR, timeout=30_000)
    await d.wait(1800)
    before = (await panel(d))["count"]
    await d.key("Control+v", "paste on this frame", label="Ctrl V", after=1200)
    after = (await settle(d))["count"]
    if after <= before:
        raise RuntimeError(f"the paste added nothing on the next frame ({before} -> {after})")
    await d.wait(900)
    # Everything pasted is selected, so the arrows move all of it at once. A third of a second on, the
    # traffic ahead sits a couple of dozen pixels to the left.
    for _ in range(2):
        await d.key("Shift+ArrowLeft", "nudge all, 10 px", label="Shift ←", after=450)
    await d.wait(500)
    await deselect(d)
    await save(d)


async def verify_paste(d: Demo, i: dict) -> str:
    src = await human_objects(i["manual_frame"])
    dst = await human_objects(i["manual_next"])
    if not dst or len(dst) != len(src):
        raise RuntimeError(f"pasted {len(dst)} objects onto the next frame, the source frame has {len(src)}")
    return f"{len(dst)} objects copied to the next frame and saved there"


# ------------------------------------------------------------------ two people on one frame
async def _colleague() -> dict:
    """The reviewer who takes the frame over, as the API knows her: a user row and a signed token."""
    import uuid as _uuid

    from core.config import get_settings
    from services.api.auth_token import mint_token

    async with get_engine().begin() as c:
        row = (await c.execute(text("select user_id::text, token_version from app_user where name=:n"),
                               {"n": COLLEAGUE})).first()
        if row is None:
            uid = str(_uuid.uuid4())
            await c.execute(text("insert into app_user (user_id, name, role) values (cast(:u as uuid), :n, 'reviewer')"),
                            {"u": uid, "n": COLLEAGUE})
            row = (uid, 1)
    token = mint_token(row[0], get_settings().auth.signing_key, token_version=int(row[1] or 1), ttl_seconds=3600)
    return {"user_id": row[0], "token": token, "tab": "demo-colleague-tab-0001"}


def _presence(colleague: dict, frame_id: str, action: str) -> dict:
    """One presence call as the colleague, straight to the API the editor itself uses."""
    from annotate import API

    req = urllib.request.Request(
        f"{API}/api/presence/frames/{frame_id}/{action}", method="POST",
        data=json.dumps({"tab": colleague["tab"], "acquire": True}).encode(),
        headers={"content-type": "application/json", "authorization": f"Bearer {colleague['token']}"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read())


# The bus on the shared frame, five frames on: nearer and larger than on the hand-annotated frame.
SHARED_BUS = (495, 418, 705, 652)


async def two_people(d: Demo, i: dict) -> None:
    """We are editing; a reviewer takes the frame over; we are told and switched to view-only; we take it back."""
    import asyncio

    who = await _colleague()
    fid = i["shared_frame"]
    await d.key("f", "fit to view", label="F", after=700)
    await d.pick_class("bus")
    await draw_box(d, "bus", SHARED_BUS, ms=1000)
    await save(d)
    await d.wait(600)

    # Meera opens the same frame and takes it over. From here she heartbeats like any open editor would.
    await asyncio.to_thread(_presence, who, fid, "lock/takeover")
    alive = True

    async def keep_alive() -> None:
        while alive:
            await asyncio.sleep(4)
            if alive:
                await asyncio.to_thread(_presence, who, fid, "heartbeat")

    beat = asyncio.ensure_future(keep_alive())
    try:
        banner = d.page.get_by_role("status").filter(has_text="view only").first
        await banner.wait_for(state="visible", timeout=20_000)
        bb = await banner.bounding_box()
        if bb:
            await d.move(bb["x"] + 200, bb["y"] + bb["height"] / 2, ms=800)
        await d.wait(3200)
        # A view-only editor does not move things: the arrows are pressed and nothing happens.
        await d.key("Control+a", "select all", label="Ctrl A", after=500)
        await d.key("ArrowRight", "view only: nothing moves", label="→", after=1300)
        take = d.page.get_by_role("button", name="Take over").first
        await d.click(take, ms=700, after=900)
        confirm_box = d.page.get_by_role("alertdialog")
        await confirm_box.wait_for(state="visible", timeout=8000)
        await d.wait(2200)
        await d.click(confirm_box.get_by_role("button", name="Take over"), ms=600, after=1200)
    finally:
        alive = False
        beat.cancel()
    await d.page.get_by_role("button", name="Take over").first.wait_for(state="hidden", timeout=15_000)
    await d.wait(600)
    # The frame is ours again, and editing works.
    await d.key("Control+a", "select all", label="Ctrl A", after=400)
    await d.key("ArrowRight", "ours again", label="→", after=500)
    await d.key("ArrowLeft", "", label="←", after=400)
    await deselect(d)
    await save(d)


async def verify_two_people(d: Demo, i: dict) -> str:
    n = len(await human_objects(i["shared_frame"]))
    took = await _one("""select count(*) from activity_event where verb = 'took_over_frame'
                          and subject_id = :f""", f=i["shared_frame"])
    if not n or not took:
        raise RuntimeError(f"expected a saved box and a recorded take-over, found {n} objects and {took} events")
    return f"{n} object saved on the shared frame; {took} take-overs recorded in the activity log"



SCENES = [
    Scene("sessions", "Real Indian roads", "Real footage",
          "/annotations",
          "Real Indian roads, seen the way a vehicle sees them. A dashcam drive through Bengaluru, following "
          "a city bus past an auto-rickshaw and a motorcycle, and a herd of cattle crossing a road in Delhi, "
          "filmed from a car. Faces and number plates are blurred before anything is stored. One copy of the "
          "Bengaluru frames is left unlabelled, for annotating by hand.",
          ready=f"text={DELHI}", run=sessions),
    Scene("import", "Real Indian roads", "Import",
          "/import",
          "Footage arrives here. A video, a folder of images or a sensor recording is split into frames, "
          "checked for quality and anonymised on the way in.",
          ready="select", run=import_page),
    Scene("editor_open", "Manual annotation", "An empty frame",
          "/frame/{manual_frame}",
          "This frame is from the Bengaluru drive, and no model has labelled it. Every object on it is "
          "annotated manually in the editor, through the same mouse gestures and shortcuts an annotator uses.",
          ready=EDITOR, run=open_editor, verify=_count_human(), before=reset),
    Scene("box_bus", "Manual annotation", "Box tool",
          "/frame/{manual_frame}",
          "First the city bus. The class is chosen from the picker by typing its name, B selects the box tool, "
          "and one drag draws it.",
          ready=EDITOR, run=box_bus, verify=_count_human("bus")),
    Scene("box_autos", "Manual annotation", "Auto-rickshaws",
          "/frame/{manual_frame}",
          "Auto-rickshaws are their own class here, not cars. The ontology has two hundred classes built for "
          "Indian roads, from auto-rickshaws and goods carriers to cattle.",
          ready=EDITOR, run=box_autos, verify=_count_human("autorickshaw")),
    Scene("box_more", "Manual annotation", "Small and half hidden",
          "/frame/{manual_frame}",
          "The jeep ahead is small, so it gets a zoom first, with the scroll wheel, and the box is drawn at "
          "its edges rather than roughly around it. Then the white car half hidden behind the median hedge.",
          ready=EDITOR, run=box_more, verify=_count_human()),
    Scene("polygon", "Manual annotation", "Polygon tool",
          "/frame/{manual_frame}",
          "Not everything is a box. G draws a polygon, one click per corner, and a double click closes it. "
          "This is the planted median that divides the road, a class of its own.",
          ready=EDITOR, run=polygon, verify=_count_human("median")),
    Scene("edit", "Manual annotation", "Editing",
          "/frame/{manual_frame}",
          "Boxes resize from their handles, and every change can be undone. Objects can be copied, pasted "
          "and deleted from the keyboard.",
          ready=EDITOR, run=edit, verify=_count_human()),
    Scene("nudge", "Manual annotation", "Arrow keys",
          "/frame/{manual_frame}",
          "A box that is nearly right does not need the mouse. The arrow keys move the selection one pixel "
          "at a time, and ten with Shift held down.",
          ready=EDITOR, run=nudge, verify=_count_human()),
    Scene("shortcuts", "Manual annotation", "Your own shortcuts",
          "/frame/{manual_frame}",
          "Every tool and action has a key, and the question mark lists them. Any of them can be changed "
          "to suit the annotator, by clicking the key and pressing a new one.",
          ready=EDITOR, run=shortcuts),
    Scene("sam_point", "AI assist", "Segment from one click",
          "/frame/{manual_frame}",
          "S turns one click into a pixel mask. Here it is the motorcycle. The model proposes the outline, "
          "Enter accepts it, and the annotator still decides what it is.",
          ready=EDITOR, run=sam_point, verify=_count_human("motorcycle")),
    Scene("sam_box", "AI assist", "Segment from a box",
          "/frame/{manual_frame}",
          "M does the same from a rough box, which is quicker for a large vehicle like the truck at the "
          "left edge.",
          ready=EDITOR, run=sam_box, verify=_count_human("truck")),
    Scene("wand", "AI assist", "Magic wand",
          "/frame/{manual_frame}",
          "The wand creates the object and its mask in a single click, with no accept step. This is the rider.",
          ready=EDITOR, run=wand, verify=_count_human("rider")),
    Scene("brush", "AI assist", "Brush and eraser",
          "/frame/{manual_frame}",
          "Where a mask is not quite right, the brush adds to it and the eraser takes away, stroke by stroke.",
          ready=EDITOR, run=brush, verify=_count_human("rider")),
    Scene("amodal", "Beyond boxes", "Occluded extent",
          "/frame/{manual_frame}",
          "The white car is partly hidden behind the hedge. K records its whole extent, as well as the "
          "part that is visible, which is what a planner needs.",
          ready=EDITOR, run=amodal, verify=_has("amodal")),
    Scene("polyline", "Beyond boxes", "Polylines",
          "/frame/{manual_frame}",
          "L draws polylines, for things with length and no area: lane markings, railings, and this "
          "painted kerb along the median.",
          ready=EDITOR, run=polyline, verify=_has("polyline")),
    Scene("keypoints", "Beyond boxes", "Pose keypoints",
          "/frame/{manual_frame}",
          "Pose mode places the seventeen body keypoints of a person, here the motorcycle rider, which is "
          "how posture and intent are labelled.",
          ready=EDITOR, run=keypoints, verify=_has("keypoints")),
    Scene("attributes", "Beyond boxes", "Attributes",
          "/frame/{manual_frame}",
          "Every object carries attributes as well as a class: occlusion, direction, lane position, load.",
          ready=EDITOR, run=attributes, verify=_count_human()),
    Scene("measure", "Beyond boxes", "Measure",
          "/frame/{manual_frame}",
          "R measures a distance on the image in pixels. On a LiDAR bird's eye view the same tool reads in metres.",
          ready=EDITOR, run=measure),
    Scene("adverse", "Beyond boxes", "Adverse regions",
          "/frame/{manual_frame}",
          "D marks a region the model should not be judged on. Here the road lies in deep shadow under the trees.",
          ready=EDITOR, run=adverse, verify=_adverse_saved),
    Scene("confirm", "Beyond boxes", "Confirm the frame",
          "/frame/{manual_frame}",
          "When every object has been checked, Confirm and next records the human verdict and moves on to "
          "the next frame.",
          ready=EDITOR, run=confirm, verify=_count_human()),
    Scene("machine_frame", "Checking the machine", "What the models proposed",
          "/frame/{machine_frame}",
          "This is the same instant, in the copy the models labelled. Nobody drew these. Three detectors "
          "proposed {proposals} objects on this one frame, and {machine_objects} across the {blr_frames} "
          "frames of the clip, along with the drivable road in green. They also called the bonnet of this "
          "car a sedan, which is why a proposal is not a label until a person has checked it.",
          ready=EDITOR, run=machine_frame),
    Scene("frame_review", "Checking the machine", "Reviewing proposals",
          "/frame/{review_frame}",
          "This Delhi frame was labelled by the models, not by hand. Review mode checks each proposal. The cows "
          "are right, the pedestrian on the right is a reflection in the car window and is rejected, and the "
          "white car is a people carrier, not a sedan, so its class is changed before it is accepted.",
          ready=EDITOR, run=frame_review, verify=verify_frame_review, before=restore_for("frame")),
    Scene("rapid", "Checking the machine", "Rapid review",
          "/review/rapid?session={delhi}&class=cattle",
          "Rapid review shows one proposal at a time, and one key decides it. Here it is every cattle proposal "
          "in the Delhi clip. Reflections of the herd in the window glass are rejected, and a motorcycle the "
          "model called a cow is reclassified.",
          ready="img", run=rapid, verify=verify_rapid, before=restore_for("rapid")),
    Scene("grid", "Checking the machine", "Bulk review",
          "/review/grid?session={delhi}&class=pedestrian",
          "The grid shows many proposals at once. Eight of the Delhi pedestrians are the striped bollards on the "
          "traffic island, so they are selected together and rejected with one key. The two crops of the man "
          "walking past are accepted the same way.",
          ready='button[title^="pedestrian"]', run=grid, verify=verify_grid, before=restore_for("grid")),
    Scene("paste_next", "Across frames", "Copy to the next frame",
          "/frame/{manual_frame}",
          "The next frame of a clip is nearly the same picture. So everything drawn here is copied, pasted "
          "onto the next frame, and nudged into place with the arrow keys, rather than drawn again.",
          ready=EDITOR, run=paste_next, verify=verify_paste),
    Scene("track", "Across frames", "One fix, every frame",
          "/track/{track}",
          "Detections are linked into tracks. This is one cow, followed through twenty-eight seconds. The model "
          "called it a light goods vehicle in one frame, and relabelling the track fixes every frame at once.",
          ready="text=relabel entire track", run=track_relabel, verify=verify_track, before=restore_for("track")),
    Scene("two_people", "Working together", "Two people, one frame",
          "/frame/{shared_frame}",
          "Only one person edits a frame at a time. While this box is being drawn, a reviewer called Meera "
          "takes the frame over. The editor says so at once and becomes view only, so nothing is "
          "overwritten. Taking a frame from someone asks first, and is recorded.",
          ready=EDITOR, run=two_people, verify=verify_two_people),
    Scene("datasets", "Delivering the data", "Export",
          "/datasets",
          "What a person has checked leaves as a sealed, versioned dataset, in the format the training pipeline "
          "expects: COCO, YOLO, Parquet, OpenLABEL or nuScenes.",
          ready="text=Sealed Datasets", run=datasets),
]
