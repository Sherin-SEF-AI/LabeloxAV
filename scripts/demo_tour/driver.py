"""Drive the real UI and record the tour.

The driver does four things in a fixed order for every scene: resolve the page, open it in a real
browser against the running application, speak the narration while the screen is captured, and cut the
segment to exactly the length the speech turned out to be.

Two decisions are worth stating.

**The numbers come out of the database at record time.** Narration strings carry named placeholders and
`facts()` fills them from one read-only query pass. A tour that hard-codes its figures becomes a lie the
first time the corpus changes, and this one is meant to stay checkable.

**A page that fails to load is narrated, not hidden.** If a route errors or times out the scene still
records, the manifest marks it, and the summary at the end lists every such page. Cutting them would
turn the recording into a selection of the pages that happened to work.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import narration  # noqa: E402
from narration import Scene, check  # noqa: E402

# Which script to record. The tour explains the product; the verification film is a record of a checking
# pass over it. Both are scene lists of the same shape, so the driver does not care which it is given.
SCENES = narration.SCENES
CHAPTERS = narration.CHAPTERS


def use_script(name: str) -> None:
    global SCENES, CHAPTERS
    if name == "tour":
        SCENES, CHAPTERS = narration.SCENES, narration.CHAPTERS
        return
    import importlib

    mod = importlib.import_module(name)
    SCENES = mod.SCENES
    CHAPTERS = getattr(mod, "CHAPTERS", narration.CHAPTERS)
    # `check()` validates the tour's own list, so the alternate script gets the same guards inline.
    keys = [x.key for x in SCENES]
    dupes = {k for k in keys if keys.count(k) > 1}
    if dupes:
        raise ValueError(f"duplicate scene keys: {sorted(dupes)}")
    for x in SCENES:
        for frag in x.narration.replace("\n", " ").split(". "):
            if frag.strip().startswith("{"):
                raise ValueError(f"{x.key}: a sentence starts with a placeholder, so it will be lower case")
        if "—" in x.narration or "–" in x.narration:
            raise ValueError(f"dash in narration for {x.key}")
from record import (  # noqa: E402
    API,
    FPS,
    OUT,
    WEB,
    H,
    W,
    check_tools,
    concat,
    concat_audio,
    fit_duration,
    mux,
    pad_audio,
    probe_duration,
    quantize,
    save_manifest,
    synth,
    write_ass,
    write_chapters,
    write_srt,
)

# The key `components/shell/Onboarding.tsx` writes when somebody finishes or skips the welcome tour.
ONBOARDED = "try { localStorage.setItem('lbx_onboarded', new Date().toISOString()); } catch (e) {}"

# A pointer drawn into the page, because a headless browser renders no cursor and a demo where things
# click themselves is hard to follow. It follows real mouse events, so it is only ever where Playwright
# actually moved the mouse, and every press leaves a ripple where it landed.
CURSOR = r"""
(() => {
  if (window.__lbxCursor) return; window.__lbxCursor = true;
  const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28">' +
    '<path d="M5 3 L5 22 L10 17.5 L13.6 25 L17 23.4 L13.5 16 L20 16 Z" fill="#fff" stroke="#111" ' +
    'stroke-width="1.6" stroke-linejoin="round"/></svg>';
  const style = document.createElement('style');
  style.textContent = '@keyframes lbxRipple{from{transform:translate(-50%,-50%) scale(.2);opacity:.85}' +
    'to{transform:translate(-50%,-50%) scale(1.9);opacity:0}}';
  const put = () => {
    if (!document.body) return setTimeout(put, 30);
    document.head && document.head.appendChild(style);
    const c = document.createElement('div');
    c.id = '__lbx_cursor';
    c.innerHTML = svg;
    c.style.cssText = 'position:fixed;left:0;top:0;width:28px;height:28px;z-index:2147483647;' +
      'pointer-events:none;filter:drop-shadow(0 2px 3px rgba(0,0,0,.45));transform:translate(-200px,-200px)';
    document.documentElement.appendChild(c);
    addEventListener('mousemove', e => { c.style.transform = `translate(${e.clientX - 5}px,${e.clientY - 3}px)`; }, true);
    addEventListener('mousedown', e => {
      const r = document.createElement('div');
      r.style.cssText = `position:fixed;left:${e.clientX}px;top:${e.clientY}px;width:44px;height:44px;` +
        'border-radius:50%;border:3px solid rgba(64,150,255,.95);background:rgba(64,150,255,.18);' +
        'z-index:2147483646;pointer-events:none;animation:lbxRipple .6s ease-out forwards';
      document.documentElement.appendChild(r);
      setTimeout(() => r.remove(), 700);
    }, true);
  };
  put();
})();
"""


def signed_in() -> str:
    """An init script that puts a signed admin session where the web app looks for it.

    Auth is always on now, so a fresh browser context opens to a sign-in wall. dev-login is the product's
    own local bootstrap (404 anywhere but a local install), and the browser is handed its answer exactly as
    the sign-in page would store it.
    """
    import urllib.request

    req = urllib.request.Request(f"{API}/api/auth/dev-login", method="POST", data=b"")
    with urllib.request.urlopen(req, timeout=15) as r:
        user = json.loads(r.read())
    return f"try {{ localStorage.setItem('lbx_user', {json.dumps(json.dumps(user))}); }} catch (e) {{}}"


async def facts() -> dict[str, str]:
    """Read every number the narration quotes, in one pass, read only.

    Formatted with thousands separators here rather than in the narration, because these strings are
    spoken as well as displayed and a bare digit string reads badly either way.
    """
    from sqlalchemy import text

    from db.session import get_sessionmaker

    queries = {
        "sessions": "select count(*) from session where origin = 'real'",
        "frames": "select count(*) from frame",
        "objects": "select count(*) from object",
        "accepted": "select count(*) from object where state = 'accepted'",
        "review_pending": "select count(*) from object where state = 'review'",
        "tracks": "select count(*) from track",
        "models": "select count(*) from model_registry",
        "embeddings": "select count(*) from frame_embedding",
        "clouds": "select count(*) from point_cloud where session_id = :kitti",
        # Corpus-wide, as distinct from the two above, which are scoped to one drive. Kept as separate
        # names because a script that says "the corpus holds" and quotes one session's count is wrong in
        # the way that is hardest to notice: the number is real, it just answers a different question.
        "clouds_all": "select count(*) from point_cloud",
        "points_all": "select coalesce(sum(point_count), 0) / 1000000 from point_cloud",
        "poses": "select count(*) from ego_pose where measured is true",
        "points": "select coalesce(sum(point_count), 0) / 1000000 from point_cloud where session_id = :kitti",
        # The recordings this tour was filmed on. Counted rather than written down: the first version of
        # this scene said "three recordings, two dashcam clips" because that was true on the afternoon the
        # script was written, and it stopped being true when a third clip was labelled. A tour that
        # insists every figure comes from the database should not make an exception for the small ones.
        "demo_recordings": "select count(*) from session where route = 'demo-tour-2026' or route like 'KITTI 2011%'",
        "demo_dashcam": "select count(*) from session where route = 'demo-tour-2026'",
        # Added for the 2026-11 film.
        "tracks_all": "select count(*) from track",
        "scenarios": "select count(*) from scenario",
        "people": "select count(*) from app_user",
        "projects": "select count(*) from label_project",
        "gnss_frames": "select count(*) from frame where gnss is not null",
        "gnss_sessions": "select count(distinct session_id) from frame where gnss is not null",
        "gold_sets": "select count(*) from gold_set",
        "human_verdicts": "select count(*) from object where state in ('accepted', 'rejected')",
        "auto_accepted": "select count(*) from object where state = 'auto_accept'",
        "pii_access": "select count(*) from pii_access_log",
        "classes": "select count(*) from ontology_class where version = (select max(version) from ontology_version)",
        "dash_objects": "select count(*) from object o join frame f on f.frame_id = o.frame_id "
                        "where f.frame_id = (select frame_id from frame where frame_id = :dash_frame)",
        "dash_track_len": "select count(*) from object where track_id = :dash_track",
    }
    # Small counts are spoken, so they are spelled. "3 are dashcam clips" is read correctly by the voice
    # but reads as a stray digit in a caption sitting beside sentences that spell everything else out.
    spell = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
             6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    out: dict[str, str] = {}
    async with get_sessionmaker()() as db:
        kitti = (await db.execute(text(
            "select session_id from session where route like 'KITTI%' order by created_at desc limit 1"
        ))).scalar()
        picked = await ids()
        for name, sql in queries.items():
            try:
                params = {"kitti": kitti, "dash_frame": picked.get("dash_frame") or None,
                          "dash_track": picked.get("dash_track") or None}
                v = (await db.execute(text(sql), {k: v for k, v in params.items() if ":" + k in sql})).scalar()
            except Exception as exc:
                # A number that cannot be read is spoken as unmeasured. It is never spoken as zero and
                # the scene is never dropped, because a missing figure is itself worth hearing.
                print(f"  fact {name} unavailable: {str(exc).splitlines()[0][:90]}")
                out[name] = "an unreadable"
                continue
            n = int(v or 0)
            out[name] = spell[n] if name.startswith("demo_") and n in spell else f"{n:,}"
    return out


async def ids() -> dict[str, str]:
    """Pick the concrete rows the dynamic pages open on.

    Preference goes to the sessions recorded for this tour, so the pages show the data the narration is
    talking about. Where the tour's own data has nothing yet, the tour falls back to the wider corpus and
    says nothing it cannot support.
    """
    from sqlalchemy import text

    from db.session import get_sessionmaker

    picks: dict[str, str] = {}
    async with get_sessionmaker()() as db:
        async def one(sql: str) -> str | None:
            try:
                v = (await db.execute(text(sql))).scalar()
            except Exception:
                return None
            return str(v) if v is not None else None

        picks["session_id"] = (
            await one("select session_id from session where route like 'KITTI%' order by created_at desc limit 1")
            or await one("select session_id from session where origin = 'real' order by created_at desc limit 1")
            or ""
        )
        # A frame worth opening is one that has labels on it; an empty canvas demonstrates nothing.
        picks["frame_id"] = (
            await one("""select f.frame_id from frame f join object o on o.frame_id = f.frame_id
                         where f.session_id in (select session_id from session
                                                where route like 'KITTI%' or route = 'demo-tour-2026')
                         group by f.frame_id order by count(*) desc limit 1""")
            or await one("""select f.frame_id from frame f join object o on o.frame_id = f.frame_id
                            group by f.frame_id order by count(*) desc limit 1""")
            or ""
        )
        picks["object_id"] = (
            await one(f"select object_id from object where frame_id = '{picks['frame_id']}' limit 1")
            if picks["frame_id"] else None
        ) or await one("select object_id from object order by created_at desc limit 1") or ""
        # The 2026-11 film opens its 2D pages on the operator's own Bengaluru dashcam drives, which carry
        # GPS read from the stamp the camera burns into its video. The busiest frame of the drive with the
        # most objects; a long single-class vehicle track ranked by tube coherence; the KITTI drive with
        # the most real point clouds.
        picks["dash_frame"] = await one("""
            with best as (select f.session_id from frame f join object o on o.frame_id = f.frame_id
                          join session s on s.session_id = f.session_id
                          where s.vehicle_id = 'DASHCAM-01' and f.gnss is not null
                          group by 1 order by count(*) desc limit 1)
            select f.frame_id from frame f join object o on o.frame_id = f.frame_id
            where f.session_id = (select session_id from best)
            group by f.frame_id, f.ts_ns order by count(*) desc, f.ts_ns asc limit 1""") or ""
        picks["dash_session"] = (await one(
            f"select session_id from frame where frame_id = '{picks['dash_frame']}'") if picks["dash_frame"] else "") or ""
        picks["dash_track"] = await one("""
            select t.track_id from track t join object o on o.track_id = t.track_id
            join frame f on f.frame_id = o.frame_id join session s on s.session_id = f.session_id
            join ontology_class c on c.id = o.class_id
            where s.vehicle_id = 'DASHCAM-01' and t.tube_score is not null
              and c.name in ('bus', 'truck', 'sedan', 'autorickshaw', 'suv', 'hatchback')
            group by t.track_id, t.tube_score
            having count(*) between 100 and 160 and count(distinct o.class_id) = 1
            order by t.tube_score desc limit 1""") or ""
        picks["kitti_session"] = await one("""
            select s.session_id from session s join point_cloud p on p.session_id = s.session_id
            where s.route like 'KITTI%' group by 1 order by count(*) desc limit 1""") or ""
        # The drive whose clouds carry cuboids lifted from 2D, so the 3D pages have boxes to show.
        picks["kitti_cuboids"] = await one(
            "select session_id from session where route like 'KITTI 2011_09_26_drive_0005%' limit 1") or ""
        picks["track_id"] = (
            await one("""select o.track_id from object o join frame f on f.frame_id = o.frame_id
                         join session s on s.session_id = f.session_id
                         where o.track_id is not null
                           and (s.route like 'KITTI%' or s.route = 'demo-tour-2026')
                         group by o.track_id order by count(*) desc limit 1""")
            or await one("""select track_id from object where track_id is not null
                            group by track_id order by count(*) desc limit 1""")
            or ""
        )
    return picks


async def act(page, actions: list, i: dict[str, str]) -> str:
    """Run a scene's interactions and report what did not work, rather than failing the scene.

    A page that needs a session id typed into it shows an empty panel until somebody types one, and an
    empty panel narrated as a point cloud viewer is the kind of thing that makes a demo worthless. So the
    tour drives the page the way a person would.

    Every step is best effort. A selector that has moved costs that one step and a line in the manifest,
    not the recording: the scene still shows the page in whatever state it reached, and the summary says
    which interactions did not land.

    Steps are `(kind, *args)`: fill(selector, value), click(selector), click_text(text),
    press(key), scroll(pixels), wait(milliseconds). Any string argument may carry an id placeholder.
    """
    def sub(v):
        if not isinstance(v, str):
            return v
        for k, val in i.items():
            v = v.replace("{" + k + "}", val)
        return v

    problems = []

    async def glide(loc) -> tuple[float, float]:
        """Move the visible pointer to the middle of `loc` the way a hand would, and return where it is."""
        await loc.scroll_into_view_if_needed(timeout=8000)
        box = await loc.bounding_box(timeout=8000)
        if box is None:
            raise RuntimeError("no box")
        x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        await page.mouse.move(x, y, steps=28)
        await page.wait_for_timeout(180)
        return x, y

    for step in actions:
        kind, args = step[0], [sub(a) for a in step[1:]]
        try:
            if kind == "fill":
                await page.fill(args[0], args[1], timeout=8000)
            elif kind in ("click", "click_text", "move", "move_text"):
                if kind in ("click", "move"):
                    loc = page.locator(args[0]).first
                else:
                    # Prefer the page's own content: a word like "Search" or "Load" is also a sidebar link,
                    # and a pointer that wanders off to the navigation looks like a mistake.
                    main = page.locator("main").get_by_text(args[0], exact=False)
                    loc = (main if await main.count() else page.get_by_text(args[0], exact=False)).first
                x, y = await glide(loc)
                if kind.startswith("click"):
                    await page.mouse.click(x, y)
            elif kind == "type":
                # Typed at a person's pace, so the viewer can read the query as it is entered.
                loc = page.locator(args[0]).first
                x, y = await glide(loc)
                await page.mouse.click(x, y)
                await loc.fill("")
                await page.keyboard.type(args[1], delay=55)
            elif kind == "select":
                # Choose the option whose text contains args[1]; option labels carry ids we cannot predict.
                loc = page.locator(args[0]).first
                await glide(loc)
                value = await loc.evaluate(
                    "(e, t) => { const o = [...e.options].find(o => o.value === t || o.text.includes(t));"
                    " return o ? o.value : null; }",
                    args[1])
                if value is None:
                    raise RuntimeError(f"no option containing {args[1]!r}")
                await loc.select_option(value)
            elif kind == "press":
                await page.keyboard.press(args[0])
            elif kind == "scroll":
                # In small steps, so it reads as a scroll rather than a cut.
                total, n = int(args[0]), 12
                for _ in range(n):
                    await page.mouse.wheel(0, total / n)
                    await page.wait_for_timeout(35)
            elif kind == "point":
                await page.mouse.move(float(args[0]), float(args[1]), steps=24)
            elif kind == "wait":
                await page.wait_for_timeout(int(args[0]))
            else:
                problems.append(f"unknown action {kind}")
        except Exception as exc:
            problems.append(f"{kind} {str(args[0])[:40] if args else ''}: {type(exc).__name__}")
    return "; ".join(problems)


def _entry(scene: Scene, n: int, path: str | None, text: str, t: float, dur: float, note: str) -> dict:
    e = asdict(scene)
    e.update({"index": n, "path_resolved": path, "text": text,
              "start": round(t, 3), "duration": round(dur, 3), "note": note})
    return e


def resolve(scene: Scene, f: dict[str, str], i: dict[str, str]) -> tuple[str | None, str]:
    path = scene.path
    if path:
        for k, v in i.items():
            path = path.replace("{" + k + "}", v)
        if "{" in path:      # an id the corpus could not supply
            return None, scene.narration.format(**f)
    return path, scene.narration.format(**f)


async def record(only: list[str] | None, skip_capture: bool) -> list[dict]:
    from playwright.async_api import async_playwright

    if SCENES is narration.SCENES:
        check()
    missing = check_tools()
    if missing:
        raise SystemExit(f"cannot record, missing: {', '.join(missing)}")

    f, i = await facts(), await ids()
    session = signed_in()
    print("facts:", ", ".join(f"{k}={v}" for k, v in f.items()))
    print("ids:", ", ".join(f"{k}={v[:8]}" for k, v in i.items() if v))

    scenes = [s for s in SCENES if not only or s.key in only]
    entries: list[dict] = []
    t = 0.0
    segdir, auddir = OUT / "segments", OUT / "audio"
    segdir.mkdir(parents=True, exist_ok=True)
    auddir.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as pw:
        # Headless, because the browser records itself rather than the desktop being grabbed. That makes
        # the capture independent of the display server, which matters here: this is a Wayland session,
        # and an X11 grab of it returns black frames.
        browser = await pw.chromium.launch(headless=True, args=["--hide-scrollbars", "--force-device-scale-factor=1"])
        for n, scene in enumerate(scenes, 1):
            path, text = resolve(scene, f, i)
            note = ""

            wav = auddir / f"{scene.key}.wav"
            speech = synth(text, wav)
            dur = quantize(speech + scene.hold_s + 0.6)

            if skip_capture:
                pad_audio(wav, auddir / f"{scene.key}.pad.wav", dur)
                entries.append(_entry(scene, n, path, text, t, dur, note))
                t += dur
                continue

            raw = segdir / scene.key
            # One context per scene, because Playwright records video per context. Creating it here and
            # closing it below is what bounds the segment, and it also means a scene that has to be
            # re-recorded is re-recorded alone.
            ctx = await browser.new_context(
                viewport={"width": W, "height": H},
                record_video_dir=str(raw), record_video_size={"width": W, "height": H},
            )
            # Each scene gets a fresh context, so each one is a first visit and the welcome tour opens
            # over the page. Marking it seen before any script runs keeps the product's own onboarding
            # out of a recording that is itself an onboarding.
            await ctx.add_init_script(ONBOARDED)
            await ctx.add_init_script(session)
            await ctx.add_init_script(CURSOR)
            page = await ctx.new_page()
            # Parked off to the side of the content, so the first glide reads as a move toward something.
            await page.mouse.move(1700, 760)
            settle = scene.settle_s
            opened = time.monotonic()
            try:
                if path:
                    await page.goto(f"{WEB}{path}", wait_until="domcontentloaded", timeout=30_000)
                    await page.wait_for_timeout(int(settle * 1000))
                    if scene.actions:
                        # Interactions happen inside the settle window, before the narration starts, so
                        # the scene opens on a page that is already showing what is being described.
                        failed = await act(page, scene.actions, i)
                        if failed:
                            note = f"interaction: {failed}"
                        await page.wait_for_timeout(1200)
                else:
                    if scene.path:
                        note = "no row in the corpus to open this page on"
                    # A scene with no page of its own continues on the previous one, so the previous
                    # page is re-opened rather than left as a blank tab.
                    prev = next((e["path_resolved"] for e in reversed(entries) if e["path_resolved"]), None)
                    if prev:
                        await page.goto(f"{WEB}{prev}", wait_until="domcontentloaded", timeout=30_000)
                        await page.wait_for_timeout(int(settle * 1000))
            except Exception as exc:
                note = f"load failed: {str(exc).splitlines()[0][:110]}"
            # What to trim from the front is measured, not assumed: with interactions the preamble is
            # the load plus however long the clicking took, and guessing it would put the narration over
            # a page that is still being set up.
            lead = time.monotonic() - opened
            live = getattr(scene, "live", None)
            if live and path:
                # Interactions the viewer should see happen inside the recorded window, while the narration
                # describes them; the scene's length is still the narration's, so whatever they took is
                # subtracted from the hold rather than added to the segment.
                started = time.monotonic()
                failed = await act(page, live, i)
                if failed:
                    note = (note + "; " if note else "") + f"live: {failed}"
                spent = time.monotonic() - started
                await page.wait_for_timeout(int(max(0.0, dur - spent) * 1000))
            else:
                await page.wait_for_timeout(int(dur * 1000))
            await ctx.close()   # the video file is only finalised on close

            vids = sorted(raw.glob("*.webm"))
            if not vids:
                note = (note + "; " if note else "") + "the browser produced no video for this scene"
                print(f"  [{n}/{len(scenes)}] {scene.key}: {note}")
            else:
                fit_duration(vids[0], segdir / f"{scene.key}.mp4", dur, start=lead)
                for v in vids:
                    v.unlink(missing_ok=True)
                raw.rmdir()

            pad_audio(wav, auddir / f"{scene.key}.pad.wav", dur)
            entries.append(_entry(scene, n, path, text, t, dur, note))
            t += dur
            print(f"  [{n}/{len(scenes)}] {scene.key:22s} {dur:6.2f}s  {path or '(continues)'}")
        await browser.close()
    return entries


async def check_pages() -> int:
    """Open every page the tour visits and report what breaks, without recording anything.

    Worth having as its own mode. A broken route found here costs one line of output; found during a
    recording it costs the whole take, because the segments after it are already timed against audio
    that has been synthesised.
    """
    from playwright.async_api import async_playwright

    if SCENES is narration.SCENES:
        check()
    f, i = await facts(), await ids()
    bad: list[tuple[str, str]] = []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1920, "height": 1080})
        await ctx.add_init_script(ONBOARDED)
        await ctx.add_init_script(signed_in())
        page = await ctx.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)[:120]))
        for n, scene in enumerate(SCENES, 1):
            path, _ = resolve(scene, f, i)
            if not path:
                if scene.path:
                    bad.append((scene.key, f"no row to fill {scene.path}"))
                continue
            errors.clear()
            try:
                resp = await page.goto(f"{WEB}{path}", wait_until="domcontentloaded", timeout=30_000)
                await page.wait_for_timeout(1200)
                status = resp.status if resp else 0
                body = (await page.inner_text("body"))[:400] if resp else ""
            except Exception as exc:
                bad.append((scene.key, f"{path}: {str(exc).splitlines()[0][:90]}"))
                continue
            why = ""
            if status >= 400:
                why = f"HTTP {status}"
            elif "Application error" in body or "Unhandled Runtime Error" in body:
                why = "runtime error on the page"
            elif errors:
                why = f"console: {errors[0]}"
            elif len(body.strip()) < 40:
                why = "page rendered almost nothing"
            if why:
                bad.append((scene.key, f"{path}: {why}"))
            print(f"  [{n}/{len(SCENES)}] {scene.key:22s} {'BAD ' if why else 'ok  '} {path}")
        await browser.close()
    print(f"\n{len(SCENES)} scenes, {len(bad)} problems")
    for k, why in bad:
        print(f"  {k:22s} {why}")
    return len(bad)


def assemble(entries: list[dict], out: Path) -> None:
    segdir, auddir = OUT / "segments", OUT / "audio"
    segs = [segdir / f"{e['key']}.mp4" for e in entries]
    auds = [auddir / f"{e['key']}.pad.wav" for e in entries]
    missing = [p.name for p in segs + auds if not p.exists()]
    if missing:
        raise SystemExit(f"cannot assemble, {len(missing)} pieces missing, first: {missing[0]}")

    # Every segment must be the length the manifest says, because the captions and the chapter marks are
    # timed from the manifest and the audio is padded to it. A segment even one frame short shifts
    # everything after it, and the shift accumulates silently across the film: the first cut of this tour
    # ended with the audio 339 milliseconds past the picture, from five short segments out of seventy
    # five. Half a frame is the tolerance; nothing correct can miss by that much.
    tol = 0.5 / FPS
    off = [(e["key"], e["duration"], round(probe_duration(p), 4))
           for e, p in zip(entries, segs, strict=True)
           if abs(probe_duration(p) - e["duration"]) > tol]
    if off:
        lines = "\n".join(f"  {k}: manifest {w:.3f}s, file {g:.4f}s" for k, w, g in off)
        raise SystemExit(f"cannot assemble, {len(off)} segments are not the length the manifest says:\n"
                         f"{lines}\nre-record them with --only, or refit them.")
    concat(segs, OUT / "video.mp4")
    concat_audio(auds, OUT / "audio.m4a")
    # The SRT goes to its own directory, never beside the video: a player auto-loads a `.srt` that
    # shares the video's name and draws it over the caption already burned into the picture.
    (OUT / "transcript").mkdir(parents=True, exist_ok=True)
    write_srt(entries, OUT / "transcript" / "tour.srt")
    write_ass(entries, OUT / "tour.ass")
    n = write_chapters(entries, OUT / "chapters.txt")
    save_manifest(entries, OUT / "manifest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    mux(OUT / "video.mp4", OUT / "audio.m4a", OUT / "tour.ass", out, chapters=OUT / "chapters.txt")
    print(f"{n} chapters written into the file")


def merge(existing: list[dict], fresh: list[dict]) -> list[dict]:
    """Fold re-recorded scenes back into the full manifest, then re-time everything after them.

    Without this, `--only` would save a manifest containing just the scenes it re-recorded and the tour
    would assemble as those scenes alone. And a re-recorded scene rarely comes back the same length,
    since the narration may have been edited, so every later scene's start has to move with it or the
    captions after the edit would all be wrong by the difference.
    """
    by_key = {e["key"]: e for e in fresh}
    out = [by_key.get(e["key"], e) for e in existing]
    t = 0.0
    for n, e in enumerate(out, 1):
        e["index"], e["start"] = n, round(t, 3)
        t += e["duration"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma separated scene keys, for re-recording a few")
    ap.add_argument("--out", default="site/demo/labeloxav-tour.mp4")
    ap.add_argument("--no-capture", action="store_true", help="narration and timing only, no screen grab")
    ap.add_argument("--assemble-only", action="store_true")
    ap.add_argument("--check-pages", action="store_true", help="load every page, record nothing")
    ap.add_argument("--script", default="tour",
                    help="scene list module: 'tour' (default) or 'verification'")
    a = ap.parse_args()
    use_script(a.script)

    if a.check_pages:
        raise SystemExit(1 if asyncio.run(check_pages()) else 0)

    if a.assemble_only:
        import json
        entries = json.loads((OUT / "manifest.json").read_text())
    else:
        only = [k for k in a.only.split(",") if k]
        entries = asyncio.run(record(only, a.no_capture))
        if only and (OUT / "manifest.json").exists():
            import json
            entries = merge(json.loads((OUT / "manifest.json").read_text()), entries)
        save_manifest(entries, OUT / "manifest.json")

    total = sum(e["duration"] for e in entries)
    failed = [e for e in entries if e["note"]]
    print(f"\n{len(entries)} scenes, {total / 60:.1f} minutes")
    if failed:
        print(f"{len(failed)} scenes recorded a page that did not load cleanly:")
        for e in failed:
            print(f"  {e['key']:22s} {e['note']}")
    if not a.no_capture:
        assemble(entries, Path(a.out))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
