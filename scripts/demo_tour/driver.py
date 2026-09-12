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
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from narration import SCENES, Scene, check  # noqa: E402
from record import (  # noqa: E402
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
    quantize,
    save_manifest,
    synth,
    write_ass,
    write_chapters,
    write_srt,
)

# The key `components/shell/Onboarding.tsx` writes when somebody finishes or skips the welcome tour.
ONBOARDED = "try { localStorage.setItem('lbx_onboarded', new Date().toISOString()); } catch (e) {}"


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
        "poses": "select count(*) from ego_pose where measured is true",
        "points": "select coalesce(sum(point_count), 0) / 1000000 from point_cloud where session_id = :kitti",
        # The recordings this tour was filmed on. Counted rather than written down: the first version of
        # this scene said "three recordings, two dashcam clips" because that was true on the afternoon the
        # script was written, and it stopped being true when a third clip was labelled. A tour that
        # insists every figure comes from the database should not make an exception for the small ones.
        "demo_recordings": "select count(*) from session where route = 'demo-tour-2026' or route like 'KITTI 2011%'",
        "demo_dashcam": "select count(*) from session where route = 'demo-tour-2026'",
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
        for name, sql in queries.items():
            try:
                v = (await db.execute(text(sql), {"kitti": kitti})).scalar()
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
    for step in actions:
        kind, args = step[0], [sub(a) for a in step[1:]]
        try:
            if kind == "fill":
                await page.fill(args[0], args[1], timeout=8000)
            elif kind == "click":
                await page.click(args[0], timeout=8000)
            elif kind == "click_text":
                await page.get_by_text(args[0], exact=False).first.click(timeout=8000)
            elif kind == "press":
                await page.keyboard.press(args[0])
            elif kind == "scroll":
                await page.mouse.wheel(0, int(args[0]))
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

    check()
    missing = check_tools()
    if missing:
        raise SystemExit(f"cannot record, missing: {', '.join(missing)}")

    f, i = await facts(), await ids()
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
            page = await ctx.new_page()
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

    check()
    f, i = await facts(), await ids()
    bad: list[tuple[str, str]] = []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1920, "height": 1080})
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
    concat(segs, OUT / "video.mp4")
    concat_audio(auds, OUT / "audio.m4a")
    write_srt(entries, OUT / "tour.srt")
    write_ass(entries, OUT / "tour.ass")
    n = write_chapters(entries, OUT / "chapters.txt")
    save_manifest(entries, OUT / "manifest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    mux(OUT / "video.mp4", OUT / "audio.m4a", OUT / "tour.srt", OUT / "tour.ass", out,
        chapters=OUT / "chapters.txt")
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
    a = ap.parse_args()

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
