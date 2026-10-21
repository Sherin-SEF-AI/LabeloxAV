"""Record a hands-on annotation demo: real Indian road footage, labelled by hand in the running product.

    .venv/bin/python scripts/demo_tour/annotate.py dry  [--only k1,k2]   run the actions, screenshot each scene
    .venv/bin/python scripts/demo_tour/annotate.py record [--only k1,k2] record scenes with narration
    .venv/bin/python scripts/demo_tour/annotate.py assemble               join the scenes and write the transcript

The tour in `driver.py` shows pages. This shows work: boxes dragged, polygons clicked vertex by vertex,
classes chosen, machine proposals accepted and rejected, all through the same mouse and keyboard events a
person would produce, against the live API and database. Every annotation in the recording is a row the
system now holds, and each scene checks that afterwards rather than trusting that the clicks landed.

Four things differ from the tour, and each is deliberate.

**The actions are the content, so they are recorded, not trimmed.** The tour clicks inside the page load
and cuts that part away. Here the scene starts once the page is ready and everything after that is kept.
A scene lasts as long as its actions or its narration, whichever is longer, and the narration plays over
the actions rather than after them.

**The pointer is drawn into the page.** A headless browser records no cursor, and a demo of drawing where
boxes appear from nowhere teaches nothing. An init script draws a pointer that follows the real mouse
events, a ripple on every press, and a badge for every shortcut, so the viewer sees what was pressed.

**Canvas positions come from the canvas itself.** Every click on the image is given in image pixels and
converted with the live Konva transform of the image node, read at the moment of the click. The editor
fits, zooms and pans, so a formula fixed at page load would put clicks in the wrong place after the first
zoom; reading the transform each time cannot drift.

**Resources are watched between scenes.** Each scene starts only while scripts/resource_watch.py has no
pause flag up. The first attempt at this demo ran an auto-label job while a browser held most of memory,
and the kernel's out-of-memory killer fired.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import sys
import time
import urllib.request
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))

import record as R  # noqa: E402
from resource_watch import wait_while_paused  # noqa: E402

OUT = ROOT / ".scratch/demo/india/tour"
FLAG = ROOT / ".scratch/watch/PAUSE.json"
WEB = "http://127.0.0.1:3000"
API = "http://127.0.0.1:8000"
W, H = 1920, 1080
# Drawing is motion, and at the tour's 15 fps a dragged box stutters. 30 matches the finished film.
FPS = 30
R.FPS = FPS
R.OUT = OUT

READY_TIMEOUT_MS = 180_000

# ---------------------------------------------------------------- page overlay

OVERLAY = r"""
(() => {
  const install = () => {
    if (!document.body || document.getElementById('__demo_cursor')) return;
    const st = document.createElement('style');
    st.textContent = `
      #__demo_cursor{position:fixed;left:-40px;top:-40px;width:30px;height:30px;pointer-events:none;z-index:2147483647;
        transform:translate(-4px,-3px);filter:drop-shadow(0 2px 3px rgba(0,0,0,.45))}
      .__demo_ripple{position:fixed;width:40px;height:40px;margin:-20px 0 0 -20px;border-radius:50%;
        border:3px solid #9C7BFF;pointer-events:none;z-index:2147483646;animation:__rip .6s ease-out forwards}
      @keyframes __rip{from{transform:scale(.25);opacity:1}to{transform:scale(1.5);opacity:0}}
      #__demo_keys{position:fixed;right:28px;bottom:72px;display:flex;flex-direction:column;align-items:flex-end;
        gap:8px;pointer-events:none;z-index:2147483647}
      .__demo_key{display:flex;align-items:center;gap:12px;font:600 21px/1 ui-monospace,SFMono-Regular,Menlo,monospace;
        color:#fff;background:rgba(17,20,29,.9);border:1px solid rgba(156,123,255,.55);border-radius:10px;
        padding:10px 14px;box-shadow:0 8px 24px rgba(0,0,0,.4);animation:__kin .2s ease-out}
      .__demo_key span{font:500 16px/1 system-ui,sans-serif;color:#D5D9E3}
      @keyframes __kin{from{transform:translateY(10px);opacity:0}to{transform:none;opacity:1}}`;
    document.head.appendChild(st);
    const c = document.createElement('div');
    c.id = '__demo_cursor';
    c.innerHTML = '<svg width="30" height="30" viewBox="0 0 30 30"><path d="M4 2 L4 24 L10 18.5 L14 27.5 L18 25.8 ' +
      'L14 17 L22.5 17 Z" fill="#fff" stroke="#15171f" stroke-width="1.8" stroke-linejoin="round"/></svg>';
    document.body.appendChild(c);
    const k = document.createElement('div');
    k.id = '__demo_keys';
    document.body.appendChild(k);
    const at = window.__demoAt || {x: -40, y: -40};
    c.style.left = at.x + 'px'; c.style.top = at.y + 'px';
  };
  window.addEventListener('mousemove', e => {
    window.__demoAt = {x: e.clientX, y: e.clientY};
    const c = document.getElementById('__demo_cursor');
    if (c) { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; }
  }, true);
  window.addEventListener('mousedown', e => {
    const r = document.createElement('div');
    r.className = '__demo_ripple'; r.style.left = e.clientX + 'px'; r.style.top = e.clientY + 'px';
    document.body.appendChild(r); setTimeout(() => r.remove(), 650);
  }, true);
  window.__demoKey = (label, hint) => {
    install();
    const box = document.getElementById('__demo_keys'); if (!box) return;
    const el = document.createElement('div'); el.className = '__demo_key';
    el.textContent = label;
    if (hint) { const s = document.createElement('span'); s.textContent = hint; el.appendChild(s); }
    box.appendChild(el); setTimeout(() => el.remove(), 1900);
  };
  // React owns the document after hydration and may replace nodes it did not render, so the overlay is
  // re-installed whenever it goes missing rather than once.
  setInterval(install, 400);
  if (document.readyState !== 'loading') install(); else document.addEventListener('DOMContentLoaded', install);
})();
"""

# Image pixel to viewport position, through the live transform of the Konva image node.
IMG_TO_CLIENT = r"""
([x, y]) => {
  const K = window.Konva;
  if (!K || !K.stages) return null;
  const stage = K.stages.filter(s => s.container() && s.container().isConnected)
    .sort((a, b) => b.width() * b.height() - a.width() * a.height())[0];
  if (!stage) return null;
  const img = stage.findOne('Image');
  if (!img || !img.image()) return null;
  const nw = img.image().naturalWidth || img.width(), nh = img.image().naturalHeight || img.height();
  const p = img.getAbsoluteTransform().point({x: x * img.width() / nw, y: y * img.height() / nh});
  const r = stage.container().getBoundingClientRect();
  return {x: r.left + p.x, y: r.top + p.y};
}
"""


def _auth() -> dict:
    req = urllib.request.Request(API + "/api/auth/dev-login", data=b"{}", method="POST",
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


class Demo:
    """One page, driven the way a person would drive it, with the pointer and keys made visible."""

    def __init__(self, page, user: dict) -> None:
        self.page = page
        self.user = user
        self.pos = (W / 2, H / 2)
        self.log: list[str] = []

    # --- pointer
    async def move(self, x: float, y: float, *, ms: int = 450) -> None:
        steps = max(int(ms / 16), 2)
        await self.page.mouse.move(x, y, steps=steps)
        self.pos = (x, y)

    async def img(self, x: float, y: float) -> tuple[float, float]:
        # A mode switch remounts the canvas, and for a moment there is no image node to measure against.
        for _ in range(50):
            p = await self.page.evaluate(IMG_TO_CLIENT, [x, y])
            if p:
                return p["x"], p["y"]
            await self.page.wait_for_timeout(100)
        raise RuntimeError("the editor canvas has no image node")

    async def click_img(self, x: float, y: float, *, ms: int = 450, shift: bool = False, pause: int = 180) -> None:
        cx, cy = await self.img(x, y)
        await self.move(cx, cy, ms=ms)
        if shift:
            await self.page.keyboard.down("Shift")
        await self.page.mouse.down()
        await self.page.wait_for_timeout(60)
        await self.page.mouse.up()
        if shift:
            await self.page.keyboard.up("Shift")
        await self.page.wait_for_timeout(pause)

    async def dblclick_img(self, x: float, y: float) -> None:
        cx, cy = await self.img(x, y)
        await self.move(cx, cy, ms=350)
        await self.page.mouse.dblclick(cx, cy)
        await self.page.wait_for_timeout(250)

    async def drag_img(self, x1: float, y1: float, x2: float, y2: float, *, ms: int = 900) -> None:
        a, b = await self.img(x1, y1), await self.img(x2, y2)
        await self.move(*a, ms=450)
        await self.page.mouse.down()
        await self.page.mouse.move(b[0], b[1], steps=max(int(ms / 16), 4))
        self.pos = b
        await self.page.wait_for_timeout(80)
        await self.page.mouse.up()
        await self.page.wait_for_timeout(300)
        await self._trace("drag")

    async def stroke_img(self, pts: list[tuple[float, float]], *, ms_per_seg: int = 180) -> None:
        """A freehand stroke through image points, for the brush and eraser."""
        first = await self.img(*pts[0])
        await self.move(*first, ms=350)
        await self.page.mouse.down()
        for p in pts[1:]:
            cx, cy = await self.img(*p)
            await self.page.mouse.move(cx, cy, steps=max(int(ms_per_seg / 16), 2))
        await self.page.mouse.up()
        await self.page.wait_for_timeout(300)

    async def zoom_at(self, x: float, y: float, clicks: int, *, pause: int = 70) -> None:
        """Wheel-zoom about an image point: positive clicks zoom in. The editor zooms about the pointer."""
        cx, cy = await self.img(x, y)
        await self.move(cx, cy, ms=400)
        await self.page.evaluate("([l, h]) => window.__demoKey && window.__demoKey(l, h)",
                                 ["Scroll", "zoom in" if clicks > 0 else "zoom out"])
        for _ in range(abs(clicks)):
            await self.page.mouse.wheel(0, -100 if clicks > 0 else 100)
            await self.page.wait_for_timeout(pause)
        await self.page.wait_for_timeout(300)

    async def select_option(self, locator, value: str, hint: str = "") -> None:
        loc = locator.first
        await self.click(loc, after=200)
        await loc.select_option(value)
        await self.page.evaluate("([l, h]) => window.__demoKey && window.__demoKey(l, h)", [value, hint])
        await self.page.wait_for_timeout(400)

    # --- keys and widgets
    async def key(self, combo: str, hint: str = "", *, label: str | None = None, after: int = 350) -> None:
        shown = label or combo.replace("Control+", "Ctrl ").replace("Shift+", "Shift ").replace("Key", "")
        await self.page.evaluate("([l, h]) => window.__demoKey && window.__demoKey(l, h)", [shown, hint])
        await self.page.keyboard.press(combo)
        await self.page.wait_for_timeout(after)
        await self._trace(f"key-{shown}")

    async def _trace(self, tag: str) -> None:
        """With DEMO_TRACE set to a directory, screenshot after every key and drag, for debugging a scene."""
        where = os.environ.get("DEMO_TRACE")
        if not where:
            return
        self._n = getattr(self, "_n", 0) + 1
        Path(where).mkdir(parents=True, exist_ok=True)
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in tag)[:40]
        await self.page.screenshot(path=f"{where}/{self._n:03d}_{safe}.png")

    async def type(self, text: str, *, delay: int = 70) -> None:
        await self.page.keyboard.type(text, delay=delay)

    async def click(self, locator, *, ms: int = 500, after: int = 350) -> None:
        loc = locator.first
        await loc.wait_for(state="visible", timeout=15_000)
        box = await loc.bounding_box()
        if box:
            await self.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, ms=ms)
        await loc.click()
        await self.page.wait_for_timeout(after)

    async def title(self, title: str, **kw) -> None:
        await self.click(self.page.get_by_title(title, exact=False), **kw)

    async def text(self, text: str, **kw) -> None:
        await self.click(self.page.get_by_text(text, exact=False), **kw)

    async def pick_class(self, name: str) -> None:
        """Choose the class for new objects through the panel's class picker, typed as a person would."""
        await self.title("pick the class for new objects")
        box = self.page.get_by_label("search or add class")
        await box.wait_for(state="visible", timeout=10_000)
        await self.type(name)
        await self.page.wait_for_timeout(300)
        await self.key("Enter", f"class: {name}", label="Enter")

    async def wait(self, ms: int) -> None:
        await self.page.wait_for_timeout(ms)

    async def api(self, path: str) -> dict | list:
        """Read from the API as the signed-in user, to check what an action actually wrote."""
        return await self.page.evaluate(
            """async ([p, t]) => { const r = await fetch(p, {headers: {authorization: 'Bearer ' + t}});
                                   return r.ok ? r.json() : {error: r.status}; }""",
            [path, self.user["token"]])


# ---------------------------------------------------------------- scenes

@dataclass
class Scene:
    key: str
    chapter: str
    title: str
    path: str                                   # route, may carry {id} placeholders from the ids dict
    narration: str                              # may carry {fact} placeholders from the facts dict
    ready: str = "body"                         # selector that means the page is usable
    run: Callable[[Demo, dict], Awaitable[None]] | None = None
    verify: Callable[[Demo, dict], Awaitable[str]] | None = None
    # Runs after the page is ready and before the cut point, so it is never in the recording: it puts the
    # data back the way the scene expects, for a scene whose actions change it permanently.
    before: Callable[[Demo, dict], Awaitable[str]] | None = None
    hold_s: float = 1.2
    settle_ms: int = 1500


@dataclass
class Entry:
    key: str
    chapter: str
    title: str
    index: int
    path: str
    text: str
    start: float
    duration: float
    action_s: float
    speech_s: float
    verified: str = ""
    note: str = ""
    extra: dict = field(default_factory=dict)


def _fill(s: str, d: dict) -> str:
    for k, v in d.items():
        s = s.replace("{" + k + "}", str(v))
    return s


async def _open(browser, user: dict, *, video_dir: Path | None):
    kw = {"viewport": {"width": W, "height": H}}
    if video_dir is not None:
        kw.update(record_video_dir=str(video_dir), record_video_size={"width": W, "height": H})
    ctx = await browser.new_context(**kw)
    await ctx.add_init_script(
        f"try {{ localStorage.setItem('lbx_user', {json.dumps(json.dumps(user))});"
        f" localStorage.setItem('lbx_onboarded', new Date().toISOString()); }} catch (e) {{}}")
    await ctx.add_init_script(OVERLAY)
    return ctx


async def _ready(page, scene: Scene) -> None:
    """Wait for the scene's ready selector; on timeout keep a screenshot of what the page showed instead."""
    try:
        await page.wait_for_selector(scene.ready, state="visible", timeout=READY_TIMEOUT_MS)
    except Exception:
        where = OUT / "shots" / f"NOT_READY_{scene.key}.png"
        where.parent.mkdir(parents=True, exist_ok=True)
        await page.screenshot(path=str(where))
        raise


async def _run_scene(browser, user: dict, scene: Scene, facts: dict, ids: dict, *, capture: bool,
                     speech_s: float, n: int) -> tuple[Entry, Path | None, float]:
    raw = OUT / "raw" / scene.key
    ctx = await _open(browser, user, video_dir=raw if capture else None)
    page = await ctx.new_page()
    demo = Demo(page, user)
    note, verified = "", ""
    opened = time.monotonic()
    path = _fill(scene.path, ids)
    await page.goto(WEB + path, wait_until="domcontentloaded", timeout=120_000)
    await _ready(page, scene)
    if scene.before:
        # The page is reloaded afterwards so it shows the data as the step left it.
        verified = await scene.before(demo, ids)
        await page.reload(wait_until="domcontentloaded")
        await _ready(page, scene)
    await page.wait_for_timeout(scene.settle_ms)
    await page.mouse.move(W * 0.62, H * 0.55)
    lead = time.monotonic() - opened
    t0 = time.monotonic()
    if scene.run:
        try:
            await scene.run(demo, ids)
        except Exception as exc:  # noqa: BLE001  a failed step is reported, the take is still kept
            note = f"action failed: {type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
    action_s = time.monotonic() - t0
    dur = R.quantize(max(action_s + scene.hold_s, speech_s + 0.8))
    remaining = dur - (time.monotonic() - t0)
    if remaining > 0:
        await page.wait_for_timeout(int(remaining * 1000))
    shot = OUT / "shots" / f"{n:02d}_{scene.key}.png"
    shot.parent.mkdir(parents=True, exist_ok=True)
    await page.screenshot(path=str(shot))
    if scene.verify:
        try:
            verified = (verified + "; " if verified else "") + await scene.verify(demo, ids)
        except Exception as exc:  # noqa: BLE001
            verified = f"verify failed: {type(exc).__name__}: {exc}"
    await ctx.close()
    entry = Entry(scene.key, scene.chapter, scene.title, n, path, _fill(scene.narration, facts), 0.0, dur,
                  round(action_s, 2), round(speech_s, 2), verified, note)
    vid = None
    if capture:
        vids = sorted(raw.glob("*.webm"))
        vid = vids[0] if vids else None
    return entry, vid, lead


async def run(mode: str, only: list[str] | None, script: str) -> None:
    from playwright.async_api import async_playwright

    mod = importlib.import_module(script)
    scenes: list[Scene] = [s for s in mod.SCENES if not only or s.key in only]
    facts, ids = await mod.facts(), await mod.ids()
    print("facts:", json.dumps(facts))
    print("ids:", json.dumps({k: str(v)[:8] for k, v in ids.items()}))
    capture = mode == "record"
    if capture:
        missing = R.check_tools()
        if missing:
            raise SystemExit(f"cannot record, missing: {', '.join(missing)}")
    (OUT / "segments").mkdir(parents=True, exist_ok=True)
    (OUT / "audio").mkdir(parents=True, exist_ok=True)
    manifest = OUT / "manifest.json"
    entries: dict[str, dict] = {}
    if manifest.exists():
        entries = {e["key"]: e for e in json.loads(manifest.read_text())}

    user = _auth()
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=["--hide-scrollbars", "--force-device-scale-factor=1"])
        for scene in scenes:
            if not wait_while_paused(FLAG, timeout_s=1800):
                raise SystemExit(f"stopping before {scene.key}: resources over their limits: {FLAG.read_text()}")
            wav = OUT / "audio" / f"{scene.key}.wav"
            speech = R.synth(_fill(scene.narration, facts), wav) if capture else 0.0
            index = [s.key for s in mod.SCENES].index(scene.key) + 1
            entry, vid, lead = await _run_scene(browser, user, scene, facts, ids, capture=capture,
                                                speech_s=speech, n=index)
            if capture:
                if vid is None:
                    entry.note = (entry.note + "; " if entry.note else "") + "no video produced"
                else:
                    R.fit_duration(vid, OUT / "segments" / f"{scene.key}.mp4", entry.duration, start=lead)
                    for v in (OUT / "raw" / scene.key).glob("*.webm"):
                        v.unlink()
                R.pad_audio(wav, OUT / "audio" / f"{scene.key}.pad.wav", entry.duration)
                entries[scene.key] = asdict(entry)
            flag = f"  NOTE {entry.note}" if entry.note else ""
            print(f"[{index:02d}] {scene.key:24s} actions {entry.action_s:5.1f}s  scene {entry.duration:5.1f}s  "
                  f"{entry.verified}{flag}", flush=True)
        await browser.close()

    if capture:
        order = [s.key for s in mod.SCENES]
        rows = [entries[k] for k in order if k in entries]
        t = 0.0
        for i, e in enumerate(rows, 1):
            e["index"], e["start"] = i, round(t, 3)
            t += e["duration"]
        manifest.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def assemble(script: str) -> None:
    mod = importlib.import_module(script)
    rows = json.loads((OUT / "manifest.json").read_text())
    order = [s.key for s in mod.SCENES]
    missing = [k for k in order if k not in {e["key"] for e in rows}]
    if missing:
        raise SystemExit(f"cannot assemble, scenes not recorded yet: {missing}")
    tol = 0.5 / FPS
    off = [(e["key"], e["duration"], R.probe_duration(OUT / "segments" / f"{e['key']}.mp4")) for e in rows]
    off = [o for o in off if abs(o[1] - o[2]) > tol]
    if off:
        raise SystemExit(f"segments not the manifest length: {off}")
    (OUT / "transcript").mkdir(parents=True, exist_ok=True)
    R.write_srt(rows, OUT / "transcript" / "demo.srt")
    total = sum(e["duration"] for e in rows)
    print(f"{len(rows)} scenes, {total / 60:.2f} min, transcript at {OUT / 'transcript' / 'demo.srt'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["dry", "record", "assemble"])
    ap.add_argument("--only", default="")
    ap.add_argument("--script", default="india_scenes")
    args = ap.parse_args()
    if args.mode == "assemble":
        assemble(args.script)
        return
    only = [k for k in args.only.split(",") if k] or None
    asyncio.run(run(args.mode, only, args.script))


if __name__ == "__main__":
    main()
