"""The look of the edited tour: canvas, scene labels, chapter cards, intro and outro.

Every graphic is HTML and CSS rendered by headless Chromium, because typography, gradients, blur and
easing are what a browser is good at and what an ffmpeg filter graph is bad at. Animated graphics are not
screen-recorded. Recording a page captures whatever frames the compositor happens to paint, which drops
frames under load and makes a two second title animation stutter. Instead every animation is paused and
each output frame is rendered by setting the animation clock directly and taking a screenshot, so frame
240 of the intro is the same image on every run.

Palette and type come from the product. Navy #1A2132 and violet #7044DF are sampled from the logo, and
Space Grotesk is the typeface the web app itself uses, so the titles look like the software on screen
rather than like a template laid over it.
"""

from __future__ import annotations

import asyncio
import html
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent / "brand"
FONTS = HERE / "fonts"
W, H, FPS = 1920, 1080, 30

# Where the recorded UI sits on the canvas. The margins are not decoration: the label lives in the top one
# and the captions in the bottom one, so neither covers the interface being narrated, which the first cut
# of this tour did on every scene.
FRAME_X, FRAME_Y, FRAME_W, FRAME_H, FRAME_R = 160, 52, 1600, 900, 16

BG = "#090C13"
NAVY = "#1A2132"
VIOLET = "#7044DF"
VIOLET_LIGHT = "#9C7BFF"
TEXT = "#EEF1F6"
MUTED = "#8C95A8"


def _css() -> str:
    return f"""
    @font-face {{ font-family: 'Space Grotesk'; src: url('file://{FONTS}/SpaceGrotesk-Variable.woff2') format('woff2');
                 font-weight: 300 700; }}
    @font-face {{ font-family: 'Space Mono'; src: url('file://{FONTS}/SpaceMono-Regular.woff2') format('woff2'); }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html, body {{ width: {W}px; height: {H}px; overflow: hidden; }}
    body {{ font-family: 'Ubuntu Sans', 'Noto Sans', sans-serif; color: {TEXT}; -webkit-font-smoothing: antialiased; }}
    .display {{ font-family: 'Space Grotesk', 'Ubuntu Sans', sans-serif; }}
    .mono {{ font-family: 'Space Mono', 'Ubuntu Mono', monospace; }}
    /* Every animation starts paused; the renderer sets its clock frame by frame. */
    * {{ animation-play-state: paused !important; }}
    @keyframes rise {{ from {{ opacity: 0; transform: translateY(26px); }} to {{ opacity: 1; transform: none; }} }}
    @keyframes fade {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
    @keyframes grow {{ from {{ transform: scaleX(0); }} to {{ transform: scaleX(1); }} }}
    @keyframes settle {{ from {{ transform: scale(1.09); }} to {{ transform: scale(1.0); }} }}
    @keyframes pop {{ from {{ opacity: 0; transform: scale(.86); }} to {{ opacity: 1; transform: none; }} }}
    @keyframes drift {{ from {{ transform: scale(1.12) translateX(-18px); }} to {{ transform: scale(1.02) translateX(18px); }} }}
    .a {{ opacity: 0; animation-fill-mode: both; animation-timing-function: cubic-bezier(.16,1,.3,1); }}
    """


def _backdrop(glow_x: str = "14%", glow_y: str = "6%", base: bool = True) -> str:
    """Glow and dot grid. `base` adds the opaque ground colour, which must be left off when the backdrop is
    laid over footage: with it, the first card render showed no screenshot at all, because the ground
    colour was painted on top of it."""
    return f"""
    {f'<div style="position:absolute;inset:0;background:{BG}"></div>' if base else ''}
    <div style="position:absolute;inset:0;background:
        radial-gradient(900px 520px at {glow_x} {glow_y}, rgba(112,68,223,.20), transparent 70%),
        radial-gradient(1100px 620px at 92% 108%, rgba(46,72,140,.22), transparent 70%)"></div>
    <div style="position:absolute;inset:0;opacity:.07;
        background-image: radial-gradient(rgba(255,255,255,.9) 1px, transparent 1.3px);
        background-size: 30px 30px;
        -webkit-mask-image: radial-gradient(1200px 700px at 50% 45%, #000 20%, transparent 80%)"></div>
    """


def canvas_html() -> str:
    """The still background every scene is composited onto: glow, faint grid, the frame's shadow, the mark."""
    return f"""<html><head><style>{_css()}</style></head><body>
    {_backdrop()}
    <div style="position:absolute;left:{FRAME_X}px;top:{FRAME_Y}px;width:{FRAME_W}px;height:{FRAME_H}px;
        border-radius:{FRAME_R}px;background:#15181f;
        box-shadow: 0 34px 90px rgba(0,0,0,.62), 0 10px 26px rgba(0,0,0,.35), 0 0 0 1px rgba(255,255,255,.07)"></div>
    <div class="display" style="position:absolute;right:{W - FRAME_X - FRAME_W}px;top:11px;height:32px;display:flex;align-items:center;gap:10px;opacity:.92">
      <img src="file://{HERE}/mark-light.png" style="height:26px">
      <span style="font-size:20px;font-weight:600;letter-spacing:-.01em">Labelox<span style="color:{VIOLET_LIGHT}">AV</span></span>
    </div>
    </body></html>"""


def mask_html() -> str:
    """White rounded rectangle on black, used as the alpha mask that gives the recording its corners."""
    return f"""<html><head><style>{_css()} body{{background:#000}}</style></head><body>
    <div style="position:absolute;left:0;top:0;width:{FRAME_W}px;height:{FRAME_H}px;border-radius:{FRAME_R}px;background:#fff"></div>
    </body></html>"""


def label_html(number: int, chapter: str, title: str) -> str:
    """Chapter and scene name for the top margin, on a transparent page so it can be animated over video."""
    return f"""<html><head><style>{_css()} html,body{{background:transparent}}</style></head><body>
    <div class="display" style="position:absolute;left:{FRAME_X}px;top:12px;height:30px;display:flex;align-items:center;gap:14px;white-space:nowrap">
      <span style="color:{VIOLET_LIGHT};font-weight:600;font-size:19px;letter-spacing:.02em">{number:02d}</span>
      <span style="color:{MUTED};font-size:14px;letter-spacing:.16em;text-transform:uppercase">{html.escape(chapter)}</span>
      <span style="width:4px;height:4px;border-radius:2px;background:rgba(255,255,255,.28)"></span>
      <span style="color:{TEXT};font-size:19px;font-weight:500">{html.escape(title)}</span>
    </div></body></html>"""


def card_html(number: int, chapter: str, blurb: str, backdrop: Path) -> str:
    """A chapter card over a blurred still of the chapter's first screen, so the card previews what follows."""
    return f"""<html><head><style>{_css()}</style></head><body style="background:{BG}">
    <div class="a" style="position:absolute;inset:-40px;opacity:1;animation:settle 3.6s cubic-bezier(.2,.7,.2,1) both">
      <img src="file://{backdrop}" style="width:100%;height:100%;object-fit:cover;filter:blur(18px) brightness(.5) saturate(.85)">
    </div>
    <div style="position:absolute;inset:0;background:linear-gradient(90deg, rgba(9,12,19,.95) 0%, rgba(9,12,19,.78) 44%, rgba(9,12,19,.30) 100%)"></div>
    <div class="a display" style="position:absolute;right:120px;top:50%;margin-top:-260px;height:520px;line-height:520px;font-size:520px;font-weight:700;
         letter-spacing:-.04em;color:transparent;-webkit-text-stroke:2px rgba(156,123,255,.30);animation:rise 1.4s .15s both">{number:02d}</div>
    {_backdrop("8%", "20%", base=False)}
    <div style="position:absolute;left:220px;top:0;bottom:0;display:flex;flex-direction:column;justify-content:center;max-width:1300px">
      <div class="a display" style="animation:rise .9s .10s both;color:{VIOLET_LIGHT};font-size:30px;font-weight:600;letter-spacing:.24em">CHAPTER {number:02d}</div>
      <div class="a display" style="animation:rise 1.0s .28s both;font-size:108px;font-weight:600;line-height:1.02;margin-top:18px;letter-spacing:-.02em;text-wrap:balance">{html.escape(chapter)}</div>
      <div class="a" style="opacity:1;animation:grow 1.0s .55s both;transform-origin:left;width:240px;height:4px;border-radius:2px;margin:34px 0 30px;
           background:linear-gradient(90deg,{VIOLET},{VIOLET_LIGHT})"></div>
      <div class="a" style="animation:rise 1.0s .72s both;font-size:36px;color:#C3C9D6;line-height:1.35;max-width:1000px;text-wrap:balance">{html.escape(blurb)}</div>
    </div>
    </body></html>"""


def intro_html(stills: list[Path]) -> str:
    """Title sequence: real footage drifting behind the mark, name, promise and what the product covers."""
    n = len(stills)
    slide = 9.5 / n
    imgs = "".join(
        f"""<div class="a" style="position:absolute;inset:0;opacity:0;animation:fade .9s {i * slide:.2f}s both">
              <img src="file://{p}" style="position:absolute;inset:-60px;width:calc(100% + 120px);height:calc(100% + 120px);object-fit:cover;
                   filter:blur(10px) brightness(.30) saturate(.8);animation:drift 9.5s linear both"></div>"""
        for i, p in enumerate(stills))
    chips = ["Auto-labelling", "Human review", "Tracking", "LiDAR and 3D", "HD maps", "Governance"]
    chip_html = "".join(
        f"""<span class="a display" style="animation:pop .7s {3.05 + i * 0.11:.2f}s both;padding:12px 24px;border-radius:999px;font-size:24px;
              background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.14);color:#DCE1EA">{c}</span>"""
        for i, c in enumerate(chips))
    return f"""<html><head><style>{_css()}</style></head><body style="background:{BG}">
    {imgs}
    <div style="position:absolute;inset:0;background:radial-gradient(1200px 760px at 50% 44%, rgba(9,12,19,.35), rgba(9,12,19,.94) 78%)"></div>
    {_backdrop("50%", "30%", base=False)}
    <div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center">
      <img class="a" src="file://{HERE}/mark-light.png" style="height:170px;animation:pop 1.1s .25s both">
      <div class="a display" style="animation:rise 1.1s .95s both;font-size:132px;font-weight:600;letter-spacing:-.03em;margin-top:28px;line-height:1">
        Labelox<span style="color:{VIOLET_LIGHT}">AV</span></div>
      <div class="a" style="animation:rise 1.0s 1.75s both;font-size:40px;color:#C3C9D6;margin-top:26px">
        A data engine for autonomous driving, built for Indian roads</div>
      <div style="display:flex;gap:14px;margin-top:52px;flex-wrap:wrap;justify-content:center;max-width:1500px">{chip_html}</div>
      <div class="a mono" style="animation:fade 1.0s 4.3s both;font-size:21px;color:{MUTED};margin-top:46px;letter-spacing:.04em">
        A narrated tour of every screen, recorded live against the running system</div>
    </div>
    </body></html>"""


def outro_html(stills: list[Path], version: str, credit: str = "") -> str:
    """End card: where to get it, how to install it, and the one claim the whole tour is built on."""
    imgs = "".join(
        f"""<img class="a" src="file://{p}" style="position:absolute;inset:-60px;width:calc(100% + 120px);height:calc(100% + 120px);object-fit:cover;
              filter:blur(14px) brightness(.22) saturate(.7);opacity:0;animation:fade 1.2s {i * 4.0:.1f}s both">"""
        for i, p in enumerate(stills))
    def row(k: str, v: str, d: float) -> str:
        return f"""<div class="a" style="animation:rise .9s {d:.2f}s both;display:flex;align-items:baseline;gap:26px">
        <span class="display" style="width:170px;text-align:right;color:{MUTED};font-size:22px;letter-spacing:.14em">{k}</span>{v}</div>"""
    return f"""<html><head><style>{_css()}</style></head><body style="background:{BG}">
    {imgs}
    <div style="position:absolute;inset:0;background:radial-gradient(1300px 800px at 50% 50%, rgba(9,12,19,.45), rgba(9,12,19,.95) 80%)"></div>
    {_backdrop("50%", "12%", base=False)}
    <div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center">
      <img class="a" src="file://{HERE}/logo-light.png" style="height:210px;animation:pop 1.1s .2s both">
      <div class="a display" style="animation:rise 1.0s .85s both;font-size:54px;font-weight:600;margin-top:40px;letter-spacing:-.01em">
        Open source, Apache 2.0</div>
      <div style="display:flex;flex-direction:column;gap:24px;margin-top:56px">
        {row("CODE", f'<span class="display" style="font-size:38px">github.com/Sherin-SEF-AI/<b style="color:{VIOLET_LIGHT};font-weight:600">LabeloxAV</b></span>', 1.35)}
        {row("INSTALL", f'<span class="mono" style="font-size:30px;padding:12px 22px;border-radius:12px;background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.13)">sudo apt install ./labeloxav_{version}_all.deb</span>', 1.55)}
        {row("DOCS", '<span class="display" style="font-size:34px">sherin-sef-ai.github.io/LabeloxAV</span>', 1.75)}
      </div>
      <div class="a" style="animation:fade 1.0s 2.6s both;margin-top:70px;font-size:26px;color:#AEB6C5">
        Every number in this tour was read from the live database at the moment it was recorded.</div>
      {f'<div class="a" style="animation:fade 1.0s 3.0s both;margin-top:26px;font-size:19px;color:{MUTED};max-width:1500px;text-align:center;line-height:1.5">{html.escape(credit)}</div>' if credit else ''}
    </div>
    </body></html>"""


@dataclass
class Job:
    html: str
    out: Path            # a directory of frames for an animation, or a single PNG for a still
    seconds: float = 0.0  # 0 renders one still
    transparent: bool = False


async def _render_one(browser, job: Job, work: Path) -> None:
    page = await browser.new_page(viewport={"width": W, "height": H})
    src = work / f"{job.out.stem}.html"
    src.write_text(job.html, encoding="utf-8")
    await page.goto(f"file://{src}")
    await page.evaluate("document.fonts.ready")
    await page.evaluate("""Promise.all(Array.from(document.images).map(i => i.complete ? 0 :
                           new Promise(r => { i.onload = r; i.onerror = r; })))""")
    if job.seconds <= 0:
        await page.evaluate("document.getAnimations().forEach(a => { a.pause(); a.currentTime = 1e7; })")
        job.out.parent.mkdir(parents=True, exist_ok=True)
        await page.screenshot(path=str(job.out), omit_background=job.transparent)
    else:
        job.out.mkdir(parents=True, exist_ok=True)
        frames = int(round(job.seconds * FPS))
        for i in range(frames):
            await page.evaluate("t => document.getAnimations().forEach(a => { a.pause(); a.currentTime = t; })",
                                i * 1000.0 / FPS)
            await page.screenshot(path=str(job.out / f"{i:05d}.jpg"), type="jpeg", quality=93)
    await page.close()


async def render(jobs: list[Job], work: Path, parallel: int = 4) -> None:
    from playwright.async_api import async_playwright

    work.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(parallel)
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--force-device-scale-factor=1", "--hide-scrollbars"])

        async def run(job: Job) -> None:
            async with sem:
                await _render_one(browser, job, work)

        await asyncio.gather(*(run(j) for j in jobs))
        await browser.close()
