"""Edit the recorded tour into a finished film: intro, chapter cards, framing, transitions, music, outro.

    .venv/bin/python scripts/demo_tour/produce.py plan        what it will build, and the framing per scene
    .venv/bin/python scripts/demo_tour/produce.py all         build everything into .scratch/demo/pro/

It edits footage that already exists and never touches the running system. The inputs are the per-scene
recordings and narration clips `driver.py` wrote to .scratch/demo/tour, and the tour's transcript, which
is where each scene's exact spoken text is recovered from. So the product does not need to be running,
and the numbers spoken in the film are the ones read from the database on the day it was recorded.

Three editorial decisions are worth knowing.

**The interface sits in a frame, not full screen.** The first cut burned captions over the bottom of every
page, covering the part of the interface being described. Here the recording is scaled into a rounded
frame on a designed canvas, the captions live in the margin below it and the chapter and scene name in
the margin above, so nothing covers the product.

**Sparse pages are zoomed, measured rather than guessed.** About a fifth of the recorded pages put a small
panel in one corner of a large empty screen, and at full frame the thing being narrated is too small to
read. Each scene is sampled, the region with content in it is found, and if it is small enough to be worth
it the frame eases in on that region after a beat. A page that already fills the screen is left alone.

**Audio is lossless until the last step.** Every intermediate carries PCM rather than AAC. An AAC stream
is padded to a whole frame, up to 21 ms, and the film joins 89 clips; left compressed, that padding would
walk the voice out of sync by up to two seconds by the end, which is exactly the class of drift an earlier
version of this tooling spent a commit removing.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import brand as B  # noqa: E402
import narration  # noqa: E402

TOUR = ROOT / ".scratch/demo/tour"
WORK = ROOT / ".scratch/demo/pro"
TRANSCRIPT = ROOT / "docs/demo/transcript/labeloxav-tour.srt"
VERSION = "0.1.1"

FPS = 30
LEAD = 0.3           # held frame before each scene, so a crossfade never lands on the first word
XF_SCENE = 0.45      # between scenes inside a chapter
XF_CARD = 0.6        # from a chapter card into its first scene
XF_CHAPTER = 0.8     # into a chapter card, and into and out of the intro and outro
CARD_S = 3.6
INTRO_S = 9.5
OUTRO_S = 12.0

# Zoom is only applied where it earns its keep: below this the page already fills the frame well enough,
# and above the cap text is magnified past the point where it stops looking like a screen.
MIN_ZOOM, MAX_ZOOM = 1.2, 1.75
ZOOM_AT, ZOOM_FOR = 1.2, 1.4   # seconds into the scene the move starts, and how long it takes
CHROME_PX = 124                # app menu, breadcrumb and page title bar, which every page shares

BLURBS = {
    "Introduction": "One data spine under every platform in the product",
    "Ingest": "Dashcam video, sensor drives, calibration and the vehicle's own motion",
    "Auto-labelling": "Models propose, a confidence gate decides what reaches a person",
    "Review": "Human verdicts, ranked by what each one is worth",
    "Tracking and multi-camera": "Identity across frames, and across the cameras on one vehicle",
    "3D and LiDAR": "Real laser points, cuboids, and a world frame that stays put while the vehicle moves",
    "Measurement": "Accuracy with honest intervals, slices and counterfactuals",
    "Autonomy": "It proposes its own work and cannot approve itself",
    "The data engine": "Deciding which frames are worth labelling next",
    "Export and privacy": "Datasets, redaction and a budget for what leaves",
    "Edge": "From a trained model to the vehicle, and back",
    "Closing": "What is real today, and what is still thin",
}


def q(seconds: float) -> float:
    """Round to a whole frame, so every duration and offset is something a 30 fps stream can represent."""
    return round(seconds * FPS) / FPS


def run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:6])} ... failed:\n{r.stderr[-2500:]}")


def probe(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.strip()
    return float(out)


@dataclass
class Cue:
    start: float
    end: float
    text: str


@dataclass
class SceneClip:
    key: str
    chapter: str
    chapter_no: int
    title: str
    seg: Path
    audio: Path
    seg_dur: float
    cues: list[Cue] = field(default_factory=list)
    zoom: dict | None = None

    @property
    def dur(self) -> float:
        return q(self.seg_dur + LEAD)


@dataclass
class Item:
    kind: str                 # intro | card | scene | outro
    dur: float
    ref: object = None        # SceneClip, or the chapter name for a card
    start: float = 0.0        # position in the finished film
    xfade_in: float = 0.0     # transition from the previous item


def _parse_srt(path: Path) -> list[Cue]:
    def t(s: str) -> float:
        h, m, rest = s.split(":")
        sec, ms = rest.split(",")
        return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000

    cues = []
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8").strip()):
        lines = block.strip().splitlines()
        if len(lines) >= 3 and "-->" in lines[1]:
            a, b = (x.strip() for x in lines[1].split("-->"))
            cues.append(Cue(t(a), t(b), " ".join(lines[2:])))
    return cues


def load_scenes() -> list[SceneClip]:
    """The recorded scenes in tour order, each with the captions that were spoken over it."""
    scenes, cursor = [], 0.0
    for s in narration.SCENES:
        seg, aud = TOUR / "segments" / f"{s.key}.mp4", TOUR / "audio" / f"{s.key}.pad.wav"
        if not seg.exists() or not aud.exists():
            raise SystemExit(f"missing recording for {s.key}; re-record it with driver.py --only {s.key}")
        d = probe(seg)
        scenes.append(SceneClip(s.key, s.chapter, narration.CHAPTERS.index(s.chapter) + 1, s.title, seg, aud, d))
        cursor += d
    # Captions are recovered from the transcript by position. The transcript was written from the same
    # per-scene durations, so a cue belongs to the scene whose window its start falls in.
    cues = _parse_srt(TRANSCRIPT)
    edges, t0 = [], 0.0
    for sc in scenes:
        edges.append((t0, t0 + sc.seg_dur))
        t0 += sc.seg_dur
    for c in cues:
        for sc, (a, b) in zip(scenes, edges, strict=True):
            if a - 0.02 <= c.start < b - 0.02:
                sc.cues.append(Cue(c.start - a, min(c.end, b) - a, c.text))
                break
    empty = [sc.key for sc in scenes if not sc.cues]
    if empty:
        raise SystemExit(f"no transcript captions matched scenes {empty}; the transcript and recordings disagree")
    if abs(cues[-1].end - cursor) > 0.25:
        raise SystemExit(f"transcript ends at {cues[-1].end:.2f}s but the recordings add up to {cursor:.2f}s")
    return scenes


def timeline(scenes: list[SceneClip]) -> list[Item]:
    items = [Item("intro", q(INTRO_S))]
    for ch in narration.CHAPTERS:
        members = [s for s in scenes if s.chapter == ch]
        items.append(Item("card", q(CARD_S), ch, xfade_in=XF_CHAPTER))
        for i, sc in enumerate(members):
            items.append(Item("scene", sc.dur, sc, xfade_in=XF_CARD if i == 0 else XF_SCENE))
    items.append(Item("outro", q(OUTRO_S), xfade_in=XF_CHAPTER))
    t = 0.0
    for i, it in enumerate(items):
        if i:
            t = t + items[i - 1].dur - it.xfade_in
        it.start = t
    return items


def film_length(items: list[Item]) -> float:
    return items[-1].start + items[-1].dur


# ------------------------------------------------------------------ framing

def _gray_frame(seg: Path, at: float) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{at:.2f}", "-i", str(seg), "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(1080, 1920).astype(np.int16)


def framing(sc: SceneClip) -> dict | None:
    """The region worth looking at, as a zoom and a centre, or None when the page already fills the frame.

    Content is anything that differs from the page's own background, sampled at three moments so a panel
    that appears part way through is included. Bounds are taken at the 1.5th and 98.5th percentile of
    content mass rather than at the outermost pixel, because one stray footer label in a far corner
    would otherwise stretch the region to the whole screen and cancel the zoom it was meant to make.
    """
    mask = np.zeros((1080 - CHROME_PX, 1904), bool)
    for f in (0.2, 0.5, 0.82):
        g = _gray_frame(sc.seg, sc.seg_dur * f)[CHROME_PX:, :1904]
        hist = np.bincount((g // 4).ravel(), minlength=64)
        bg = int(hist.argmax()) * 4 + 2
        mask |= np.abs(g - bg) > 24
    if mask.sum() < 400:
        return None
    rows, cols = mask.sum(axis=1).astype(float), mask.sum(axis=0).astype(float)

    def bounds(v: np.ndarray) -> tuple[int, int]:
        c = np.cumsum(v) / v.sum()
        return int(np.searchsorted(c, 0.015)), int(np.searchsorted(c, 0.985))

    y0, y1 = bounds(rows)
    x0, x1 = bounds(cols)
    y0, y1 = y0 + CHROME_PX, y1 + CHROME_PX
    pad = 56
    bw, bh = (x1 - x0) + 2 * pad, (y1 - y0) + 2 * pad
    cw = max(bw, bh * 16 / 9)
    z = min(MAX_ZOOM, 1920.0 / cw)
    if z < MIN_ZOOM:
        return None
    cw, ch = 1920.0 / z, 1080.0 / z
    cx = min(max((x0 + x1) / 2, cw / 2), 1920 - cw / 2)
    cy = min(max((y0 + y1) / 2, ch / 2), 1080 - ch / 2)
    return {"z": round(z, 3), "cx": round(cx, 1), "cy": round(cy, 1), "box": [x0, y0, x1, y1]}


# ------------------------------------------------------------------ captions

def _wrap(text: str, width: int = 74) -> str:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    if len(lines) > 2:
        # Never three lines: the margin under the frame holds two. Rebalance into two even lines instead.
        half = len(text) / 2
        cut = min((m.start() for m in re.finditer(" ", text)), key=lambda i: abs(i - half))
        lines = [text[:cut].strip(), text[cut:].strip()]
    return r"\N".join(lines)


def _ass_time(t: float) -> str:
    cs = max(0, int(round(t * 100)))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def write_ass(sc: SceneClip, path: Path) -> None:
    head = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Ubuntu Sans,34,&H00F6F1EE,&H00F6F1EE,&H00000000,&H96000000,0,0,0,0,100,100,0.3,0,1,0,2.2,2,220,220,24,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    body = [f"Dialogue: 0,{_ass_time(c.start + LEAD)},{_ass_time(c.end + LEAD)},Cap,,0,0,0,,{{\\fad(140,110)}}{_wrap(c.text)}"
            for c in sc.cues]
    path.write_text(head + "\n".join(body) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ rendering

def _still(seg: Path, at: float, out: Path) -> Path:
    if not out.exists():
        run(["ffmpeg", "-y", "-v", "error", "-ss", f"{at:.2f}", "-i", str(seg), "-frames:v", "1", "-q:v", "2", str(out)])
    return out


# Where the scoring picks something colourful but off-topic. The LiDAR chapter scores the street map highest
# because map tiles are saturated, and a card announcing point clouds should show one.
CARD_BACKDROP = {"3D and LiDAR": "lidar_linked"}


def _richest(members: list[SceneClip], still) -> str:
    """The scene in a chapter with the most to look at, for the card's blurred backdrop.

    The first scene of a chapter is often its emptiest page. Chapter 6 opens on a LiDAR viewer that has not
    loaded a session yet, and the first card render used it and came out as a flat dark rectangle. Colour
    saturation and edge density both separate real footage, point clouds and image grids from panels of
    grey text, and both survive the heavy blur the card applies, so their product ranks the candidates.
    """
    from PIL import Image, ImageFilter

    def score(sc: SceneClip) -> float:
        im = Image.open(still(sc.key)).convert("RGB").resize((320, 180))
        hsv = np.asarray(im.convert("HSV")).astype(np.float32)
        edges = np.asarray(im.convert("L").filter(ImageFilter.FIND_EDGES)).astype(np.float32)
        return float(hsv[..., 1].mean() * hsv[..., 2].mean() / 255.0 + 0.6 * edges.mean())

    pinned = CARD_BACKDROP.get(members[0].chapter)
    if pinned and any(m.key == pinned for m in members):
        return pinned
    return max(members, key=score).key


def render_graphics(scenes: list[SceneClip]) -> None:
    g = WORK / "graphics"
    stills = g / "stills"
    stills.mkdir(parents=True, exist_ok=True)
    by_key = {s.key: s for s in scenes}

    def st(key: str, frac: float = 0.55) -> Path:
        s = by_key[key]
        return _still(s.seg, s.seg_dur * frac, stills / f"{key}.jpg")

    jobs = [B.Job(B.canvas_html(), g / "canvas.png"), B.Job(B.mask_html(), g / "mask.png")]
    for sc in scenes:
        jobs.append(B.Job(B.label_html(sc.chapter_no, sc.chapter, sc.title), g / "labels" / f"{sc.key}.png", transparent=True))
    for n, ch in enumerate(narration.CHAPTERS, 1):
        members = [s for s in scenes if s.chapter == ch]
        jobs.append(B.Job(B.card_html(n, ch, BLURBS[ch], st(_richest(members, st), 0.55)), g / "cards" / f"{n:02d}",
                          seconds=CARD_S))
    jobs.append(B.Job(B.intro_html([st("lidar_annotate"), st("frame_editor"), st("discovery"), st("review_grid")]),
                      g / "intro", seconds=INTRO_S))
    jobs.append(B.Job(B.outro_html([st("lidar_linked"), st("map"), st("analytics")], VERSION, CREDIT), g / "outro",
                      seconds=OUTRO_S))
    todo = [j for j in jobs if not (j.out.exists() and (j.seconds <= 0 or len(list(j.out.glob("*.jpg"))) == int(round(j.seconds * FPS))))]
    print(f"graphics: {len(jobs) - len(todo)} cached, rendering {len(todo)}")
    if todo:
        asyncio.run(B.render(todo, g / "html", parallel=6))
    # The mask is used as a luma matte, so it has to be exactly the frame's size.
    run(["ffmpeg", "-y", "-v", "error", "-i", str(g / "mask.png"), "-vf", f"crop={B.FRAME_W}:{B.FRAME_H}:0:0,format=gray",
         str(g / "mask-frame.png")])


ENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p", "-r", str(FPS),
       "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2"]


def render_scene(sc: SceneClip, item: Item, total: float) -> Path:
    out = WORK / "clips" / f"{item.start:08.3f}_{sc.key}.mkv"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    g = WORK / "graphics"
    ass = WORK / "captions" / f"{sc.key}.ass"
    ass.parent.mkdir(parents=True, exist_ok=True)
    write_ass(sc, ass)
    D = sc.dur
    if sc.zoom:
        # The move interpolates the edges of the visible rectangle, not a centre and a zoom level. Those two
        # do not move together, and the first render doing it that way swung the view sideways part way
        # through, cutting off the start of the very panel it was zooming towards before swinging back.
        # A rectangle whose edges move linearly between two rectangles that both contain the content
        # contains it at every frame in between, so this cannot crop what is being narrated.
        z = sc.zoom["z"]
        rw = 3200.0 / z
        left = (sc.zoom["cx"] - 960.0 / z) * 5 / 3
        top = (sc.zoom["cy"] - 540.0 / z) * 5 / 3
        p_ = f"clip((on/{FPS}-{LEAD + ZOOM_AT})/{ZOOM_FOR}\\,0\\,1)"
        e = f"({p_}*{p_}*(3-2*{p_}))"
        screen = (f"scale=3200:1800:flags=lanczos,"
                  f"zoompan=z='3200/(3200+({rw:.3f}-3200)*{e})':"
                  f"x='{left:.3f}*{e}':y='{top:.3f}*{e}':"
                  f"d=1:s={B.FRAME_W}x{B.FRAME_H}:fps={FPS}")
    else:
        screen = f"scale={B.FRAME_W}:{B.FRAME_H}:flags=lanczos"
    s0 = item.start
    graph = (
        f"[0:v]settb=AVTB,fps={FPS},tpad=start_duration={LEAD}:start_mode=clone:stop_duration=2:stop_mode=clone,"
        f"trim=duration={D},setpts=PTS-STARTPTS,{screen},setsar=1,format=rgba[scr];"
        f"[2:v]format=gray,setsar=1[m];[scr][m]alphamerge[win];"
        f"[1:v]format=rgba,setsar=1[bg];[bg][win]overlay={B.FRAME_X}:{B.FRAME_Y}:shortest=1[c1];"
        f"[3:v]format=rgba,fade=t=in:st=0.3:d=0.6:alpha=1[lab];"
        f"[c1][lab]overlay=x='-34*pow(1-min(1\\,max(0\\,(t-0.3)/0.7))\\,3)':y=0[c2];"
        f"[c2]drawbox=x=0:y=1076:w=1920:h=4:color=white@0.07:t=fill[c3];"
        f"[4:v]format=rgba[bar];[c3][bar]overlay=x='-w+W*min(1\\,({s0:.3f}+t)/{total:.3f})':y=1076[c4];"
        f"[c4]ass='{ass}',format=yuv420p[v];"
        f"[5:a]aformat=sample_rates=48000:channel_layouts=stereo,adelay={int(LEAD * 1000)}:all=1,apad,atrim=duration={D}[a]"
    )
    run(["ffmpeg", "-y", "-v", "error", "-threads", "4",
         "-i", str(sc.seg),
         "-loop", "1", "-framerate", str(FPS), "-t", f"{D}", "-i", str(g / "canvas.png"),
         "-loop", "1", "-framerate", str(FPS), "-t", f"{D}", "-i", str(g / "mask-frame.png"),
         "-loop", "1", "-framerate", str(FPS), "-t", f"{D}", "-i", str(g / "labels" / f"{sc.key}.png"),
         "-f", "lavfi", "-t", f"{D}", "-i", f"color=c=0x8B63F5:s=1920x4:r={FPS}",
         "-i", str(sc.audio),
         "-filter_complex", graph, "-map", "[v]", "-map", "[a]", "-t", f"{D}", *ENC, str(out)])
    return out


def render_graphic_clip(name: str, frames: Path, dur: float) -> Path:
    out = WORK / "clips" / f"{name}.mkv"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-y", "-v", "error", "-framerate", str(FPS), "-i", str(frames / "%05d.jpg"),
         "-f", "lavfi", "-t", f"{dur}", "-i", "anullsrc=r=48000:cl=stereo",
         "-map", "0:v", "-map", "1:a", "-t", f"{dur}", *ENC, str(out)])
    return out


def xfade_chain(parts: list[tuple[Path, float]], fades: list[float], out: Path, final: bool = False,
                extra_inputs: list[str] | None = None, audio_tail: str = "") -> None:
    """Join clips with crossfades on picture and sound together. `fades[i]` joins part i to part i+1."""
    args, graph = [], []
    for i, (p, _d) in enumerate(parts):
        args += ["-i", str(p)]
        graph.append(f"[{i}:v]settb=AVTB,fps={FPS},format=yuv420p,setsar=1[v{i}];"
                     f"[{i}:a]aformat=sample_rates=48000:channel_layouts=stereo[a{i}]")
    vcur, acur, t = "v0", "a0", 0.0
    for i in range(1, len(parts)):
        t += parts[i - 1][1] - fades[i - 1]
        graph.append(f"[{vcur}][v{i}]xfade=transition=fade:duration={fades[i - 1]}:offset={q(t):.4f}[vx{i}];"
                     f"[{acur}][a{i}]acrossfade=d={fades[i - 1]}:c1=tri:c2=tri[ax{i}]")
        vcur, acur = f"vx{i}", f"ax{i}"
    graph.append(f"[{vcur}]null[vout];[{acur}]anull[aout]" if not audio_tail else f"[{vcur}]null[vout];{audio_tail.format(a=acur)}")
    enc = (["-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(FPS), "-movflags", "+faststart",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000"] if final else ENC)
    run(["ffmpeg", "-y", "-v", "error", *args, *(extra_inputs or []), "-filter_complex", ";".join(graph),
         "-map", "[vout]", "-map", "[aout]", *enc, str(out)])


# ------------------------------------------------------------------ music

# Composed tracks by Kevin MacLeod, from incompetech.com under Creative Commons Attribution 4.0, which
# requires the credit that CREDIT carries onto the outro. Chosen by measurement rather than by ear, since
# this was produced without listening: steady level (loudness range 1.8 to 4.4 LU), no vocals or novelty
# instruments, and enough forward motion to carry a product demo without sounding like a meditation app.
# Tracks that swung 14 dB between quiet passages and swells, or sat bright enough to crowd the voice's
# frequencies, were measured and left out.
#
# Each entry is the chapter whose card the track takes over at, the track, and where in the track to
# start. Hand-overs happen on chapter cards because nobody is speaking there, so the change is heard as
# a new section rather than as a cut under a sentence. Beauty Flow returns for the last act from further
# into the piece, so the film ends on the theme it opened with without repeating its first bars.
MUSIC_SOURCE = "https://incompetech.com/music/royalty-free/mp3-royaltyfree/"
MUSIC_PLAN = [("Intro", "Beauty Flow", 0.0), ("Review", "Space Jazz", 0.0),
              ("Measurement", "Sincerely", 0.0), ("Export and privacy", "Beauty Flow", 150.0)]
MUSIC_XF = 3.0
CREDIT = ("Music: \u201cBeauty Flow\u201d, \u201cSpace Jazz\u201d, \u201cSincerely\u201d by Kevin MacLeod (incompetech.com). "
          "Licensed under Creative Commons: By Attribution 4.0.")


def _track(title: str) -> Path:
    import urllib.parse
    import urllib.request

    path = WORK / "music" / f"{title}.mp3"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MUSIC_SOURCE + urllib.parse.quote(f"{title}.mp3"), path)
    return path


def _loudness(path: Path) -> float:
    err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    return float(next(ln.split(":")[1].split()[0] for ln in reversed(err.splitlines()) if ln.strip().startswith("I:")))


def music_bed(items: list[Item], total: float, path: Path) -> None:
    """Lay the planned tracks end to end, each levelled to the same loudness, crossfading on chapter cards."""
    if path.exists():
        return
    starts = []
    for chapter, title, offset in MUSIC_PLAN:
        at = 0.0 if chapter == "Intro" else next(it.start for it in items if it.kind == "card" and it.ref == chapter)
        starts.append((at, title, offset))
    args, graph = [], []
    for i, (at, title, offset) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else total
        # Each piece runs past its hand-over by half the crossfade and starts that much early, so the fade
        # is centred on the card rather than on the scene before it.
        length = (end - at) + (MUSIC_XF if 0 < i + 1 < len(starts) else MUSIC_XF / 2) + (MUSIC_XF / 2 if i else 0)
        track = _track(title)
        gain = -20.0 - _loudness(track)
        args += ["-ss", f"{offset:.2f}", "-t", f"{length:.2f}", "-i", str(track)]
        graph.append(f"[{i}:a]aformat=sample_rates=48000:channel_layouts=stereo,volume={gain:.2f}dB[m{i}]")
    cur = "m0"
    for i in range(1, len(starts)):
        graph.append(f"[{cur}][m{i}]acrossfade=d={MUSIC_XF}:c1=qsin:c2=qsin[j{i}]")
        cur = f"j{i}"
    graph.append(f"[{cur}]apad,atrim=duration={total:.3f}[out]")
    path.parent.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-y", "-v", "error", *args, "-filter_complex", ";".join(graph), "-map", "[out]",
         "-c:a", "pcm_s16le", str(path)])


# ------------------------------------------------------------------ steps

def plan(write: bool = True) -> tuple[list[SceneClip], list[Item]]:
    scenes = load_scenes()
    cache = WORK / "framing.json"
    known = json.loads(cache.read_text()) if cache.exists() else {}
    with ThreadPoolExecutor(8) as ex:
        results = dict(zip([s.key for s in scenes],
                           ex.map(lambda s: known[s.key] if s.key in known else framing(s), scenes), strict=True))
    for s in scenes:
        s.zoom = results[s.key]
    items = timeline(scenes)
    if write:
        WORK.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(results, indent=1))
    return scenes, items


def build() -> Path:
    scenes, items = plan()
    total = film_length(items)
    print(f"{len(scenes)} scenes, {sum(1 for s in scenes if s.zoom)} zoomed, film {total / 60:.2f} min")
    render_graphics(scenes)

    g = WORK / "graphics"
    clip_paths: dict[int, Path] = {}
    with ThreadPoolExecutor(6) as ex:
        futs = {}
        for idx, it in enumerate(items):
            if it.kind == "scene":
                futs[idx] = ex.submit(render_scene, it.ref, it, total)
            elif it.kind == "card":
                n = narration.CHAPTERS.index(it.ref) + 1
                futs[idx] = ex.submit(render_graphic_clip, f"card{n:02d}", g / "cards" / f"{n:02d}", it.dur)
            else:
                futs[idx] = ex.submit(render_graphic_clip, it.kind, g / it.kind, it.dur)
        done = 0
        for idx, f in futs.items():
            clip_paths[idx] = f.result()
            done += 1
            if done % 10 == 0:
                print(f"  clips {done}/{len(items)}")

    # Chapters first, each a card and its scenes, so no single join has to hold 89 decoders open.
    chapters = WORK / "chapters"
    chapters.mkdir(parents=True, exist_ok=True)
    top: list[tuple[Path, float]] = [(clip_paths[0], items[0].dur)]
    idx = 1
    for n, ch in enumerate(narration.CHAPTERS, 1):
        members = [idx]
        idx += 1
        while idx < len(items) and items[idx].kind == "scene" and items[idx].ref.chapter == ch:
            members.append(idx)
            idx += 1
        parts = [(clip_paths[i], items[i].dur) for i in members]
        fades = [items[i].xfade_in for i in members[1:]]
        out = chapters / f"{n:02d}.mkv"
        length = sum(d for _p, d in parts) - sum(fades)
        if not out.exists():
            xfade_chain(parts, fades, out)
        top.append((out, q(length)))
        print(f"  chapter {n:02d} {ch}: {len(members) - 1} scenes, {length:.1f}s")
    top.append((clip_paths[len(items) - 1], items[-1].dur))

    import hashlib

    bed = WORK / f"music-{hashlib.sha1(repr(MUSIC_PLAN).encode()).hexdigest()[:8]}.wav"
    music_bed(items, total, bed)
    chapters_meta = WORK / "chapters.txt"
    lines = [";FFMETADATA1", "title=LabeloxAV: a tour of every screen", "artist=Sherin Joseph Roy"]
    marks = [("Intro", 0.0)] + [(ch, next(it.start for it in items if it.kind == "card" and it.ref == ch))
                                for ch in narration.CHAPTERS] + [("Outro", items[-1].start)]
    for i, (name, st) in enumerate(marks):
        end = marks[i + 1][1] if i + 1 < len(marks) else total
        lines += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={int(st * 1000)}", f"END={int(end * 1000)}", f"title={name}"]
    chapters_meta.write_text("\n".join(lines) + "\n", encoding="utf-8")
    yt = [f"{int(st // 60)}:{int(st % 60):02d} {name}" for name, st in marks]
    (WORK / "youtube-chapters.txt").write_text("\n".join(yt) + "\n", encoding="utf-8")

    final = WORK / "labeloxav-tour-pro.mp4"
    m = len(top)
    # Levels were measured rather than set by ear, because this was produced without listening, and measured
    # as ungated RMS: integrated LUFS gates out quiet passages, so the harder the music ducks under speech
    # the more of it the gate discards, and the reading rises as the music falls.
    #
    # The ducking is deliberately gentle. A 14:1 compressor put the bed 16 dB under the voice in one passage
    # and 30 dB under in a denser one, so the music vanished under long sentences and swelled back in every
    # pause, which is the pumping that makes a narrated film sound amateur. At 4:1 with a soft knee it sits
    # 14.2, 17.8 and 16.0 dB under the voice across three quite different passages. The intro and outro get
    # about 4 dB more, ramped over a second and a half, so the opening carries at close to voice level
    # (2.4 dB under) without that lift ever sitting under a sentence.
    outro_at = items[-1].start
    lift = f"0.5*(1+0.6*clip((9.5-t)/1.5\\,0\\,1)+0.6*clip((t-{outro_at - 0.75:.2f})/1.5\\,0\\,1))"
    tail = (f"[{m}:a]aformat=sample_rates=48000:channel_layouts=stereo,volume='{lift}':eval=frame,"
            f"afade=t=in:d=2,afade=t=out:st={max(0, total - 3.5):.2f}:d=3.5[mus];"
            f"[{{a}}]asplit=2[nar][key];"
            f"[mus][key]sidechaincompress=threshold=0.03:ratio=4:attack=40:release=900:knee=6[duck];"
            f"[nar][duck]amix=inputs=2:duration=first:normalize=0,"
            f"loudnorm=I=-15:TP=-1.5:LRA=11[aout]")
    xfade_chain(top, [XF_CHAPTER] * (m - 1), final, final=True,
                extra_inputs=["-i", str(bed), "-i", str(chapters_meta), "-map_metadata", str(m + 1)],
                audio_tail=tail)
    return final


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "plan"
    if cmd == "plan":
        scenes, items = plan()
        for s in scenes:
            z = s.zoom
            print(f"  {s.chapter_no:02d} {s.key:22s} {s.dur:6.2f}s  cues {len(s.cues):2d}  "
                  + (f"zoom {z['z']:.2f} at ({z['cx']:.0f},{z['cy']:.0f})" if z else "full frame"))
        print(f"{len(scenes)} scenes, {sum(1 for s in scenes if s.zoom)} zoomed, {len(items)} clips, "
              f"film {film_length(items) / 60:.2f} min")
    elif cmd == "all":
        out = build()
        print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {probe(out) / 60:.2f} min)")
    else:
        raise SystemExit("usage: produce.py plan | all")


if __name__ == "__main__":
    main()
