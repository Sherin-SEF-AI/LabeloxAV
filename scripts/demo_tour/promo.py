"""Cut a two-minute promo from the recorded hands-on demo.

    .venv/bin/python scripts/demo_tour/promo.py plan      what it will build, shot by shot
    .venv/bin/python scripts/demo_tour/promo.py build     build .scratch/demo/india/promo/labeloxav-promo.mp4

The demo film explains the editor tool by tool. This is the other kind of film: the same footage, cut
fast, with one line per shot and music in place of a voice. It makes nothing up to do that. Every shot is
a stretch of a scene `annotate.py record` recorded in the running product, played faster, and each line
says what is happening in the shot it is over.

Three decisions.

**Each shot is a whole action, sped up, not a slice of one.** A two-second cut of a box half drawn shows
nothing. A shot here covers the scene from the first move to the result and is played at whatever speed
fits it into its slot, capped so the pointer can still be followed; a scene too long for that cap starts
late rather than running faster.

**The line sits on the picture, at the bottom, over a fade.** The film puts captions in a margin so they
never cover the interface. A promo is watched at a glance, often small and muted, so the line is large and
on the footage, and the fade behind it is what keeps it legible over a bright road.

**Music is the only sound.** One track, levelled, faded in and out. The scenes' narration is not used:
sped up it would be noise, and at its own speed it would not fit.
"""

from __future__ import annotations

import asyncio
import html
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import brand as B  # noqa: E402

TOUR = ROOT / ".scratch/demo/india/tour"
WORK = ROOT / ".scratch/demo/india/promo"
MUSIC = ROOT / ".scratch/demo/india/pro/music/Inspired.mp3"
OUT = WORK / "labeloxav-promo.mp4"
SOURCE_URL = "github.com/Sherin-SEF-AI/LabeloxAV"

FPS = 30
TARGET_S = 120.0
OPEN_S, CLOSE_S = 5.0, 8.0
XF = 0.4                 # between every two pieces
MAX_SPEED = 3.0          # past this the pointer cannot be followed
MUSIC_OFFSET_S = 6.0     # into the track, past its quiet opening bars
ENC = ["-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", "-r", str(FPS)]


@dataclass(frozen=True)
class Shot:
    key: str              # the recorded scene
    line: str             # what the shot shows, in a few words
    sub: str = ""         # a second, smaller line
    weight: float = 1.0   # share of the running time, relative to the others
    skip_s: float = 0.0   # seconds of the scene to leave off the front


SHOTS = [
    Shot("box_bus", "Draw a box in one drag", "real dashcam footage from Bengaluru"),
    Shot("box_autos", "200 classes built for Indian roads", "auto-rickshaws are not cars"),
    Shot("polygon", "Polygons for what is not a box"),
    Shot("sam_point", "One click becomes a pixel mask", "AI assist, corrected by hand"),
    Shot("keypoints", "Pose, polylines, occlusion, attributes"),
    Shot("nudge", "Fine-tune from the keyboard", "arrow keys move the selection a pixel at a time"),
    Shot("machine_frame", "Models propose every object", "three detectors, one frame"),
    Shot("frame_review", "People decide what is true", "accept, reject or correct each proposal", weight=1.2),
    Shot("grid", "Reject eight wrong labels with one key"),
    Shot("paste_next", "Copy a frame's work to the next"),
    Shot("track", "Fix one frame, fix the whole track"),
    Shot("two_people", "Two people, one frame, nothing overwritten", weight=1.2, skip_s=9.0),
    Shot("datasets", "Export COCO, YOLO, Parquet, OpenLABEL, nuScenes"),
]


def run(cmd: list[str]) -> None:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(f"{' '.join(cmd[:6])} ... failed:\n{res.stderr[-1500:]}")


def probe(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.strip()
    return float(out)


def slots() -> list[tuple[Shot, float, float, float]]:
    """(shot, seconds on screen, speed, seconds skipped at the front) for every shot.

    The running time left after the cards is shared out by weight. A crossfade overlaps two pieces, so each
    one is lengthened by the fade it loses and the total still lands on the target.
    """
    pieces = len(SHOTS) + 2
    body = TARGET_S - OPEN_S - CLOSE_S + XF * (pieces - 1)
    total_w = sum(s.weight for s in SHOTS)
    out = []
    for s in SHOTS:
        seg = TOUR / "segments" / f"{s.key}.mp4"
        if not seg.exists():
            raise SystemExit(f"no recording for scene {s.key!r}; run annotate.py record first")
        on_screen = round(body * s.weight / total_w * FPS) / FPS
        have = probe(seg) - s.skip_s
        speed = min(MAX_SPEED, max(1.0, have / on_screen))
        # A scene longer than the cap allows starts later, so the shot ends on the result, not before it.
        skip = s.skip_s + max(0.0, have - on_screen * speed)
        out.append((s, on_screen, speed, skip))
    return out


# ------------------------------------------------------------------ graphics

def line_html(shot: Shot, n: int, of: int) -> str:
    sub = (f'<div style="font-size:30px;color:#C9CFDB;margin-top:10px">{html.escape(shot.sub)}</div>'
           if shot.sub else "")
    return f"""<html><head><style>{B._css()}</style></head><body style="background:transparent">
    <div style="position:absolute;left:0;right:0;bottom:0;height:330px;
                background:linear-gradient(to top, rgba(9,12,19,.94) 0%, rgba(9,12,19,.80) 42%, rgba(9,12,19,0) 100%)"></div>
    <div style="position:absolute;left:96px;right:96px;bottom:70px;display:flex;align-items:flex-end;gap:28px">
      <div style="width:8px;align-self:stretch;border-radius:4px;background:{B.VIOLET_LIGHT}"></div>
      <div style="flex:1">
        <div class="display" style="font-size:64px;font-weight:600;letter-spacing:-.02em;line-height:1.08">{html.escape(shot.line)}</div>
        {sub}
      </div>
      <div class="mono" style="font-size:22px;color:{B.MUTED};letter-spacing:.06em">{n:02d} / {of:02d}</div>
    </div>
    </body></html>"""


def open_html(still: Path) -> str:
    return f"""<html><head><style>{B._css()}</style></head><body style="background:{B.BG}">
    <img src="file://{still}" style="position:absolute;inset:-60px;width:calc(100% + 120px);height:calc(100% + 120px);
         object-fit:cover;filter:blur(8px) brightness(.32) saturate(.85);animation:drift {OPEN_S}s linear both">
    <div style="position:absolute;inset:0;background:radial-gradient(1200px 760px at 50% 44%, rgba(9,12,19,.30), rgba(9,12,19,.92) 78%)"></div>
    {B._backdrop("50%", "30%", base=False)}
    <div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center">
      <img class="a" src="file://{B.HERE}/mark-light.png" style="height:150px;animation:pop .9s .15s both">
      <div class="a display" style="animation:rise .9s .6s both;font-size:128px;font-weight:600;letter-spacing:-.03em;margin-top:24px;line-height:1">
        Labelox<span style="color:{B.VIOLET_LIGHT}">AV</span></div>
      <div class="a display" style="animation:rise .9s 1.2s both;font-size:50px;color:#DCE1EA;margin-top:28px;font-weight:500">
        A data engine for autonomous driving</div>
      <div class="a" style="animation:rise .9s 1.7s both;font-size:40px;color:{B.VIOLET_LIGHT};margin-top:14px">
        built for Indian roads</div>
    </div>
    </body></html>"""


def close_html(still: Path) -> str:
    return f"""<html><head><style>{B._css()}</style></head><body style="background:{B.BG}">
    <img src="file://{still}" style="position:absolute;inset:-60px;width:calc(100% + 120px);height:calc(100% + 120px);
         object-fit:cover;filter:blur(10px) brightness(.26) saturate(.8);animation:drift {CLOSE_S}s linear both">
    <div style="position:absolute;inset:0;background:radial-gradient(1200px 760px at 50% 44%, rgba(9,12,19,.35), rgba(9,12,19,.94) 78%)"></div>
    {B._backdrop("50%", "30%", base=False)}
    <div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center">
      <div class="a display" style="animation:rise .9s .2s both;font-size:76px;font-weight:600;letter-spacing:-.02em;line-height:1.12">
        Drives in. Labels checked.<br>Models out.</div>
      <div class="a display" style="animation:rise .9s .9s both;font-size:112px;font-weight:600;letter-spacing:-.03em;margin-top:44px;line-height:1">
        Labelox<span style="color:{B.VIOLET_LIGHT}">AV</span></div>
      <div class="a mono" style="animation:rise .9s 1.5s both;font-size:36px;color:#DCE1EA;margin-top:30px;
           padding:14px 28px;border-radius:14px;background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.16)">
        {html.escape(SOURCE_URL)}</div>
      <div class="a" style="animation:fade .9s 2.2s both;font-size:28px;color:#C3C9D6;margin-top:26px">
        Open source. Runs on one GPU box. Every shot recorded in the running product.</div>
      <div class="a mono" style="animation:fade .9s 3.0s both;position:absolute;bottom:44px;left:0;right:0;font-size:17px;color:{B.MUTED};line-height:1.6">
        Footage: the operator's own dashcam drive in Bengaluru, and "Stray Cattle in Lutyens Delhi" by Fowler&amp;fowler, Wikimedia Commons, CC BY-SA 3.0.<br>
        Music: "Inspired" by Kevin MacLeod (incompetech.com). Licensed under Creative Commons: By Attribution 4.0.</div>
    </div>
    </body></html>"""


def still(seg: Path, at: float, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-y", "-v", "error", "-ss", f"{at:.2f}", "-i", str(seg), "-frames:v", "1", "-q:v", "2", str(out)])
    return out


def render_graphics(plan: list) -> None:
    gfx = WORK / "graphics"
    first = TOUR / "segments" / "adverse.mp4"
    last = TOUR / "segments" / "frame_review.mp4"
    s1 = still(first if first.exists() else TOUR / "segments" / f"{SHOTS[0].key}.mp4", 4.0, gfx / "open_still.jpg")
    s2 = still(last if last.exists() else TOUR / "segments" / f"{SHOTS[-1].key}.mp4", 8.0, gfx / "close_still.jpg")
    jobs = [B.Job(open_html(s1), gfx / "open", OPEN_S), B.Job(close_html(s2), gfx / "close", CLOSE_S)]
    jobs += [B.Job(line_html(shot, i + 1, len(plan)), gfx / f"line_{shot.key}.png", transparent=True)
             for i, (shot, _d, _s, _k) in enumerate(plan)]
    asyncio.run(B.render(jobs, gfx / "_html"))


# ------------------------------------------------------------------ pieces

def card_clip(name: str, dur: float) -> Path:
    out = WORK / "clips" / f"{name}.mkv"
    out.parent.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-y", "-v", "error", "-framerate", str(FPS), "-i", str(WORK / "graphics" / name / "%05d.jpg"),
         "-t", f"{dur}", *ENC, "-an", str(out)])
    return out


def shot_clip(shot: Shot, dur: float, speed: float, skip: float) -> Path:
    """One shot: the scene from `skip`, played `speed` times faster for `dur` seconds, with its line on it."""
    out = WORK / "clips" / f"shot_{shot.key}.mkv"
    seg = TOUR / "segments" / f"{shot.key}.mp4"
    line = WORK / "graphics" / f"line_{shot.key}.png"
    graph = (f"[0:v]trim=start={skip:.3f}:duration={dur * speed:.3f},setpts=(PTS-STARTPTS)/{speed:.4f},"
             f"fps={FPS},scale={B.W}:{B.H}:flags=lanczos,setsar=1[v];"
             # The line arrives a beat after the cut, so the eye lands on the picture first.
             f"[1:v]format=rgba,fade=t=in:st=0.25:d=0.35:alpha=1[l];[v][l]overlay=0:0:format=auto,format=yuv420p[o]")
    run(["ffmpeg", "-y", "-v", "error", "-i", str(seg), "-loop", "1", "-t", f"{dur + 0.5:.3f}", "-i", str(line),
         "-filter_complex", graph, "-map", "[o]", "-t", f"{dur:.3f}", *ENC, "-an", str(out)])
    return out


def assemble(parts: list[tuple[Path, float]]) -> float:
    """Crossfade every piece into the next, lay the music under the whole of it, and encode for upload."""
    args, graph = [], []
    for i, (p, _d) in enumerate(parts):
        args += ["-i", str(p)]
        graph.append(f"[{i}:v]settb=AVTB,fps={FPS},format=yuv420p,setsar=1[v{i}]")
    cur, t = "v0", 0.0
    for i in range(1, len(parts)):
        t += parts[i - 1][1] - XF
        graph.append(f"[{cur}][v{i}]xfade=transition=fade:duration={XF}:offset={t:.4f}[x{i}]")
        cur = f"x{i}"
    total = t + parts[-1][1]
    m = len(parts)
    graph.append(f"[{cur}]fade=t=in:st=0:d=0.5,fade=t=out:st={total - 0.8:.3f}:d=0.8[vout]")
    # Levelled to -14 LUFS, the loudness video sites normalise to, with the true peak held under -1 dB.
    graph.append(f"[{m}:a]aformat=sample_rates=48000:channel_layouts=stereo,atrim=duration={total:.3f},"
                 f"afade=t=in:st=0:d=1.2,afade=t=out:st={total - 3.0:.3f}:d=3.0,"
                 f"loudnorm=I=-14:TP=-1.5:LRA=9[aout]")
    run(["ffmpeg", "-y", "-v", "error", *args, "-ss", f"{MUSIC_OFFSET_S}", "-i", str(MUSIC),
         "-filter_complex", ";".join(graph), "-map", "[vout]", "-map", "[aout]",
         "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(FPS),
         "-movflags", "+faststart", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
         "-metadata", "title=LabeloxAV: annotating real Indian roads, in two minutes", "-t", f"{total:.3f}", str(OUT)])
    return total


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "plan"
    plan = slots()
    if mode == "plan":
        for shot, dur, speed, skip in plan:
            print(f"{shot.key:16s} {dur:5.2f}s on screen  x{speed:4.2f}  from {skip:5.1f}s   {shot.line}")
        pieces = len(plan) + 2
        print(f"total {OPEN_S + CLOSE_S + sum(d for _s, d, _v, _k in plan) - XF * (pieces - 1):.1f}s")
        return
    if not MUSIC.exists():
        raise SystemExit(f"the music track is not cached at {MUSIC}; produce.py fetches it for the film")
    WORK.mkdir(parents=True, exist_ok=True)
    render_graphics(plan)
    parts = [(card_clip("open", OPEN_S), OPEN_S)]
    parts += [(shot_clip(shot, dur, speed, skip), dur) for shot, dur, speed, skip in plan]
    parts.append((card_clip("close", CLOSE_S), CLOSE_S))
    total = assemble(parts)
    (WORK / "promo.json").write_text(json.dumps(
        {"seconds": round(total, 2), "shots": [{"scene": s.key, "line": s.line, "seconds": d, "speed": round(v, 2)}
                                               for s, d, v, _k in plan]}, indent=2))
    print(f"wrote {OUT} ({total:.1f}s)")


if __name__ == "__main__":
    main()
