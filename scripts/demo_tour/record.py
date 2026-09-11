"""Record the tour: drive the real UI, capture the real screen, and speak the same words as the subtitles.

Three properties this is built around.

**Nothing is mocked.** The browser talks to the running API against the live database, so every number on
screen is the one the system actually holds. If a page is slow or empty, that is what the recording shows,
and the narration says so.

**The audio and the subtitles come from one source.** Both are generated from the same `Scene.narration`
string, so a subtitle can never disagree with the voice. Timing comes from measuring each synthesised clip
rather than guessing, so the captions stay aligned however long a sentence turns out to take.

**The video is assembled from per-scene segments.** A single long screen capture would have to be cut
blind afterwards; recording each scene separately means every segment is exactly as long as its narration
plus its hold, and a scene that needs re-recording does not cost the whole tour.

**The browser records itself, rather than the screen being grabbed.** This machine runs a Wayland
session, where an X11 screen grab of `:0` sees only the XWayland root and returns black frames, which is
exactly what the first take produced. Chromium writing its own video has no such dependency: it captures
the page rather than the desktop, so the output is pixel exact at the requested size, it runs headless
without taking over the display, and nothing from the rest of the desktop can appear in the recording.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import wave
from pathlib import Path

OUT = Path(".scratch/demo/tour")
VOICE = Path(".scratch/demo/voices/en_US-lessac-medium.onnx")
WEB = "http://localhost:3000"
API = "http://localhost:8000"
# 1920x1080, which is what the browser is told to render, so nothing is rescaled. 15 fps because a tour
# of a web application is mostly still pages: the rate that matters for legibility is the resolution, and
# doubling the frame rate would double the file for footage that barely moves. Segment duration is pinned
# exactly at assembly, so the rate never affects synchronisation with the voice.
W, H, FPS = 1920, 1080, 15


def synth(text: str, path: Path) -> float:
    """Speak one scene and return how long it takes, which is what drives the segment length."""
    voice = _voice()
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        voice.synthesize_wav(text, w)
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


_VOICE = None


def _voice():
    global _VOICE
    if _VOICE is None:
        from piper import PiperVoice

        _VOICE = PiperVoice.load(str(VOICE))
    return _VOICE


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _cues(entries: list[dict]) -> list[tuple[float, float, str]]:
    """Split every scene into per-sentence cues with start and end times.

    Split on sentences rather than at a fixed character count, because a caption cut mid-clause is harder
    to read than a slightly long one, and the narration is written in short sentences for this reason.
    Each sentence is timed in proportion to its length, which tracks speech duration closely enough that
    a caption never trails its audio by more than a fraction of a second.
    """
    out: list[tuple[float, float, str]] = []
    for e in entries:
        sentences = [x.strip() for x in e["text"].replace("\n", " ").split(". ") if x.strip()]
        if not sentences:
            continue
        total = sum(len(x) for x in sentences) or 1
        t = e["start"]
        for i, sent in enumerate(sentences):
            share = (len(sent) / total) * e["duration"]
            text = sent if sent.endswith((".", "?", "!")) or i == len(sentences) - 1 else sent + "."
            out.append((t, t + share, _wrap(text)))
            t += share
    return out


def write_srt(entries: list[dict], path: Path) -> None:
    """The selectable soft subtitle track."""
    out = []
    for n, (start, end, text) in enumerate(_cues(entries), 1):
        out.append(f"{n}\n{srt_time(start)} --> {srt_time(end)}\n{text}\n")
    path.write_text("\n".join(out), encoding="utf-8")


def write_ass(entries: list[dict], path: Path) -> None:
    """The burned-in subtitle track, written as ASS with an explicit play resolution.

    Burning from the SRT looked wrong and the reason is worth recording. libass assumes a 384 by 288 play
    resolution for a format that carries none, then scales everything it draws by the ratio to the real
    frame, so a font size chosen as a pixel height comes out nearly four times too big. Stating
    PlayResX and PlayResY here makes every number below a real pixel on a 1920 by 1080 frame.

    The SRT is still written alongside, as the selectable soft track. The two are generated from the same
    entries, so they cannot disagree.
    """
    head = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}",
        "WrapStyle: 0", "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour,"
        " Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline,"
        " Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        # BorderStyle 3 is a filled box behind the text rather than an outline. Over a dark dashboard an
        # outlined glyph still competes with the panels underneath; a box is readable over anything.
        "Style: Tour,DejaVu Sans,38,&H00FFFFFF,&H00FFFFFF,&H00000000,&HB4000000,"
        "0,0,0,0,100,100,0,0,3,5,0,2,140,140,54,1", "",
        "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    body = []
    for start, end, text in _cues(entries):
        body.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Tour,,0,0,0,,"
                    + text.replace("\n", "\\N"))
    path.write_text("\n".join(head + body) + "\n", encoding="utf-8")


def _ass_time(t: float) -> str:
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _wrap(text: str, width: int = 62) -> str:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines[:2]) if len(lines) <= 2 else "\n".join([" ".join(lines[:-1]), lines[-1]])


def quantize(seconds: float) -> float:
    """Round a duration up to a whole frame.

    A video file cannot be 5.213 seconds long at 15 fps; it is 78 frames, or 5.2. Deciding the scene
    length on a frame boundary up front means the picture, the voice and the caption are all built from
    one number that every one of them can represent exactly, instead of three that drift apart by a
    frame per scene.
    """
    return math.ceil(seconds * FPS) / FPS


def probe_duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def fit_duration(src: Path, dst: Path, seconds: float, start: float = 0.0) -> None:
    """Re-encode a captured segment to exactly `seconds`, from `start`, holding the last frame if short.

    `start` drops the page load. Recording begins when the browser context is created, which is before
    the page has been asked for, so the opening moment of every raw segment is a blank tab resolving into
    the page. The narration is written to start on a settled page, so that part is cut rather than
    narrated over.

    `tpad` freezes the final frame rather than looping or blanking. A frozen UI reads as a pause for the
    narration to finish, which is what it is; a black gap reads as a fault in the recording.
    """
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{max(start, 0.0):.3f}", "-i", str(src),
         "-vf", f"tpad=stop_mode=clone:stop_duration={max(seconds, 0.1):.3f},fps={FPS},"
                f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
                f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black",
         "-t", f"{seconds:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-pix_fmt", "yuv420p", "-an", str(dst)], check=True)


def pad_audio(src: Path, dst: Path, seconds: float) -> None:
    """Pad a narration clip with silence to exactly the scene length.

    The hold at the end of a scene is silence over a still page, so the audio track has to carry that
    silence explicitly. Concatenating unpadded clips would pull every later scene's voice forward.
    """
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-af", "apad",
         "-t", f"{seconds:.3f}", "-ar", "22050", "-ac", "1", "-c:a", "pcm_s16le", str(dst)],
        check=True)


def concat(segments: list[Path], path: Path) -> None:
    listing = OUT / "segments.txt"
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in segments))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(listing), "-c", "copy", str(path)], check=True)


def concat_audio(clips: list[Path], path: Path) -> None:
    listing = OUT / "audio.txt"
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in clips))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(listing), "-c:a", "aac", "-b:a", "192k", str(path)], check=True)


def write_chapters(entries: list[dict], path: Path) -> int:
    """An ffmpeg metadata file with one chapter per section of the tour.

    A half-hour video with no chapter marks is searched by dragging the scrubber. With them the player
    shows twelve named sections, so somebody who only wants the LiDAR chapter can go straight to it.
    """
    lines = [";FFMETADATA1"]
    n = 0
    for i, e in enumerate(entries):
        if i and e["chapter"] == entries[i - 1]["chapter"]:
            continue
        end = next((x["start"] for x in entries[i:] if x["chapter"] != e["chapter"]),
                   entries[-1]["start"] + entries[-1]["duration"])
        lines += ["[CHAPTER]", "TIMEBASE=1/1000",
                  f"START={int(e['start'] * 1000)}", f"END={int(end * 1000)}",
                  f"title={e['chapter']}"]
        n += 1
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return n


def mux(video: Path, audio: Path, srt: Path, ass: Path, out: Path, burn: bool = True,
        chapters: Path | None = None) -> None:
    """Combine picture, voice and captions.

    Captions are burned in by default and also attached as a soft track: a burned caption always shows,
    which is what makes a muted autoplay watchable, and the soft track stays selectable and searchable.
    """
    # Input order fixes the map indices below, so it is written out once rather than computed: 0 video,
    # 1 audio, 2 subtitles, 3 chapter metadata when present.
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-i", str(video), "-i", str(audio), "-i", str(srt)]
    if chapters is not None:
        cmd += ["-i", str(chapters), "-map_metadata", "3"]
    # `-dn` because the chapter metadata input otherwise arrives as a fourth, empty data stream in the
    # output. Harmless, but a player that lists streams would show it, and it is not a stream.
    cmd += ["-map", "0:v:0", "-map", "1:a:0", "-map", "2:s:0", "-c:s", "mov_text", "-dn",
            "-metadata:s:s:0", "language=eng"]
    if burn:
        cmd += ["-vf", f"ass={ass}", "-c:v", "libx264",
                "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p"]
    else:
        cmd += ["-c:v", "copy"]
    cmd += ["-c:a", "aac", "-b:a", "192k", str(out)]
    subprocess.run(cmd, check=True)


def check_tools() -> list[str]:
    missing = []
    if not shutil.which("ffmpeg"):
        missing.append("ffmpeg")
    if not VOICE.exists():
        missing.append(f"voice model at {VOICE}")
    return missing


def save_manifest(entries: list[dict], path: Path) -> None:
    """What was recorded, how long each scene ran, and the exact words spoken.

    Written beside the video because a tour is a claim about a system, and somebody should be able to
    check a claim against the page it was made on without re-watching.
    """
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


__all__ = ["synth", "write_srt", "write_ass", "write_chapters", "quantize", "probe_duration", "fit_duration",
           "pad_audio", "concat", "concat_audio", "mux", "check_tools", "save_manifest",
           "OUT", "WEB", "API", "W", "H", "FPS"]
