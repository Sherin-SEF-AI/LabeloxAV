"""Record the auto-labeller annotating a real drive live, with Groq as its vision verifier, beside the sensors.

    .venv/bin/python scripts/demo_tour/groq_live.py trial            3 frames, prints timings, writes nothing
    .venv/bin/python scripts/demo_tour/groq_live.py record           the full run, into .scratch/demo/groq_live/
    .venv/bin/python scripts/demo_tour/groq_live.py finish           title and end cards around the recording

The drive is a phone recording from Bengaluru (Whitefield, 2 km, 6.6 minutes): trip.mp4 plus trip.mcap with
GPS fixes, IMU and the phone's own road-event detector. Nothing here is simulated or replayed from a cache:

- Each sampled frame is read from the video, and faces and number plates are blurred with the platform's own
  anonymiser before the frame is shown or any crop of it leaves the machine.
- The product's stage-one detectors (YOLO, YOLO-World with SAM) run on it, fusion and the gate run exactly as
  in services/autolabel/runner.py, and every object the gate sends to the vision verifier is sent to Groq
  through the product's VlmVerifier. The verdict is applied with the product's apply_vlm and the object is
  re-gated, so the colours on screen are the states the platform would store.
- GPS, speed, the IMU trace and road events come from trip.mcap and events.csv, aligned to the video by the
  camera's frame timestamps.

**Real time means the recording's clock is the wall clock.** The console redraws whenever something changes
(a frame is detected, a crop is sent, a verdict comes back), and each drawing is held in the film for exactly
as long as it was on screen while the pipeline ran. A one-second Groq call is a one-second pause. The run is
not sped up, and the latencies printed on screen are measured, not estimated.

Nothing is written to the database: this shows the pipeline's decisions without adding a session.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from resource_watch import wait_while_paused  # noqa: E402

DRIVE = Path("/home/jo/Downloads/datacollection2")
OUT = ROOT / ".scratch/demo/groq_live"
FLAG = ROOT / ".scratch/watch/PAUSE.json"
W, H = 1920, 1080

# The stretches of the drive with traffic in them, in seconds of video, and the sampling step within each.
SEGMENTS = [(56.0, 76.0), (136.0, 190.0), (252.0, 280.0), (334.0, 392.0)]
STEP_S = 2.0
# Verifier calls allowed per frame. The product's default is 2 (models.vlm.max_calls_per_frame); this demo
# allows 4 so more of Groq's work is visible, and says so on screen.
CALLS_PER_FRAME = 4

# The phone was held upright on the dashboard: the video is portrait (1080 wide, 1920 tall, stored landscape with
# a rotation flag) and the lower half of every frame is the dashboard. Detection runs on the whole upright
# frame; only the display is cropped to the road above the dashboard.
VIEW_BOTTOM = 0.52

# Palette
BG = (9, 12, 19)
PANEL = (17, 21, 31)
LINE = (38, 45, 60)
TEXT = (230, 234, 242)
MUTED = (140, 149, 168)
VIOLET = (156, 123, 255)
CYAN = (86, 204, 242)
GREEN = (86, 211, 100)
AMBER = (240, 180, 60)
RED = (248, 110, 110)
GREY = (150, 155, 165)

FONT = "/usr/share/fonts/truetype/ubuntu/UbuntuSans[wdth,wght].ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def font(size: int, weight: int = 400, mono: bool = False) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(MONO if mono else FONT, size)
    if not mono:
        try:
            f.set_variation_by_axes([100, weight])
        except (OSError, ValueError):
            pass
    return f


# ------------------------------------------------------------------ sensors

@dataclass
class Sensors:
    gps_t: np.ndarray          # video seconds
    lat: np.ndarray
    lon: np.ndarray
    speed_t: np.ndarray
    speed: np.ndarray          # m/s, from consecutive fixes
    imu_t: np.ndarray
    imu_v: np.ndarray          # vertical linear acceleration, m/s^2, 20 Hz
    events: list[tuple[float, str]]  # (video seconds, type)

    def at(self, arr_t: np.ndarray, arr: np.ndarray, t: float) -> float:
        return float(np.interp(t, arr_t, arr))


def _haversine(lat1, lon1, lat2, lon2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def load_sensors() -> Sensors:
    """Read GPS and IMU from trip.mcap and put everything on the video's clock.

    Video time zero is the first /camera/front frame's log time: the camera topic carries one timing record per
    video frame, and events.csv confirms the offset (every event's t_s minus video_t_s is the same 1.22 s).
    """
    cache = OUT / "sensors.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return Sensors(*(z[k] for k in ("gps_t", "lat", "lon", "speed_t", "speed", "imu_t", "imu_v")),
                       events=[tuple(e) for e in z["events"].tolist()])
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    cam0 = None
    gps, imu = [], []
    with open(DRIVE / "trip.mcap", "rb") as fh:
        r = make_reader(fh, decoder_factories=[DecoderFactory()])
        for _sch, ch, msg, dec in r.iter_decoded_messages(
                topics=["/camera/front", "/gps/fix", "/imu/linear_acceleration"]):
            if ch.topic == "/camera/front":
                if cam0 is None:
                    cam0 = msg.log_time
            elif ch.topic == "/gps/fix":
                gps.append((msg.log_time, dec.latitude, dec.longitude))
            else:
                imu.append((msg.log_time, dec.z))
    g = np.array(gps, dtype=np.float64)
    gps_t = (g[:, 0] - cam0) / 1e9
    sp_t, sp = [], []
    for i in range(1, len(g)):
        dt = (g[i, 0] - g[i - 1, 0]) / 1e9
        if dt > 0:
            sp_t.append((gps_t[i] + gps_t[i - 1]) / 2)
            sp.append(_haversine(g[i - 1, 1], g[i - 1, 2], g[i, 1], g[i, 2]) / dt)
    sp = np.convolve(np.array(sp), np.ones(3) / 3, mode="same")        # 3 s moving average against GPS jitter
    a = np.array(imu, dtype=np.float64)
    imu_t = (a[:, 0] - cam0) / 1e9
    # 20 Hz is enough for a trace a few hundred pixels wide, and keeps every redraw cheap.
    bins = np.floor(imu_t * 20).astype(int)
    _, first = np.unique(bins, return_index=True)
    imu_t_ds = imu_t[first]
    imu_v_ds = np.array([a[s:e, 1].mean() for s, e in zip(first, list(first[1:]) + [len(a)], strict=True)])
    events = []
    with open(DRIVE / "events.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            events.append((float(row["video_t_s"]), row["type"]))
    s = Sensors(gps_t, g[:, 1], g[:, 2], np.array(sp_t), sp, imu_t_ds, imu_v_ds, events)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(cache, gps_t=s.gps_t, lat=s.lat, lon=s.lon, speed_t=s.speed_t, speed=s.speed, imu_t=s.imu_t,
             imu_v=s.imu_v, events=np.array(s.events, dtype=object))
    return s


# ------------------------------------------------------------------ pipeline

@dataclass
class Shown:
    """One object as the console shows it."""
    key: int
    bbox: tuple
    cls: str
    conf: float
    state: str
    detector_cls: str
    asked: bool = False
    asking: bool = False
    verdict: str = ""          # confirm | overruled | unsure | failed
    groq_cls: str = ""


@dataclass
class Call:
    thumb: Image.Image
    detector_cls: str
    detector_conf: float
    groq_cls: str
    verdict: str
    latency_ms: float
    confident: bool
    drive_t: float


@dataclass
class Console:
    sensors: Sensors
    model: str
    frame: np.ndarray | None = None
    drive_t: float = 0.0
    objects: list[Shown] = field(default_factory=list)
    calls: list[Call] = field(default_factory=list)
    status: str = ""
    n_frames: int = 0
    n_objects: int = 0
    n_calls: int = 0
    n_changed: int = 0
    n_failed: int = 0
    wall0: float = 0.0
    pii: tuple[int, int] = (0, 0)


class Pipeline:
    """The product's per-frame auto-label path, without the database."""

    def __init__(self) -> None:
        from core.config import get_settings
        from services.anonymize.anonymizer import get_anonymizer
        from services.autolabel.fusion import FusionEngine
        from services.autolabel.grounding import supported_core_ids
        from services.autolabel.ontology import get_ontology
        from services.autolabel.paths.path_c_vlm import VlmVerifier
        from services.autolabel.runner import StagedRunner
        from services.llm.router import GroqVlmClient

        self.s = get_settings()
        self.onto = get_ontology()
        sup = supported_core_ids(self.onto)
        self.anon = get_anonymizer()
        self.runner = StagedRunner(None, sup)
        self.runner.open_stage1()
        self.engine = FusionEngine(self.s, self.onto)
        # Groq directly rather than through the router, so a failed call shows as a failure on screen instead of
        # being answered by the local model.
        self.client = GroqVlmClient(self.s)
        if not self.client.available():
            raise SystemExit("no Groq API key configured (GROQ_API_KEY)")
        self.verifier = VlmVerifier(self.client, self.onto, self.s, supported_ids=sup)
        self.model = self.s.groq.vision_model

    def close(self) -> None:
        self.runner.close_stage1()


def _state(v) -> str:
    return getattr(v, "value", str(v))


# ------------------------------------------------------------------ rendering

def _chip(d: ImageDraw.ImageDraw, xy, text, fnt, fg, bg, pad=(8, 4)) -> int:
    x, y = xy
    w = d.textlength(text, font=fnt)
    d.rounded_rectangle([x, y, x + w + 2 * pad[0], y + fnt.size + 2 * pad[1]], 6, fill=bg)
    d.text((x + pad[0], y + pad[1] - 1), text, font=fnt, fill=fg)
    return int(w + 2 * pad[0])


def _box_colour(o: Shown) -> tuple:
    if o.asking:
        return CYAN
    if o.verdict == "overruled":
        return VIOLET
    if o.state == "auto_accept":
        return GREEN
    if o.state == "review":
        return AMBER
    return GREY


class Renderer:
    def __init__(self) -> None:
        self.f = {k: font(*v) for k, v in {
            "h1": (30, 650), "h2": (22, 600), "b": (19, 450), "bs": (17, 450), "s": (15, 450),
            "big": (40, 700)}.items()}
        # Box labels carry an arrow and check marks, which Ubuntu Sans does not have: the first take drew them
        # as empty boxes. DejaVu Sans covers them.
        self.f["label"] = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        self.fm = font(15, mono=True)
        self.fms = font(13, mono=True)

    def draw(self, c: Console) -> Image.Image:
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        self._header(d, c)
        self._video(im, d, c)
        self._sensors(im, d, c)
        self._feed(im, d, c)
        return im

    def _header(self, d, c: Console) -> None:
        d.rectangle([0, 0, W, 64], fill=PANEL)
        d.ellipse([28, 24, 44, 40], fill=RED)
        d.text((56, 15), "LIVE", font=self.f["h1"], fill=RED)
        d.text((140, 15), "Auto-labelling a real Bengaluru drive", font=self.f["h1"], fill=TEXT)
        x = 700
        x += _chip(d, (x, 18), f"vision verifier: Groq  {c.model}", self.f["bs"], TEXT, (52, 40, 96)) + 10
        _chip(d, (x, 18), "detectors: YOLO + YOLO-World + SAM, on this GPU", self.f["bs"], TEXT, (32, 40, 56))
        wall = time.monotonic() - c.wall0 if c.wall0 else 0.0
        d.text((W - 360, 12), f"drive  {int(c.drive_t // 60)}:{c.drive_t % 60:04.1f}", font=self.f["h2"], fill=TEXT)
        d.text((W - 360, 36), f"wall clock  {int(wall // 60)}:{wall % 60:04.1f}  (not sped up)", font=self.f["s"],
               fill=MUTED)

    def _video(self, im, d, c: Console) -> None:
        x0, y0, vw, vh = 24, 80, 840, 776
        d.rectangle([x0 - 1, y0 - 1, x0 + vw, y0 + vh], outline=LINE)
        if c.frame is None:
            return
        fh, fw = c.frame.shape[:2]
        bottom = int(fh * VIEW_BOTTOM)
        crop = c.frame[:bottom, :]
        rgb = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).resize((vw, vh), Image.LANCZOS)
        im.paste(rgb, (x0, y0))
        sx, sy = vw / fw, vh / bottom
        # Smaller boxes last, so their labels sit on top.
        for o in sorted(c.objects, key=lambda o: -(o.bbox[2] - o.bbox[0]) * (o.bbox[3] - o.bbox[1])):
            x1, y1, x2, y2 = o.bbox
            if y1 >= bottom:
                continue
            bx1, by1 = x0 + x1 * sx, y0 + y1 * sy
            bx2, by2 = x0 + x2 * sx, y0 + min(y2, bottom) * sy
            col = _box_colour(o)
            d.rectangle([bx1, by1, bx2, by2], outline=col, width=4 if (o.asked or o.asking) else 2)
            label = o.cls
            if o.verdict == "overruled":
                label = f"{o.detector_cls} \u2192 {o.cls}"
            if o.asking:
                label = f"{o.cls}  asking Groq\u2026"
            txt = f"{label} {o.conf:.2f}"
            tw = d.textlength(txt, font=self.f["label"])
            ly = max(y0, by1 - 22)
            lx = min(bx1, x0 + vw - tw - 12)
            d.rectangle([lx, ly, lx + tw + 12, ly + 21], fill=col)
            d.text((lx + 6, ly + 1), txt, font=self.f["label"], fill=(10, 12, 18))
            if o.asked and not o.asking:
                mark = {"confirm": "G \u2713", "overruled": "G \u270e", "unsure": "G ?", "failed": "G \u00d7"}[o.verdict]
                _chip(d, (bx2 - 44, by2 - 24), mark, self.f["label"], (10, 12, 18), col, pad=(6, 2))
        d.text((x0 + 12, y0 + vh - 26), f"faces {c.pii[0]}, plates {c.pii[1]} blurred on this frame",
               font=self.f["s"], fill=TEXT)
        # Legend
        ly = y0 + vh + 12
        lx = x0
        for col, name in ((GREEN, "auto-accepted"), (AMBER, "to a person"), (GREY, "needs drawing"),
                          (CYAN, "asking Groq"), (VIOLET, "Groq changed it")):
            d.rectangle([lx, ly + 3, lx + 14, ly + 17], fill=col)
            d.text((lx + 20, ly), name, font=self.f["s"], fill=TEXT)
            lx += 34 + d.textlength(name, font=self.f["s"])

    def _sensors(self, im, d, c: Console) -> None:
        s = c.sensors
        x0, y0 = 888, 80
        # Map
        mw, mh = 440, 440
        d.rectangle([x0, y0, x0 + mw, y0 + mh], fill=PANEL)
        d.text((x0 + 14, y0 + 10), "GPS, Whitefield, Bengaluru", font=self.f["h2"], fill=TEXT)
        lat, lon = s.lat, s.lon
        la0, la1, lo0, lo1 = lat.min(), lat.max(), lon.min(), lon.max()
        kx = math.cos(math.radians((la0 + la1) / 2))
        span = max((lo1 - lo0) * kx, la1 - la0)
        ox, oy, size = x0 + 26, y0 + 50, min(mw, mh - 60) - 52

        def pt(la, lo):
            return (ox + ((lo - lo0) * kx) / span * size + (size - (lo1 - lo0) * kx / span * size) / 2,
                    oy + (la1 - la) / span * size + (size - (la1 - la0) / span * size) / 2)
        path = [pt(a, b) for a, b in zip(lat, lon, strict=True)]
        d.line(path, fill=LINE, width=4)
        done = [p for p, t in zip(path, s.gps_t, strict=True) if t <= c.drive_t]
        if len(done) > 1:
            d.line(done, fill=VIOLET, width=4)
        for seg in SEGMENTS:
            for t in seg:
                p = pt(s.at(s.gps_t, lat, t), s.at(s.gps_t, lon, t))
                d.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=MUTED)
        cur = pt(s.at(s.gps_t, lat, c.drive_t), s.at(s.gps_t, lon, c.drive_t))
        d.ellipse([cur[0] - 9, cur[1] - 9, cur[0] + 9, cur[1] + 9], fill=TEXT)
        d.ellipse([cur[0] - 5, cur[1] - 5, cur[0] + 5, cur[1] + 5], fill=VIOLET)
        m_per_px = span * 111_320 / size
        bar = 200 / m_per_px
        d.line([x0 + 20, y0 + mh - 20, x0 + 20 + bar, y0 + mh - 20], fill=MUTED, width=3)
        d.text((x0 + 26 + bar, y0 + mh - 30), "200 m", font=self.f["s"], fill=MUTED)
        la_now, lo_now = s.at(s.gps_t, lat, c.drive_t), s.at(s.gps_t, lon, c.drive_t)
        d.text((x0 + mw - 210, y0 + mh - 30), f"{la_now:.5f}, {lo_now:.5f}", font=self.fms, fill=MUTED)
        # Charts, under the video
        self._chart(d, (24, 904, 412, 152), s.speed_t, s.speed * 3.6, c.drive_t, "speed from GPS", "km/h",
                    (0, max(60.0, float(s.speed.max() * 3.6))), GREEN)
        self._chart(d, (452, 904, 412, 152), s.imu_t, s.imu_v, c.drive_t, "IMU vertical accel.",
                    "m/s²", (-6, 6), CYAN)
        # Events, under the map
        cx0, cw, ey = x0, mw, y0 + mh + 16
        d.rectangle([cx0, ey, cx0 + cw, H - 24], fill=PANEL)
        d.text((cx0 + 14, ey + 10), "road events, phone IMU", font=self.f["h2"], fill=TEXT)
        recent = [e for e in s.events if c.drive_t - 12 <= e[0] <= c.drive_t]
        total = sum(1 for e in s.events if e[0] <= c.drive_t)
        d.text((cx0 + cw - 150, ey + 14), f"{total} of {len(s.events)}", font=self.f["s"], fill=MUTED)
        colours = {"POTHOLE_IMPACT": RED, "SPEED_BUMP": AMBER, "RAPID_LANE_CHANGE": CYAN, "SHARP_TURN": VIOLET,
                   "AGGRESSIVE_DRIVING": (240, 130, 90)}
        yy = ey + 46
        for t, kind in recent[-12:][::-1]:
            if yy > H - 60:
                break
            _chip(d, (cx0 + 14, yy), kind.replace("_", " ").lower(), self.f["bs"], (10, 12, 18),
                  colours.get(kind, GREY))
            d.text((cx0 + 260, yy + 3), f"{c.drive_t - t:4.1f} s ago", font=self.f["bs"], fill=MUTED)
            yy += 30
        if not recent:
            d.text((cx0 + 14, yy), "none in the last 12 s", font=self.f["bs"], fill=MUTED)

    def _chart(self, d, box, ts, vs, now, title, unit, yr, col) -> None:
        x0, y0, w, h = box
        d.rectangle([x0, y0, x0 + w, y0 + h], fill=PANEL)
        d.text((x0 + 14, y0 + 8), title, font=self.f["bs"], fill=TEXT)
        cur = float(np.interp(now, ts, vs))
        d.text((x0 + w - 124, y0 + 6), f"{cur:5.1f} {unit}", font=self.f["b"], fill=col)
        px0, py0, pw, ph = x0 + 14, y0 + 36, w - 28, h - 48
        win = 30.0
        m = (ts >= now - win) & (ts <= now)
        lo, hi = yr
        d.line([px0, py0 + ph * (hi / (hi - lo)) if lo < 0 else py0 + ph, px0 + pw,
                py0 + ph * (hi / (hi - lo)) if lo < 0 else py0 + ph], fill=LINE, width=1)
        if m.sum() > 1:
            pts = [(px0 + (t - (now - win)) / win * pw, py0 + (hi - min(max(v, lo), hi)) / (hi - lo) * ph)
                   for t, v in zip(ts[m], vs[m], strict=True)]
            d.line(pts, fill=col, width=2)
        d.text((px0, py0 + ph + 1), "last 30 s", font=self.fms, fill=MUTED)

    def _feed(self, im, d, c: Console) -> None:
        x0, y0, fw = 1344, 80, 552
        d.rectangle([x0, y0, x0 + fw, H - 24], fill=PANEL)
        d.text((x0 + 16, y0 + 12), "What the pipeline is doing", font=self.f["h2"], fill=TEXT)
        d.text((x0 + 16, y0 + 44), c.status, font=self.f["b"], fill=CYAN)
        # Stats
        sy = y0 + 84
        avg = [k.latency_ms for k in c.calls if k.verdict != "failed"]
        stats = [("frames", c.n_frames), ("objects", c.n_objects), ("Groq calls", c.n_calls),
                 ("classes changed", c.n_changed), ("median latency",
                                                    f"{int(np.median(avg))} ms" if avg else "-")]
        for i, (k, v) in enumerate(stats):
            cx = x0 + 16 + (i % 3) * 178
            cy = sy + (i // 3) * 66
            d.text((cx, cy), str(v), font=self.f["big"], fill=TEXT)
            d.text((cx, cy + 44), k, font=self.f["s"], fill=MUTED)
        d.line([x0 + 16, sy + 140, x0 + fw - 16, sy + 140], fill=LINE)
        d.text((x0 + 16, sy + 150), "Groq verdicts, newest first", font=self.f["bs"], fill=MUTED)
        yy = sy + 180
        for k in c.calls[::-1][:7]:
            if yy > H - 120:
                break
            th = k.thumb.copy()
            th.thumbnail((96, 96))
            im.paste(th, (x0 + 16 + (96 - th.width) // 2, yy + (96 - th.height) // 2))
            d.rectangle([x0 + 16, yy, x0 + 112, yy + 96], outline=LINE)
            tx = x0 + 128
            d.text((tx, yy + 2), f"detector  {k.detector_cls}  {k.detector_conf:.2f}", font=self.f["bs"], fill=MUTED)
            col = {"confirm": GREEN, "overruled": VIOLET, "unsure": GREY, "failed": RED}[k.verdict]
            line = {"confirm": f"Groq  confirmed  {k.groq_cls}",
                    "overruled": f"Groq  changed it to  {k.groq_cls}",
                    # Not confident enough to change the class, so the detector's class stands.
                    "unsure": f"Groq  unsure ({k.groq_cls}?), kept {k.detector_cls}",
                    "failed": "Groq  call failed"}[k.verdict]
            d.text((tx, yy + 28), line, font=self.f["b"], fill=col)
            d.text((tx, yy + 58), f"{k.latency_ms:.0f} ms   at {k.drive_t:.0f} s"
                   + ("   confident" if k.confident else ""), font=self.fm, fill=MUTED)
            yy += 110
        d.text((x0 + 16, H - 74), "a class Groq changes still goes to a person to check;",
               font=self.fms, fill=MUTED)
        d.text((x0 + 16, H - 58), f"up to {CALLS_PER_FRAME} verifier calls per frame (product default 2);",
               font=self.fms, fill=MUTED)
        d.text((x0 + 16, H - 42), "crops leave the machine only after face and plate blurring",
               font=self.fms, fill=MUTED)


# ------------------------------------------------------------------ recording

class Tape:
    """Frames held for their real on-screen duration: the recording's clock is the wall clock."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        for p in folder.glob("*.jpg"):
            p.unlink()
        self.items: list[tuple[Path, float]] = []
        self.last: float | None = None

    def show(self, im: Image.Image) -> None:
        now = time.monotonic()
        if self.items and self.last is not None:
            p, _ = self.items[-1]
            self.items[-1] = (p, now - self.last)
        p = self.folder / f"{len(self.items):06d}.jpg"
        im.save(p, quality=92)
        self.items.append((p, 0.0))
        self.last = now

    def close(self, tail_s: float = 2.0) -> None:
        if self.items:
            p, _ = self.items[-1]
            self.items[-1] = (p, tail_s)

    def encode(self, out: Path) -> float:
        listing = self.folder / "concat.txt"
        lines = []
        for p, dur in self.items:
            lines += [f"file '{p.name}'", f"duration {max(dur, 1 / 30):.4f}"]
        lines.append(f"file '{self.items[-1][0].name}'")
        listing.write_text("\n".join(lines) + "\n")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-vf", "fps=30,format=yuv420p", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                        str(out)], check=True, cwd=self.folder)
        return sum(max(d, 1 / 30) for _, d in self.items)


# ------------------------------------------------------------------ the run

def run(mode: str) -> None:
    from services.autolabel.gate import gate_object, needs_vlm
    from services.autolabel.paths.path_c_vlm import apply_vlm, crop_object
    from services.autolabel.quality_reviewer import review_object_quality
    from services.llm.groq_client import GroqError

    sensors = load_sensors()
    times = [t for a, b in SEGMENTS for t in np.arange(a, b, STEP_S)]
    if mode == "trial":
        times = times[:3]
    pipe = Pipeline()
    con = Console(sensors=sensors, model=pipe.model)
    rend = Renderer()
    tape = Tape(OUT / "tape") if mode == "record" else None
    cap = cv2.VideoCapture(str(DRIVE / "trip.mp4"))
    # The file is stored landscape with a rotation flag; without this OpenCV returns the frames sideways, and
    # the detectors were being shown a bus lying on its side.
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    log: list[dict] = []

    def show() -> None:
        if tape is not None:
            tape.show(rend.draw(con))

    con.wall0 = time.monotonic()
    try:
        for t in times:
            if not wait_while_paused(FLAG, timeout_s=900):
                raise SystemExit(f"stopping at {t:.0f} s: resources over their limits: {FLAG.read_text()}")
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                continue
            if frame.shape[0] < frame.shape[1]:
                raise SystemExit(f"frame is landscape {frame.shape}: the rotation flag was not applied")
            con.drive_t = t
            t0 = time.monotonic()
            pii = pipe.anon.anonymize(frame)          # in place: nothing unblurred is shown or sent
            con.pii = (pii.n_faces, pii.n_plates)
            con.frame = frame
            con.objects = []
            con.status = f"detecting objects in the frame at {t:.0f} s"
            show()
            dets_a, dets_b = pipe.runner.run_stage1_frame(frame)
            import uuid
            fused = [g for g in pipe.engine.fuse_frame(uuid.uuid4(), dets_a, dets_b) if pipe.onto.is_thing(g.obj.class_id)]
            objs = [g.obj for g in fused]
            h, w = frame.shape[:2]
            qok = {}
            for o in objs:
                qok[id(o)] = review_object_quality(o, [x for x in objs if x is not o], pipe.onto, w, h,
                                                   pipe.s.quality).ok
                o.state = gate_object(o, pipe.onto, pipe.s.gate, quality_ok=qok[id(o)])
            det_ms = (time.monotonic() - t0) * 1000
            shown = {id(o): Shown(id(o), tuple(o.bbox.as_list()), o.class_name, float(o.conf), _state(o.state),
                                  o.class_name) for o in objs}
            con.objects = list(shown.values())
            con.n_frames += 1
            con.n_objects += len(objs)
            asks = [o for o in sorted(objs, key=lambda o: o.conf)
                    if needs_vlm(o, pipe.onto, pipe.s.gate, quality_ok=qok[id(o)])][:CALLS_PER_FRAME]
            con.status = f"{len(objs)} objects in {det_ms:.0f} ms; {len(asks)} uncertain, sending to Groq"
            show()
            rec = {"t": t, "objects": len(objs), "detect_ms": round(det_ms), "calls": []}
            for o in asks:
                so = shown[id(o)]
                so.asking = True
                con.status = f"asking Groq about a {o.class_name} ({o.conf:.2f})"
                show()
                crop = crop_object(frame, tuple(o.bbox.as_list()), pipe.s.models.vlm.crop_margin)
                thumb = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) if crop.size else Image.new("RGB", (8, 8))
                c0 = time.monotonic()
                before = o.class_name
                try:
                    res = pipe.verifier.verify_object(frame, tuple(o.bbox.as_list()), o.class_id)
                    lat = (time.monotonic() - c0) * 1000
                    apply_vlm(o, res, pipe.onto, pipe.model)
                    o.state = gate_object(o, pipe.onto, pipe.s.gate, quality_ok=qok[id(o)])
                    verdict = next((p.verdict for p in reversed(o.provenance.proposals)
                                    if p.path == "path_c_qwen3vl"), "unsure") or "unsure"
                    verdict = verdict if verdict in ("confirm", "overruled") else "unsure"
                    groq_cls = res.class_name or "-"
                    confident = bool(res.confident)
                except GroqError as exc:
                    lat = (time.monotonic() - c0) * 1000
                    verdict, groq_cls, confident = "failed", "", False
                    con.n_failed += 1
                    rec.setdefault("errors", []).append(str(exc)[:120])
                so.asking, so.asked, so.verdict = False, True, verdict
                so.cls, so.conf, so.state = o.class_name, float(o.conf), _state(o.state)
                con.n_calls += 1
                if verdict == "overruled":
                    con.n_changed += 1
                con.calls.append(Call(thumb, before, so.conf if verdict == "failed" else float(o.conf), groq_cls,
                                      verdict, lat, confident, t))
                rec["calls"].append({"detector": before, "groq": groq_cls, "verdict": verdict,
                                     "final": o.class_name, "state": _state(o.state), "ms": round(lat),
                                     "confident": confident})
                con.status = f"Groq answered in {lat:.0f} ms: {verdict}"
                show()
            log.append(rec)
            print(f"{t:6.1f}s  {len(objs):2d} objects  detect {det_ms:5.0f} ms  "
                  + "  ".join(f"{c['detector']}->{c['groq']}:{c['verdict']}({c['ms']}ms)" for c in rec["calls"]),
                  flush=True)
    finally:
        pipe.close()
        cap.release()
    if tape is not None:
        con.status = "done"
        show()
        tape.close()
        dur = tape.encode(OUT / "groq-live-raw.mp4")
        (OUT / "run.json").write_text(json.dumps({
            "model": pipe.model, "frames": con.n_frames, "objects": con.n_objects, "groq_calls": con.n_calls,
            "classes_changed": con.n_changed, "failed_calls": con.n_failed, "film_seconds": round(dur, 1),
            "per_frame": log}, indent=1))
        print(f"recorded {dur:.0f} s of real time to {OUT / 'groq-live-raw.mp4'}")


def finish() -> Path:
    """Put the branded title sequence and end card around the live recording. The recording is not cut."""
    import asyncio

    sys.path.insert(0, str(HERE))
    import brand as B

    run_log = json.loads((OUT / "run.json").read_text())
    raw = OUT / "groq-live-raw.mp4"
    tape = sorted((OUT / "tape").glob("*.jpg"))
    stills = [tape[int(len(tape) * f)] for f in (0.15, 0.4, 0.65, 0.9)]
    lat = [c["ms"] for r in run_log["per_frame"] for c in r["calls"] if c["verdict"] != "failed"]
    note = (f"{run_log['frames']} frames, {run_log['objects']} objects, {run_log['groq_calls']} Groq calls, "
            f"median {int(np.median(lat))} ms each, {run_log['classes_changed']} classes changed by Groq. "
            "Recorded live, not sped up.")
    intro_s, outro_s = 9.5, 10.0
    g = OUT / "graphics"
    jobs = [
        B.Job(B.intro_html(stills, "Groq annotating a real Bengaluru drive, live",
                           ("Real Indian driving data", "GPS and IMU", "YOLO + SAM detectors",
                            f"Groq {run_log['model']}", "Faces and plates blurred"),
                           "Every pause in this film is the pipeline really working"), g / "intro", seconds=intro_s),
        B.Job(B.outro_html(stills[1:], "0.1.1", "Drive: Whitefield, Bengaluru, recorded on a phone with GPS and IMU.",
                           note), g / "outro", seconds=outro_s),
    ]
    asyncio.run(B.render(jobs, g / "html", parallel=2))
    clips = []
    for name, secs in (("intro", intro_s), ("outro", outro_s)):
        out = OUT / f"{name}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i", str(g / name / "%05d.jpg"),
                        "-t", f"{secs}", "-vf", "format=yuv420p", "-c:v", "libx264", "-crf", "18", str(out)],
                       check=True)
        clips.append(out)
    raw_s = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                  str(raw)], capture_output=True, text=True).stdout)
    xf = 0.8
    final = OUT / "labeloxav-groq-live.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(clips[0]), "-i", str(raw), "-i", str(clips[1]),
        "-filter_complex",
        f"[0:v]fps=30,settb=AVTB[a];[1:v]fps=30,settb=AVTB[b];[2:v]fps=30,settb=AVTB[c];"
        f"[a][b]xfade=transition=fade:duration={xf}:offset={intro_s - xf:.3f}[ab];"
        f"[ab][c]xfade=transition=fade:duration={xf}:offset={intro_s + raw_s - 2 * xf:.3f},format=yuv420p[v]",
        "-map", "[v]", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-movflags", "+faststart", str(final)],
        check=True)
    print(f"wrote {final}")
    return final


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "trial"
    if mode == "finish":
        finish()
    else:
        run(mode)
