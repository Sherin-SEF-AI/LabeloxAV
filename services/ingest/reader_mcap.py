"""MCAP reader. MCAP is the native container (Principle 07): sensor streams in one indexed,
timestamped file. Camera images plus GNSS/IMU/CAN side channels are demuxed and each frame is
mapped to UTC nanoseconds from the message log time.

Protobuf messages are decoded generically from the schema embedded in the file, so this works
with Foxglove well-known schemas (CompressedImage, CompressedVideo, LocationFix) without
compile-time stubs. Two camera encodings are supported: per-message still images
(CompressedImage: JPEG/PNG via cv2.imdecode) and an encoded video stream (CompressedVideo:
H.264/H.265/AV1/VP9, decoded with PyAV while holding decoder state across frames, since delta
frames reference earlier ones).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from core.logging import get_logger
from services.ingest.types import RawFrame

log = get_logger(__name__)

# CompressedVideo.format -> the FFmpeg/PyAV decoder name. "h265" is the Foxglove spelling; the
# codec is "hevc". Anything not here is unsupported and logged once.
_VIDEO_CODECS = {
    "h264": "h264", "avc": "h264", "avc1": "h264",
    "h265": "hevc", "hevc": "hevc",
    "av1": "av1", "vp9": "vp9", "vp8": "vp8",
}
# Topic segments that name the medium, not the camera, so they fall back to the rig default id.
_GENERIC_CAM_SEGMENTS = {
    "video", "image", "image_raw", "image_color", "compressed", "compressedimage",
    "compressedvideo", "color", "rgb", "camera", "cam",
}


def _decode_image(msg) -> np.ndarray | None:
    data = getattr(msg, "data", None)
    if not data:
        return None
    buf = np.frombuffer(bytes(data), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _is_image_schema(name: str) -> bool:
    return "CompressedImage" in name or name.endswith(".Image")


def _is_video_schema(name: str) -> bool:
    return "CompressedVideo" in name


def _is_gnss_schema(name: str) -> bool:
    return "LocationFix" in name or "NavSatFix" in name


def _cam_id_from_topic(topic: str, default: str) -> str:
    """Pick the camera id from a topic, skipping medium words: /camera/video -> default,
    /cam_l/compressed -> cam_l, /sensors/front/image_raw -> front."""
    segs = [s for s in topic.strip("/").split("/") if s]
    for s in reversed(segs):
        if s.lower() not in _GENERIC_CAM_SEGMENTS:
            return s
    return default


class _VideoTrack:
    """Holds one camera's video decoder. Each CompressedVideo message is one access unit; feeding
    them in order keeps the inter-frame prediction state intact. Output order equals input order
    for the low-latency (no B-frame) streams these cameras produce, so side-channel state and
    timestamps are paired to frames through a FIFO."""

    def __init__(self, codec_name: str) -> None:
        import av

        self._ctx = av.CodecContext.create(codec_name, "r")
        self._av = av
        self._pending: deque[tuple] = deque()

    def feed(self, data: bytes, meta: tuple) -> Iterator[tuple]:
        self._pending.append(meta)
        packet = self._av.Packet(data)
        try:
            frames = self._ctx.decode(packet)
        except Exception as exc:  # noqa: BLE001  (a corrupt access unit must not abort the clip)
            log.warning("mcap.video_decode_failed", error=str(exc))
            if self._pending:
                self._pending.popleft()
            return
        for frame in frames:
            yield self._emit(frame)

    def flush(self) -> Iterator[tuple]:
        try:
            frames = self._ctx.decode(None)  # drain buffered frames at end of stream
        except Exception:  # noqa: BLE001
            return
        for frame in frames:
            yield self._emit(frame)

    def _emit(self, frame) -> tuple:
        meta = self._pending.popleft() if self._pending else (None, None, None, None)
        return meta, frame.to_ndarray(format="bgr24")


def read_mcap(
    mcap_path: str | Path,
    target_fps: float,
    default_cam_id: str = "cam_f",
) -> Iterator[RawFrame]:
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    mcap_path = Path(mcap_path)
    if not mcap_path.exists():
        raise FileNotFoundError(mcap_path)

    interval_ns = int(1e9 / target_fps) if target_fps > 0 else 0
    last_kept: dict[str, int] = {}

    # Forward-filled latest side-channel state, attached to camera frames as they arrive.
    cur_lat: float | None = None
    cur_lon: float | None = None
    cur_speed: float | None = None

    tracks: dict[str, _VideoTrack] = {}
    unsupported_logged: set[str] = set()

    def _keep(cam_id: str, ts_ns: int) -> bool:
        # Decode every frame to keep predictor state, but only label at the target cadence.
        if interval_ns and (ts_ns - last_kept.get(cam_id, -interval_ns)) < interval_ns:
            return False
        last_kept[cam_id] = ts_ns
        return True

    with open(mcap_path, "rb") as fh:
        reader = make_reader(fh, decoder_factories=[DecoderFactory()])
        for schema, channel, message, proto in reader.iter_decoded_messages():
            ts_ns = int(message.log_time)
            sname = schema.name if schema else ""

            if _is_gnss_schema(sname):
                cur_lat = getattr(proto, "latitude", cur_lat)
                cur_lon = getattr(proto, "longitude", cur_lon)
                continue

            # Vehicle speed: a float-bearing message on a speed topic.
            if "speed" in channel.topic.lower():
                for attr in ("ego_speed", "speed", "value"):
                    if hasattr(proto, attr):
                        cur_speed = float(getattr(proto, attr))
                        break
                continue

            if _is_image_schema(sname):
                cam_id = channel.topic.strip("/").split("/")[-1] or default_cam_id
                if not _keep(cam_id, ts_ns):
                    continue
                img = _decode_image(proto)
                if img is None:
                    log.warning("mcap.image_decode_failed", topic=channel.topic, ts_ns=ts_ns)
                    last_kept.pop(cam_id, None)
                    continue
                yield RawFrame(ts_ns=ts_ns, cam_id=cam_id, image_bgr=img,
                               lat=cur_lat, lon=cur_lon, ego_speed=cur_speed)
                continue

            if _is_video_schema(sname):
                cam_id = _cam_id_from_topic(channel.topic, default_cam_id)
                fmt = str(getattr(proto, "format", "")).lower()
                codec = _VIDEO_CODECS.get(fmt)
                if codec is None:
                    if fmt not in unsupported_logged:
                        log.warning("mcap.video_format_unsupported", topic=channel.topic, format=fmt)
                        unsupported_logged.add(fmt)
                    continue
                track = tracks.get(cam_id)
                if track is None:
                    track = tracks[cam_id] = _VideoTrack(codec)
                data = bytes(getattr(proto, "data", b""))
                if not data:
                    continue
                for (mts, mlat, mlon, mspeed), img in track.feed(
                        data, (ts_ns, cur_lat, cur_lon, cur_speed)):
                    if not _keep(cam_id, mts):
                        continue
                    yield RawFrame(ts_ns=mts, cam_id=cam_id, image_bgr=img,
                                   lat=mlat, lon=mlon, ego_speed=mspeed)
                continue

        # Drain any frames the decoders were still holding at end of stream.
        for cam_id, track in tracks.items():
            for (mts, mlat, mlon, mspeed), img in track.flush():
                if not _keep(cam_id, mts):
                    continue
                yield RawFrame(ts_ns=mts, cam_id=cam_id, image_bgr=img,
                               lat=mlat, lon=mlon, ego_speed=mspeed)
