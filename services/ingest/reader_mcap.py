"""MCAP reader. MCAP is the native container (Principle 07): sensor streams in one indexed,
timestamped file. Camera images plus GNSS/IMU/CAN side channels are demuxed and each frame is
mapped to UTC nanoseconds from the message log time.

Three camera encodings are handled, dispatched on the schemas present in the file:

  1. Still images in the file   - CompressedImage (JPEG/PNG via cv2.imdecode). Protobuf (Foxglove)
     and ROS 2 (sensor_msgs/CompressedImage, CDR) are both decoded.
  2. An encoded video stream in the file - CompressedVideo (H.264/H.265/AV1/VP9 via PyAV), with
     decoder state held across access units since delta frames reference earlier ones.
  3. Camera metadata in the file + the video in a sibling file - the BluRabbit DriveLogger format
     writes blurabbit.CameraFrameMeta (timestamps, frame_id, video_uri) on the camera topic and the
     pixels in an adjacent trip.mp4. Frames are decoded from that MP4 and paired to the meta by
     frame_id, with GNSS/IMU forward-filled from the same file.

Messages are decoded generically from the schema embedded in the file, so this works with the
well-known schemas without compile-time stubs.
"""

from __future__ import annotations

import bisect
from collections import deque
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from core.logging import get_logger
from services.ingest.types import RawFrame

log = get_logger(__name__)

# CompressedVideo.format / MP4 codec -> the FFmpeg/PyAV decoder name. "h265" is the Foxglove
# spelling; the codec is "hevc". Anything not here is unsupported and logged once.
_VIDEO_CODECS = {
    "h264": "h264", "avc": "h264", "avc1": "h264",
    "h265": "hevc", "hevc": "hevc",
    "av1": "av1", "vp9": "vp9", "vp8": "vp8",
}
# Topic segments that name the medium, not the camera, so they fall back to the rig default id.
_GENERIC_CAM_SEGMENTS = {
    "video", "image", "image_raw", "image_color", "compressed", "compressedimage",
    "compressedvideo", "color", "rgb", "camera", "cam", "front",
}


def _decoder_factories() -> list:
    """Protobuf always; ROS 2 (CDR) when its support package is installed."""
    from mcap_protobuf.decoder import DecoderFactory as ProtobufFactory

    factories: list = [ProtobufFactory()]
    try:
        from mcap_ros2.decoder import DecoderFactory as Ros2Factory

        factories.append(Ros2Factory())
    except ImportError:
        log.debug("mcap.ros2_decoder_absent")  # CDR-encoded bags will be skipped until installed
    return factories


def _decode_image(msg) -> np.ndarray | None:
    data = getattr(msg, "data", None)
    if not data:
        return None
    buf = np.frombuffer(bytes(data), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _is_image_schema(name: str) -> bool:
    return "CompressedImage" in name or name.endswith(".Image") or name.endswith("/Image")


def _is_video_schema(name: str) -> bool:
    return "CompressedVideo" in name


def _is_frame_meta_schema(name: str) -> bool:
    # Camera metadata whose pixels live in a sibling video file (BluRabbit DriveLogger).
    return "CameraFrameMeta" in name


def _is_gnss_schema(name: str) -> bool:
    return "LocationFix" in name or "NavSatFix" in name


def _cam_id_from_topic(topic: str, default: str) -> str:
    """Pick the camera id from a topic, skipping medium words: /camera/video -> default,
    /cam_l/compressed -> cam_l, /sensors/front/image_raw -> default (front is generic here)."""
    segs = [s for s in topic.strip("/").split("/") if s]
    for s in reversed(segs):
        if s.lower() not in _GENERIC_CAM_SEGMENTS:
            return s
    return default


class _VideoTrack:
    """Holds one camera's video-stream decoder. Each CompressedVideo message is one access unit;
    feeding them in order keeps inter-frame prediction state intact. Output order equals input
    order for the low-latency (no B-frame) streams these cameras produce, so side-channel state
    and timestamps are paired to frames through a FIFO."""

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


def _peek_schema_names(mcap_path: Path) -> set[str]:
    from mcap.reader import make_reader

    with open(mcap_path, "rb") as fh:
        summary = make_reader(fh).get_summary()
    if not summary or not summary.schemas:
        return set()
    return {sc.name for sc in summary.schemas.values()}


def read_mcap(
    mcap_path: str | Path,
    target_fps: float,
    default_cam_id: str = "cam_f",
) -> Iterator[RawFrame]:
    mcap_path = Path(mcap_path)
    if not mcap_path.exists():
        raise FileNotFoundError(mcap_path)

    # Pick the path from what the file actually contains. The external-video (DriveLogger) format
    # carries only camera metadata, so it must be detected before the in-file decoders run.
    if any(_is_frame_meta_schema(n) for n in _peek_schema_names(mcap_path)):
        yield from _read_external_video(mcap_path, target_fps, default_cam_id)
    else:
        yield from _read_in_file(mcap_path, target_fps, default_cam_id)


def _read_in_file(mcap_path: Path, target_fps: float, default_cam_id: str) -> Iterator[RawFrame]:
    """Cameras whose pixels are in the MCAP: CompressedImage (still) or CompressedVideo (stream)."""
    from mcap.reader import make_reader

    interval_ns = int(1e9 / target_fps) if target_fps > 0 else 0
    last_kept: dict[str, int] = {}
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
        reader = make_reader(fh, decoder_factories=_decoder_factories())
        for schema, channel, message, proto in reader.iter_decoded_messages():
            ts_ns = int(message.log_time)
            sname = schema.name if schema else ""

            if _is_gnss_schema(sname):
                cur_lat = getattr(proto, "latitude", cur_lat)
                cur_lon = getattr(proto, "longitude", cur_lon)
                continue

            if "speed" in channel.topic.lower():
                for attr in ("ego_speed", "speed", "value"):
                    if hasattr(proto, attr):
                        cur_speed = float(getattr(proto, attr))
                        break
                continue

            if _is_image_schema(sname):
                cam_id = _cam_id_from_topic(channel.topic, default_cam_id)
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
                for (mts, mlat, mlon, mspd), img in track.feed(data, (ts_ns, cur_lat, cur_lon, cur_speed)):
                    if not _keep(cam_id, mts):
                        continue
                    yield RawFrame(ts_ns=mts, cam_id=cam_id, image_bgr=img,
                                   lat=mlat, lon=mlon, ego_speed=mspd)
                continue

        for cam_id, track in tracks.items():  # drain decoder buffers at end of stream
            for (mts, mlat, mlon, mspd), img in track.flush():
                if not _keep(cam_id, mts):
                    continue
                yield RawFrame(ts_ns=mts, cam_id=cam_id, image_bgr=img,
                               lat=mlat, lon=mlon, ego_speed=mspd)


def _read_external_video(mcap_path: Path, target_fps: float, default_cam_id: str) -> Iterator[RawFrame]:
    """BluRabbit DriveLogger: camera metadata in the MCAP, pixels in a sibling MP4. Pair each
    decoded MP4 frame to its CameraFrameMeta by frame_id, and forward-fill GNSS by timestamp."""
    import av
    from mcap.reader import make_reader

    # One pass over the metadata: frame_id -> (utc_ns, cam_id), the sibling video name, GNSS fixes.
    frame_ts: dict[int, int] = {}
    frame_cam: dict[int, str] = {}
    gnss: list[tuple[int, float, float]] = []
    video_uri: str | None = None
    with open(mcap_path, "rb") as fh:
        reader = make_reader(fh, decoder_factories=_decoder_factories())
        for schema, _channel, message, proto in reader.iter_decoded_messages():
            sname = schema.name if schema else ""
            if _is_frame_meta_schema(sname):
                fid = int(getattr(proto, "frame_id", -1))
                if fid < 0:
                    continue
                frame_ts[fid] = int(message.log_time)  # log_time is real UTC ns; unified_ns is a boot clock
                frame_cam[fid] = str(getattr(proto, "camera_id", "") or "") or default_cam_id
                video_uri = video_uri or getattr(proto, "video_uri", None)
            elif _is_gnss_schema(sname):
                lat, lon = getattr(proto, "latitude", None), getattr(proto, "longitude", None)
                if lat is not None and lon is not None:
                    gnss.append((int(message.log_time), float(lat), float(lon)))

    if not frame_ts:
        raise RuntimeError("camera metadata present but no frame records were decoded")
    if not video_uri:
        raise RuntimeError("camera metadata has no video_uri to locate the sibling video")
    video_path = (mcap_path.parent / video_uri).resolve()
    if not video_path.exists():
        raise FileNotFoundError(
            f"BluRabbit capture references {video_uri!r} but it is not next to the .mcap "
            f"(expected {video_path}); ingest the capture directory so the video travels with it")

    gnss.sort(key=lambda g: g[0])
    gnss_ts = [g[0] for g in gnss]

    def _gnss_at(ts_ns: int) -> tuple[float | None, float | None]:
        if not gnss:
            return None, None
        i = bisect.bisect_right(gnss_ts, ts_ns) - 1
        return (gnss[i][1], gnss[i][2]) if i >= 0 else (None, None)

    interval_ns = int(1e9 / target_fps) if target_fps > 0 else 0
    last_kept: dict[str, int] = {}
    log.info("mcap.external_video", video=video_path.name, frames=len(frame_ts), gnss=len(gnss))

    with av.open(str(video_path)) as container:
        # container.decode yields frames in presentation order, matching the 0..N-1 frame_id index.
        for idx, frame in enumerate(container.decode(video=0)):
            ts_ns = frame_ts.get(idx)
            if ts_ns is None:  # MP4 can hold one trailing frame with no metadata record
                continue
            cam_id = frame_cam.get(idx, default_cam_id)
            if interval_ns and (ts_ns - last_kept.get(cam_id, -interval_ns)) < interval_ns:
                continue
            last_kept[cam_id] = ts_ns
            lat, lon = _gnss_at(ts_ns)
            yield RawFrame(ts_ns=ts_ns, cam_id=cam_id, image_bgr=frame.to_ndarray(format="bgr24"),
                           lat=lat, lon=lon, ego_speed=None)
