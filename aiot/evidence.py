"""Evidence clips: keep the last few seconds of annotated frames in a ring buffer
and, when an event fires, write a short video covering pre_s before and post_s
after the event.

Frames are kept JPEG-encoded so a 10 s buffer stays small (~50 KB/frame instead
of ~1.3 MB raw at 768x576).
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class _PendingClip:
    path: Path
    until: float
    frames: list = field(default_factory=list)  # [(t, jpeg_bytes)]


class ClipRecorder:
    """One instance per source. Call push() for every processed frame."""

    def __init__(self, out_dir: Path, pre_s: float = 5.0, post_s: float = 5.0,
                 max_buffer_frames: int = 600, jpeg_quality: int = 80,
                 on_status: Optional[Callable[[str, str, Optional[str], bool], None]] = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.pre_s = pre_s
        self.post_s = post_s
        self.jpeg_quality = jpeg_quality
        self.on_status = on_status
        self._ring: deque[tuple[float, bytes]] = deque(maxlen=max_buffer_frames)
        self._pending: list[_PendingClip] = []
        self._writers: list[threading.Thread] = []
        self.clips_written = 0
        self.clips_failed = 0

    def push(self, frame: np.ndarray, t: float) -> None:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return
        item = (t, buf.tobytes())
        self._ring.append(item)
        while self._ring and t - self._ring[0][0] > self.pre_s:
            self._ring.popleft()
        for clip in self._pending:
            clip.frames.append(item)
        self.finish_due(t)

    def finish_due(self, t: float) -> None:
        """Finish the post-event window even if a camera stops supplying frames."""
        for clip in [clip for clip in self._pending if t >= clip.until]:
            self._pending.remove(clip)
            self._write_async(clip)

    def start_clip(self, name: str, t: float) -> str:
        """Begin a clip around time t; returns its file name (written once post_s has passed)."""
        path = self.out_dir / f"{name}.mp4"
        self._pending.append(_PendingClip(path, t + self.post_s, list(self._ring)))
        return path.name

    def flush(self) -> None:
        """Write unfinished clips with whatever frames they have (call on shutdown)."""
        for clip in self._pending:
            self._write_async(clip)
        self._pending.clear()
        for th in self._writers:
            th.join(timeout=30)
        self._writers = [th for th in self._writers if th.is_alive()]

    def _write_async(self, clip: _PendingClip) -> None:
        th = threading.Thread(target=self._write, args=(clip,), name=f"clip-{clip.path.stem}", daemon=True)
        th.start()
        self._writers = [w for w in self._writers if w.is_alive()] + [th]

    def _write(self, clip: _PendingClip) -> None:
        raw = clip.path.with_suffix(".recording.mp4")
        converted = clip.path.with_suffix(".h264.mp4")
        status, error, playable = "failed", None, False
        try:
            if not clip.frames:
                raise ValueError("clip has no frames")
            span = clip.frames[-1][0] - clip.frames[0][0]
            fps = (len(clip.frames) - 1) / span if span > 0 else 10.0
            first = cv2.imdecode(np.frombuffer(clip.frames[0][1], np.uint8), cv2.IMREAD_COLOR)
            if first is None:
                raise ValueError("cannot decode clip frame")
            h, w = first.shape[:2]
            h, w = h - h % 2, w - w % 2
            fps = min(max(1.0, fps), 60.0)
            writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if not writer.isOpened():
                writer.release()
                raise RuntimeError("cannot open video writer")
            written = 0
            try:
                for _, jpg in clip.frames:
                    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        writer.write(img[:h, :w])
                        written += 1
            finally:
                writer.release()
            if not written or not raw.is_file() or raw.stat().st_size == 0:
                raise RuntimeError("video writer produced no frames")
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg:
                try:
                    subprocess.run([ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-i", str(raw),
                                    "-an", "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p",
                                    "-movflags", "+faststart", str(converted)],
                                   check=True, capture_output=True, timeout=120)
                    if not converted.is_file() or converted.stat().st_size == 0:
                        raise RuntimeError("H.264 conversion produced no file")
                    converted.replace(clip.path)
                    playable = True
                except (OSError, subprocess.SubprocessError, RuntimeError):
                    error = "H.264 conversion failed; original MPEG-4 clip is available for download."
                    raw.replace(clip.path)
            else:
                error = "FFmpeg is unavailable; MPEG-4 clip may require downloading to play."
                raw.replace(clip.path)
            status = "ready"
            self.clips_written += 1
            log.info("clip written %s (%d frames, %.1f s)", clip.path.name, written, span)
        except Exception as exc:  # report encoder and disk errors to the event lifecycle
            error = str(exc)
            self.clips_failed += 1
            log.exception("failed to write clip %s", clip.path.name)
        finally:
            for path in (raw, converted):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    log.warning("could not remove temporary clip %s", path)
            if self.on_status:
                try:
                    self.on_status(clip.path.name, status, error, playable)
                except Exception:
                    log.exception("could not persist clip status %s", clip.path.name)


def percentile(values, p: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)
