"""Data streaming block: capture a camera/video stream, timestamp and buffer
frames, reconnect on failure and expose source health.

Works with webcams (uri "0"), RTSP/HTTP streams and video files. A video file
with realtime=True is paced at its native FPS so it behaves like a live camera.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


class State:
    CONNECTING = "connecting"
    RUNNING = "running"
    RECONNECTING = "reconnecting"
    ENDED = "ended"
    STOPPED = "stopped"


@dataclass
class FramePacket:
    source_id: str
    seq: int
    measured_at: float  # capture time (epoch s); cameras give no trusted clock, so this is grab time
    received_at: float  # time the ingest thread pushed it to the buffer
    frame: np.ndarray
    quality: str = "ok"


def default_open(uri):
    """Open a capture; network streams get open/read timeouts so a dead camera can't hang the thread."""
    if isinstance(uri, str) and "://" in uri:
        return cv2.VideoCapture(uri, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000,
        ])
    return cv2.VideoCapture(uri)


class StreamSource:
    def __init__(
        self,
        source_id: str,
        uri,
        *,
        buffer_size: int = 4,
        reconnect_initial_s: float = 1.0,
        reconnect_max_s: float = 30.0,
        stale_after_s: float = 3.0,
        loop: bool = False,
        realtime: bool = True,
        open_fn: Optional[Callable] = None,
    ):
        self.source_id = source_id
        uri = str(uri)
        self.uri = int(uri) if uri.isdigit() else uri
        self.is_file = isinstance(self.uri, str) and "://" not in self.uri
        self.reconnect_initial_s = reconnect_initial_s
        self.reconnect_max_s = reconnect_max_s
        self.stale_after_s = stale_after_s
        self.loop = loop
        self.realtime = realtime
        self._open_fn = open_fn or default_open

        self._buf: deque[FramePacket] = deque(maxlen=max(1, buffer_size))
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.state = State.CONNECTING
        self.frames_received = 0
        self.frames_dropped = 0
        self.reconnects = 0
        self.last_frame_at: Optional[float] = None
        self.last_error: Optional[str] = None
        self._fps_ema: Optional[float] = None
        self._seq = 0

    # ---- public API -------------------------------------------------------
    def start(self) -> "StreamSource":
        self._thread = threading.Thread(target=self._run, name=f"stream-{self.source_id}", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread:
            self._thread.join(timeout=5)
        self.state = State.STOPPED

    def read(self, timeout: float = 1.0) -> Optional[FramePacket]:
        """Oldest buffered frame, or None if nothing arrived within timeout."""
        with self._cond:
            if not self._buf:
                self._cond.wait(timeout)
            return self._buf.popleft() if self._buf else None

    def health(self) -> dict:
        now = time.time()
        age = None if self.last_frame_at is None else now - self.last_frame_at
        healthy = self.state == State.RUNNING and age is not None and age <= self.stale_after_s
        return {
            "source_id": self.source_id,
            "uri": str(self.uri),
            "state": self.state,
            "healthy": healthy,
            "fps": round(self._fps_ema, 1) if self._fps_ema else 0.0,
            "frames_received": self.frames_received,
            "frames_dropped": self.frames_dropped,
            "buffered": len(self._buf),
            "reconnects": self.reconnects,
            "last_frame_age_s": round(age, 2) if age is not None else None,
            "last_error": self.last_error,
        }

    # ---- ingest thread ----------------------------------------------------
    def _run(self) -> None:
        backoff = self.reconnect_initial_s
        while not self._stop.is_set():
            cap = self._open()
            if cap is not None:
                self.state = State.RUNNING
                ended, got_frames = self._pump(cap)
                cap.release()
                if self._stop.is_set():
                    break
                if got_frames:
                    backoff = self.reconnect_initial_s
                if ended:
                    if self.loop:
                        continue
                    self.state = State.ENDED
                    with self._cond:
                        self._cond.notify_all()
                    log.info("[%s] end of file", self.source_id)
                    return
            self.state = State.RECONNECTING
            self.reconnects += 1
            log.warning("[%s] %s; reconnect #%d in %.1fs", self.source_id, self.last_error, self.reconnects, backoff)
            self._stop.wait(backoff)
            backoff = min(backoff * 2, self.reconnect_max_s)

    def _open(self):
        try:
            cap = self._open_fn(self.uri)
        except Exception as e:  # noqa: BLE001 - any backend error means "not connected"
            self.last_error = f"open error: {e}"
            return None
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            self.last_error = "open failed"
            return None
        return cap

    def _pump(self, cap) -> tuple[bool, bool]:
        """Read until failure. Returns (end_of_file, received_any_frame)."""
        period = 0.0
        if self.is_file and self.realtime:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            period = 1.0 / fps if fps > 0 else 0.0
        t0 = time.time()
        n = 0
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                if self.is_file:
                    return True, n > 0
                self.last_error = "read failed"
                return False, n > 0
            if period:
                wait = t0 + n * period - time.time()
                if wait > 0:
                    self._stop.wait(wait)
            n += 1
            self._push(frame, time.time())
        return False, n > 0

    def _push(self, frame: np.ndarray, now: float) -> None:
        if self.last_frame_at is not None:
            dt = now - self.last_frame_at
            if dt > 0:
                inst = 1.0 / dt
                self._fps_ema = inst if self._fps_ema is None else 0.9 * self._fps_ema + 0.1 * inst
        self.last_frame_at = now
        self.frames_received += 1
        self._seq += 1
        pkt = FramePacket(self.source_id, self._seq, now, now, frame)
        with self._cond:
            if len(self._buf) == self._buf.maxlen:
                self.frames_dropped += 1  # latest-frame policy: drop oldest when consumer is slow
            self._buf.append(pkt)
            self._cond.notify()
