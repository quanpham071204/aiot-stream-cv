"""Run the (slow) YOLO tracker off the main loop so it never stalls the light OpenCV analyzers.

``AsyncTracker`` has the same ``update(frame) -> list[Track]`` method as ``Tracker``,
so ``SourcePipeline`` does not change: each call hands the newest frame to a worker
thread (older pending frames are replaced, never queued) and immediately returns the
most recent finished result. ``NullTracker`` is used when object detection is off.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

from .vision import Track

log = logging.getLogger(__name__)


class NullTracker:
    def update(self, frame: np.ndarray) -> list[Track]:
        return []


class AsyncTracker:
    def __init__(self, tracker, max_age_s: float = 5.0):
        self.tracker = tracker
        self.max_age_s = max_age_s  # results older than this are dropped (worker stuck or failing)
        self._cv = threading.Condition()
        self._pending: Optional[np.ndarray] = None
        self._tracks: list[Track] = []
        self._result_at: Optional[float] = None
        self.infer_ms: Optional[float] = None
        self.runs = 0
        self.errors = 0
        self.last_error: Optional[str] = None
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="async-tracker", daemon=True)
        self._thread.start()

    def update(self, frame: np.ndarray) -> list[Track]:
        with self._cv:
            self._pending = frame
            self._cv.notify()
            if self._result_at is None or time.time() - self._result_at > self.max_age_s:
                return []
            return list(self._tracks)

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._pending is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                frame, self._pending = self._pending, None
            t0 = time.time()
            try:
                tracks = self.tracker.update(frame)
            except Exception as e:  # noqa: BLE001 - keep the worker alive, surface the error
                log.exception("async tracker failed")
                with self._cv:
                    self.errors += 1
                    self.last_error = str(e)
                continue
            with self._cv:
                self._tracks = tracks
                self._result_at = time.time()
                self.infer_ms = (self._result_at - t0) * 1000
                self.runs += 1
