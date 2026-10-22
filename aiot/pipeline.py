"""Per-source worker: stream -> tracker -> rules -> events, plus source-health events."""
from __future__ import annotations

import logging
import threading
import time
from collections import Counter, deque
from typing import Optional

import numpy as np

from .events import EventStore
from .evidence import ClipRecorder, percentile
from .rules import ZoneIntrusionRule
from .stream import State, StreamSource
from .timeutil import iso
from .vision import draw

log = logging.getLogger(__name__)


class SourcePipeline:
    def __init__(self, source: StreamSource, tracker, rules: list[ZoneIntrusionRule], store: EventStore,
                 clips: Optional[ClipRecorder] = None, offline_grace_s: float = 10.0, stats_window: int = 1000):
        self.source = source
        self.tracker = tracker
        self.rules = rules
        self.store = store
        self.clips = clips
        # a source that has never delivered a frame is reported offline after this long
        self.offline_grace_s = offline_grace_s
        self.latest_observation: Optional[dict] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.processing_fps = 0.0
        self.frames_processed = 0
        self.inference_errors = 0
        self.event_counts: Counter = Counter()
        self._infer_ms: deque[float] = deque(maxlen=stats_window)
        self._latency_ms: deque[float] = deque(maxlen=stats_window)
        self._last_done: Optional[float] = None
        self._prev_status: Optional[str] = None  # "online" / "offline" / "ended"
        self._started_at: Optional[float] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "SourcePipeline":
        self._started_at = time.time()
        self.source.start()
        self._thread = threading.Thread(target=self._run, name=f"pipe-{self.source.source_id}", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self.source.stop()
        if self._thread:
            self._thread.join(timeout=5)
        if self.clips:
            self.clips.flush()

    @property
    def finished(self) -> bool:
        return self.source.state == State.ENDED and self._thread is not None and not self._thread.is_alive()

    def health(self) -> dict:
        return {**self.source.health(), "processing_fps": round(self.processing_fps, 1)}

    def metrics(self) -> dict:
        """Run statistics for the report: throughput, latency percentiles, drops, events."""
        def stats(values):
            v = list(values)
            return {"p50": _r(percentile(v, 50)), "p95": _r(percentile(v, 95)), "max": _r(max(v) if v else None)}

        h = self.source.health()
        received = h["frames_received"]
        return {
            "source_id": self.source.source_id,
            "uptime_s": round(time.time() - self._started_at, 1) if self._started_at else 0.0,
            "frames_received": received,
            "frames_processed": self.frames_processed,
            "frames_dropped": h["frames_dropped"],
            "drop_rate": round(h["frames_dropped"] / received, 3) if received else None,
            "input_fps": h["fps"],
            "processing_fps": round(self.processing_fps, 1),
            "infer_ms": stats(self._infer_ms),
            "latency_ms": stats(self._latency_ms),
            "reconnects": h["reconnects"],
            "inference_errors": self.inference_errors,
            "events": dict(self.event_counts),
            "clips_written": self.clips.clips_written if self.clips else 0,
            "clips_failed": self.clips.clips_failed if self.clips else 0,
        }

    def _emit(self, event_type: str, t: float, result: dict, frame: Optional[np.ndarray] = None) -> dict:
        event_id = EventStore.new_id(self.source.source_id, t)
        clip = None
        if self.clips is not None and frame is not None:
            clip = self.clips.start_clip(event_id, t)
        ev = self.store.emit(self.source.source_id, event_type, t, result, frame, event_id=event_id, clip_name=clip)
        self.event_counts[event_type] += 1
        return ev

    def _run(self) -> None:
        sid = self.source.source_id
        while not self._stop.is_set():
            self._check_health()
            pkt = self.source.read(timeout=0.5)
            if pkt is None:
                if self.source.state == State.ENDED:
                    self._check_health()
                    break
                continue
            t0 = time.time()
            try:
                tracks = self.tracker.update(pkt.frame)
            except Exception as e:  # noqa: BLE001 - report, never treat as "nothing detected"
                log.exception("[%s] inference failed", sid)
                self.inference_errors += 1
                self.latest_observation = {"source_id": sid, "status": "error", "error": str(e),
                                           "measured_at": iso(pkt.measured_at)}
                continue
            infer_ms = (time.time() - t0) * 1000

            fired = [(rule, ev) for rule in self.rules for ev in rule.update(tracks, pkt.measured_at)]
            alert_ids = set().union(*(r.inside_ids for r in self.rules)) if self.rules else set()
            annotated = draw(pkt.frame, tracks, [(r.zone_id, r.polygon) for r in self.rules], alert_ids)
            if self.clips is not None:
                self.clips.push(annotated, pkt.measured_at)
            for rule, payload in fired:
                self._emit(rule.event_type, pkt.measured_at, payload, annotated)

            done = time.time()
            if self._last_done is not None and done > self._last_done:
                inst = 1.0 / (done - self._last_done)
                self.processing_fps = 0.9 * self.processing_fps + 0.1 * inst if self.processing_fps else inst
            self._last_done = done
            latency_ms = (done - pkt.received_at) * 1000
            self.frames_processed += 1
            self._infer_ms.append(infer_ms)
            self._latency_ms.append(latency_ms)
            self.latest_frame = annotated
            self.latest_observation = {
                "source_id": sid,
                "status": "ok",
                "seq": pkt.seq,
                "measured_at": iso(pkt.measured_at),
                "counts": dict(Counter(t.cls_name for t in tracks)),
                "tracks": [
                    {"track_id": t.track_id, "class": t.cls_name, "confidence": round(t.conf, 3),
                     "bbox": [round(v, 1) for v in t.bbox],
                     "zones": [r.zone_id for r in self.rules if t.track_id in r.inside_ids]}
                    for t in tracks
                ],
                "infer_ms": round(infer_ms, 1),
                "latency_ms": round(latency_ms, 1),
            }

    def _check_health(self) -> None:
        h = self.source.health()
        if self.source.state == State.ENDED:
            status = "ended"
        elif h["healthy"]:
            status = "online"
        elif self._prev_status is None:
            # never seen a frame yet: only call it offline once the grace period is over,
            # so a slow-starting camera does not raise a false alarm
            if self._started_at is None or time.time() - self._started_at < self.offline_grace_s:
                return
            status = "offline"
        else:
            status = "offline"
        prev, self._prev_status = self._prev_status, status
        if status == prev or (prev is None and status == "online"):
            return
        event_type = {"online": "source_recovered", "offline": "source_offline", "ended": "source_ended"}[status]
        detail = {k: h[k] for k in ("state", "last_frame_age_s", "reconnects", "last_error")}
        self._emit(event_type, time.time(), detail)


def _r(v: Optional[float]) -> Optional[float]:
    return None if v is None else round(v, 1)
