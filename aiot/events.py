"""Event + evidence store (JSONL file + JPEG snapshots).

Event fields follow the group's agreed sample: event id, source, time,
event type, result, evidence path (snapshot) + evidence clip.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .timeutil import iso

log = logging.getLogger(__name__)


class EventStore:
    def __init__(self, out_dir: str, keep: int = 500):
        self.dir = Path(out_dir)
        (self.dir / "evidence").mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "events.jsonl"
        self._recent: deque[dict] = deque(maxlen=keep)
        self._lock = threading.Lock()

    @staticmethod
    def new_id(source_id: str, occurred_at: float) -> str:
        return f"{source_id}-{time.strftime('%Y%m%dT%H%M%S', time.localtime(occurred_at))}-{uuid.uuid4().hex[:6]}"

    def emit(self, source_id: str, event_type: str, occurred_at: float, result: dict,
             frame: Optional[np.ndarray] = None, event_id: Optional[str] = None,
             clip_name: Optional[str] = None) -> dict:
        """Append one event. clip_name is a file in evidence/ that the ClipRecorder
        finishes writing post_s seconds after the event."""
        event_id = event_id or self.new_id(source_id, occurred_at)
        evidence = None
        if frame is not None:
            rel = f"evidence/{event_id}.jpg"
            if cv2.imwrite(str(self.dir / rel), frame):
                evidence = rel
            else:
                log.error("failed to write evidence %s", rel)
        ev = {
            "event_id": event_id,
            "source_id": source_id,
            "event_type": event_type,
            "occurred_at": iso(occurred_at),
            "created_at": iso(time.time()),
            "result": result,
            "evidence_path": evidence,
            "evidence_clip": f"evidence/{clip_name}" if clip_name else None,
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            self._recent.append(ev)
        log.info("EVENT %s %s %s", event_type, source_id, result)
        return ev

    def recent(self, limit: int = 50) -> list[dict]:
        with self._lock:
            return list(self._recent)[-limit:][::-1]
