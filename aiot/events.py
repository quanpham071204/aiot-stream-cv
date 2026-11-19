"""Append-only JSONL events with paged history and durable evidence lifecycle metadata."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .timeutil import iso

log = logging.getLogger(__name__)


def timestamp(value: str) -> float:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return dt.timestamp()


class EventStore:
    def __init__(self, out_dir: str, keep: int = 500, retention_days: Optional[float] = None,
                 max_evidence_mb: Optional[float] = None):
        self.dir = Path(out_dir)
        (self.dir / "evidence").mkdir(parents=True, exist_ok=True)
        self._metadata = self.dir / "evidence-metadata"
        self._metadata.mkdir(exist_ok=True)
        self.path = self.dir / "events.jsonl"
        self._lock = threading.RLock()
        self._active_clips: set[str] = set()
        for value in (retention_days, max_evidence_mb):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError("evidence retention limits must be positive or null")
        self.retention_days = retention_days
        self.max_evidence_bytes = None if max_evidence_mb is None else int(max_evidence_mb * 1024 * 1024)
        self._recent: deque[dict] = deque(self._records(), maxlen=keep)

    @staticmethod
    def new_id(source_id: str, occurred_at: float) -> str:
        safe_source = re.sub(r"[^\w.-]", "_", source_id)
        return f"{safe_source}-{time.strftime('%Y%m%dT%H%M%S', time.localtime(occurred_at))}-{uuid.uuid4().hex[:6]}"

    def _records(self) -> list[dict]:
        records = []
        if not self.path.exists():
            return records
        with self.path.open(encoding="utf-8", errors="replace") as stream:
            for number, line in enumerate(stream, 1):
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict) or not isinstance(event.get("result"), dict):
                        raise ValueError("invalid event object")
                    for key in ("event_id", "source_id", "event_type", "occurred_at", "created_at"):
                        if not isinstance(event.get(key), str):
                            raise ValueError(f"invalid {key}")
                    timestamp(event["occurred_at"])
                    timestamp(event["created_at"])
                    event.setdefault("evidence_path", None)
                    event.setdefault("evidence_clip", None)
                    records.append(event)
                except (ValueError, TypeError, OverflowError):
                    log.warning("skipping invalid event log line %s", number)
        return records

    def _meta_path(self, event_id: str) -> Path:
        return self._metadata / f"{hashlib.sha256(event_id.encode()).hexdigest()}.json"

    def _meta(self, event_id: str) -> dict:
        try:
            value = json.loads(self._meta_path(event_id).read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_meta(self, event_id: str, changes: dict) -> None:
        data = {**self._meta(event_id), **changes}
        path = self._meta_path(event_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def evidence_file(self, relative: Optional[str]) -> Optional[Path]:
        if not isinstance(relative, str) or not relative:
            return None
        root = (self.dir / "evidence").resolve()
        candidate = (self.dir / relative).resolve()
        return candidate if candidate.parent == root else None

    def _decorate(self, event: dict) -> dict:
        meta = self._meta(event["event_id"])
        result = {**event, "clip_error": meta.get("clip_error"), "clip_playable": meta.get("clip_playable")}
        for kind, field in (("image", "evidence_path"), ("clip", "evidence_clip")):
            path = self.evidence_file(event.get(field))
            state = meta.get(f"{kind}_status")
            if state in ("expired", "failed"):
                status = state
            elif kind == "clip" and event["event_id"] in self._active_clips:
                status = "pending"
            elif path is not None and path.is_file():
                status = "ready"
            elif state == "pending":
                status = "interrupted"
            elif path is not None:
                status = "missing"
            else:
                status = "none"
            result[f"{kind}_status"] = status
        return result

    def emit(self, source_id: str, event_type: str, occurred_at: float, result: dict,
             frame: Optional[np.ndarray] = None, event_id: Optional[str] = None,
             clip_name: Optional[str] = None) -> dict:
        event_id = event_id or self.new_id(source_id, occurred_at)
        evidence = None
        if frame is not None:
            rel = f"evidence/{event_id}.jpg"
            path = self.evidence_file(rel)
            try:
                if path is not None and cv2.imwrite(str(path), frame):
                    evidence = rel
                else:
                    log.error("failed to write evidence %s", rel)
            except (cv2.error, OSError):
                log.exception("failed to write evidence %s", rel)
        ev = {
            "event_id": event_id, "source_id": source_id, "event_type": event_type,
            "occurred_at": iso(occurred_at), "created_at": iso(time.time()), "result": result,
            "evidence_path": evidence, "evidence_clip": f"evidence/{clip_name}" if clip_name else None,
        }
        with self._lock:
            self._save_meta(event_id, {
                "image_status": "ready" if evidence else "failed" if frame is not None else "none",
                "clip_status": "pending" if clip_name else "none",
            })
            if clip_name:
                self._active_clips.add(event_id)
            # Separate a crash-truncated last line from the next valid event.
            with self.path.open("a+b") as stream:
                if stream.tell():
                    stream.seek(-1, 2)
                    if stream.read(1) != b"\n":
                        stream.write(b"\n")
                stream.write((json.dumps(ev, ensure_ascii=False) + "\n").encode("utf-8"))
            self._recent.append(ev)
        log.info("EVENT %s %s %s", event_type, source_id, result)
        return ev

    def update_clip(self, name: str, status: str, error: Optional[str] = None, playable: bool = False) -> None:
        with self._lock:
            event_id = Path(name).stem
            self._save_meta(event_id, {"clip_status": status, "clip_error": error, "clip_playable": playable})
            self._active_clips.discard(event_id)

    def recent(self, limit: int = 50) -> list[dict]:
        with self._lock:
            return [self._decorate(ev) for ev in list(self._recent)[-limit:][::-1]] if limit > 0 else []

    def get(self, event_id: str) -> Optional[dict]:
        with self._lock:
            event = next((event for event in self._records() if event["event_id"] == event_id), None)
            return self._decorate(event) if event else None

    def history(self, limit: Optional[int] = 20, offset: int = 0, source_id: Optional[str] = None,
                event_type: Optional[str] = None, start: Optional[float] = None, end: Optional[float] = None,
                snapshot: Optional[int] = None) -> dict:
        with self._lock:
            records = self._records()
            snapshot = len(records) if snapshot is None else min(snapshot, len(records))
            records = records[:snapshot]
            sources = sorted({ev["source_id"] for ev in records})
            types = sorted({ev["event_type"] for ev in records})
            filtered = [ev for ev in records
                        if (not source_id or ev["source_id"] == source_id)
                        and (not event_type or ev["event_type"] == event_type)
                        and (start is None or timestamp(ev["occurred_at"]) >= start)
                        and (end is None or timestamp(ev["occurred_at"]) <= end)]
            filtered.sort(key=lambda ev: (timestamp(ev["occurred_at"]), timestamp(ev["created_at"]), ev["event_id"]),
                          reverse=True)
            page = filtered[offset:] if limit is None else filtered[offset:offset + limit]
            return {"items": [self._decorate(ev) for ev in page], "total": len(filtered), "limit": limit,
                    "offset": offset, "snapshot": snapshot, "source_ids": sources, "event_types": types}

    def cleanup(self, now: Optional[float] = None) -> int:
        """Opt-in age/size retention of referenced evidence; JSONL history is never deleted."""
        if self.retention_days is None and self.max_evidence_bytes is None:
            return 0
        now = time.time() if now is None else now
        with self._lock:
            files = {}
            for ev in self._records():
                for kind, field in (("image", "evidence_path"), ("clip", "evidence_clip")):
                    path = self.evidence_file(ev.get(field))
                    if path is not None and path.is_file():
                        files.setdefault(path, []).append((ev["event_id"], kind))
            candidates = []
            total = 0
            for path, owners in files.items():
                stat = path.stat()
                total += stat.st_size
                if not any(event_id in self._active_clips for event_id, _ in owners):
                    candidates.append((stat.st_mtime, path, stat.st_size, owners))
            removed = 0
            for modified, path, size, owners in sorted(candidates):
                too_old = self.retention_days is not None and modified < now - self.retention_days * 86400
                too_large = self.max_evidence_bytes is not None and total > self.max_evidence_bytes
                if not too_old and not too_large:
                    continue
                try:
                    path.unlink()
                    for event_id, kind in owners:
                        self._save_meta(event_id, {f"{kind}_status": "expired"})
                    total -= size
                    removed += 1
                except OSError:
                    log.exception("could not expire evidence %s", path.name)
            if self.max_evidence_bytes is not None and total > self.max_evidence_bytes:
                log.warning("evidence exceeds size limit; active clips are protected from cleanup")
            return removed
