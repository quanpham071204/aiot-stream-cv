"""Small local HTTP API so the web/workflow side can pull results (stdlib only).

GET /health                  source health for every stream
GET /observations            latest tracks/counts per source
GET /zones                   effective restricted-zone configuration per source
GET /events?limit=N          most recent events (newest first)
GET /events/history          filtered, paged history
GET /events/export           complete filtered CSV/JSON attachment
GET /events/<id>             event detail with current evidence state
GET /snapshot/<source>.jpg   latest annotated frame
GET /metrics                 run statistics per source (fps, latency p50/p95, drops, events)
GET /evidence/<file>         evidence image (.jpg) or clip (.mp4) referenced by an event
"""
from __future__ import annotations

import csv
import io
import json
import logging
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

import cv2

from .events import EventStore, timestamp
from .pipeline import SourcePipeline
from .timeutil import iso

log = logging.getLogger(__name__)


def history_options(query: dict) -> dict:
    options = {}
    for key, default, minimum, maximum in (("limit", 20, 1, 200), ("offset", 0, 0, None),
                                            ("snapshot", None, 0, None)):
        raw = query.get(key, [default])[0]
        value = None if raw is None else int(raw)
        if value is not None and (value < minimum or (maximum is not None and value > maximum)):
            raise ValueError(f"{key} is out of range")
        options[key] = value
    for key in ("source_id", "event_type"):
        options[key] = query.get(key, [None])[0]
    for param, key in (("from", "start"), ("to", "end")):
        raw = query.get(param, [None])[0]
        options[key] = timestamp(raw) if raw else None
    if options["start"] is not None and options["end"] is not None and options["start"] > options["end"]:
        raise ValueError("from must not be after to")
    return options


def export_csv(events: list[dict]) -> bytes:
    output = io.StringIO(newline="")
    columns = ["event_id", "source_id", "event_type", "occurred_at", "created_at", "zone_id", "track_id",
               "class", "confidence", "dwell_s", "evidence_path", "evidence_clip", "image_status", "clip_status",
               "result"]
    writer = csv.writer(output)
    writer.writerow(columns)
    for event in events:
        row = {**event, **{key: event["result"].get(key) for key in
                          ("zone_id", "track_id", "class", "confidence", "dwell_s")},
               "result": json.dumps(event["result"], ensure_ascii=False)}
        values = []
        for column in columns:
            value = "" if row.get(column) is None else str(row[column])
            if value.lstrip().startswith(("=", "+", "-", "@")):
                value = "'" + value
            values.append(value)
        writer.writerow(values)
    return ("\ufeff" + output.getvalue()).encode("utf-8")


def make_server(host: str, port: int, pipelines: dict[str, SourcePipeline], store: EventStore) -> ThreadingHTTPServer:

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # route to logging instead of stderr
            log.debug("api: " + fmt, *args)

        def _send(self, code: int, body: bytes, ctype: str, headers: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Expose-Headers", "Content-Disposition, Content-Range, Accept-Ranges")
            self.send_header("Cache-Control", "no-store")
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):  # noqa: N802
            url = urlparse(self.path)
            path = url.path.rstrip("/")
            if path == "/health":
                return self._json({"time": iso(time.time()), "sources": [p.health() for p in pipelines.values()]})
            if path == "/observations":
                return self._json({sid: p.latest_observation or {"source_id": sid, "status": "waiting"}
                                   for sid, p in pipelines.items()})
            if path == "/zones":
                return self._json({"time": iso(time.time()), "sources": [
                    {"source_id": sid, "zones": [
                        {"zone_id": rule.zone_id, "polygon": rule.polygon,
                         "min_dwell_s": rule.min_dwell_s, "lost_timeout_s": rule.lost_timeout_s,
                         "classes": sorted(rule.classes) if rule.classes is not None else None}
                        for rule in p.rules]}
                    for sid, p in pipelines.items()]})
            if path == "/metrics":
                return self._json({"time": iso(time.time()), "sources": [p.metrics() for p in pipelines.values()]})
            if path == "/events":
                try:
                    limit = int(parse_qs(url.query).get("limit", ["50"])[0])
                except ValueError:
                    return self._json({"error": "limit must be an integer"}, 400)
                if not 1 <= limit <= 500:
                    return self._json({"error": "limit must be between 1 and 500"}, 400)
                return self._json(store.recent(limit))
            if path in ("/events/history", "/events/export"):
                try:
                    query = parse_qs(url.query)
                    options = history_options(query)
                    if path == "/events/history":
                        return self._json(store.history(**options))
                    fmt = query.get("format", ["json"])[0]
                    if fmt not in ("json", "csv"):
                        raise ValueError("format must be json or csv")
                    options.update(limit=None, offset=0)
                    events = store.history(**options)["items"]
                    body = (export_csv(events) if fmt == "csv"
                            else json.dumps(events, ensure_ascii=False, indent=2).encode())
                    return self._send(200, body, "text/csv; charset=utf-8" if fmt == "csv" else "application/json",
                                      {"Content-Disposition": f'attachment; filename="events.{fmt}"'})
                except (ValueError, TypeError, OverflowError) as exc:
                    return self._json({"error": str(exc)}, 400)
            if path.startswith("/events/"):
                event = store.get(unquote(path[len("/events/"):]))
                return self._json(event) if event else self._json({"error": "event not found"}, 404)
            if path.startswith("/snapshot/") and path.endswith(".jpg"):
                p = pipelines.get(unquote(path[len("/snapshot/"):-4]))
                if p is None or p.latest_frame is None:
                    return self._json({"error": "no frame"}, 404)
                ok, buf = cv2.imencode(".jpg", p.latest_frame)
                if not ok:
                    return self._json({"error": "encode failed"}, 500)
                return self._send(200, buf.tobytes(), "image/jpeg")
            if path.startswith("/evidence/"):
                name = unquote(path[len("/evidence/"):])
                f = store.evidence_file(f"evidence/{name}") if Path(name).name == name else None
                if f is None or not f.is_file() or f.suffix not in (".jpg", ".mp4"):
                    return self._json({"error": "not found"}, 404)
                ctype = "video/mp4" if f.suffix == ".mp4" else "image/jpeg"
                try:
                    size = f.stat().st_size
                except FileNotFoundError:
                    return self._json({"error": "evidence expired or missing"}, 404)
                headers = {"Accept-Ranges": "bytes"}
                if parse_qs(url.query).get("download") == ["1"]:
                    headers["Content-Disposition"] = f"attachment; filename*=UTF-8''{quote(f.name)}"
                start, end, code = 0, size - 1, 200
                requested = self.headers.get("Range")
                if requested:
                    match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
                    if not match or not any(match.groups()) or size == 0 or len(requested) > 64:
                        return self._send(416, b"", ctype, {"Content-Range": f"bytes */{size}"})
                    left, right = match.groups()
                    start = int(left) if left else max(0, size - int(right))
                    end = min(int(right), size - 1) if left and right else size - 1
                    if start >= size or end < start:
                        return self._send(416, b"", ctype, {"Content-Range": f"bytes */{size}"})
                    headers["Content-Range"] = f"bytes {start}-{end}/{size}"
                    code = 206
                try:
                    with f.open("rb") as stream:
                        stream.seek(start)
                        body = stream.read(end - start + 1)
                except FileNotFoundError:
                    return self._json({"error": "evidence expired or missing"}, 404)
                return self._send(code, body, ctype, headers)
            return self._json({"error": "not found",
                               "routes": ["/health", "/metrics", "/observations", "/zones", "/events",
                                          "/events/history", "/events/export", "/events/<id>",
                                          "/snapshot/<id>.jpg", "/evidence/<file>"]}, 404)

    return ThreadingHTTPServer((host, port), Handler)


def serve_in_background(server: ThreadingHTTPServer) -> threading.Thread:
    t = threading.Thread(target=server.serve_forever, name="api", daemon=True)
    t.start()
    log.info("API on http://%s:%d", *server.server_address[:2])
    return t
