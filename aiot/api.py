"""Small local HTTP API so the web/workflow side can pull results (stdlib only).

GET /health                  source health for every stream
GET /observations            latest tracks/counts per source
GET /events?limit=N          most recent events (newest first)
GET /snapshot/<source>.jpg   latest annotated frame
GET /metrics                 run statistics per source (fps, latency p50/p95, drops, events)
GET /evidence/<file>         evidence image (.jpg) or clip (.mp4) referenced by an event
"""
from __future__ import annotations

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

from .events import EventStore
from .pipeline import SourcePipeline
from .timeutil import iso

log = logging.getLogger(__name__)


def make_server(host: str, port: int, pipelines: dict[str, SourcePipeline], store: EventStore) -> ThreadingHTTPServer:
    evidence_dir = (store.dir / "evidence").resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # route to logging instead of stderr
            log.debug("api: " + fmt, *args)

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
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
                return self._json({sid: p.latest_observation for sid, p in pipelines.items()})
            if path == "/metrics":
                return self._json({"time": iso(time.time()), "sources": [p.metrics() for p in pipelines.values()]})
            if path == "/events":
                try:
                    limit = int(parse_qs(url.query).get("limit", ["50"])[0])
                except ValueError:
                    return self._json({"error": "limit must be an integer"}, 400)
                return self._json(store.recent(limit))
            if path.startswith("/snapshot/") and path.endswith(".jpg"):
                p = pipelines.get(path[len("/snapshot/"):-4])
                if p is None or p.latest_frame is None:
                    return self._json({"error": "no frame"}, 404)
                ok, buf = cv2.imencode(".jpg", p.latest_frame)
                if not ok:
                    return self._json({"error": "encode failed"}, 500)
                return self._send(200, buf.tobytes(), "image/jpeg")
            if path.startswith("/evidence/"):
                f = (evidence_dir / Path(path).name).resolve()
                if f.parent != evidence_dir or not f.is_file():
                    return self._json({"error": "not found"}, 404)
                ctype = "video/mp4" if f.suffix == ".mp4" else "image/jpeg"
                return self._send(200, f.read_bytes(), ctype)
            return self._json({"error": "not found",
                               "routes": ["/health", "/metrics", "/observations", "/events",
                                          "/snapshot/<id>.jpg", "/evidence/<file>"]}, 404)

    return ThreadingHTTPServer((host, port), Handler)


def serve_in_background(server: ThreadingHTTPServer) -> threading.Thread:
    t = threading.Thread(target=server.serve_forever, name="api", daemon=True)
    t.start()
    log.info("API on http://%s:%d", *server.server_address[:2])
    return t
