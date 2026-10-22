"""Run the stream + CV pipeline from a YAML config.

    python run.py                       # config.yaml, API on :8080
    python run.py --show                # also open a preview window
    python run.py --duration 60         # stop after 60 s
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import cv2
import yaml

from aiot.api import make_server, serve_in_background
from aiot.events import EventStore
from aiot.evidence import ClipRecorder
from aiot.pipeline import SourcePipeline
from aiot.rules import ZoneIntrusionRule
from aiot.stream import StreamSource
from aiot.vision import Tracker


def build(cfg: dict) -> tuple[dict[str, SourcePipeline], EventStore]:
    store = EventStore(cfg.get("events", {}).get("dir", "output"))
    st, m, ev = cfg.get("stream", {}), cfg.get("model", {}), cfg.get("events", {})
    pipelines = {}
    for s in cfg["sources"]:
        source = StreamSource(
            s["id"], s["uri"],
            buffer_size=st.get("buffer_size", 4),
            reconnect_initial_s=st.get("reconnect_initial_s", 1.0),
            reconnect_max_s=st.get("reconnect_max_s", 30.0),
            stale_after_s=st.get("stale_after_s", 3.0),
            loop=s.get("loop", False),
            realtime=s.get("realtime", True),
        )
        tracker = Tracker(m.get("weights", "yolov8n.pt"), m.get("classes"), m.get("conf", 0.35),
                          m.get("imgsz", 640), m.get("device"))
        rules = [ZoneIntrusionRule(z["id"], z["polygon"], z.get("min_dwell_s", 1.0),
                                   z.get("lost_timeout_s", 2.0), z.get("classes"))
                 for z in s.get("zones", [])]
        clips = None
        if ev.get("clip", True):
            clips = ClipRecorder(store.dir / "evidence", ev.get("clip_pre_s", 5.0), ev.get("clip_post_s", 5.0))
        pipelines[s["id"]] = SourcePipeline(source, tracker, rules, store, clips,
                                            offline_grace_s=st.get("offline_grace_s", 10.0))
    return pipelines, store


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--show", action="store_true", help="preview window (press q to quit)")
    ap.add_argument("--duration", type=float, default=0, help="seconds to run, 0 = until Ctrl+C")
    ap.add_argument("--no-api", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("run")
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    pipelines, store = build(cfg)
    server = None
    if not args.no_api:
        api = cfg.get("api", {})
        server = make_server(api.get("host", "127.0.0.1"), api.get("port", 8080), pipelines, store)
        serve_in_background(server)
    for p in pipelines.values():
        p.start()

    t_start = last_report = time.time()
    try:
        while True:
            now = time.time()
            if args.duration and now - t_start >= args.duration:
                break
            if all(p.finished for p in pipelines.values()):
                log.info("all sources ended")
                break
            if now - last_report >= 5:
                last_report = now
                for p in pipelines.values():
                    log.info("HEALTH %s", json.dumps(p.health()))
            if args.show:
                for sid, p in pipelines.items():
                    if p.latest_frame is not None:
                        cv2.imshow(sid, p.latest_frame)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in pipelines.values():
            p.stop()
        if server:
            server.shutdown()
        cv2.destroyAllWindows()
        metrics = [p.metrics() for p in pipelines.values()]
        (store.dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        for m in metrics:
            log.info("METRICS %s", json.dumps(m))
        log.info("stopped; events in %s, metrics in %s", store.path, store.dir / "metrics.json")


if __name__ == "__main__":
    main()
