"""SourcePipeline end to end with fake sources: health events, evidence clips, metrics."""
import json
import time

import cv2
import numpy as np
from fakes import FakeCapture, wait_until

from aiot.events import EventStore
from aiot.evidence import ClipRecorder, percentile
from aiot.pipeline import SourcePipeline
from aiot.rules import ZoneIntrusionRule
from aiot.stream import StreamSource
from aiot.vision import Track

ZONE = [[0, 0], [100, 0], [100, 100], [0, 100]]


class BigFakeCapture(FakeCapture):
    """FakeCapture with frames large enough for the video encoder."""

    def read(self):
        ok, _ = super().read()
        return (True, np.zeros((48, 64, 3), np.uint8)) if ok else (False, None)


class FakeTracker:
    """Always reports one person standing inside ZONE."""

    def update(self, frame):
        return [Track(1, "person", 0.9, (45, 30, 55, 50))]


def events(store):
    if not store.path.exists():
        return []
    return [json.loads(line) for line in store.path.read_text(encoding="utf-8").splitlines()]


def test_offline_from_start_is_reported(tmp_path):
    src = StreamSource("dead", "rtsp://fake", reconnect_initial_s=0.01, reconnect_max_s=0.05,
                       open_fn=lambda _uri: FakeCapture(0, opened=False))
    store = EventStore(str(tmp_path))
    p = SourcePipeline(src, FakeTracker(), [], store, offline_grace_s=0.2).start()
    try:
        assert wait_until(lambda: any(e["event_type"] == "source_offline" for e in events(store)))
        time.sleep(0.3)
        assert [e["event_type"] for e in events(store)] == ["source_offline"]  # reported once, not every check
    finally:
        p.stop()


def test_no_false_offline_during_grace(tmp_path):
    src = StreamSource("slow", "rtsp://fake", reconnect_initial_s=0.01, reconnect_max_s=0.05,
                       open_fn=lambda _uri: FakeCapture(0, opened=False))
    store = EventStore(str(tmp_path))
    p = SourcePipeline(src, FakeTracker(), [], store, offline_grace_s=5.0).start()
    try:
        time.sleep(0.3)
        assert events(store) == []
    finally:
        p.stop()


def test_offline_then_recovered(tmp_path):
    attempts = []

    def open_fn(_uri):
        attempts.append(1)
        return FakeCapture(10_000, opened=len(attempts) > 20)

    src = StreamSource("cam", "rtsp://fake", reconnect_initial_s=0.02, reconnect_max_s=0.02, open_fn=open_fn)
    store = EventStore(str(tmp_path))
    p = SourcePipeline(src, FakeTracker(), [], store, offline_grace_s=0.1).start()
    try:
        assert wait_until(lambda: "source_recovered" in [e["event_type"] for e in events(store)])
        assert [e["event_type"] for e in events(store)][:2] == ["source_offline", "source_recovered"]
    finally:
        p.stop()


def test_zone_event_has_snapshot_clip_and_metrics(tmp_path):
    src = StreamSource("cam", "rtsp://fake", buffer_size=100, reconnect_initial_s=0.01,
                       open_fn=lambda _uri: BigFakeCapture(50))
    store = EventStore(str(tmp_path))
    clips = ClipRecorder(tmp_path / "evidence", pre_s=1.0, post_s=0.05)
    rule = ZoneIntrusionRule("z", ZONE, min_dwell_s=0.0)
    p = SourcePipeline(src, FakeTracker(), [rule], store, clips).start()
    try:
        assert wait_until(lambda: any(e["event_type"] == "zone_intrusion" for e in events(store)))
        assert wait_until(lambda: clips.clips_written >= 1)
    finally:
        p.stop()
    ev = next(e for e in events(store) if e["event_type"] == "zone_intrusion")
    assert (tmp_path / ev["evidence_path"]).is_file()
    clip = tmp_path / ev["evidence_clip"]
    assert clip.is_file() and clip.stem == ev["event_id"]
    m = p.metrics()
    assert m["frames_processed"] > 0 and m["events"]["zone_intrusion"] == 1
    assert m["latency_ms"]["p50"] is not None and m["clips_written"] >= 1


def test_clip_contains_frames_before_and_after(tmp_path):
    rec = ClipRecorder(tmp_path, pre_s=1.0, post_s=0.5)
    frame = np.zeros((48, 64, 3), np.uint8)
    for i in range(30):  # 10 fps, 3 s
        rec.push(frame, i * 0.1)
    name = rec.start_clip("ev1", 2.9)
    for i in range(30, 40):
        rec.push(frame, i * 0.1)
    rec.flush()
    cap = cv2.VideoCapture(str(tmp_path / name))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert rec.clips_written == 1
    assert 14 <= n <= 17  # ~1 s before + ~0.5 s after at 10 fps


def test_percentile():
    assert percentile([], 50) is None
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 95) == 5
