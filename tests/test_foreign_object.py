"""foreign_object: a zone over the tank surface whose events are named foreign_object (no camera or model)."""
import json
import time

import numpy as np
import pytest
import yaml

from aiot.events import EventStore
from aiot.pipeline import SourcePipeline
from aiot.rules import ZoneIntrusionRule
from aiot.stream import StreamSource
from aiot.vision import Track

TANK = [[0, 0], [100, 0], [100, 100], [0, 100]]


class Capture:
    def __init__(self, frames):
        self.left = frames

    def isOpened(self):  # noqa: N802 - mirrors cv2 API
        return True

    def read(self):
        if self.left <= 0:
            return False, None
        self.left -= 1
        return True, np.zeros((48, 64, 3), np.uint8)

    def get(self, _prop):
        return 0

    def release(self):
        pass


class FloatingBottle:
    """One bottle on the tank surface and one person walking by outside it."""

    def update(self, frame):
        return [Track(4, "bottle", 0.7, (40, 40, 50, 60)), Track(9, "person", 0.9, (200, 10, 230, 90))]


def wait_until(cond, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def read_events(store):
    if not store.path.exists():
        return []
    return [json.loads(line) for line in store.path.read_text(encoding="utf-8").splitlines()]


def test_tank_zone_emits_foreign_object_once_with_the_usual_payload(tmp_path):
    src = StreamSource("tank01", "rtsp://fake", buffer_size=100, reconnect_initial_s=0.01,
                       open_fn=lambda _uri: Capture(40))
    store = EventStore(str(tmp_path))
    rule = ZoneIntrusionRule("tank_surface", TANK, min_dwell_s=0.0, event_type="foreign_object")
    p = SourcePipeline(src, FloatingBottle(), [rule], store).start()
    try:
        assert wait_until(lambda: any(e["event_type"] == "foreign_object" for e in read_events(store)))
        assert wait_until(lambda: p.metrics()["frames_processed"] >= 20)
    finally:
        p.stop()
    evs = [e for e in read_events(store) if e["event_type"] == "foreign_object"]
    assert len(evs) == 1                                             # once per object visit, not per frame
    r = evs[0]["result"]
    assert r["zone_id"] == "tank_surface" and r["class"] == "bottle" and r["track_id"] == 4
    assert set(r) == {"zone_id", "track_id", "class", "confidence", "dwell_s", "bbox"}
    assert not any(e["event_type"] == "zone_intrusion" for e in read_events(store))
    obs = p.latest_observation
    assert [t["zones"] for t in obs["tracks"] if t["track_id"] == 4] == [["tank_surface"]]
    assert [t["zones"] for t in obs["tracks"] if t["track_id"] == 9] == [[]]


def test_class_filter_keeps_people_out_of_foreign_object():
    rule = ZoneIntrusionRule("tank_surface", TANK, min_dwell_s=0.0, classes=["bottle"], event_type="foreign_object")
    inside_person = Track(1, "person", 0.9, (40, 40, 50, 60))
    assert rule.update([inside_person], 0.0) == []
    assert len(rule.update([Track(2, "bottle", 0.6, (40, 40, 50, 60))], 0.1)) == 1


def test_default_stays_zone_intrusion_and_bad_config_is_rejected():
    assert ZoneIntrusionRule("restricted_a", TANK).event_type == "zone_intrusion"
    with pytest.raises(ValueError, match="event_type"):
        ZoneIntrusionRule("tank", TANK, event_type="foreign-object")
    with pytest.raises(ValueError, match="polygon"):
        ZoneIntrusionRule("tank", [[0, 0], [1, 1]])


def test_example_config_zone_builds_a_foreign_object_rule():
    from pathlib import Path

    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.analyzers.example.yaml").read_text("utf-8"))
    zones = cfg["sources"][0]["zones"]
    rules = [ZoneIntrusionRule(z["id"], z["polygon"], z.get("min_dwell_s", 1.0), z.get("lost_timeout_s", 2.0),
                               z.get("classes"), z.get("event_type", "zone_intrusion")) for z in zones]
    assert [r.event_type for r in rules] == ["foreign_object"]
