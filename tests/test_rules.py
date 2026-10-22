"""ZoneIntrusionRule: dwell time, border jitter, re-entry and class filtering."""
from aiot.rules import ZoneIntrusionRule, point_in_polygon
from aiot.vision import Track

ZONE = [[0, 0], [100, 0], [100, 100], [0, 100]]


def person(tid, x, y):
    # anchor = bottom-centre = (x, y)
    return Track(tid, "person", 0.9, (x - 5, y - 20, x + 5, y))


def test_point_in_polygon():
    assert point_in_polygon((50, 50), ZONE)
    assert not point_in_polygon((150, 50), ZONE)


def test_fires_once_after_dwell():
    r = ZoneIntrusionRule("z", ZONE, min_dwell_s=1.0)
    assert r.update([person(1, 50, 50)], 0.0) == []
    assert r.update([person(1, 50, 50)], 0.5) == []
    ev = r.update([person(1, 50, 50)], 1.0)
    assert len(ev) == 1 and ev[0]["track_id"] == 1
    assert r.update([person(1, 50, 50)], 2.0) == []


def test_border_jitter_does_not_refire():
    r = ZoneIntrusionRule("z", ZONE, min_dwell_s=0.5, lost_timeout_s=2.0)
    r.update([person(1, 50, 50)], 0.0)
    assert len(r.update([person(1, 50, 50)], 0.6)) == 1
    r.update([person(1, 150, 50)], 0.8)  # briefly outside
    assert r.update([person(1, 50, 50)], 1.5) == []


def test_reentry_after_leaving_fires_again():
    r = ZoneIntrusionRule("z", ZONE, min_dwell_s=0.5, lost_timeout_s=1.0)
    r.update([person(1, 50, 50)], 0.0)
    assert len(r.update([person(1, 50, 50)], 0.6)) == 1
    r.update([], 3.0)  # gone longer than lost_timeout
    r.update([person(1, 50, 50)], 3.1)
    assert len(r.update([person(1, 50, 50)], 3.7)) == 1


def test_class_filter():
    r = ZoneIntrusionRule("z", ZONE, min_dwell_s=0.0, classes=["person"])
    car = Track(2, "car", 0.9, (40, 30, 60, 50))
    assert r.update([car], 0.0) == []
