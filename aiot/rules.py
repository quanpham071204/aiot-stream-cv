"""Turn tracks into application observations/events. Pure Python, no model needed."""
from __future__ import annotations

from typing import Iterable, Optional

from .vision import Track


def point_in_polygon(pt: tuple[float, float], poly: list) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


EVENT_TYPES = ("zone_intrusion", "foreign_object")


class ZoneIntrusionRule:
    """Fire once when a tracked object stays inside a zone for min_dwell_s.

    A track counts as having left only after it has been outside (or unseen)
    for lost_timeout_s, so detector jitter at the border does not re-fire.

    ``event_type`` names what the zone means: ``zone_intrusion`` (a person in a restricted area,
    the default) or ``foreign_object`` (an object detected on the tank surface; the zone is the tank
    and ``classes`` are the foreign-object classes of the model). The logic and payload are the same.
    """

    def __init__(self, zone_id: str, polygon: list, min_dwell_s: float = 1.0,
                 lost_timeout_s: float = 2.0, classes: Optional[Iterable[str]] = None,
                 event_type: str = "zone_intrusion"):
        if event_type not in EVENT_TYPES:
            raise ValueError(f"zone {zone_id!r}: event_type must be one of {list(EVENT_TYPES)}, got {event_type!r}")
        if polygon is None or len(polygon) < 3:
            raise ValueError(f"zone {zone_id!r}: polygon needs at least 3 points")
        self.event_type = event_type
        self.zone_id = zone_id
        self.polygon = polygon
        self.min_dwell_s = min_dwell_s
        self.lost_timeout_s = lost_timeout_s
        self.classes = set(classes) if classes else None
        self._state: dict[int, dict] = {}
        self.inside_ids: set[int] = set()

    def update(self, tracks: Iterable[Track], t: float) -> list[dict]:
        events = []
        self.inside_ids = set()
        # Expire before observing returning tracks, including gaps with no rule updates.
        for tid in [k for k, s in self._state.items() if t - s["last_seen"] > self.lost_timeout_s]:
            del self._state[tid]
        for tr in tracks:
            if self.classes and tr.cls_name not in self.classes:
                continue
            if not point_in_polygon(tr.anchor, self.polygon):
                continue
            self.inside_ids.add(tr.track_id)
            st = self._state.setdefault(tr.track_id, {"entered_at": t, "fired": False})
            st["last_seen"] = t
            dwell = t - st["entered_at"]
            if not st["fired"] and dwell >= self.min_dwell_s:
                st["fired"] = True
                events.append({
                    "zone_id": self.zone_id,
                    "track_id": tr.track_id,
                    "class": tr.cls_name,
                    "confidence": round(tr.conf, 3),
                    "dwell_s": round(dwell, 2),
                    "bbox": [round(v, 1) for v in tr.bbox],
                })
        return events
