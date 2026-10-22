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


class ZoneIntrusionRule:
    """Fire once when a tracked object stays inside a zone for min_dwell_s.

    A track counts as having left only after it has been outside (or unseen)
    for lost_timeout_s, so detector jitter at the border does not re-fire.
    """

    event_type = "zone_intrusion"

    def __init__(self, zone_id: str, polygon: list, min_dwell_s: float = 1.0,
                 lost_timeout_s: float = 2.0, classes: Optional[Iterable[str]] = None):
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
        for tid in [k for k, s in self._state.items() if t - s["last_seen"] > self.lost_timeout_s]:
            del self._state[tid]
        return events
