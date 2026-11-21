"""Overflow: watch a check zone OUTSIDE the tank wall (e.g. a dark drip tray).

The zone's empty look is learnt from the first ``learn_frames`` frames. A pixel
counts as changed when its grey level differs from that reference by more than
``diff_thresh`` - this catches clear water (wet surface darkens/reflects) as well
as foam. Hysteresis + consecutive-frame confirmation + cooldown make one event per
spill instead of one per frame:

- alarm starts when changed ratio >= ``on_ratio`` for ``confirm_frames`` frames
  in a row and no overflow event was raised in the last ``cooldown_s`` seconds;
- alarm clears when the ratio drops below ``off_ratio`` (no event on clear).

While the alarm is off and the zone is quiet the reference adapts slowly
(``adapt_rate``) to follow lighting drift.
"""
from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from . import Analyzer, AnalyzerResult


class Overflow(Analyzer):
    name = "overflow"
    event_type = "overflow"

    def __init__(self, roi: Sequence[Sequence[float]], zone_id: str = "overflow_zone", diff_thresh: int = 30,
                 on_ratio: float = 0.15, off_ratio: float = 0.05, confirm_frames: int = 3,
                 cooldown_s: float = 30.0, learn_frames: int = 10, adapt_rate: float = 0.02, blur_px: int = 5):
        if roi is None or len(roi) < 3:
            raise ValueError("overflow.roi needs a polygon with at least 3 points")
        if not 0 <= off_ratio < on_ratio <= 1:
            raise ValueError("need 0 <= off_ratio < on_ratio <= 1")
        self.zone_id = zone_id
        self.polygon = np.array(roi, dtype=np.int32)
        self.diff_thresh = diff_thresh
        self.on_ratio, self.off_ratio = on_ratio, off_ratio
        self.confirm_frames = confirm_frames
        self.cooldown_s = cooldown_s
        self.learn_frames = learn_frames
        self.adapt_rate = adapt_rate
        self.blur_px = blur_px | 1
        self._mask: Optional[np.ndarray] = None
        self._learn: list[np.ndarray] = []
        self.reference: Optional[np.ndarray] = None
        self.active = False
        self.ratio = 0.0
        self._streak = 0
        self._last_fired: Optional[float] = None

    def _gray(self, frame: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(g, (self.blur_px, self.blur_px), 0).astype(np.float32)

    def process(self, frame: np.ndarray, t: float) -> AnalyzerResult:
        g = self._gray(frame)
        if self._mask is None or self._mask.shape != g.shape:
            self._mask = np.zeros(g.shape, np.uint8)
            cv2.fillPoly(self._mask, [self.polygon], 1)
            self._learn, self.reference = [], None
        if self.reference is None:
            self._learn.append(g)
            if len(self._learn) >= self.learn_frames:
                self.reference = np.median(np.stack(self._learn), axis=0).astype(np.float32)
                self._learn = []
            return AnalyzerResult(measurement={"status": "learning", "zone_id": self.zone_id})

        inside = self._mask.astype(bool)
        changed = np.abs(g - self.reference) > self.diff_thresh
        self.ratio = float(np.count_nonzero(changed & inside)) / max(1, int(np.count_nonzero(inside)))

        events = []
        if self.active:
            if self.ratio < self.off_ratio:
                self.active, self._streak = False, 0
        else:
            self._streak = self._streak + 1 if self.ratio >= self.on_ratio else 0
            if self._streak >= self.confirm_frames:
                self.active = True
                if self._last_fired is None or t - self._last_fired >= self.cooldown_s:
                    self._last_fired = t
                    events.append({"zone_id": self.zone_id, "changed_ratio": round(self.ratio, 3),
                                   "on_ratio": self.on_ratio})
            elif self.ratio < self.off_ratio:
                cv2.accumulateWeighted(g, self.reference, self.adapt_rate)
        return AnalyzerResult(
            measurement={"status": "ok", "zone_id": self.zone_id, "active": self.active,
                         "changed_ratio": round(self.ratio, 3)},
            events=events,
        )

    def draw(self, frame: np.ndarray) -> None:
        color = (0, 0, 255) if self.active else (0, 200, 255)
        cv2.polylines(frame, [self.polygon], True, color, 2)
        x, y = self.polygon[0]
        cv2.putText(frame, f"{self.zone_id} {self.ratio:.0%}", (int(x), max(12, int(y) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
