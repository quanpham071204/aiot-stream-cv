"""Foam coverage (% of the tank surface covered by foam), plain OpenCV.

1. Warp the tank surface (4-point ROI) to a fixed top-down rectangle, so every
   pixel stands for about the same area and railings/sky are cut away.
2. A pixel is foam if its HSV colour falls inside one of the configured ranges
   (presets ``white``, ``brown``, ``grey``, ``shadow``), or - after calibration on
   the water - if it is clearly brighter than the water. Calibration is strongly
   recommended (it stops bright brown water from counting as brown foam):
   ``calibrate: first_frames`` learns the water brightness from the first
   ``calibrate_frames`` frames after start (start the camera on a calm, foam-free
   tank); ``calibrate: <path to image>`` uses a photo of the clean tank taken by the
   same camera at the same resolution.
3. Morphological open/close removes speckle and fills foam patches.
4. Coverage = foam pixels / ROI pixels, smoothed with an exponential moving average.

No alert is raised: phase 1 only reports the measurement.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from ..timeutil import iso
from . import Analyzer, AnalyzerResult

# OpenCV HSV: H 0-179, S 0-255, V 0-255. Tuned on the advisor's 29 reference photos with a
# tank ROI each (scripts/foam_eval.py --rois scripts/foam_rois.json).
# "grey" and "shadow" catch foam under overcast sky / in shade; they also match grey concrete
# and bright grey water, so use them only with a tight tank-surface ROI.
PRESETS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "white": ((0, 0, 165), (179, 60, 255)),     # bright, nearly colourless
    "brown": ((5, 40, 135), (30, 200, 255)),    # tan / beige / golden foam, lighter than brown water
    "grey": ((0, 0, 115), (179, 30, 255)),      # dull grey-white foam (overcast light)
    "shadow": ((95, 0, 150), (125, 110, 255)),  # white foam tinted blue by shade / sky
}


class FoamCoverage(Analyzer):
    name = "foam_coverage"

    def __init__(self, roi: Optional[Sequence[Sequence[float]]] = None, warp_size: Sequence[int] = (320, 240),
                 colors: Sequence = ("white",), brighter_than_water: float = 40.0, kernel_px: int = 5,
                 smooth_alpha: float = 0.3, reset_after_s: float = 10.0, calibrate: Optional[str] = None,
                 calibrate_frames: int = 10):
        if roi is not None and len(roi) != 4:
            raise ValueError("foam_coverage.roi needs exactly 4 corner points")
        if not 0 < smooth_alpha <= 1:
            raise ValueError("foam_coverage.smooth_alpha must be in (0, 1]")
        self.roi = _order_corners(roi) if roi is not None else None
        self.reset_after_s = float(reset_after_s)  # gap longer than this: restart smoothing from the new value
        self._last_t: Optional[float] = None
        self.warp_size = (int(warp_size[0]), int(warp_size[1]))
        self.ranges = [_hsv_range(c) for c in colors]
        self.brighter_than_water = float(brighter_than_water)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px)) if kernel_px > 1 else None
        self.smooth_alpha = float(smooth_alpha)
        self.water_v: Optional[float] = None
        self.value: Optional[float] = None
        self.last_mask: Optional[np.ndarray] = None
        # calibration source: None (off), "first_frames", or a clean-tank image applied on the first frame
        self._cal_image: Optional[np.ndarray] = None
        self._cal_samples: list[float] = []
        self._cal_mode: Optional[str] = None
        if calibrate is not None:
            if calibrate_frames < 1:
                raise ValueError("foam_coverage.calibrate_frames must be >= 1")
            self.calibrate_frames = int(calibrate_frames)
            if calibrate == "first_frames":
                self._cal_mode = "first_frames"
            else:
                img = None
                if Path(calibrate).is_file():  # imdecode(fromfile): unicode-safe paths on Windows
                    img = cv2.imdecode(np.fromfile(str(calibrate), np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"foam_coverage.calibrate: cannot read clean-tank image {calibrate!r} "
                                     "(use 'first_frames' or an image path)")
                self._cal_mode, self._cal_image = "image", img

    # --- helpers -------------------------------------------------------------------------------------------------
    def surface(self, frame: np.ndarray) -> np.ndarray:
        """Top-down view of the tank surface (whole frame resized when no ROI is set)."""
        w, h = self.warp_size
        if self.roi is None:
            return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        m = cv2.getPerspectiveTransform(self.roi, dst)
        return cv2.warpPerspective(frame, m, (w, h), flags=cv2.INTER_AREA)

    def _water_level(self, frame: np.ndarray) -> float:
        return float(np.median(cv2.cvtColor(self.surface(frame), cv2.COLOR_BGR2HSV)[:, :, 2]))

    def calibrate(self, clean_frame: np.ndarray) -> float:
        """Remember the brightness of the clean (foam-free) water surface."""
        self.water_v = self._water_level(clean_frame)
        return self.water_v

    def mask(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(self.surface(frame), cv2.COLOR_BGR2HSV)
        m = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in self.ranges:
            m |= cv2.inRange(hsv, lo, hi)
        if self.water_v is not None:
            above = hsv[:, :, 2].astype(np.float32) - self.water_v
            # calibrated: a colour match must also be somewhat brighter than the clean water
            # (bright sky reflections on clean water are not foam) ...
            m &= (above >= self.brighter_than_water / 2).astype(np.uint8) * 255
            # ... and anything clearly brighter than the water counts, whatever its colour (grey foam)
            m |= (above >= self.brighter_than_water).astype(np.uint8) * 255
        if self.kernel is not None:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.kernel)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self.kernel)
        return m

    def coverage(self, frame: np.ndarray) -> float:
        """Instant (unsmoothed) coverage in percent."""
        self.last_mask = self.mask(frame)
        return float(np.count_nonzero(self.last_mask)) * 100.0 / self.last_mask.size

    # --- Analyzer API ----------------------------------------------------------------------------------------------
    def process(self, frame: np.ndarray, t: float) -> AnalyzerResult:
        if self._cal_mode == "image":
            if self._cal_image.shape != frame.shape:
                raise ValueError(f"foam_coverage.calibrate image is {self._cal_image.shape[1::-1]} but the stream "
                                 f"is {frame.shape[1::-1]}; take the clean-tank photo with the same camera setting")
            self.calibrate(self._cal_image)
            self._cal_mode, self._cal_image = None, None
        elif self._cal_mode == "first_frames":
            self._cal_samples.append(self._water_level(frame))
            if len(self._cal_samples) < self.calibrate_frames:
                return AnalyzerResult(measurement={"status": "calibrating", "calibrated": False,
                                                   "frames_left": self.calibrate_frames - len(self._cal_samples),
                                                   "updated_at": iso(t)})
            self.water_v = float(np.median(self._cal_samples))
            self._cal_mode, self._cal_samples = None, []
        raw = self.coverage(frame)
        a = self.smooth_alpha
        stale = self._last_t is None or t - self._last_t > self.reset_after_s
        self.value = raw if self.value is None or stale else a * raw + (1 - a) * self.value
        self._last_t = t
        return AnalyzerResult(measurement={
            "status": "ok",
            "foam_coverage_pct": round(self.value, 1),
            "raw_pct": round(raw, 1),
            "calibrated": self.water_v is not None,
            "updated_at": iso(t),
        })

    def draw(self, frame: np.ndarray) -> None:
        if self.roi is not None:
            cv2.polylines(frame, [self.roi.astype(np.int32)], True, (255, 200, 0), 2)
        if self.value is not None:
            cv2.putText(frame, f"foam {self.value:.1f}%", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)


def _hsv_range(c) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(c, str):
        if c not in PRESETS:
            raise ValueError(f"unknown foam colour preset {c!r}; known: {sorted(PRESETS)}")
        lo, hi = PRESETS[c]
    else:
        lo, hi = c["lo"], c["hi"]
    limits = (179, 255, 255)
    if len(lo) != 3 or len(hi) != 3 or any(not 0 <= x <= m for v in (lo, hi) for x, m in zip(v, limits, strict=True)) \
            or any(a > b for a, b in zip(lo, hi, strict=True)):
        raise ValueError(f"bad HSV range {c!r}: need lo <= hi, H 0-179, S/V 0-255 (no hue wrap-around)")
    return np.array(lo, np.uint8), np.array(hi, np.uint8)


def _order_corners(pts) -> np.ndarray:
    """Return the 4 points as top-left, top-right, bottom-right, bottom-left, whatever order they came in.

    Sorted by angle around the centroid (clockwise on screen), then rotated so the corner
    nearest the image origin comes first. Works for skewed / perspective quads where a
    plain x+y / y-x rule picks the same point twice.
    """
    p = np.array(pts, dtype=np.float32)
    c = p.mean(axis=0)
    p = p[np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))]
    p = np.roll(p, -int(np.argmin(p.sum(axis=1))), axis=0)
    if len({tuple(q) for q in p.tolist()}) != 4 or cv2.contourArea(p) < 1 or not cv2.isContourConvex(p):
        raise ValueError("foam_coverage.roi corners must form a convex quadrilateral")
    return p
