"""Computer vision block: detections -> tracks (YOLOv8 + ByteTrack)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import cv2
import numpy as np


@dataclass
class Track:
    track_id: int
    cls_name: str
    conf: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2

    @property
    def anchor(self) -> tuple[float, float]:
        """Bottom-centre of the box (~ foot point), used for zone tests."""
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)


def _hw(imgsz) -> tuple[int, int]:
    """(h, w) from an ultralytics imgsz: a number (also numpy) or a 1-2 item sequence."""
    size = np.atleast_1d(np.asarray(imgsz))
    if size.dtype.kind not in "iuf" or size.ndim != 1 or not 1 <= size.size <= 2 or (size <= 0).any():
        raise ValueError(f"model.imgsz must be a positive number or [h, w], got {imgsz!r}")
    return int(size[0]), int(size[-1])


class Tracker:
    """One instance per source: ByteTrack state is kept inside the model (persist=True)."""

    def __init__(self, weights: str = "yolov8n.pt", classes: Optional[list[int]] = None,
                 conf: float = 0.35, imgsz: int = 640, device=None, tracker_cfg: str = "bytetrack.yaml",
                 warmup: bool = True):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.classes = classes
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.tracker_cfg = tracker_cfg
        if warmup:
            # The first inference loads CUDA kernels / allocates buffers (measured 8.4 s on the
            # team laptop's GPU). Pay it here, before the stream starts, instead of on the first
            # frames (sync mode dropped ~200 frames). predict() leaves the ByteTrack state untouched.
            # imgsz may be an int or [h, w] (both valid for ultralytics); use a full-size dummy frame so the
            # warmed-up kernels match the real input size.
            self.model.predict(np.zeros((*_hw(imgsz), 3), np.uint8), imgsz=imgsz, device=device, verbose=False)

    def update(self, frame: np.ndarray) -> list[Track]:
        r = self.model.track(frame, persist=True, tracker=self.tracker_cfg, classes=self.classes,
                             conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        boxes = r.boxes
        if boxes is None or boxes.id is None:
            return []
        names = r.names
        return [
            Track(int(i), names[int(c)], float(p), tuple(b))
            for i, c, p, b in zip(boxes.id.int().tolist(), boxes.cls.int().tolist(),
                                  boxes.conf.tolist(), boxes.xyxy.tolist(), strict=True)
        ]


def draw(frame: np.ndarray, tracks: Iterable[Track], zones: Iterable[tuple[str, list]], alert_ids=()) -> np.ndarray:
    out = frame.copy()
    for zone_id, poly in zones:
        pts = np.array(poly, dtype=np.int32)
        cv2.polylines(out, [pts], True, (0, 0, 255), 2)
        cv2.putText(out, zone_id, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    alert_ids = set(alert_ids)
    for t in tracks:
        x1, y1, x2, y2 = map(int, t.bbox)
        color = (0, 0, 255) if t.track_id in alert_ids else (0, 200, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"#{t.track_id} {t.cls_name} {t.conf:.2f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return out
