"""Quick set-up for the demo tank: grab a frame, click the regions, get the YAML to paste.

    python scripts/pick_roi.py --source 0                 # webcam 0 (or a video file / rtsp:// URL)
    python scripts/pick_roi.py --source 0 --clean clean.jpg   # also save a clean-tank photo for
                                                              # foam_coverage.calibrate: clean.jpg
    python scripts/pick_roi.py --image frame.jpg --points "10,10 300,10 300,200 10,200" \
        --overflow "0,210 320,210 320,240 0,240"        # no window: points given on the command line

In the window: left click adds a corner, U undoes, N moves on to the next region
(1: tank surface, 4 corners; 2: overflow tray, 3+ corners), Enter / Esc finishes.
Coordinates are in the camera's own pixels, which is what config.yaml expects.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REGIONS = [("foam_coverage.roi (tank surface, 4 corners)", (255, 200, 0)),
           ("overflow.roi (tray outside the wall, 3+ corners)", (0, 0, 255))]


def grab(source: str, warm_frames: int = 15) -> np.ndarray:
    """Read one frame; skip the first few so webcam exposure / white balance can settle."""
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        sys.exit(f"cannot open source {source!r}")
    frame = None
    for _ in range(warm_frames if source.isdigit() else 1):
        ok, f = cap.read()
        if ok:
            frame = f
        time.sleep(0.03)
    cap.release()
    if frame is None:
        sys.exit(f"no frame from source {source!r}")
    return frame


def parse_points(s: str) -> list[list[int]]:
    return [[int(float(v)) for v in p.split(",")] for p in s.split()]


def to_yaml(foam: list, overflow: list, frame_shape) -> str:
    h, w = frame_shape[:2]
    lines = [f"# frame size {w}x{h}: keep the camera at this resolution, or measure again"]
    lines += ["analyzers:", "  foam_coverage:", "    enabled: true", f"    roi: {foam}",
              "    colors: [white, brown, grey, shadow]", "    calibrate: first_frames"]
    if overflow:
        lines += ["  overflow:", "    enabled: true", "    zone_id: tray_front", f"    roi: {overflow}"]
    return "\n".join(lines)


def check(foam: list, overflow: list) -> None:
    if len(foam) != 4:
        sys.exit(f"tank surface needs exactly 4 corners, got {len(foam)}")
    if overflow and len(overflow) < 3:
        sys.exit(f"overflow tray needs at least 3 corners, got {len(overflow)}")
    from aiot.analyzers.foam import FoamCoverage  # same validation as the service (convex quad)

    FoamCoverage(roi=foam)


def click(frame: np.ndarray) -> tuple[list, list]:
    regions: list[list] = [[], []]
    cur = 0
    win = "pick_roi - click corners, N next, U undo, Enter done"

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            regions[cur].append([x, y])

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        view = frame.copy()
        for i, pts in enumerate(regions):
            color = REGIONS[i][1]
            for p in pts:
                cv2.circle(view, tuple(p), 4, color, -1)
            if len(pts) > 1:
                cv2.polylines(view, [np.array(pts, np.int32)], len(pts) > 2, color, 2)
        cv2.putText(view, f"{cur + 1}: {REGIONS[cur][0]}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(win, view)
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("u"), ord("U")) and regions[cur]:
            regions[cur].pop()
        elif k in (ord("n"), ord("N")) and cur < len(REGIONS) - 1:
            cur += 1
        elif k in (13, 27):
            break
    cv2.destroyAllWindows()
    return regions[0], regions[1]


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--source", help="webcam index, video file or rtsp:// URL")
    src.add_argument("--image", help="use a saved frame instead of a live source")
    ap.add_argument("--points", help='tank corners without a window: "x,y x,y x,y x,y"')
    ap.add_argument("--overflow", default="", help='overflow tray corners without a window: "x,y x,y x,y"')
    ap.add_argument("--clean", help="save the grabbed frame here as the clean-tank calibration photo")
    ap.add_argument("--save", help="save a preview with the regions drawn")
    args = ap.parse_args()

    if args.image:
        frame = cv2.imdecode(np.fromfile(args.image, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            sys.exit(f"cannot read {args.image!r}")
    else:
        frame = grab(args.source)
    if args.clean:
        cv2.imencode(Path(args.clean).suffix or ".jpg", frame)[1].tofile(args.clean)
        print(f"saved clean-tank photo {args.clean} (use: calibrate: {args.clean})", file=sys.stderr)

    if args.points:
        foam, overflow = parse_points(args.points), parse_points(args.overflow) if args.overflow else []
    else:
        foam, overflow = click(frame)
    check(foam, overflow)
    if args.save:
        view = frame.copy()
        for pts, (_, color) in zip((foam, overflow), REGIONS, strict=True):
            if pts:
                cv2.polylines(view, [np.array(pts, np.int32)], True, color, 2)
        cv2.imencode(Path(args.save).suffix or ".jpg", view)[1].tofile(args.save)
    print(to_yaml(foam, overflow, frame.shape))


if __name__ == "__main__":
    main()
