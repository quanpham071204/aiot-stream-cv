"""Run FoamCoverage on still images and save a contact sheet (image | foam mask overlay | %).

    python scripts/foam_eval.py IMG_DIR [IMG_DIR ...] --out foam_sheet.jpg --colors white brown grey shadow
        --rois scripts/foam_rois.json    (one command line)

Without --rois the whole image is measured, so railings, walkways and sky count as
"tank". With --rois each image is warped to its tank-surface quad (corners given as
fractions of width/height) and compared with the file's ``ref_pct``; the mean absolute
error is printed at the end.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aiot.analyzers.foam import FoamCoverage  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TILE = (320, 240)


def tile(img: np.ndarray, label: str) -> np.ndarray:
    t = cv2.resize(img, TILE, interpolation=cv2.INTER_AREA)
    cv2.rectangle(t, (0, 0), (TILE[0], 22), (0, 0, 0), -1)
    cv2.putText(t, label[:44], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="foam_sheet.jpg")
    ap.add_argument("--colors", nargs="+", default=["white"])
    ap.add_argument("--rois", help="JSON: {image_name: {roi: [[fx, fy] x4], ref_pct: float}}")
    args = ap.parse_args()

    rois = json.loads(Path(args.rois).read_text(encoding="utf-8")) if args.rois else {}
    rows, table = [], []
    for d in args.dirs:
        for p in sorted(Path(d).rglob("*")):
            if p.suffix.lower() not in EXTS:
                continue
            img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)  # unicode-safe on Windows
            if img is None:
                print(f"skip unreadable {p}", file=sys.stderr)
                continue
            spec = rois.get(p.name, {})
            h, w = img.shape[:2]
            roi = [[fx * (w - 1), fy * (h - 1)] for fx, fy in spec["roi"]] if "roi" in spec else None
            fc = FoamCoverage(roi=roi, colors=args.colors, warp_size=TILE)
            pct = fc.coverage(img)
            ref = spec.get("ref_pct")
            surf = fc.surface(img)
            over = surf.copy()
            over[fc.last_mask > 0] = (0.4 * over[fc.last_mask > 0] + 0.6 * np.array([255, 0, 255])).astype(np.uint8)
            label = f"{pct:.1f}%" + (f"  ref {ref:.0f}%" if ref is not None else "")
            rows.append(np.hstack([tile(surf, p.name), tile(over, label)]))
            table.append((Path(d).name, p.name, round(pct, 1), ref, None if ref is None else round(pct - ref, 1)))
    if not rows:
        sys.exit("no images found")
    pairs = [np.hstack(rows[i:i + 2]) if i + 1 < len(rows) else np.hstack([rows[i], np.zeros_like(rows[i])])
             for i in range(0, len(rows), 2)]
    cv2.imencode(Path(args.out).suffix or ".jpg", np.vstack(pairs))[1].tofile(args.out)
    with open(Path(args.out).with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([("folder", "image", "foam_pct", "ref_pct", "error"), *table])
    errs = [r[4] for r in table if r[4] is not None]
    if errs:
        print(f"MAE {np.mean(np.abs(errs)):.1f}  bias {np.mean(errs):+.1f}  (n={len(errs)})")
    for row in table:
        print(*row, sep="\t")


if __name__ == "__main__":
    main()
