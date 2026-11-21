"""Run FoamCoverage on still images and save a contact sheet (image | foam mask overlay | %).

    python scripts/foam_eval.py IMG_DIR [IMG_DIR ...] --out foam_sheet.jpg --colors white brown

No ROI is applied (whole image), so railings, walkways and sky count as "tank" -
use it to compare colour presets, not as a final accuracy number.
"""
from __future__ import annotations

import argparse
import csv
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
    args = ap.parse_args()

    fc = FoamCoverage(colors=args.colors, warp_size=TILE)
    rows, table = [], []
    for d in args.dirs:
        for p in sorted(Path(d).rglob("*")):
            if p.suffix.lower() not in EXTS:
                continue
            img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)  # unicode-safe on Windows
            if img is None:
                print(f"skip unreadable {p}", file=sys.stderr)
                continue
            pct = fc.coverage(img)
            surf = fc.surface(img)
            over = surf.copy()
            over[fc.last_mask > 0] = (0.4 * over[fc.last_mask > 0] + 0.6 * np.array([255, 0, 255])).astype(np.uint8)
            rows.append(np.hstack([tile(surf, p.name), tile(over, f"{Path(d).name}: {pct:.1f}%")]))
            table.append((Path(d).name, p.name, round(pct, 1)))
    if not rows:
        sys.exit("no images found")
    pairs = [np.hstack(rows[i:i + 2]) if i + 1 < len(rows) else np.hstack([rows[i], np.zeros_like(rows[i])])
             for i in range(0, len(rows), 2)]
    cv2.imencode(Path(args.out).suffix or ".jpg", np.vstack(pairs))[1].tofile(args.out)
    with open(Path(args.out).with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([("folder", "image", "foam_pct"), *table])
    for row in table:
        print(*row, sep="\t")


if __name__ == "__main__":
    main()
