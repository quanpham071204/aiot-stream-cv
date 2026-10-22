"""Download the sample video and the pretrained YOLOv8n weights used by config.yaml.

    python scripts/download_assets.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIDEO_URL = "https://github.com/opencv/opencv/raw/4.x/samples/data/vtest.avi"


def fetch(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"ok        {dest.relative_to(ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"download  {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    print(f"saved     {dest.relative_to(ROOT)} ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    fetch(VIDEO_URL, ROOT / "data" / "vtest.avi")
    weights = ROOT / "yolov8n.pt"
    if weights.exists():
        print(f"ok        {weights.relative_to(ROOT)}")
    else:
        from ultralytics import YOLO  # downloads the official release asset on first use

        YOLO(str(weights))
        print(f"saved     {weights.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
