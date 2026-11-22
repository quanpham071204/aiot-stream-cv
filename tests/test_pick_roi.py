"""scripts/pick_roi.py without a window: points on the command line -> YAML the service accepts."""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from aiot.analyzers import build_analyzers

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pick_roi.py"


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=60)


def test_points_give_yaml_that_builds_analyzers(tmp_path):
    img = tmp_path / "frame.png"
    cv2.imwrite(str(img), np.zeros((240, 320, 3), np.uint8))
    clean = tmp_path / "clean.png"
    r = run("--image", str(img), "--points", "300,10 10,10 10,200 300,200", "--overflow", "0,210 319,210 319,239",
            "--clean", str(clean), "--save", str(tmp_path / "preview.png"))
    assert r.returncode == 0, r.stderr
    cfg = yaml.safe_load(r.stdout)["analyzers"]
    names = [a.name for a in build_analyzers(cfg)]
    assert names == ["foam_coverage", "overflow"]
    assert clean.exists() and (tmp_path / "preview.png").exists()


def test_bad_corner_count_is_rejected(tmp_path):
    img = tmp_path / "frame.png"
    cv2.imwrite(str(img), np.zeros((240, 320, 3), np.uint8))
    r = run("--image", str(img), "--points", "0,0 10,0 10,10")
    assert r.returncode != 0 and "4 corners" in r.stderr
