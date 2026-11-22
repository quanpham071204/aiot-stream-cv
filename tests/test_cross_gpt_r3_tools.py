"""T059: adversarial calibration, fake-YOLO warmup and headless ROI CLI checks.

All images are synthetic. CLI sources are local files, never camera indices.
The YOLO module is replaced before Tracker construction; no model is loaded.
"""
import os
import subprocess
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from aiot.analyzers import build_analyzers
from aiot.analyzers.foam import FoamCoverage
from aiot.vision import Tracker

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "pick_roi.py"
POINTS = "1,1 90,1 90,60 1,60"
ORDERED = [[1, 1], [90, 1], [90, 60], [1, 60]]


def write_png(path, pixels):
    """Use imencode/tofile so Windows Unicode paths do not depend on imwrite."""
    ok, encoded = cv2.imencode(".png", pixels)
    assert ok
    encoded.tofile(str(path))
    return path


def run_roi(*args):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    return subprocess.run(
        [sys.executable, "-B", str(SCRIPT), *map(str, args)],
        cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=3.5,
    )


def load_output(result):
    assert result.returncode == 0, result.stderr
    cfg = yaml.safe_load(result.stdout)["analyzers"]
    return cfg, build_analyzers(cfg)


@pytest.fixture
def image(tmp_path):
    return write_png(tmp_path / "frame.png", np.full((64, 96, 3), 180, np.uint8))


def test_calibrate_grayscale_png_is_applied_before_process(tmp_path):
    gray = np.full((64, 96), 180, np.uint8)
    path = write_png(tmp_path / "gray.png", gray)
    decoded = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded.ndim == 2  # A genuinely single-channel calibration file.
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    analyzer = FoamCoverage(calibrate=str(path), kernel_px=1)
    assert analyzer.water_v == pytest.approx(180)
    assert analyzer.coverage(frame) == 0
    result = analyzer.process(frame, 0).measurement
    assert result["calibrated"] and result["status"] == "ok"
    assert result["raw_pct"] == 0


@pytest.mark.parametrize("alpha", [0, 128, 255], ids=["transparent", "partial", "opaque"])
def test_calibrate_alpha_png_uses_color_channels(tmp_path, alpha):
    bgra = np.full((64, 96, 4), 180, np.uint8)
    bgra[:, :, 3] = alpha
    path = write_png(tmp_path / "alpha.png", bgra)
    assert cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED).shape[2] == 4
    analyzer = FoamCoverage(calibrate=str(path), kernel_px=1)
    assert analyzer.water_v == pytest.approx(180)
    assert analyzer.coverage(bgra[:, :, :3]) == 0
    assert analyzer.process(bgra[:, :, :3], 0).measurement["calibrated"]


def test_calibrate_unicode_and_space_path_builds_analyzer(tmp_path):
    folder = tmp_path / "bể nước sạch"
    folder.mkdir()
    frame = np.full((64, 96, 3), 180, np.uint8)
    path = write_png(folder / "ảnh mốc nước.png", frame)
    (analyzer,) = build_analyzers({"foam_coverage": {"calibrate": str(path), "kernel_px": 1}})
    assert analyzer.water_v == pytest.approx(180)
    assert analyzer.coverage(frame) == 0
    assert analyzer.process(frame, 0).measurement["raw_pct"] == 0


@pytest.mark.parametrize("shape", [(32, 96, 3), (64, 48, 3), (96, 64, 3)])
def test_calibrate_checks_size_after_valid_frames_and_recovers(image, shape):
    analyzer = FoamCoverage(calibrate=str(image), kernel_px=1)
    frame = np.full((64, 96, 3), 180, np.uint8)
    assert analyzer.process(frame, 0).measurement["status"] == "ok"
    assert analyzer.process(frame, 1).measurement["calibrated"]
    # Repeated mismatches ensure validation is not consumed by the first frame/error.
    for t in (2, 3):
        with pytest.raises(ValueError, match=r"calibrate image.*stream"):
            analyzer.process(np.full(shape, 180, np.uint8), t)
    assert analyzer.process(frame, 4).measurement["raw_pct"] == 0
    assert analyzer.water_v == pytest.approx(180)


@pytest.mark.parametrize("kind", ["missing", "corrupt"])
def test_calibrate_unreadable_image_raises_valueerror(tmp_path, kind):
    path = tmp_path / "unreadable.png"
    if kind == "corrupt":
        path.write_bytes(b"not a PNG")
    with pytest.raises(ValueError, match="cannot read clean-tank image"):
        FoamCoverage(calibrate=str(path))


@pytest.fixture
def fake_yolo(monkeypatch):
    instances = []

    class FakeYOLO:
        def __init__(self, weights):
            self.weights = weights
            self.predictions = []
            self.tracks = []
            instances.append(self)

        def predict(self, frame, **kwargs):
            self.predictions.append((frame, kwargs))
            return []

        def track(self, frame, **kwargs):
            self.tracks.append((frame, kwargs))
            return [types.SimpleNamespace(boxes=None)]

    module = types.ModuleType("ultralytics")
    module.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    return instances


@pytest.mark.parametrize(
    ("imgsz", "shape"),
    [(96, (96, 96, 3)), (96.0, (96, 96, 3)), (96.75, (96, 96, 3)),
     ([48, 96], (48, 96, 3)), ((48, 96), (48, 96, 3)), ([96], (96, 96, 3))],
    ids=["int", "float", "fractional-float", "list-hw", "tuple-hw", "list-one"],
)
def test_tracker_warmup_shape_and_options_with_fake_yolo(fake_yolo, imgsz, shape):
    tracker = Tracker(weights="T059_FAKE_ONLY.pt", imgsz=imgsz, device="cpu", warmup=True)
    assert len(fake_yolo) == 1 and tracker.model is fake_yolo[0]
    model = fake_yolo[0]
    assert model.weights == "T059_FAKE_ONLY.pt"
    assert model.tracks == []  # Warmup must not create ByteTrack history.
    assert len(model.predictions) == 1
    dummy, kwargs = model.predictions[0]
    assert dummy.shape == shape and dummy.dtype == np.uint8
    assert np.count_nonzero(dummy) == 0
    assert kwargs == {"imgsz": imgsz, "device": "cpu", "verbose": False}


def test_tracker_warmup_false_skips_prediction_and_preserves_update_options(fake_yolo):
    tracker = Tracker(weights="T059_FAKE_ONLY.pt", imgsz=(48, 96), classes=[0], conf=0.6,
                      device="cpu", tracker_cfg="fake_tracker.yaml", warmup=False)
    model = fake_yolo[0]
    assert model.predictions == [] and model.tracks == []
    frame = np.zeros((48, 96, 3), np.uint8)
    assert tracker.update(frame) == []
    assert len(model.tracks) == 1 and model.tracks[0][0] is frame
    assert model.tracks[0][1] == {
        "persist": True, "tracker": "fake_tracker.yaml", "classes": [0], "conf": 0.6,
        "imgsz": (48, 96), "device": "cpu", "verbose": False,
    }


@pytest.mark.parametrize(
    "points",
    ["-1,1 90,1 90,60 -1,60", "1,-1 90,-1 90,60 1,60",
     "1,1 120,1 120,60 1,60", "1,1 90,1 90,80 1,80"],
    ids=["negative-x", "negative-y", "x-beyond-width", "y-beyond-height"],
)
def test_pick_roi_rejects_points_outside_frame(image, points):
    result = run_roi("--image", image, f"--points={points}")
    assert result.returncode != 0, f"CLI accepted out-of-frame ROI:\n{result.stdout}"
    assert result.stderr.strip()
    assert "Traceback" not in result.stderr


def test_pick_roi_fractional_points_roundtrip_to_integer_corners(image):
    # parse_points deliberately converts float text to integer pixel coordinates.
    result = run_roi("--image", image, "--points=1.9,1.2 90.8,1.9 90.1,60.9 1.1,60.1")
    cfg, analyzers = load_output(result)
    assert cfg["foam_coverage"]["roi"] == ORDERED
    assert analyzers[0].process(np.full((64, 96, 3), 180, np.uint8), 0).measurement["status"] == "calibrating"


@pytest.mark.parametrize("bad", ["10", "10,2,3", "10,", ",10", "word,10", "nan,10", "inf,10"],
                         ids=["missing-comma", "extra-comma", "missing-y", "missing-x", "text", "nan", "inf"])
def test_pick_roi_malformed_points_fail_without_traceback(image, bad):
    result = run_roi("--image", image, f"--points={bad} 90,1 90,60 1,60")
    assert result.returncode != 0
    assert "bad point" in result.stderr and "x,y" in result.stderr, result.stderr
    assert "Traceback" not in result.stderr
    assert "analyzers:" not in result.stdout


def test_pick_roi_malformed_overflow_point_is_validated(image):
    result = run_roi("--image", image, f"--points={POINTS}", "--overflow=1,1 90,1 90,60,2")
    assert result.returncode != 0
    assert "bad point" in result.stderr and "x,y" in result.stderr
    assert "Traceback" not in result.stderr


def test_pick_roi_concave_quad_is_rejected(image):
    result = run_roi("--image", image, "--points=1,1 90,1 30,20 1,60")
    assert result.returncode != 0
    assert "convex quadrilateral" in result.stderr
    assert "analyzers:" not in result.stdout


def test_pick_roi_z_order_is_reordered_and_yaml_builds_both_analyzers(image):
    # The four corners form a convex tank; only the supplied traversal crosses.
    result = run_roi("--image", image, "--points=1,1 90,60 90,1 1,60", "--overflow=2,40 20,40 20,60 2,60")
    cfg, analyzers = load_output(result)
    assert cfg["foam_coverage"]["roi"] == ORDERED
    assert [a.name for a in analyzers] == ["foam_coverage", "overflow"]
    assert analyzers[0].roi.astype(int).tolist() == ORDERED


def test_pick_roi_missing_image_fails_without_traceback(tmp_path):
    result = run_roi("--image", tmp_path / "missing.png", f"--points={POINTS}")
    assert result.returncode != 0
    assert "no such image" in result.stderr and "Traceback" not in result.stderr


def test_pick_roi_broken_source_is_a_local_file_not_a_camera(tmp_path):
    source = tmp_path / "broken.mp4"
    source.write_bytes(b"not a video")
    result = run_roi("--source", source, f"--points={POINTS}")
    assert result.returncode != 0
    assert "cannot open source" in result.stderr and "Traceback" not in result.stderr
    assert "analyzers:" not in result.stdout


@pytest.mark.parametrize("name", ["ảnh mốc nước.png", "ảnh mốc # nước.png"], ids=["unicode-spaces", "hash-space"])
def test_pick_roi_clean_path_yaml_roundtrip_and_immediate_calibration(tmp_path, name):
    image = write_png(tmp_path / "ảnh nguồn.png", np.full((64, 96, 3), 180, np.uint8))
    clean = tmp_path / name
    result = run_roi("--image", image, f"--points={POINTS}", "--clean", clean)
    assert result.returncode == 0, result.stderr
    cfg = yaml.safe_load(result.stdout)["analyzers"]
    assert cfg["foam_coverage"]["calibrate"] == str(clean), result.stdout
    assert clean.is_file()
    (analyzer,) = build_analyzers(cfg)
    assert analyzer.water_v == pytest.approx(180)
    assert analyzer.process(np.full((64, 96, 3), 180, np.uint8), 0).measurement["raw_pct"] == 0


def test_pick_roi_clean_missing_parent_has_controlled_outcome(image, tmp_path):
    clean = tmp_path / "not-created" / "nested" / "clean.png"
    assert not clean.parent.exists()
    result = run_roi("--image", image, f"--points={POINTS}", "--clean", clean)
    # Either create the parent and succeed, or report a concise user-facing error.
    if result.returncode == 0:
        cfg, analyzers = load_output(result)
        assert clean.is_file() and cfg["foam_coverage"]["calibrate"] == str(clean)
        assert analyzers[0].water_v == pytest.approx(180)
    else:
        assert result.stderr.strip()
        assert "Traceback" not in result.stderr, result.stderr
        assert "analyzers:" not in result.stdout
