"""Analyzer framework, foam coverage, overflow and async tracker (synthetic frames, no model)."""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from aiot.analyzers import Analyzer, AnalyzerResult, build_analyzers
from aiot.analyzers.foam import FoamCoverage
from aiot.analyzers.overflow import Overflow
from aiot.async_tracker import AsyncTracker, NullTracker
from aiot.pipeline import SourcePipeline

WHITE = (240, 240, 240)
WATER = (40, 35, 30)        # dark water, BGR
BROWN_FOAM = (120, 170, 200)  # tan foam, BGR
BROWN_WATER = (30, 50, 75)


def frame(color=WATER, h=120, w=160):
    f = np.zeros((h, w, 3), np.uint8)
    f[:] = color
    return f


# --- framework ---------------------------------------------------------------------------------------------------
def test_build_analyzers_respects_enabled_flag_and_rejects_unknown():
    a = build_analyzers({"foam_coverage": {"enabled": True}, "overflow": {"enabled": False, "roi": [[0, 0]]}})
    assert [x.name for x in a] == ["foam_coverage"]
    assert build_analyzers(None) == []
    with pytest.raises(ValueError):
        build_analyzers({"no_such": {}})


# --- foam coverage -----------------------------------------------------------------------------------------------
def test_foam_zero_half_full():
    fc = FoamCoverage(kernel_px=1)
    assert fc.coverage(frame()) == 0
    half = frame()
    half[:, :80] = WHITE
    assert fc.coverage(half) == pytest.approx(50, abs=1)
    assert fc.coverage(frame(WHITE)) == 100


def test_foam_roi_crops_outside_area():
    f = frame(WHITE)                      # bright railings / sky everywhere ...
    f[40:100, 40:120] = WATER             # ... except the tank surface
    fc = FoamCoverage(roi=[[40, 40], [119, 40], [119, 99], [40, 99]], kernel_px=1)
    assert fc.coverage(f) < 5
    assert FoamCoverage(kernel_px=1).coverage(f) > 50


def test_brown_preset_separates_tan_foam_from_brown_water():
    f = frame(BROWN_WATER)
    f[:60] = BROWN_FOAM
    assert FoamCoverage(colors=["white"], kernel_px=1).coverage(f) < 5
    assert FoamCoverage(colors=["white", "brown"], kernel_px=1).coverage(f) == pytest.approx(50, abs=1)


def test_calibration_uses_brightness_above_clean_water():
    fc = FoamCoverage(colors=[], brighter_than_water=40, kernel_px=1)
    fc.calibrate(frame((60, 60, 60)))
    f = frame((60, 60, 60))
    f[:30] = (120, 120, 120)              # grey foam, not in any colour range
    assert fc.coverage(f) == pytest.approx(25, abs=1)


def test_foam_measurement_is_smoothed():
    fc = FoamCoverage(kernel_px=1, smooth_alpha=0.5)
    fc.process(frame(), 0.0)
    m = fc.process(frame(WHITE), 1.0).measurement
    assert m["raw_pct"] == 100 and m["foam_coverage_pct"] == 50 and m["status"] == "ok"


def test_foam_roi_needs_four_points():
    with pytest.raises(ValueError):
        FoamCoverage(roi=[[0, 0], [1, 0], [1, 1]])


# --- overflow ----------------------------------------------------------------------------------------------------
ROI = [[0, 0], [159, 0], [159, 59], [0, 59]]  # top half of the frame


def spill():
    f = frame()
    f[:60] = WHITE
    return f


def feed(ov, f, t, n=1, dt=0.1):
    events = []
    for i in range(n):
        events += ov.process(f, t + i * dt).events
    return events


def test_overflow_learns_then_fires_once_per_spill():
    ov = Overflow(ROI, learn_frames=3, confirm_frames=2, cooldown_s=0)
    assert feed(ov, frame(), 0, 3) == []                      # learning
    assert ov.process(frame(), 1).measurement["status"] == "ok"
    ev = feed(ov, spill(), 2, 10)
    assert len(ev) == 1 and ev[0]["zone_id"] == "overflow_zone" and ov.active
    assert feed(ov, frame(), 4, 1) == [] and not ov.active      # cleared
    assert len(feed(ov, spill(), 5, 5)) == 1                    # new spill, cooldown 0


def test_overflow_cooldown_blocks_repeat_alerts():
    ov = Overflow(ROI, learn_frames=1, confirm_frames=1, cooldown_s=30)
    feed(ov, frame(), 0)
    assert len(feed(ov, spill(), 1)) == 1
    feed(ov, frame(), 2)
    assert feed(ov, spill(), 10) == []                          # within 30 s
    feed(ov, frame(), 20)
    assert len(feed(ov, spill(), 40)) == 1


def test_overflow_ignores_single_frame_flicker():
    ov = Overflow(ROI, learn_frames=1, confirm_frames=3)
    feed(ov, frame(), 0)
    for t in range(1, 10):
        assert feed(ov, spill() if t % 2 else frame(), t) == []


# --- async tracker -----------------------------------------------------------------------------------------------
class SlowTracker:
    def __init__(self, delay):
        self.delay, self.calls = delay, 0
        self.started = threading.Event()

    def update(self, f):
        self.started.set()
        time.sleep(self.delay)
        self.calls += 1
        return [f"tracks-{self.calls}"]


def test_async_tracker_never_blocks_and_returns_latest():
    slow = SlowTracker(0.3)
    at = AsyncTracker(slow)
    try:
        t0 = time.time()
        assert at.update(frame()) == []                         # nothing finished yet
        assert time.time() - t0 < 0.05
        assert slow.started.wait(1)
        for _ in range(20):                                     # frames arriving faster than the model
            at.update(frame())
        time.sleep(0.8)
        assert at.update(frame()) and slow.calls <= 3           # older frames were skipped, not queued
    finally:
        at.stop()


def test_null_tracker():
    assert NullTracker().update(frame()) == []


# --- pipeline integration ----------------------------------------------------------------------------------------
class Boom(Analyzer):
    name = "boom"

    def process(self, f, t):
        raise RuntimeError("bad roi")


class Alerting(Analyzer):
    name = "alerting"
    event_type = "overflow"

    def process(self, f, t):
        return AnalyzerResult(measurement={"status": "ok"}, events=[{"zone_id": "z"}])


def test_pipeline_runs_analyzers_and_isolates_failures():
    p = SourcePipeline(SimpleNamespace(source_id="cam"), NullTracker(), [], store=None,
                       analyzers=[Boom(), FoamCoverage(kernel_px=1), Alerting()])
    meas, events = p._run_analyzers(frame(WHITE), 0.0, frame(WHITE))
    assert meas["boom"]["status"] == "error"                    # error, not a fake 0 %
    assert meas["foam_coverage"]["foam_coverage_pct"] == 100
    assert events == [("overflow", {"zone_id": "z"})]
    assert p.analyzer_errors["boom"] == 1


def test_overflow_slow_creeping_spill_is_not_absorbed_into_reference():
    """Regression (Argon #3): quiet-zone adaptation must skip pixels that already changed."""
    ov = Overflow(ROI, learn_frames=1, confirm_frames=3, on_ratio=0.15, off_ratio=0.05, adapt_rate=0.2)
    feed(ov, frame(), 0)
    events, f = [], frame()
    for i in range(200):                      # wet patch grows by one column every 2 frames
        f[:60, : i // 2] = WHITE
        events += ov.process(f.copy(), 1 + 0.1 * i).events
    assert len(events) == 1


# --- ROI corner order / overcast & shade presets (found on the advisor's photos with a tank ROI) --------------------
def test_skewed_perspective_roi_is_ordered_not_rejected():
    """x+y / y-x picked the same corner twice for this shape (sh-bot-trang-04: far edge short, near edge wide)."""
    quad = [[0, 39], [53, 37], [159, 119], [0, 119]]
    fc = FoamCoverage(roi=[quad[2], quad[0], quad[3], quad[1]], kernel_px=1)   # shuffled
    assert fc.roi.tolist() == [[0, 39], [53, 37], [159, 119], [0, 119]]
    with pytest.raises(ValueError):                                          # bow-tie is not a quad
        FoamCoverage(roi=[[0, 0], [100, 0], [0, 100], [50, 10]])


def test_grey_preset_catches_overcast_foam_but_not_dark_water():
    f = frame((70, 75, 80))                     # dark grey water
    f[:60] = (140, 145, 148)                    # dull grey-white foam, V ~ 148 < white's 165
    assert FoamCoverage(colors=["white"], kernel_px=1).coverage(f) < 5
    assert FoamCoverage(colors=["white", "grey"], kernel_px=1).coverage(f) == pytest.approx(50, abs=1)


def test_shadow_preset_catches_blue_tinted_foam():
    f = frame(WATER)
    f[:60] = (225, 190, 160)                    # foam in shade, bluish (S ~ 74 > white's 60)
    assert FoamCoverage(colors=["white"], kernel_px=1).coverage(f) < 5
    assert FoamCoverage(colors=["white", "shadow"], kernel_px=1).coverage(f) == pytest.approx(50, abs=1)


# --- calibration from config (calibrate: first_frames | clean-tank image) ------------------------------------------
LIGHT_BROWN_WATER = (95, 150, 185)            # sunlit mixed liquor: inside the brown preset, but it is water


def test_calibrate_first_frames_reports_calibrating_then_stops_water_counting_as_foam():
    uncal = FoamCoverage(colors=["white", "brown"], kernel_px=1)
    assert uncal.coverage(frame(LIGHT_BROWN_WATER)) > 90                  # the problem seen on Pexels video 1
    fc = FoamCoverage(colors=["white", "brown"], kernel_px=1, calibrate="first_frames", calibrate_frames=3)
    statuses = [fc.process(frame(LIGHT_BROWN_WATER), i).measurement["status"] for i in range(3)]
    assert statuses == ["calibrating", "calibrating", "ok"]
    assert fc.water_v is not None
    f = frame(LIGHT_BROWN_WATER)
    f[:60] = WHITE                                                        # real foam still counts
    m = fc.process(f, 3.0).measurement
    assert m["calibrated"] and m["raw_pct"] == pytest.approx(50, abs=1)


def test_calibrate_from_clean_tank_image(tmp_path):
    import cv2
    clean = tmp_path / "clean.png"
    cv2.imwrite(str(clean), frame(LIGHT_BROWN_WATER))
    fc = FoamCoverage(colors=["white", "brown"], kernel_px=1, calibrate=str(clean))
    m = fc.process(frame(LIGHT_BROWN_WATER), 0.0).measurement
    assert m["status"] == "ok" and m["calibrated"] and m["raw_pct"] < 5


def test_calibrate_bad_inputs_fail_loudly(tmp_path):
    import cv2
    with pytest.raises(ValueError):
        FoamCoverage(calibrate=str(tmp_path / "missing.png"))
    with pytest.raises(ValueError):
        FoamCoverage(calibrate="first_frames", calibrate_frames=0)
    small = tmp_path / "small.png"
    cv2.imwrite(str(small), frame(h=60, w=80))
    with pytest.raises(ValueError):                                       # photo from another resolution
        FoamCoverage(calibrate=str(small)).process(frame(), 0.0)


def test_calibrate_option_reaches_analyzer_via_config():
    (fc,) = build_analyzers({"foam_coverage": {"calibrate": "first_frames", "calibrate_frames": 2}})
    assert fc.process(frame(), 0.0).measurement["status"] == "calibrating"


def test_darker_than_local_drops_brown_water_cells_between_brown_foam():
    f = frame((150, 190, 225))                 # light tan foam everywhere ...
    f[45:75, 60:100] = (110, 150, 185)         # ... with a darker brown-water cell that still matches "brown"
    plain = FoamCoverage(colors=["brown"], kernel_px=1)
    local = FoamCoverage(colors=["brown"], kernel_px=1, darker_than_local=10)
    assert plain.coverage(f) > 95
    assert local.coverage(f) == pytest.approx(93.75, abs=2)       # 30x40 cell = 6.25 % of the 120x160 frame
    assert local.coverage(frame((150, 190, 225))) > 95            # uniform foam is untouched
    with pytest.raises(ValueError):
        FoamCoverage(darker_than_local=-1)


def test_brighter_than_local_catches_dull_foam_veins_no_preset_matches():
    f = frame((60, 65, 70))                    # dark grey-brown water
    f[:, 70:80] = (85, 95, 105)                # dull foam vein: no colour preset matches (V ~105, S ~49)
    f[55:65, :] = (85, 95, 105)
    vein_pct = (10 * 120 + 10 * 160 - 100) * 100 / (120 * 160)
    assert FoamCoverage(colors=["white", "brown", "grey", "shadow"], kernel_px=1).coverage(f) < 1
    fc = FoamCoverage(colors=["white", "brown", "grey", "shadow"], kernel_px=1, brighter_than_local=8)
    assert fc.coverage(f) == pytest.approx(vein_pct, abs=3)
    assert fc.coverage(frame((60, 65, 70))) < 1                    # flat water: nothing is locally bright
    with pytest.raises(ValueError):
        FoamCoverage(brighter_than_local=0)


# --- cross-test round 2 (Argon T055): confirmed bugs, now regression tests -----------------------------------------
def test_t055_2_calibrate_image_applies_to_plain_coverage_calls(tmp_path):
    import cv2
    clean = np.full((240, 320, 3), (50, 110, 150), np.uint8)       # bright brown water, inside the brown preset
    p = tmp_path / "clean.png"
    cv2.imwrite(str(p), clean)
    fc = FoamCoverage(colors=("brown",), calibrate=str(p))
    assert fc.water_v == pytest.approx(150.0)
    assert fc.coverage(clean) == 0.0                                # was 100 %: calibration only ran in process()
    with pytest.raises(ValueError):                                 # size checked on every frame, not just the first
        fc.process(np.zeros((120, 160, 3), np.uint8), 0.0)


def test_t055_3_brighter_than_local_ignores_water_darker_than_calibrated_level():
    fc = FoamCoverage(colors=("white",), brighter_than_local=8.0)
    fc.calibrate(np.full((240, 320, 3), 100, np.uint8))
    shaded = np.full((240, 320, 3), 70, np.uint8)
    shaded[:, :160] = 40                                            # shadow edge on clean water
    assert fc.coverage(shaded) == 0.0                               # was 5.9 %


def test_t055_4_darker_than_local_keeps_shaded_band_of_white_foam():
    full = np.full((240, 320, 3), 250, np.uint8)
    full[:, 145:175] = 190                                          # 30 px band of shaded white foam
    fc = FoamCoverage(colors=("white",), darker_than_local=10.0)
    fc.calibrate(np.full((240, 320, 3), 60, np.uint8))
    assert fc.coverage(full) == pytest.approx(100.0, abs=1.0)      # was 90.6 %: 130 levels above water = foam
    # uncalibrated, only V >= BRIGHT_FOAM_V (200) is safe from the filter: lower would bring back grey water
    # between foam (sh-bot-trang-07). Known limit: a V=190 band is still cut without calibration.
    uncal = FoamCoverage(colors=("white",), darker_than_local=10.0)
    assert uncal.coverage(full) == pytest.approx(90.6, abs=1.0)
    full[:, 145:175] = 205
    assert uncal.coverage(full) == pytest.approx(100.0, abs=1.0)


def test_t055_6_first_frames_on_a_tank_already_60pct_foam():
    fc = FoamCoverage(colors=("white",), calibrate="first_frames", calibrate_frames=3)
    f = np.full((240, 320, 3), 50, np.uint8)
    f[:144] = 210                                                   # 60 % white foam at start-up
    for t in range(3):
        m = fc.process(f, float(t)).measurement
    assert fc.water_v == pytest.approx(50.0)                        # 25th percentile, not the median (= foam)
    assert m["raw_pct"] == pytest.approx(60.0, abs=1.0)             # was 0 %


def test_t055_manual_calibrate_cancels_pending_first_frames():
    fc = FoamCoverage(calibrate="first_frames", calibrate_frames=5)
    fc.calibrate(frame())
    assert fc.process(frame(), 0.0).measurement["status"] == "ok"


def test_t055_5_window_and_warp_size_limits():
    fc = FoamCoverage(warp_size=(320, 240), local_window_frac=1.0, darker_than_local=10.0)
    assert fc.local_window[0] <= 240 and fc.local_window[0] % 2 == 1
    assert FoamCoverage(warp_size=(16, 12), kernel_px=5).local_window[0] >= 11   # >= kernel..., <= surface
    with pytest.raises(ValueError):
        FoamCoverage(warp_size=(0, 240))


def test_t055_7_tracker_warmup_accepts_hw_imgsz(monkeypatch):
    import sys
    import types

    from aiot.vision import Tracker
    calls = []

    class FakeYOLO:
        def __init__(self, weights):
            pass

        def predict(self, img, **kw):
            calls.append((img.shape, kw["imgsz"]))

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    Tracker(weights="x.pt", imgsz=[384, 640])                       # was TypeError
    Tracker(weights="x.pt", imgsz=640)
    assert calls == [((384, 640, 3), [384, 640]), ((640, 640, 3), 640)]
