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
