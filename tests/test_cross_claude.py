"""Cross-test round 1 (Claude): property / metamorphic tests on real-looking noisy frames.

Instead of exact numbers these check rules any correct implementation must obey:
more foam never lowers the %, resizing barely changes it, a uniform brightness shift on
clean water does not create foam out of nothing, and failures never look like 0 %.
"""
import numpy as np
import pytest

from aiot.analyzers.foam import FoamCoverage
from aiot.analyzers.overflow import Overflow

RNG = np.random.default_rng(261)


def water(h=240, w=320, base=(45, 40, 35), noise=8):
    f = np.empty((h, w, 3), np.int16)
    f[:] = base
    f += RNG.integers(-noise, noise + 1, size=f.shape, dtype=np.int16)
    return np.clip(f, 0, 255).astype(np.uint8)


def add_foam_blobs(f, n, seed, color=(235, 235, 235), r=(6, 18)):
    rng = np.random.default_rng(seed)
    out = f.copy()
    h, w = f.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(n):
        cy, cx, rad = rng.integers(0, h), rng.integers(0, w), rng.integers(*r)
        out[(yy - cy) ** 2 + (xx - cx) ** 2 <= rad * rad] = color
    return out


def test_more_foam_never_lowers_coverage():
    fc = FoamCoverage()
    base = water()
    values = [fc.coverage(add_foam_blobs(base, n, seed=7)) for n in (0, 5, 15, 40, 120)]
    assert values[0] < 1
    assert all(b >= a - 0.5 for a, b in zip(values, values[1:], strict=False)), values
    assert values[-1] > 50


@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_resolution_change_keeps_coverage(scale):
    import cv2

    f = add_foam_blobs(water(), 30, seed=3)
    big = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    a, b = FoamCoverage().coverage(f), FoamCoverage().coverage(big)
    assert abs(a - b) < 3, (a, b)


def test_roi_on_scaled_frame_matches_scaled_roi():
    import cv2

    f = add_foam_blobs(water(), 25, seed=11)
    roi = [[40, 30], [280, 30], [280, 210], [40, 210]]
    a = FoamCoverage(roi=roi).coverage(f)
    big = cv2.resize(f, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    b = FoamCoverage(roi=[[2 * x, 2 * y] for x, y in roi]).coverage(big)
    assert abs(a - b) < 3, (a, b)


def test_roi_corner_order_does_not_matter_much():
    f = add_foam_blobs(water(), 25, seed=5)
    cw = [[40, 30], [280, 30], [280, 210], [40, 210]]
    rotated = cw[1:] + cw[:1]                 # same quad, different starting corner
    assert abs(FoamCoverage(roi=cw).coverage(f) - FoamCoverage(roi=rotated).coverage(f)) < 1


def test_brighter_lighting_on_clean_water_is_not_foam():
    fc = FoamCoverage(colors=["white", "brown"])
    f = water()
    for shift in (0, 40, 80):
        lit = np.clip(f.astype(np.int16) + shift, 0, 255).astype(np.uint8)
        assert fc.coverage(lit) < 5, shift


def test_calibrated_lighting_jump_is_a_known_limit():
    """Documented limit: after calibration a global brightness jump looks like foam.

    This test pins current behaviour so a future fix (e.g. re-calibration) is a
    deliberate change, not an accident.
    """
    fc = FoamCoverage(colors=[], brighter_than_water=40)
    fc.calibrate(water())
    lit = np.clip(water().astype(np.int16) + 60, 0, 255).astype(np.uint8)
    assert fc.coverage(lit) > 90


def test_ema_converges_and_is_bounded():
    fc = FoamCoverage(smooth_alpha=0.3)
    full = add_foam_blobs(water(), 400, seed=1)
    vals = [fc.process(full if i >= 5 else water(), float(i)).measurement["foam_coverage_pct"] for i in range(40)]
    assert all(0 <= v <= 100 for v in vals)
    assert vals[4] < 1 and vals[-1] == pytest.approx(fc.coverage(full), abs=0.5)
    assert all(b >= a for a, b in zip(vals[5:], vals[6:], strict=False))      # monotone rise, no overshoot


# --- overflow ------------------------------------------------------------------------------------------------------
TRAY = [[0, 180], [319, 180], [319, 239], [0, 239]]


def spilled(f):
    out = f.copy()
    out[190:240, 40:280] = (200, 200, 200)
    return out


def run(ov, frames, t0=0.0, dt=0.1):
    events = []
    for i, f in enumerate(frames):
        events += ov.process(f, t0 + i * dt).events
    return events


def test_overflow_noise_alone_never_fires():
    ov = Overflow(TRAY, learn_frames=5, confirm_frames=2)
    assert run(ov, [water() for _ in range(60)]) == []


def test_overflow_one_event_per_spill_over_many_spills():
    ov = Overflow(TRAY, learn_frames=5, confirm_frames=2, cooldown_s=0)
    run(ov, [water() for _ in range(5)])
    t, total = 1.0, 0
    for _ in range(4):
        total += len(run(ov, [spilled(water()) for _ in range(15)], t0=t))
        t += 2
        run(ov, [water() for _ in range(10)], t0=t)
        t += 2
    assert total == 4


def test_overflow_slow_lighting_drift_does_not_fire():
    ov = Overflow(TRAY, learn_frames=5, confirm_frames=3, diff_thresh=30)
    frames = [np.clip(water().astype(np.int16) + i // 4, 0, 255).astype(np.uint8) for i in range(200)]
    assert run(ov, frames) == []                                  # +50 grey levels over 20 s


def test_overflow_measurement_never_reports_active_while_learning():
    ov = Overflow(TRAY, learn_frames=5)
    for i in range(4):
        m = ov.process(spilled(water()), i).measurement
        assert m["status"] == "learning" and "active" not in m
