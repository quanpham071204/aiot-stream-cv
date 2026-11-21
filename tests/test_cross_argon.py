"""Cross-test round 1 (Gemini 4 Argon, T053): draft tests that found real bugs, kept as regression tests.

Not kept: #3 uniform whole-zone brightening (treated as lighting drift by design), #6B uniform
brightening after calibration and #7 grey/blue-shadow foam without calibration (known limits).
"""
import numpy as np
import pytest

from aiot.analyzers.foam import FoamCoverage
from aiot.analyzers.overflow import Overflow


def test_overflow_resize_resets_streak_and_active():
    roi = [[10, 10], [50, 10], [50, 50], [10, 50]]
    clean_a = np.full((100, 100, 3), 40, dtype=np.uint8)
    spill_a = np.full((100, 100, 3), 200, dtype=np.uint8)
    clean_b = np.full((120, 120, 3), 40, dtype=np.uint8)
    spill_b = np.full((120, 120, 3), 200, dtype=np.uint8)
    # Nhánh A: _streak = 2 trước khi đổi kích thước không được cộng dồn sang kích thước mới
    ov_a = Overflow(roi=roi, diff_thresh=30, on_ratio=0.15, off_ratio=0.05,
                    confirm_frames=3, cooldown_s=30.0, learn_frames=3, blur_px=3)
    for t in range(3):
        ov_a.process(clean_a, float(t))
    for t in (3.0, 4.0):
        assert ov_a.process(spill_a, t).events == []
    for t in (5.0, 6.0, 7.0):
        assert ov_a.process(clean_b, t).measurement["status"] == "learning"
    res_first = ov_a.process(spill_b, 8.0)
    assert res_first.events == []
    assert res_first.measurement["active"] is False
    # Nhánh B: active = True trước khi đổi kích thước không được chặn đợt tràn mới
    ov_b = Overflow(roi=roi, diff_thresh=30, on_ratio=0.15, off_ratio=0.05,
                    confirm_frames=3, cooldown_s=5.0, learn_frames=3, blur_px=3)
    for t in range(3):
        ov_b.process(clean_a, float(t))
    res = None
    for t in (3.0, 4.0, 5.0):
        res = ov_b.process(spill_a, t)
    assert len(res.events) == 1 and res.measurement["active"] is True
    for t in (20.0, 21.0, 22.0):
        ov_b.process(clean_b, t)
    events_after = []
    for t in (23.0, 24.0, 25.0):
        events_after.extend(ov_b.process(spill_b, t).events)
    assert len(events_after) == 1
def test_overflow_spill_starting_in_cooldown_alerts_once_cooldown_expires():
    roi = [[10, 10], [50, 10], [50, 50], [10, 50]]
    clean = np.full((80, 80, 3), 40, dtype=np.uint8)
    spill = np.full((80, 80, 3), 200, dtype=np.uint8)
    ov = Overflow(roi=roi, diff_thresh=30, on_ratio=0.15, off_ratio=0.05,
                  confirm_frames=3, cooldown_s=30.0, learn_frames=3, blur_px=3)
    for t in range(3):
        ov.process(clean, float(t))
    ev1 = []
    for t in (3.0, 4.0, 5.0):
        ev1.extend(ov.process(spill, t).events)
    assert len(ev1) == 1
    res_clear = ov.process(clean, 6.0)
    assert res_clear.measurement["active"] is False
    ev_in_cd = []
    for t in (10.0, 11.0, 12.0):
        ev_in_cd.extend(ov.process(spill, t).events)
    assert ev_in_cd == []
    ev_after_cd = []
    for t in (35.0, 36.0, 37.0, 40.0):
        ev_after_cd.extend(ov.process(spill, t).events)
    assert len(ev_after_cd) >= 1
def test_overflow_flickering_around_on_ratio_still_triggers():
    roi = [[10, 10], [109, 10], [109, 109], [10, 109]]
    clean = np.full((120, 120, 3), 40, dtype=np.uint8)
    high = clean.copy()
    high[10:30, 10:110] = 200  # 2000 / 10000 = 0.20 >= on_ratio(0.15)
    mid = clean.copy()
    mid[10:23, 10:110] = 200   # 1300 / 10000 = 0.13 trong [0.05, 0.15)
    ov = Overflow(roi=roi, diff_thresh=30, on_ratio=0.15, off_ratio=0.05,
                  confirm_frames=3, learn_frames=3, blur_px=1)
    for t in range(3):
        ov.process(clean, float(t))
    events = []
    for i in range(30):
        frame = mid if (i % 3 == 2) else high
        events.extend(ov.process(frame, 3.0 + i).events)
    assert len(events) >= 1
def test_foam_z_order_roi_rejected_or_warped_correctly():
    white_tank = np.full((240, 320, 3), 255, dtype=np.uint8)
    z_roi = [[0, 0], [319, 0], [0, 239], [319, 239]]  # TL, TR, BL, BR (tự cắt)
    try:
        fc = FoamCoverage(roi=z_roi)
    except ValueError:
        return  # Chấp nhận nếu từ chối ROI sai thứ tự góc
    cov = fc.coverage(white_tank)
    assert cov == pytest.approx(100.0, abs=2.0)
def test_foam_ema_validates_alpha_and_respects_time_gap():
    for bad_alpha in (0.0, -0.2, 1.5):
        with pytest.raises(ValueError):
            FoamCoverage(smooth_alpha=bad_alpha)
    dark = np.full((240, 320, 3), 20, dtype=np.uint8)
    white = np.full((240, 320, 3), 255, dtype=np.uint8)
    fc = FoamCoverage(smooth_alpha=0.3)
    r1 = fc.process(dark, 0.0)
    r2 = fc.process(white, 600.0)  # gián đoạn 10 phút
    assert r1.measurement["foam_coverage_pct"] == 0.0
    assert r2.measurement["foam_coverage_pct"] > 90.0
