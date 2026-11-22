"""T058: deterministic, model-free adversarial checks for foam round 3.

Inputs are BGR unless a test explicitly exercises grayscale/BGRA. Tests named
``known_limit`` document ambiguous brightness-only scenes, not physical accuracy.
No image files, model downloads, or other persistent test artifacts are needed.
"""

import cv2
import numpy as np
import pytest

from aiot.analyzers.foam import FoamCoverage

WIDTH, HEIGHT = 100, 60


def frame(value, width=WIDTH, height=HEIGHT):
    return np.full((height, width, 3), value, dtype=np.uint8)


def foam(**options):
    settings = dict(warp_size=(WIDTH, HEIGHT), kernel_px=1, smooth_alpha=1)
    settings.update(options)
    return FoamCoverage(**settings)


def test_water_level_ignores_a_70_percent_white_foam_layer():
    mixed = frame(220)
    mixed[:, :30] = 60
    analyzer = foam()
    assert analyzer.calibrate(mixed) == 60  # Median would incorrectly be 220.
    assert analyzer.coverage(mixed) == pytest.approx(70)


def test_bounded_water_noise_does_not_trigger_either_local_filter():
    # Each intensity occupies 25%; water level = median 104 (round 3 / T057: median of non-white pixels).
    clean = frame(88)
    for column, value in enumerate((88, 100, 108, 112)):
        clean[:, column * 25:(column + 1) * 25] = value
    analyzer = foam(darker_than_local=5, brighter_than_local=8)
    assert analyzer.calibrate(clean) == pytest.approx(104)
    assert analyzer.coverage(clean) == 0


def test_partly_shaded_clean_water_is_not_foam():
    # Was a known limit with the Q25 water level (T057 A1/B1: sunlit 70 % counted as foam).
    # The median of non-white pixels follows the larger, sunlit part, and the shade is below it.
    clean = frame(105)
    clean[:, :30] = 60
    analyzer = foam(colors=[])
    assert analyzer.calibrate(clean) == 105
    assert analyzer.coverage(clean) == 0


def test_first_frames_takes_the_temporal_median_of_spatial_percentiles():
    analyzer = foam(calibrate="first_frames", calibrate_frames=3)
    measurements = []
    # Keep the last water level < median+40, avoiding a separate lighting jump.
    for t, water in enumerate((30, 50, 70)):
        mixed = frame(230)
        mixed[:, :30] = water
        measurements.append(analyzer.process(mixed, float(t)).measurement)
    assert [m["status"] for m in measurements] == ["calibrating", "calibrating", "ok"]
    assert [m["frames_left"] for m in measurements[:2]] == [2, 1]
    assert all(m["calibrated"] is False for m in measurements[:2])
    assert analyzer.water_v == 50
    assert measurements[-1]["calibrated"] is True
    assert measurements[-1]["raw_pct"] == pytest.approx(70)


def test_manual_calibration_discards_already_collected_startup_samples():
    analyzer = foam(calibrate="first_frames", calibrate_frames=4)
    for t in (0, 1):
        assert analyzer.process(frame(200), t).measurement["status"] == "calibrating"
    assert analyzer.calibrate(frame(60)) == 60
    # Continue beyond the original warmup: old samples must never replace 60.
    for t in range(2, 8):
        result = analyzer.process(frame(220), t).measurement
        assert result["status"] == "ok"
        assert result["calibrated"] is True
        assert result["raw_pct"] == 100
        assert analyzer.water_v == 60


def test_full_foam_startup_is_a_known_calibration_limit():
    analyzer = foam(calibrate="first_frames", calibrate_frames=2,
                    darker_than_local=5, brighter_than_local=8)
    assert analyzer.process(frame(240), 0).measurement["status"] == "calibrating"
    result = analyzer.process(frame(240), 1).measurement
    assert result["status"] == "ok" and result["calibrated"] is True
    assert analyzer.water_v == 240
    assert result["raw_pct"] == 0  # No water exists in the reference to learn.
    assert analyzer.process(frame(60), 2).measurement["raw_pct"] == 0
    assert analyzer.process(frame(240), 3).measurement["raw_pct"] == 0
    analyzer.calibrate(frame(60))  # A real clean reference restores detection.
    assert analyzer.process(frame(240), 4).measurement["raw_pct"] == 100


@pytest.mark.parametrize("value", [60, 170, 210])
def test_both_local_filters_separate_calibrated_full_water_and_full_foam(value):
    analyzer = foam(darker_than_local=5, brighter_than_local=8)
    analyzer.calibrate(frame(value))
    assert analyzer.coverage(frame(value)) == 0
    assert analyzer.coverage(frame(value + 40)) == 100


def test_both_local_filters_handle_uncalibrated_full_dark_water_and_white_foam():
    analyzer = foam(darker_than_local=5, brighter_than_local=8)
    assert analyzer.coverage(frame(60)) == 0
    assert analyzer.coverage(frame(240)) == 100


@pytest.mark.parametrize("value", [200, 255], ids=["bright-water", "overexposed-water"])
def test_bright_water_without_a_reference_has_a_known_false_positive_limit(value):
    clean = frame(value)
    analyzer = foam(darker_than_local=5, brighter_than_local=8)
    assert analyzer.coverage(clean) == 100
    analyzer.calibrate(clean)
    assert analyzer.coverage(clean) == 0


@pytest.mark.parametrize(("band_v", "expected_pct"), [(199, 98), (200, 100), (201, 100)])
def test_uncalibrated_bright_foam_threshold_is_inclusive(band_v, expected_pct):
    full = frame(255)
    full[:, 49:51] = band_v  # Narrow shaded band is darker than its neighbours.
    analyzer = foam(darker_than_local=5, brighter_than_local=8)
    assert analyzer.coverage(full) == pytest.approx(expected_pct)
    assert np.all(analyzer.last_mask[:, 49:51] == (255 if band_v >= 200 else 0))


def test_bright_foam_threshold_only_preserves_existing_colour_matches():
    saturated_red = frame((0, 0, 240))
    analyzer = foam(darker_than_local=5, brighter_than_local=8)
    # V>=200 is not an unconditional detector; red matches neither white nor dull.
    assert analyzer.coverage(saturated_red) == 0


@pytest.mark.parametrize(("band_v", "expected_pct"), [(99, 98), (100, 100)])
def test_calibrated_absolute_brightness_protects_a_locally_dark_band(band_v, expected_pct):
    full = frame(160)
    full[:, 49:51] = band_v
    analyzer = foam(colors=[{"lo": [0, 0, 90], "hi": [179, 255, 255]}],
                    darker_than_local=5, brighter_than_local=8)
    analyzer.calibrate(frame(60))
    # Both bands pass the colour/half-threshold gate, but only +40 is protected.
    assert analyzer.coverage(full) == pytest.approx(expected_pct)


@pytest.mark.parametrize(("vein_v", "expected_pct"), [(107, 0), (108, 2), (109, 2)])
def test_local_addition_obeys_the_inclusive_water_threshold(vein_v, expected_pct):
    shaded = frame(60)
    shaded[:, 49:51] = vein_v
    analyzer = foam(colors=[], darker_than_local=5, brighter_than_local=8)
    analyzer.calibrate(frame(100))
    # Empty presets isolate local addition; since T057 A5 a vein must be >= brighter_than_local (8) above water:
    # +7 fails and +8/+9 pass.
    assert analyzer.coverage(shaded) == pytest.approx(expected_pct)


@pytest.mark.parametrize(("size", "kernel", "fraction", "expected_window"), [
    ((8, 8), 1, 0.001, 3),
    ((8, 8), 2, 1.0, 7),
    ((17, 64), 5, 0.01, 15),
    ((64, 9), 3, 0.01, 9),
    ((31, 17), 5, 1.0, 17),
    ((101, 80), 1, 0.1, 9),
])
def test_small_and_extreme_warps_with_large_kernels(size, kernel, fraction, expected_window):
    analyzer = foam(warp_size=size, kernel_px=kernel, local_window_frac=fraction,
                    darker_than_local=5, brighter_than_local=8)
    assert analyzer.local_window == (expected_window, expected_window)
    assert 3 <= expected_window <= min(size) and expected_window % 2 == 1
    analyzer.calibrate(frame(60))
    assert analyzer.coverage(frame(60)) == 0
    assert analyzer.coverage(frame(240)) == 100
    assert analyzer.last_mask.shape == (size[1], size[0])


@pytest.mark.parametrize(("size", "kernel"), [((8, 8), 31), ((9, 64), 31), ((64, 8), 31), ((64, 9), 5), ((31, 17), 9)])
def test_local_filters_reject_a_surface_too_small_for_the_kernel(size, kernel):
    # T057 A6: the window used to be clamped below 3 x kernel, and the open step then erased the local filters
    with pytest.raises(ValueError, match="too small for kernel_px"):
        foam(warp_size=size, kernel_px=kernel, darker_than_local=5, brighter_than_local=8)
    foam(warp_size=size, kernel_px=kernel)                      # without the local filters it is fine


@pytest.mark.parametrize("size", [(7, 8), (8, 7), (1, 64), (0, 8), (-1, 8)])
def test_warps_smaller_than_eight_are_rejected(size):
    with pytest.raises(ValueError, match="at least 8x8"):
        foam(warp_size=size)


@pytest.mark.parametrize("shape", [(HEIGHT, WIDTH), (HEIGHT, WIDTH, 1)])
def test_grayscale_rejection_preserves_the_last_valid_measurement(shape):
    # Pipeline frames are BGR; unsupported grayscale must fail recoverably.
    analyzer = foam(darker_than_local=5, brighter_than_local=8)
    assert analyzer.process(frame(60), 0).measurement["raw_pct"] == 0
    previous_mask = analyzer.last_mask.copy()
    with pytest.raises(cv2.error, match="(?i)channel"):
        analyzer.process(np.full(shape, 240, np.uint8), 1)
    assert analyzer.value == 0
    np.testing.assert_array_equal(analyzer.last_mask, previous_mask)
    assert analyzer.process(frame(240), 2).measurement["raw_pct"] == 100


def test_grayscale_during_startup_does_not_consume_a_calibration_sample():
    analyzer = foam(calibrate="first_frames", calibrate_frames=3)
    assert analyzer.process(frame(60), 0).measurement["frames_left"] == 2
    with pytest.raises(cv2.error, match="(?i)channel"):
        analyzer.process(np.full((HEIGHT, WIDTH), 240, np.uint8), 1)
    result = analyzer.process(frame(60), 2).measurement
    assert result["status"] == "calibrating" and result["frames_left"] == 1
    assert analyzer.process(frame(60), 3).measurement["raw_pct"] == 0
    assert analyzer.water_v == 60


@pytest.mark.parametrize("alpha", [0, 255, "gradient"])
def test_bgra_alpha_does_not_change_calibration_or_the_foam_mask(alpha):
    clean = frame(60)
    mixed = clean.copy()
    mixed[:, :50] = 240
    bgra_clean = cv2.cvtColor(clean, cv2.COLOR_BGR2BGRA)
    bgra_mixed = cv2.cvtColor(mixed, cv2.COLOR_BGR2BGRA)
    alpha_plane = np.arange(WIDTH, dtype=np.uint8)[None, :] if alpha == "gradient" else alpha
    bgra_clean[:, :, 3] = alpha_plane
    bgra_mixed[:, :, 3] = alpha_plane
    original = bgra_mixed.copy()
    bgr = foam(darker_than_local=5, brighter_than_local=8)
    bgra = foam(darker_than_local=5, brighter_than_local=8)
    assert bgr.calibrate(clean) == bgra.calibrate(bgra_clean) == 60
    assert bgr.process(mixed, 0).measurement["raw_pct"] == 50
    assert bgra.process(bgra_mixed, 0).measurement["raw_pct"] == 50
    np.testing.assert_array_equal(bgr.last_mask, bgra.last_mask)
    np.testing.assert_array_equal(bgra_mixed, original)
