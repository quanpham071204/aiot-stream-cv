"""T052 adversarial checks using synthetic frames and model-free trackers.

Policy: one alert per continuous spill, even after cooldown; clearing rearms it.
A geometry change starts a new learning session, including confirmation/alarm state.
Grayscale is tested as an unsupported input with an explicit, recoverable rejection;
BGRA is supported by OpenCV and alpha alone must not count as a spill.
"""

import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import aiot.async_tracker as async_module
from aiot.analyzers import build_analyzers
from aiot.analyzers.overflow import Overflow
from aiot.async_tracker import AsyncTracker
from aiot.vision import Track

ROI = [[0, 0], [31, 0], [31, 23], [0, 23]]


def frame(value=40, shape=(24, 32, 3)):
    return np.full(shape, value, dtype=np.uint8)


def overflow(**options):
    settings = dict(roi=ROI, learn_frames=1, confirm_frames=1, cooldown_s=0, blur_px=1)
    settings.update(options)
    return Overflow(**settings)


def wait_until(predicate, timeout=0.7):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "worker did not reach the expected state"
        threading.Event().wait(0.001)


def test_overflow_resize_relearns_without_shape_errors():
    analyzer = overflow(learn_frames=2)
    for t in (0, 1):
        assert analyzer.process(frame(), t).measurement["status"] == "learning"
    assert analyzer.process(frame(), 2).measurement["changed_ratio"] == 0
    resized = frame(80, shape=(40, 48, 3))
    for t in (3, 4):
        result = analyzer.process(resized, t)
        assert result.measurement["status"] == "learning"
        assert result.events == []
    result = analyzer.process(resized, 5)
    assert result.measurement["changed_ratio"] == 0
    assert result.measurement["active"] is False
    assert result.events == []
    assert len(analyzer.process(frame(200, resized.shape), 6).events) == 1


def test_overflow_resize_discards_unfinished_confirmation():
    analyzer = overflow(confirm_frames=3)
    analyzer.process(frame(), 0)
    for t in (1, 2):
        assert analyzer.process(frame(200), t).events == []
    assert analyzer.process(frame(shape=(40, 48, 3)), 3).measurement["status"] == "learning"
    for t in (4, 5):
        result = analyzer.process(frame(200, shape=(40, 48, 3)), t)
        assert result.events == [], "old geometry must not contribute confirmation frames"
        assert result.measurement["active"] is False
    assert len(analyzer.process(frame(200, shape=(40, 48, 3)), 6).events) == 1


def test_overflow_resize_discards_active_alarm_for_new_baseline():
    analyzer = overflow()
    analyzer.process(frame(), 0)
    assert len(analyzer.process(frame(200), 1).events) == 1
    assert analyzer.active
    assert analyzer.process(frame(shape=(40, 48, 3)), 2).measurement["status"] == "learning"
    result = analyzer.process(frame(200, shape=(40, 48, 3)), 3)
    assert result.measurement["changed_ratio"] == 1
    assert len(result.events) == 1, "an alarm from the old geometry must not suppress a new spill"


def test_overflow_grayscale_rejection_preserves_bgr_state():
    # No grayscale support is promised by the BGR pipeline; reject rather than silently misread it.
    analyzer = overflow()
    analyzer.process(frame(), 0)
    with pytest.raises(cv2.error, match="(?i)channel"):
        analyzer.process(frame(200, shape=(24, 32)), 1)
    assert analyzer.process(frame(), 2).measurement["changed_ratio"] == 0
    assert len(analyzer.process(frame(200), 3).events) == 1


def test_overflow_bgra_ignores_alpha_but_detects_color_change():
    analyzer = overflow()
    clean = frame(shape=(24, 32, 4))
    clean[:, :, 3] = 0
    analyzer.process(clean, 0)
    alpha_only = clean.copy()
    alpha_only[:, :, 3] = 255
    result = analyzer.process(alpha_only, 1)
    assert result.measurement["changed_ratio"] == 0
    assert result.events == []
    spill = alpha_only.copy()
    spill[:, :, :3] = 200
    result = analyzer.process(spill, 2)
    assert result.measurement["changed_ratio"] == 1
    assert len(result.events) == 1


def test_overflow_partial_roi_uses_only_visible_pixels_as_denominator():
    # Clipping leaves exactly 10 x 10 ROI pixels; change exactly half of them.
    analyzer = overflow(roi=[[-10, -10], [9, -10], [9, 9], [-10, 9]], on_ratio=0.4)
    clean = frame(shape=(20, 20, 3))
    analyzer.process(clean, 0)
    spill = clean.copy()
    spill[:10, :5] = 200
    spill[10:, 10:] = 200  # A change outside the visible ROI must not contribute.
    result = analyzer.process(spill, 1)
    assert result.measurement["changed_ratio"] == pytest.approx(0.5)
    assert len(result.events) == 1


@pytest.mark.parametrize("origin", [-40, 40], ids=["negative", "beyond-frame"])
def test_overflow_wholly_outside_roi_has_finite_zero_ratio(origin):
    roi = [[origin, origin], [origin + 5, origin], [origin + 5, origin + 5], [origin, origin + 5]]
    analyzer = overflow(roi=roi)
    analyzer.process(frame(), 0)
    for t in (1, 2, 100):
        result = analyzer.process(frame(200), t)
        assert np.isfinite(result.measurement["changed_ratio"])
        assert result.measurement["changed_ratio"] == 0
        assert result.measurement["active"] is False
        assert result.events == []


def test_overflow_confirm_one_fires_on_first_changed_frame_after_learning():
    analyzer = overflow(confirm_frames=1)
    assert analyzer.process(frame(), 0).events == []
    result = analyzer.process(frame(200), 0.1)
    assert result.measurement["active"] is True
    assert result.events == [{"zone_id": "overflow_zone", "changed_ratio": 1.0, "on_ratio": 0.15}]


def test_overflow_continuous_spill_does_not_refire_after_cooldown():
    analyzer = overflow(cooldown_s=5)
    analyzer.process(frame(), 0)
    assert len(analyzer.process(frame(200), 1).events) == 1
    for t in (2, 5.9, 6, 7, 50):
        result = analyzer.process(frame(200), t)
        assert result.measurement["active"] is True
        assert result.events == [], "cooldown expiry alone must not create a second spill"
    cleared = analyzer.process(frame(), 51)
    assert cleared.measurement["active"] is False
    assert cleared.events == []
    assert len(analyzer.process(frame(200), 52).events) == 1


@pytest.mark.parametrize("direction", [-1, 1], ids=["dimming", "brightening"])
def test_overflow_slow_lighting_drift_adapts_without_false_alarm(direction):
    analyzer = overflow(confirm_frames=3)
    analyzer.process(frame(100), 0)
    # One gray level per four frames: steady adaptation lag ~12.5 < diff_thresh=30.
    # Total drift reaches 50, so an unadapted reference would falsely raise an alarm.
    for step in range(1, 201):
        result = analyzer.process(frame(100 + direction * (step // 4)), step / 10)
        assert result.events == [], f"false overflow during lighting drift at step {step}"
        assert result.measurement["active"] is False
        assert result.measurement["changed_ratio"] == 0
    # Adaptation must still allow detection of an abrupt spill after the drift.
    assert analyzer.process(frame(250), 21).events == []
    assert analyzer.process(frame(250), 22).events == []
    assert len(analyzer.process(frame(250), 23).events) == 1


class FailingTracker:
    def update(self, image):
        raise RuntimeError("synthetic persistent failure")


class GatedTracker:
    """Hold one inference until released; all waits are bounded for safe cleanup."""

    def __init__(self, block_on=1):
        self.block_on = block_on
        self.started = threading.Event()
        self.release = threading.Event()
        self.seen = []

    def update(self, image):
        value = int(image[0, 0, 0])
        self.seen.append(value)
        if len(self.seen) == self.block_on:
            self.started.set()
            if not self.release.wait(1):
                raise RuntimeError("test did not release synthetic inference")
        return [Track(value, "synthetic", 1.0, (0, 0, 1, 1))]


@pytest.fixture
def async_factory():
    instances = []

    def create(tracker, **options):
        wrapper = AsyncTracker(tracker, **options)
        instances.append((wrapper, tracker))
        return wrapper

    yield create
    for wrapper, tracker in instances:
        if isinstance(tracker, GatedTracker):
            tracker.release.set()
        wrapper.stop()
        assert not wrapper._thread.is_alive(), "async worker leaked after cleanup"


def test_async_tracker_persistent_errors_keep_worker_alive_and_updates_nonblocking(async_factory):
    wrapper = async_factory(FailingTracker())
    for expected_errors in range(1, 5):
        started = time.monotonic()
        assert wrapper.update(frame()) == []
        assert time.monotonic() - started < 0.2, "update blocked on the failing tracker"
        wait_until(lambda n=expected_errors: wrapper.errors >= n)
        assert wrapper._thread.is_alive()
    assert wrapper.errors == 4
    assert wrapper.runs == 0
    assert wrapper.last_error == "synthetic persistent failure"


def test_async_tracker_stop_during_inference_joins_worker_and_discards_pending(async_factory):
    tracker = GatedTracker()
    wrapper = async_factory(tracker)
    wrapper.update(frame(1))
    assert tracker.started.wait(0.5)
    wrapper.update(frame(2))  # Pending work must not start after stop is requested.
    stopped = threading.Event()

    def stop():
        wrapper.stop()
        stopped.set()

    stopper = threading.Thread(target=stop, daemon=True)
    stopper.start()
    try:
        wait_until(lambda: wrapper._stop)
        assert not stopped.is_set(), "stop returned before the active inference finished"
        tracker.release.set()
        assert stopped.wait(0.5)
        assert not wrapper._thread.is_alive()
        assert tracker.seen == [1]
    finally:
        tracker.release.set()
        stopper.join(timeout=0.5)


def test_async_tracker_expires_cached_results_and_recovers_after_new_result(async_factory, monkeypatch):
    # Replace only this module's clock; do not alter the process-wide time module.
    clock = [1000.0]
    monkeypatch.setattr(async_module, "time", SimpleNamespace(time=lambda: clock[0]))
    tracker = GatedTracker(block_on=2)
    wrapper = async_factory(tracker, max_age_s=1)
    wrapper.update(frame(7))
    wait_until(lambda: wrapper.runs == 1)
    fresh = wrapper.update(frame(8))
    assert [track.track_id for track in fresh] == [7]
    assert tracker.started.wait(0.5)
    fresh.clear()  # A caller must not be able to clear the cached result list.
    clock[0] = 1001.0
    assert [track.track_id for track in wrapper.update(frame(8))] == [7]
    clock[0] = 1001.01
    assert wrapper.update(frame(8)) == [], "results older than max_age_s must be dropped"
    tracker.release.set()
    wait_until(lambda: wrapper.runs >= 2)
    assert [track.track_id for track in wrapper.update(frame(9))] == [8]


def test_async_tracker_concurrent_updates_are_nonblocking_and_keep_newest_pending(async_factory):
    tracker = GatedTracker()
    wrapper = async_factory(tracker)
    wrapper.update(frame(0))
    assert tracker.started.wait(0.5)
    start = threading.Event()
    errors = []
    completed = []

    def submit(value):
        try:
            assert start.wait(0.5)
            for _ in range(10):
                assert wrapper.update(frame(value)) == []
            completed.append(value)
        except Exception as exc:
            errors.append(exc)

    producers = [threading.Thread(target=submit, args=(value,), daemon=True) for value in range(1, 9)]
    for producer in producers:
        producer.start()
    try:
        deadline = time.monotonic() + 0.6
        start.set()
        for producer in producers:
            producer.join(timeout=max(0, deadline - time.monotonic()))
        assert all(not producer.is_alive() for producer in producers), "concurrent update blocked on inference"
        assert errors == []
        assert sorted(completed) == list(range(1, 9))
        assert tracker.seen == [0], "the underlying tracker must run only in the worker"
        assert wrapper.update(frame(99)) == []  # Deterministically the newest pending frame.
        tracker.release.set()
        wait_until(lambda: wrapper.runs == 2)
        assert tracker.seen == [0, 99], "pending frames must be replaced rather than queued"
        assert wrapper.errors == 0
        assert wrapper._thread.is_alive()
    finally:
        start.set()
        tracker.release.set()
        for producer in producers:
            producer.join(timeout=0.1)


@pytest.mark.parametrize(
    ("name", "options", "class_name"),
    [
        ("overflow", {"roii": ROI}, "Overflow"),
        ("overflow", {"roi": ROI, "roii": ROI}, "Overflow"),
        ("foam_coverage", {"roii": ROI}, "FoamCoverage"),
    ],
    ids=["misspelled-required-roi", "extra-misspelled-option", "foam-misspelled-option"],
)
def test_build_analyzers_reports_misspelled_parameter(name, options, class_name):
    with pytest.raises(TypeError, match="roii") as caught:
        build_analyzers({name: options})
    assert class_name in str(caught.value), "the error must identify which analyzer rejected roii"
