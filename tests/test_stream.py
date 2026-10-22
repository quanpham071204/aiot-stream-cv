"""StreamSource: reconnect, buffering, health and end-of-file behaviour."""
from fakes import FakeCapture, wait_until

from aiot.stream import State, StreamSource


def make(open_fn, **kw):
    return StreamSource("t", "rtsp://fake", reconnect_initial_s=0.01, reconnect_max_s=0.05, open_fn=open_fn, **kw)


def test_reconnects_after_failed_opens():
    attempts = []

    def open_fn(_uri):
        attempts.append(1)
        return FakeCapture(5, opened=len(attempts) > 2)

    s = make(open_fn, buffer_size=100).start()
    try:
        assert wait_until(lambda: s.frames_received >= 5)
        assert s.reconnects >= 2
        assert s.read(timeout=1).seq == 1
    finally:
        s.stop()


def test_recovers_after_stream_drop():
    s = make(lambda _uri: FakeCapture(3), buffer_size=100).start()
    try:
        # each capture dies after 3 frames; source must keep reconnecting and delivering
        assert wait_until(lambda: s.frames_received >= 9)
        assert s.reconnects >= 2
        assert s.last_error == "read failed"
    finally:
        s.stop()


def test_buffer_drops_oldest_when_consumer_slow():
    s = make(lambda _uri: FakeCapture(10), buffer_size=2).start()
    try:
        assert wait_until(lambda: s.frames_received >= 10)
        assert s.frames_dropped >= 8
        assert s.read(timeout=1).seq >= 9  # newest frames kept
    finally:
        s.stop()


def test_health_goes_stale_when_no_frames():
    s = make(lambda _uri: FakeCapture(0, opened=False), stale_after_s=0.05).start()
    try:
        assert wait_until(lambda: s.reconnects >= 1)
        h = s.health()
        assert h["healthy"] is False and h["state"] in (State.RECONNECTING, State.CONNECTING)
        assert h["last_error"] == "open failed"
    finally:
        s.stop()


def test_file_without_loop_ends(tmp_path):
    s = StreamSource("f", str(tmp_path / "clip.avi"), realtime=False, open_fn=lambda _u: FakeCapture(3))
    s.start()
    try:
        assert wait_until(lambda: s.state == State.ENDED)
        assert s.frames_received == 3
    finally:
        s.stop()
