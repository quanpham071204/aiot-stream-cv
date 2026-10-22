"""Test doubles shared by the test modules (no camera or model needed)."""
import time

import numpy as np


class FakeCapture:
    """Yields `frames` frames then fails a read (simulates a camera dropping)."""

    def __init__(self, frames: int, opened: bool = True):
        self.left = frames
        self.opened = opened

    def isOpened(self):  # noqa: N802 - mirrors cv2 API
        return self.opened

    def read(self):
        if self.left <= 0:
            return False, None
        self.left -= 1
        return True, np.zeros((4, 4, 3), np.uint8)

    def get(self, _prop):
        return 0

    def release(self):
        pass


def wait_until(cond, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False
