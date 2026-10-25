"""Grayscale LUT and the INTER_AREA 240x256 -> 84x84 port vs OpenCV."""
import numpy as np
import pytest
from conftest import scaled

import fast_mario_env as fme

cv2 = pytest.importorskip("cv2")


def test_gray_lut_matches_cv2():
    want = cv2.cvtColor(fme.palette_rgb()[None], cv2.COLOR_RGB2GRAY)[0]
    assert np.array_equal(fme.gray_lut()[:64], want)


@pytest.mark.parametrize("fast", [False, True])
def test_resize_matches_cv2(fast):
    rng = np.random.default_rng(0)
    lut = fme.gray_lut()
    bad = 0
    n = scaled(500)
    for t in range(n):
        if t % 2:
            img = rng.integers(0, 256, (240, 256), np.uint8)
        else:  # palette-like blocky images, like real NES frames
            img = lut[rng.integers(0, 64, (30, 32))].repeat(8, 0).repeat(8, 1)
            img = np.roll(img, rng.integers(0, 8, 2), (0, 1)).copy()
        want = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        bad += not np.array_equal(fme.resize_area(img, fast=fast), want)
    assert bad == 0, f"{bad}/{n} images differ (cv2 {cv2.__version__})"
