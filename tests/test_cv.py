import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "flaskserver"))

import cv as flaskcv

def load_checkerboard():
    path = Path(__file__).parent / f"resources/checkerboard-11r-17c-50s.png"
    img = cv2.imread(str(path))
    # to rgb
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

class TestDetectCorners:
    def test_no_corners_on_blank_image(self):
        blank = np.full((200, 300, 3), 128, dtype=np.uint8)
        result = flaskcv.detect_corners(blank)
        assert result is None

    def test_no_corners_on_noise(self):
        rng = np.random.default_rng(42)
        noise = rng.integers(0, 256, (200, 300, 3), dtype=np.uint8)
        result = flaskcv.detect_corners(noise)
        assert result is None

    def test_corners_on_chessboard(self):
        corners_per_row = 11-1
        corners_per_col = 17-1
        img = load_checkerboard()
        corners = flaskcv.detect_corners(img)
        assert corners is not None
        assert corners.shape == (corners_per_row * corners_per_col, 1, 2)
