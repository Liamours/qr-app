"""
Shared fixtures for the face-filter backend tests.

VIDEO_PATH  — the reference clip; every frame contains a face.
SAMPLE_STEP — stride used when sampling frames (keeps the suite fast).
LANDMARK_INDICES — the six indices renderer.py depends on.
"""

import sys
import os
import pytest
import cv2
import numpy as np

# Make the project root importable regardless of where pytest is invoked from.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

VIDEO_PATH = os.path.join(os.path.dirname(__file__), "video", "WIN_20260216_18_00_23_Pro.mp4")
SAMPLE_STEP = 30          # ~1 frame per second at 30 fps → 28 frames
LANDMARK_INDICES = [10, 234, 454, 61, 291, 164]


def _load_frames(path: str, step: int) -> list:
    cap = cv2.VideoCapture(path)
    assert cap.isOpened(), f"Cannot open video: {path}"
    frames = []
    idx = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        if idx % step == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
        idx += 1
    cap.release()
    return frames


@pytest.fixture(scope="session")
def sample_frames():
    """~28 RGB frames sampled at 1 fps from the reference video."""
    frames = _load_frames(VIDEO_PATH, SAMPLE_STEP)
    assert len(frames) > 0, "No frames could be read from the video."
    return frames


@pytest.fixture(scope="session")
def all_frames():
    """All 857 RGB frames — used for high-confidence detection-rate checks."""
    frames = _load_frames(VIDEO_PATH, 1)
    assert len(frames) > 0
    return frames


@pytest.fixture(scope="session")
def first_frame(sample_frames):
    return sample_frames[0]


# ── synthetic edge-case frames ────────────────────────────────────────────────

@pytest.fixture(scope="session")
def black_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture(scope="session")
def tiny_frame():
    return np.zeros((32, 32, 3), dtype=np.uint8)


@pytest.fixture(scope="session")
def noise_frame():
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
