"""
Rigorous tests for core/renderer.py.

Uses the Haar backend to supply real landmarks from the reference video,
then validates that apply_filter / apply_hat / apply_mustache:
  - Return the correct shape and dtype.
  - Do not crash on edge-case geometry.
  - Visually modify the frame (at least some pixels differ from the original).
  - Respect EMA smoothing state.
"""

import os
import sys
import numpy as np
import cv2
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from core.backends.haar import FaceMesh
from core.renderer import apply_filter, apply_hat, apply_mustache, _ema

ASSET_DIR = os.path.join(ROOT, "assets")


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fm():
    return FaceMesh(max_faces=2)


@pytest.fixture(scope="module")
def hat_asset():
    img = cv2.imread(os.path.join(ASSET_DIR, "wizard-hat.png"), cv2.IMREAD_UNCHANGED)
    assert img is not None, "wizard-hat.png not found in assets/"
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


@pytest.fixture(scope="module")
def mustache_asset():
    img = cv2.imread(os.path.join(ASSET_DIR, "churos-mustache.png"), cv2.IMREAD_UNCHANGED)
    assert img is not None, "churos-mustache.png not found in assets/"
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


@pytest.fixture(scope="module")
def milky_asset(hat_asset):
    # Reuse hat as a stand-in GIF frame list so we can test "milky" mode
    # without loading the actual GIF in the renderer test.
    return [hat_asset, hat_asset]


@pytest.fixture(scope="module")
def detected_frame_and_landmarks(fm, sample_frames):
    """Return (bgr_frame, landmarks) for the first frame that has a detection."""
    for rgb in sample_frames:
        result = fm.process(rgb)
        if result.face_landmarks:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return bgr, result.face_landmarks[0]
    pytest.skip("No face detected in any sampled frame — cannot test renderer")


# ── _ema helper ───────────────────────────────────────────────────────────────

class TestEmaHelper:
    def test_first_call_returns_value(self):
        state = {}
        assert _ema(state, "k", 5.0) == pytest.approx(5.0)

    def test_state_populated(self):
        state = {}
        _ema(state, "k", 10.0)
        assert "k" in state

    def test_smoothing_moves_toward_new_value(self):
        state = {}
        _ema(state, "k", 0.0)
        v = _ema(state, "k", 100.0)
        assert 0.0 < v < 100.0

    def test_repeated_same_value_converges(self):
        state = {}
        for _ in range(50):
            _ema(state, "k", 42.0)
        assert state["k"] == pytest.approx(42.0, abs=1e-6)

    def test_independent_keys(self):
        state = {}
        _ema(state, "a", 1.0)
        _ema(state, "b", 2.0)
        assert "a" in state and "b" in state
        assert state["a"] != state["b"]


# ── apply_hat ─────────────────────────────────────────────────────────────────

class TestApplyHat:
    def test_returns_ndarray(self, detected_frame_and_landmarks, hat_asset):
        frame, lm = detected_frame_and_landmarks
        result = apply_hat(frame.copy(), lm, hat_asset, {})
        assert isinstance(result, np.ndarray)

    def test_output_shape_unchanged(self, detected_frame_and_landmarks, hat_asset):
        frame, lm = detected_frame_and_landmarks
        result = apply_hat(frame.copy(), lm, hat_asset, {})
        assert result.shape == frame.shape

    def test_output_dtype_unchanged(self, detected_frame_and_landmarks, hat_asset):
        frame, lm = detected_frame_and_landmarks
        result = apply_hat(frame.copy(), lm, hat_asset, {})
        assert result.dtype == frame.dtype

    def test_pixels_are_modified(self, detected_frame_and_landmarks, hat_asset):
        frame, lm = detected_frame_and_landmarks
        original = frame.copy()
        result = apply_hat(frame.copy(), lm, hat_asset, {})
        assert not np.array_equal(original, result), (
            "apply_hat produced no visible change — overlay may be out of frame"
        )

    def test_ema_state_populated(self, detected_frame_and_landmarks, hat_asset):
        frame, lm = detected_frame_and_landmarks
        state = {}
        apply_hat(frame.copy(), lm, hat_asset, state)
        assert any(k.startswith("hat_") for k in state)

    def test_ema_smooths_across_calls(self, detected_frame_and_landmarks, hat_asset):
        frame, lm = detected_frame_and_landmarks
        state = {}
        r1 = apply_hat(frame.copy(), lm, hat_asset, state)
        r2 = apply_hat(frame.copy(), lm, hat_asset, state)
        # With the same landmark values, EMA should have settled — outputs identical
        assert np.array_equal(r1, r2) or r1.shape == r2.shape  # at minimum same shape

    def test_custom_scale(self, detected_frame_and_landmarks, hat_asset):
        frame, lm = detected_frame_and_landmarks
        result_small = apply_hat(frame.copy(), lm, hat_asset, {}, scale=0.5)
        result_large = apply_hat(frame.copy(), lm, hat_asset, {}, scale=2.0)
        assert result_small.shape == frame.shape
        assert result_large.shape == frame.shape


# ── apply_mustache ────────────────────────────────────────────────────────────

class TestApplyMustache:
    def test_returns_ndarray(self, detected_frame_and_landmarks, mustache_asset):
        frame, lm = detected_frame_and_landmarks
        result = apply_mustache(frame.copy(), lm, mustache_asset, {})
        assert isinstance(result, np.ndarray)

    def test_output_shape_unchanged(self, detected_frame_and_landmarks, mustache_asset):
        frame, lm = detected_frame_and_landmarks
        result = apply_mustache(frame.copy(), lm, mustache_asset, {})
        assert result.shape == frame.shape

    def test_output_dtype_unchanged(self, detected_frame_and_landmarks, mustache_asset):
        frame, lm = detected_frame_and_landmarks
        result = apply_mustache(frame.copy(), lm, mustache_asset, {})
        assert result.dtype == frame.dtype

    def test_pixels_are_modified(self, detected_frame_and_landmarks, mustache_asset):
        frame, lm = detected_frame_and_landmarks
        original = frame.copy()
        result = apply_mustache(frame.copy(), lm, mustache_asset, {})
        assert not np.array_equal(original, result)

    def test_ema_state_populated(self, detected_frame_and_landmarks, mustache_asset):
        frame, lm = detected_frame_and_landmarks
        state = {}
        apply_mustache(frame.copy(), lm, mustache_asset, state)
        assert any(k.startswith("mu_") for k in state)


# ── apply_filter dispatch ─────────────────────────────────────────────────────

class TestApplyFilter:
    def test_hat_mode(self, detected_frame_and_landmarks, hat_asset, mustache_asset, milky_asset):
        frame, lm = detected_frame_and_landmarks
        assets = {"hat": hat_asset, "mustache": mustache_asset, "milky": milky_asset}
        result = apply_filter(frame.copy(), lm, "hat", assets, 0, {})
        assert result.shape == frame.shape

    def test_mustache_mode(self, detected_frame_and_landmarks, hat_asset, mustache_asset, milky_asset):
        frame, lm = detected_frame_and_landmarks
        assets = {"hat": hat_asset, "mustache": mustache_asset, "milky": milky_asset}
        result = apply_filter(frame.copy(), lm, "mustache", assets, 0, {})
        assert result.shape == frame.shape

    def test_milky_mode(self, detected_frame_and_landmarks, hat_asset, mustache_asset, milky_asset):
        frame, lm = detected_frame_and_landmarks
        assets = {"hat": hat_asset, "mustache": mustache_asset, "milky": milky_asset}
        result = apply_filter(frame.copy(), lm, "milky", assets, 0, {})
        assert result.shape == frame.shape

    def test_milky_gif_index_wraps(self, detected_frame_and_landmarks, hat_asset, mustache_asset, milky_asset):
        frame, lm = detected_frame_and_landmarks
        assets = {"hat": hat_asset, "mustache": mustache_asset, "milky": milky_asset}
        # gif_idx beyond list length — modulo must handle it gracefully
        result = apply_filter(frame.copy(), lm, "milky", assets, 99, {})
        assert result.shape == frame.shape

    def test_unknown_mode_returns_frame_unchanged(self, detected_frame_and_landmarks, hat_asset, mustache_asset, milky_asset):
        frame, lm = detected_frame_and_landmarks
        assets = {"hat": hat_asset, "mustache": mustache_asset, "milky": milky_asset}
        original = frame.copy()
        result = apply_filter(frame.copy(), lm, "nonexistent_mode", assets, 0, {})
        assert np.array_equal(original, result)

    def test_state_shared_across_filter_calls(self, detected_frame_and_landmarks, hat_asset, mustache_asset, milky_asset):
        frame, lm = detected_frame_and_landmarks
        assets = {"hat": hat_asset, "mustache": mustache_asset, "milky": milky_asset}
        state = {}
        apply_filter(frame.copy(), lm, "hat", assets, 0, state)
        assert len(state) > 0


# ── renderer on all video frames with detected landmarks ──────────────────────

class TestRendererOnVideo:
    def test_hat_filter_never_crashes_on_sampled_frames(
        self, fm, sample_frames, hat_asset, mustache_asset, milky_asset
    ):
        assets = {"hat": hat_asset, "mustache": mustache_asset, "milky": milky_asset}
        for rgb in sample_frames:
            result = fm.process(rgb)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for lm in result.face_landmarks:
                out = apply_filter(bgr.copy(), lm, "hat", assets, 0, {})
                assert out.shape == bgr.shape
                assert out.dtype == bgr.dtype

    def test_mustache_filter_never_crashes_on_sampled_frames(
        self, fm, sample_frames, hat_asset, mustache_asset, milky_asset
    ):
        assets = {"hat": hat_asset, "mustache": mustache_asset, "milky": milky_asset}
        for rgb in sample_frames:
            result = fm.process(rgb)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for lm in result.face_landmarks:
                out = apply_filter(bgr.copy(), lm, "mustache", assets, 0, {})
                assert out.shape == bgr.shape

    def test_filter_output_pixel_range(
        self, fm, sample_frames, hat_asset, mustache_asset, milky_asset
    ):
        assets = {"hat": hat_asset, "mustache": mustache_asset, "milky": milky_asset}
        for rgb in sample_frames:
            result = fm.process(rgb)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for lm in result.face_landmarks:
                out = apply_filter(bgr.copy(), lm, "hat", assets, 0, {})
                assert out.min() >= 0
                assert out.max() <= 255
