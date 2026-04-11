"""
Rigorous tests for the Haar Cascade backend.

Detection-rate threshold is intentionally conservative (≥25%) because Haar
is sensitive to pose and lighting, but every sampled frame contains a face so
a well-tuned cascade should detect most of them.
"""

import numpy as np
import pytest
from conftest import LANDMARK_INDICES

from core.backends.haar import FaceMesh, _DetectionResult, _FaceLandmarks, _Landmark


# ── construction ──────────────────────────────────────────────────────────────

class TestHaarInit:
    def test_instantiates(self):
        fm = FaceMesh()
        assert fm is not None

    def test_default_max_faces(self):
        fm = FaceMesh()
        assert fm._max_faces == 2

    def test_custom_max_faces(self):
        fm = FaceMesh(max_faces=5)
        assert fm._max_faces == 5

    def test_detector_loaded(self):
        fm = FaceMesh()
        assert fm._detector is not None, (
            "Haar cascade failed to load — check opencv-python-headless installation"
        )

    def test_extra_kwargs_accepted(self):
        # Dispatcher may pass keyword args meant for other backends.
        FaceMesh(max_faces=1, min_detect=0.5, min_track=0.5)

    def test_close_is_noop(self):
        fm = FaceMesh()
        fm.close()  # must not raise


# ── _DetectionResult ──────────────────────────────────────────────────────────

class TestDetectionResult:
    def test_empty_result(self):
        r = _DetectionResult([])
        assert r.face_landmarks == []

    def test_result_holds_landmarks(self):
        r = _DetectionResult(["a", "b"])
        assert len(r.face_landmarks) == 2


# ── _FaceLandmarks ────────────────────────────────────────────────────────────

class TestFaceLandmarks:
    def _make(self, x=100, y=50, w=200, h=220, img_w=640, img_h=480):
        return _FaceLandmarks(x, y, w, h, img_w, img_h)

    def test_all_required_indices_present(self):
        fl = self._make()
        for idx in LANDMARK_INDICES:
            lm = fl[idx]
            assert isinstance(lm, _Landmark), f"Index {idx} returned wrong type"

    def test_landmarks_normalised_in_unit_range(self):
        fl = self._make()
        for idx in LANDMARK_INDICES:
            lm = fl[idx]
            assert 0.0 <= lm.x <= 1.0, f"idx {idx}: x={lm.x} out of [0,1]"
            assert 0.0 <= lm.y <= 1.0, f"idx {idx}: y={lm.y} out of [0,1]"

    def test_z_defaults_to_zero(self):
        fl = self._make()
        for idx in LANDMARK_INDICES:
            assert fl[idx].z == 0.0

    def test_left_edge_x_is_zero(self):
        # Index 234 uses fx=0.0 → x == bbox_left / img_w
        fl = self._make(x=0, y=0, w=200, h=200, img_w=640, img_h=480)
        assert fl[234].x == pytest.approx(0.0)

    def test_right_edge_x_is_one(self):
        # Index 454 uses fx=1.0 → x == (bbox_left + bbox_w) / img_w
        fl = self._make(x=440, y=0, w=200, h=200, img_w=640, img_h=480)
        assert fl[454].x == pytest.approx(1.0)

    def test_forehead_y_less_than_mouth_y(self):
        fl = self._make()
        assert fl[10].y < fl[164].y, "Forehead should be above philtrum"

    def test_left_mouth_x_less_than_right_mouth_x(self):
        fl = self._make()
        assert fl[61].x < fl[291].x

    def test_left_edge_x_less_than_right_edge_x(self):
        fl = self._make()
        assert fl[234].x < fl[454].x


# ── process() on real video frames ───────────────────────────────────────────

class TestHaarProcessReal:
    @pytest.fixture(scope="class")
    def fm(self):
        return FaceMesh(max_faces=2)

    def test_returns_detection_result(self, fm, first_frame):
        result = fm.process(first_frame)
        assert isinstance(result, _DetectionResult)

    def test_face_landmarks_is_list(self, fm, first_frame):
        result = fm.process(first_frame)
        assert isinstance(result.face_landmarks, list)

    def test_detected_landmarks_have_required_indices(self, fm, sample_frames):
        for frame in sample_frames:
            result = fm.process(frame)
            for face in result.face_landmarks:
                for idx in LANDMARK_INDICES:
                    lm = face[idx]
                    assert 0.0 <= lm.x <= 1.0
                    assert 0.0 <= lm.y <= 1.0

    def test_detection_rate_on_sampled_frames(self, fm, sample_frames):
        detected = sum(1 for f in sample_frames if fm.process(f).face_landmarks)
        rate = detected / len(sample_frames)
        assert rate >= 0.25, (
            f"Haar detection rate too low: {rate:.1%} on {len(sample_frames)} frames "
            f"(expected ≥25%). Detected in {detected}/{len(sample_frames)} frames."
        )

    def test_max_faces_respected(self, fm, sample_frames):
        for frame in sample_frames:
            result = fm.process(frame)
            assert len(result.face_landmarks) <= 2

    def test_detection_rate_on_all_frames(self, fm, all_frames):
        detected = sum(1 for f in all_frames if fm.process(f).face_landmarks)
        rate = detected / len(all_frames)
        assert rate >= 0.20, (
            f"Haar full-video detection rate: {rate:.1%} "
            f"({detected}/{len(all_frames)} frames). Expected ≥20%."
        )


# ── process() on synthetic / edge-case frames ─────────────────────────────────

class TestHaarProcessEdgeCases:
    @pytest.fixture(scope="class")
    def fm(self):
        return FaceMesh()

    def test_black_frame_returns_empty_or_result(self, fm, black_frame):
        result = fm.process(black_frame)
        assert isinstance(result, _DetectionResult)

    def test_black_frame_no_face(self, fm, black_frame):
        result = fm.process(black_frame)
        assert result.face_landmarks == []

    def test_noise_frame_does_not_crash(self, fm, noise_frame):
        result = fm.process(noise_frame)
        assert isinstance(result, _DetectionResult)

    def test_tiny_frame_does_not_crash(self, fm, tiny_frame):
        result = fm.process(tiny_frame)
        assert isinstance(result, _DetectionResult)

    def test_single_row_frame(self, fm):
        frame = np.zeros((1, 640, 3), dtype=np.uint8)
        result = fm.process(frame)
        assert isinstance(result, _DetectionResult)

    def test_non_square_frame(self, fm, sample_frames):
        frame = sample_frames[0]
        h, w = frame.shape[:2]
        cropped = frame[:h, :w // 2]
        result = fm.process(cropped)
        assert isinstance(result, _DetectionResult)

    def test_repeated_calls_are_stable(self, fm, first_frame):
        results = [fm.process(first_frame) for _ in range(5)]
        counts = [len(r.face_landmarks) for r in results]
        # Should give the same count each time (deterministic cascade)
        assert len(set(counts)) == 1, f"Inconsistent results across calls: {counts}"
