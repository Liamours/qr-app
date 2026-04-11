"""
Rigorous tests for the MediaPipe FaceMesh backend.

The entire module is skipped automatically when mediapipe is not installed.
Install with:
    pip install mediapipe
"""

import numpy as np
import pytest

pytest.importorskip("mediapipe", reason="mediapipe not installed — skipping mediapipe tests")

from conftest import LANDMARK_INDICES
from core.backends.mediapipe_backend import FaceMesh, _DetectionResult, _FaceLandmarksWrapper, _Landmark


# ── construction ──────────────────────────────────────────────────────────────

class TestMediaPipeInit:
    def test_instantiates(self):
        fm = FaceMesh()
        assert fm is not None

    def test_default_max_faces(self):
        # MediaPipe stores this inside face_mesh, not a direct attr —
        # just confirm no crash and the object is valid.
        fm = FaceMesh(max_faces=2)
        assert fm._face_mesh is not None

    def test_custom_max_faces(self):
        fm = FaceMesh(max_faces=4)
        assert fm is not None

    def test_extra_kwargs_accepted(self):
        FaceMesh(max_faces=1, min_detect=0.6, min_track=0.6)

    def test_close_releases_resources(self):
        fm = FaceMesh()
        fm.close()  # must not raise


# ── _FaceLandmarksWrapper ─────────────────────────────────────────────────────

class TestMediaPipeLandmarkWrapper:
    @pytest.fixture(scope="class")
    def fm(self):
        return FaceMesh(max_faces=2)

    def test_required_indices_present(self, fm, first_frame):
        result = fm.process(first_frame)
        if not result.face_landmarks:
            pytest.skip("No face detected in first_frame")
        face = result.face_landmarks[0]
        for idx in LANDMARK_INDICES:
            lm = face[idx]
            assert isinstance(lm, _Landmark)

    def test_landmarks_in_unit_range(self, fm, sample_frames):
        for frame in sample_frames:
            result = fm.process(frame)
            for face in result.face_landmarks:
                for idx in LANDMARK_INDICES:
                    lm = face[idx]
                    assert 0.0 <= lm.x <= 1.0, f"idx {idx} x={lm.x} out of [0,1]"
                    assert 0.0 <= lm.y <= 1.0, f"idx {idx} y={lm.y} out of [0,1]"

    def test_forehead_above_philtrum(self, fm, sample_frames):
        for frame in sample_frames:
            result = fm.process(frame)
            for face in result.face_landmarks:
                assert face[10].y < face[164].y, "Forehead (10) should be above philtrum (164)"

    def test_left_mouth_left_of_right_mouth(self, fm, sample_frames):
        for frame in sample_frames:
            result = fm.process(frame)
            for face in result.face_landmarks:
                assert face[61].x < face[291].x

    def test_left_cheek_left_of_right_cheek(self, fm, sample_frames):
        for frame in sample_frames:
            result = fm.process(frame)
            for face in result.face_landmarks:
                assert face[234].x < face[454].x

    def test_z_attribute_exists(self, fm, first_frame):
        result = fm.process(first_frame)
        if not result.face_landmarks:
            pytest.skip("No face detected")
        for idx in LANDMARK_INDICES:
            lm = result.face_landmarks[0][idx]
            assert hasattr(lm, "z")


# ── process() on real video frames ───────────────────────────────────────────

class TestMediaPipeProcessReal:
    @pytest.fixture(scope="class")
    def fm(self):
        return FaceMesh(max_faces=2)

    def test_returns_detection_result(self, fm, first_frame):
        result = fm.process(first_frame)
        assert isinstance(result, _DetectionResult)

    def test_detection_rate_on_sampled_frames(self, fm, sample_frames):
        detected = sum(1 for f in sample_frames if fm.process(f).face_landmarks)
        rate = detected / len(sample_frames)
        assert rate >= 0.70, (
            f"MediaPipe detection rate too low: {rate:.1%} on {len(sample_frames)} frames "
            f"(expected ≥70%). Every frame contains a face."
        )

    def test_detection_rate_on_all_frames(self, fm, all_frames):
        detected = sum(1 for f in all_frames if fm.process(f).face_landmarks)
        rate = detected / len(all_frames)
        assert rate >= 0.70, (
            f"MediaPipe full-video detection rate: {rate:.1%} "
            f"({detected}/{len(all_frames)} frames). Expected ≥70%."
        )

    def test_max_faces_respected(self, fm, sample_frames):
        for frame in sample_frames:
            result = fm.process(frame)
            assert len(result.face_landmarks) <= 2

    def test_repeated_calls_stable(self, fm, first_frame):
        counts = [len(fm.process(first_frame).face_landmarks) for _ in range(5)]
        assert len(set(counts)) == 1, f"Inconsistent results: {counts}"

    def test_468_landmarks_accessible(self, fm, first_frame):
        """MediaPipe provides 468 points — spot-check a broad range of indices."""
        result = fm.process(first_frame)
        if not result.face_landmarks:
            pytest.skip("No face detected")
        face = result.face_landmarks[0]
        for idx in [0, 100, 200, 300, 400, 467]:
            lm = face[idx]
            assert 0.0 <= lm.x <= 1.0
            assert 0.0 <= lm.y <= 1.0


# ── process() edge cases ──────────────────────────────────────────────────────

class TestMediaPipeEdgeCases:
    @pytest.fixture(scope="class")
    def fm(self):
        return FaceMesh()

    def test_black_frame_no_crash(self, fm, black_frame):
        result = fm.process(black_frame)
        assert isinstance(result, _DetectionResult)

    def test_black_frame_no_face(self, fm, black_frame):
        assert fm.process(black_frame).face_landmarks == []

    def test_noise_frame_no_crash(self, fm, noise_frame):
        assert isinstance(fm.process(noise_frame), _DetectionResult)

    def test_tiny_frame_no_crash(self, fm, tiny_frame):
        assert isinstance(fm.process(tiny_frame), _DetectionResult)

    def test_single_row_no_crash(self, fm):
        frame = np.zeros((1, 640, 3), dtype=np.uint8)
        assert isinstance(fm.process(frame), _DetectionResult)
