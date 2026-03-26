import mediapipe as mp


class _DetectionResult:
    """Adapts legacy multi_face_landmarks to the interface expected by processor.py."""
    __slots__ = ("face_landmarks",)

    def __init__(self, multi_face_landmarks):
        # Each face.landmark is a RepeatedCompositeContainer whose elements
        # expose .x, .y, .z — same interface as the Tasks API NormalizedLandmark.
        self.face_landmarks = (
            [face.landmark for face in multi_face_landmarks]
            if multi_face_landmarks
            else []
        )


class FaceMesh:
    def __init__(self, max_faces: int = 2, min_detect: float = 0.5, min_track: float = 0.5):
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=False,
            min_detection_confidence=min_detect,
            min_tracking_confidence=min_track,
        )

    def process(self, rgb_frame) -> _DetectionResult:
        results = self._mesh.process(rgb_frame)
        return _DetectionResult(results.multi_face_landmarks)

    def close(self) -> None:
        self._mesh.close()
