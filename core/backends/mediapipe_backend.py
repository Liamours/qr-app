"""
Backend: MediaPipe FaceMesh (468 landmarks).

Requirements:
  pip install mediapipe

Landmark indices used by renderer.py are native MediaPipe indices,
so no remapping is needed:
  10  → forehead top-centre
  234 → left cheek edge
  454 → right cheek edge
  61  → left mouth corner
  291 → right mouth corner
  164 → philtrum (below nose)
"""


class _Landmark:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


class _FaceLandmarksWrapper:
    """Thin wrapper so renderer.py can do landmarks[idx] identically."""

    def __init__(self, mp_face_landmarks) -> None:
        self._lm = mp_face_landmarks

    def __getitem__(self, idx: int) -> _Landmark:
        lm = self._lm.landmark[idx]
        return _Landmark(lm.x, lm.y, lm.z)


class _DetectionResult:
    __slots__ = ("face_landmarks",)

    def __init__(self, face_landmarks: list) -> None:
        self.face_landmarks = face_landmarks


_EMPTY = _DetectionResult([])


class FaceMesh:
    def __init__(self, max_faces: int = 2,
                 min_detect: float = 0.5,
                 min_track: float = 0.5,
                 **kwargs) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError(
                "mediapipe is not installed. Run: pip install mediapipe"
            ) from exc

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=False,
            min_detection_confidence=min_detect,
            min_tracking_confidence=min_track,
        )

    def process(self, rgb_frame) -> _DetectionResult:
        try:
            result = self._face_mesh.process(rgb_frame)
            if not result.multi_face_landmarks:
                return _EMPTY
            wrapped = [
                _FaceLandmarksWrapper(lm)
                for lm in result.multi_face_landmarks
            ]
            return _DetectionResult(wrapped)
        except Exception:
            return _EMPTY

    def close(self) -> None:
        self._face_mesh.close()
