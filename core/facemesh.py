"""
Face detection via OpenCV Haar cascade.

Replaces MediaPipe entirely — no native shared-library dependencies,
no model downloads, works on every platform OpenCV supports.

The public interface (FaceMesh.process → result.face_landmarks) is
kept identical to the previous MediaPipe wrapper so that processor.py
and renderer.py need no changes.
"""

import cv2


class _Landmark:
    """Normalised (x, y) point — same attribute interface as MediaPipe."""
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


class _FaceLandmarks:
    """
    Sparse landmark estimates derived from a face bounding box.

    Only the indices consumed by renderer.py are defined; every value
    is a fraction of (box_origin + box_size) normalised to the frame.

    Index reference (MediaPipe 468-point topology):
        10  → forehead top-centre
        234 → left cheek edge
        454 → right cheek edge
        61  → left mouth corner
        291 → right mouth corner
        164 → philtrum (just below nose)
    """

    _FRACS: dict[int, tuple[float, float]] = {
        10:  (0.50, 0.02),
        234: (0.00, 0.45),
        454: (1.00, 0.45),
        61:  (0.28, 0.75),
        291: (0.72, 0.75),
        164: (0.50, 0.68),
    }

    def __init__(self, x: int, y: int, w: int, h: int,
                 img_w: int, img_h: int) -> None:
        self._lm: dict[int, _Landmark] = {
            idx: _Landmark((x + w * fx) / img_w, (y + h * fy) / img_h)
            for idx, (fx, fy) in self._FRACS.items()
        }

    def __getitem__(self, idx: int) -> _Landmark:
        return self._lm[idx]


class _DetectionResult:
    __slots__ = ("face_landmarks",)

    def __init__(self, face_landmarks: list) -> None:
        self.face_landmarks = face_landmarks


class FaceMesh:
    """
    Drop-in replacement for the MediaPipe FaceLandmarker wrapper.

    Uses OpenCV's built-in frontal-face Haar cascade — always available
    with opencv-python-headless, no extra packages required.
    """

    def __init__(self, max_faces: int = 2,
                 min_detect: float = 0.5,
                 min_track: float = 0.5) -> None:
        cascade_path = (
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._detector = cv2.CascadeClassifier(cascade_path)
        self._max_faces = max_faces

    def process(self, rgb_frame) -> _DetectionResult:
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        img_h, img_w = gray.shape

        faces = self._detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        landmarks: list[_FaceLandmarks] = []
        if len(faces) > 0:
            for (x, y, w, h) in faces[: self._max_faces]:
                landmarks.append(_FaceLandmarks(x, y, w, h, img_w, img_h))

        return _DetectionResult(landmarks)

    def close(self) -> None:
        pass  # Haar cascade holds no persistent resources
