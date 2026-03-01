import av
import cv2
import threading
from core.facemesh import FaceMesh
from core.renderer import apply_hat, apply_mustache, apply_gif
from streamlit_webrtc import VideoProcessorBase


class FaceFilterProcessor(VideoProcessorBase):
    def __init__(self, assets):
        self.face_mesh = FaceMesh()
        self.assets = assets
        self.mode = "hat"
        self._lock = threading.Lock()
        self._snapshot = None
        self._gif_idx = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        if result.face_landmarks:
            for landmarks in result.face_landmarks:
                if self.mode == "hat":
                    img = apply_hat(img, landmarks, self.assets["hat"])
                elif self.mode == "mustache":
                    img = apply_mustache(img, landmarks, self.assets["mustache"])
                elif self.mode == "milky":
                    img = apply_gif(img, landmarks, self.assets["milky"], self._gif_idx)

        if self.mode == "milky":
            self._gif_idx += 1

        with self._lock:
            self._snapshot = img.copy()

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def get_snapshot(self):
        with self._lock:
            return self._snapshot
