import av
import cv2
import numpy as np
from streamlit_webrtc import VideoProcessorBase
from core.facemesh import FaceMesh
from core.renderer import draw_landmarks, apply_hat, apply_sunglasses


class FaceFilterProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = FaceMesh()
        self.show_mesh = False
        self.active_filter = "none"
        self.hat_asset = None
        self.glasses_asset = None
        self.hat_scale = 1.3
        self.glasses_scale = 1.1

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if self.show_mesh:
                    img = draw_landmarks(img, results)

                if self.active_filter == "hat" and self.hat_asset is not None:
                    img = apply_hat(img, face_landmarks, self.hat_asset, self.hat_scale)

                elif self.active_filter == "sunglasses" and self.glasses_asset is not None:
                    img = apply_sunglasses(img, face_landmarks, self.glasses_asset, self.glasses_scale)

                elif self.active_filter == "both":
                    if self.hat_asset is not None:
                        img = apply_hat(img, face_landmarks, self.hat_asset, self.hat_scale)
                    if self.glasses_asset is not None:
                        img = apply_sunglasses(img, face_landmarks, self.glasses_asset, self.glasses_scale)

        return av.VideoFrame.from_ndarray(img, format="bgr24")
