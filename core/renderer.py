import cv2
import numpy as np

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


def overlay_asset(frame, asset_img, cx, ty, face_width, angle, scale):
    w = int(face_width * scale)
    h = int(w * asset_img.shape[0] / asset_img.shape[1])
    if w <= 0 or h <= 0:
        return frame

    asset_resized = cv2.resize(asset_img, (w, h))
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    asset_rotated = cv2.warpAffine(asset_resized, M, (w, h))

    x1, y1 = cx - w // 2, ty - h
    x2, y2 = x1 + w, y1 + h

    x1c, y1c = max(x1, 0), max(y1, 0)
    x2c, y2c = min(x2, frame.shape[1]), min(y2, frame.shape[0])
    if x1c >= x2c or y1c >= y2c:
        return frame

    ax1, ay1 = x1c - x1, y1c - y1
    ax2, ay2 = ax1 + (x2c - x1c), ay1 + (y2c - y1c)

    alpha = asset_rotated[ay1:ay2, ax1:ax2, 3:4] / 255.0
    roi   = frame[y1c:y2c, x1c:x2c]
    rgb   = asset_rotated[ay1:ay2, ax1:ax2, :3]

    frame[y1c:y2c, x1c:x2c] = (alpha * rgb + (1 - alpha) * roi).astype(np.uint8)
    return frame


def _lm(landmarks, idx, fw, fh):
    lm = landmarks[idx]
    return int(lm.x * fw), int(lm.y * fh)


def draw_landmarks(frame, detection_result):
    if not detection_result.face_landmarks:
        return frame
    fh, fw = frame.shape[:2]
    for landmarks in detection_result.face_landmarks:
        for lm in landmarks:
            cx, cy = int(lm.x * fw), int(lm.y * fh)
            cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)
    return frame


def apply_hat(frame, landmarks, asset_img, scale=1.3):
    fh, fw = frame.shape[:2]
    lx, ly = _lm(landmarks, 234, fw, fh)
    rx, ry = _lm(landmarks, 454, fw, fh)
    tx, ty = _lm(landmarks, 10,  fw, fh)

    face_width = int(np.linalg.norm([rx - lx, ry - ly]))
    angle = -np.degrees(np.arctan2(ry - ly, rx - lx))
    cx = (lx + rx) // 2

    return overlay_asset(frame, asset_img, cx, ty, face_width, angle, scale)


def apply_sunglasses(frame, landmarks, asset_img, scale=1.1):
    fh, fw = frame.shape[:2]
    lx, ly = _lm(landmarks, 33,  fw, fh)
    rx, ry = _lm(landmarks, 263, fw, fh)

    eye_width = int(np.linalg.norm([rx - lx, ry - ly]))
    angle = -np.degrees(np.arctan2(ry - ly, rx - lx))
    cx = (lx + rx) // 2
    cy = (ly + ry) // 2

    glasses_w = int(eye_width * scale)
    glasses_h = int(glasses_w * asset_img.shape[0] / asset_img.shape[1])
    if glasses_w <= 0 or glasses_h <= 0:
        return frame

    asset_resized = cv2.resize(asset_img, (glasses_w, glasses_h))
    M = cv2.getRotationMatrix2D((glasses_w // 2, glasses_h // 2), angle, 1.0)
    asset_rotated = cv2.warpAffine(asset_resized, M, (glasses_w, glasses_h))

    x1, y1 = cx - glasses_w // 2, cy - glasses_h // 2
    x2, y2 = x1 + glasses_w, y1 + glasses_h

    x1c, y1c = max(x1, 0), max(y1, 0)
    x2c, y2c = min(x2, frame.shape[1]), min(y2, frame.shape[0])
    if x1c >= x2c or y1c >= y2c:
        return frame

    ax1, ay1 = x1c - x1, y1c - y1
    ax2, ay2 = ax1 + (x2c - x1c), ay1 + (y2c - y1c)

    alpha = asset_rotated[ay1:ay2, ax1:ax2, 3:4] / 255.0
    roi   = frame[y1c:y2c, x1c:x2c]
    rgb   = asset_rotated[ay1:ay2, ax1:ax2, :3]

    frame[y1c:y2c, x1c:x2c] = (alpha * rgb + (1 - alpha) * roi).astype(np.uint8)
    return frame

def apply_mustache(frame, landmarks, asset_img, scale=1.5):
    fh, fw = frame.shape[:2]
    lx, ly = _lm(landmarks, 61,  fw, fh)
    rx, ry = _lm(landmarks, 291, fw, fh)
    _, ny = _lm(landmarks, 164, fw, fh)

    mouth_width = int(np.linalg.norm([rx - lx, ry - ly]))
    angle = -np.degrees(np.arctan2(ry - ly, rx - lx))
    cx = (lx + rx) // 2
    cy = ny

    mw = int(mouth_width * scale)
    mh = int(mw * asset_img.shape[0] / asset_img.shape[1])
    if mw <= 0 or mh <= 0:
        return frame

    asset_resized = cv2.resize(asset_img, (mw, mh))
    M = cv2.getRotationMatrix2D((mw // 2, mh // 2), angle, 1.0)
    asset_rotated = cv2.warpAffine(asset_resized, M, (mw, mh))

    x1, y1 = cx - mw // 2, cy - mh // 2
    x2, y2 = x1 + mw, y1 + mh

    x1c, y1c = max(x1, 0), max(y1, 0)
    x2c, y2c = min(x2, frame.shape[1]), min(y2, frame.shape[0])
    if x1c >= x2c or y1c >= y2c:
        return frame

    ax1, ay1 = x1c - x1, y1c - y1
    ax2, ay2 = ax1 + (x2c - x1c), ay1 + (y2c - y1c)

    alpha = asset_rotated[ay1:ay2, ax1:ax2, 3:4] / 255.0
    roi   = frame[y1c:y2c, x1c:x2c]
    rgb   = asset_rotated[ay1:ay2, ax1:ax2, :3]

    frame[y1c:y2c, x1c:x2c] = (alpha * rgb + (1 - alpha) * roi).astype(np.uint8)
    return frame


def apply_gif(frame, landmarks, gif_frames, frame_idx, scale=1.8):
    asset_img = gif_frames[frame_idx % len(gif_frames)]
    return apply_hat(frame, landmarks, asset_img, scale=scale)