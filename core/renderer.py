"""
Overlay renderer — applies hat, mustache, or GIF filters onto a frame.

EMA smoothing state is passed in/out explicitly so that each
FaceFilterProcessor instance owns its own state (no shared globals).
"""

import cv2
import numpy as np

_EMA_ALPHA = 0.4


def _ema(state: dict, key: str, value: float) -> float:
    if key not in state:
        state[key] = value
    else:
        state[key] = _EMA_ALPHA * value + (1 - _EMA_ALPHA) * state[key]
    return state[key]


def _lm(landmarks, idx: int, fw: int, fh: int):
    lm = landmarks[idx]
    return int(lm.x * fw), int(lm.y * fh)


def _overlay(frame, asset_img, cx: int, ty: int,
             face_width: int, angle: float, scale: float):
    w = int(face_width * scale)
    h = int(w * asset_img.shape[0] / asset_img.shape[1])
    if w <= 0 or h <= 0:
        return frame

    asset_resized = cv2.resize(asset_img, (w, h))
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    asset_rot = cv2.warpAffine(asset_resized, M, (w, h))

    x1, y1 = cx - w // 2, ty - h
    x2, y2 = x1 + w, y1 + h

    x1c = max(x1, 0);  y1c = max(y1, 0)
    x2c = min(x2, frame.shape[1]);  y2c = min(y2, frame.shape[0])
    if x1c >= x2c or y1c >= y2c:
        return frame

    ax1, ay1 = x1c - x1, y1c - y1
    ax2, ay2 = ax1 + (x2c - x1c), ay1 + (y2c - y1c)

    alpha = asset_rot[ay1:ay2, ax1:ax2, 3:4] / 255.0
    roi   = frame[y1c:y2c, x1c:x2c]
    rgb   = asset_rot[ay1:ay2, ax1:ax2, :3]
    frame[y1c:y2c, x1c:x2c] = (alpha * rgb + (1 - alpha) * roi).astype(np.uint8)
    return frame


def _hat_angle(landmarks, fw: int, fh: int) -> float:
    lx, ly = _lm(landmarks, 234, fw, fh)
    rx, ry = _lm(landmarks, 454, fw, fh)
    return -np.degrees(np.arctan2(ry - ly, rx - lx))


def apply_hat(frame, landmarks, asset_img, state: dict, scale: float = 1.4):
    fh, fw = frame.shape[:2]
    lx, ly = _lm(landmarks, 234, fw, fh)
    rx, ry = _lm(landmarks, 454, fw, fh)
    tx, ty = _lm(landmarks, 10,  fw, fh)

    face_width = int(np.linalg.norm([rx - lx, ry - ly]))
    raw_cx    = (lx + rx) // 2
    raw_ty    = ty - int(face_width * 0.05)
    raw_angle = _hat_angle(landmarks, fw, fh)

    cx    = int(_ema(state, "hat_cx",    raw_cx))
    ty_s  = int(_ema(state, "hat_ty",    raw_ty))
    angle = float(_ema(state, "hat_ang", raw_angle))
    fw_s  = int(_ema(state, "hat_fw",    face_width))

    return _overlay(frame, asset_img, cx, ty_s, fw_s, angle, scale)


def apply_mustache(frame, landmarks, asset_img, state: dict, scale: float = 1.5):
    fh, fw = frame.shape[:2]
    lx, ly = _lm(landmarks, 61,  fw, fh)
    rx, ry = _lm(landmarks, 291, fw, fh)
    _, ny  = _lm(landmarks, 164, fw, fh)

    mouth_w   = int(np.linalg.norm([rx - lx, ry - ly]))
    raw_angle = -np.degrees(np.arctan2(ry - ly, rx - lx))
    raw_cx    = (lx + rx) // 2

    cx    = int(_ema(state, "mu_cx",  raw_cx))
    cy    = int(_ema(state, "mu_cy",  ny))
    angle = float(_ema(state, "mu_a", raw_angle))
    mw_s  = int(_ema(state, "mu_mw", mouth_w))

    mw = int(mw_s * scale)
    mh = int(mw * asset_img.shape[0] / asset_img.shape[1])
    if mw <= 0 or mh <= 0:
        return frame

    asset_resized = cv2.resize(asset_img, (mw, mh))
    M = cv2.getRotationMatrix2D((mw // 2, mh // 2), angle, 1.0)
    asset_rot = cv2.warpAffine(asset_resized, M, (mw, mh))

    x1, y1 = cx - mw // 2, cy - mh // 2
    x2, y2 = x1 + mw, y1 + mh
    x1c = max(x1, 0);  y1c = max(y1, 0)
    x2c = min(x2, frame.shape[1]);  y2c = min(y2, frame.shape[0])
    if x1c >= x2c or y1c >= y2c:
        return frame

    ax1, ay1 = x1c - x1, y1c - y1
    ax2, ay2 = ax1 + (x2c - x1c), ay1 + (y2c - y1c)

    alpha = asset_rot[ay1:ay2, ax1:ax2, 3:4] / 255.0
    roi   = frame[y1c:y2c, x1c:x2c]
    rgb   = asset_rot[ay1:ay2, ax1:ax2, :3]
    frame[y1c:y2c, x1c:x2c] = (alpha * rgb + (1 - alpha) * roi).astype(np.uint8)
    return frame


def apply_filter(frame, landmarks, mode: str, assets: dict,
                 gif_idx: int, state: dict):
    """Single dispatch entry point called by processor.py."""
    if mode == "hat":
        return apply_hat(frame, landmarks, assets["hat"], state)
    elif mode == "mustache":
        return apply_mustache(frame, landmarks, assets["mustache"], state)
    elif mode == "milky":
        asset_img = assets["milky"][gif_idx % len(assets["milky"])]
        return apply_hat(frame, landmarks, asset_img, state, scale=1.8)
    return frame
