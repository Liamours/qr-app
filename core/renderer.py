import cv2
import numpy as np
from mediapipe.python.solutions import drawing_utils as _drawing
from mediapipe.python.solutions import drawing_styles as _styles
from mediapipe.python.solutions import face_mesh as _mp_mesh

TESSELATION_SPEC = _styles.get_default_face_mesh_tesselation_style()
CONTOUR_SPEC = _styles.get_default_face_mesh_contours_style()
IRIS_SPEC = _styles.get_default_face_mesh_iris_connections_style()


def draw_landmarks(frame, results):
    if not results.multi_face_landmarks:
        return frame
    for face_landmarks in results.multi_face_landmarks:
        _drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=_mp_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=TESSELATION_SPEC,
        )
        _drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=_mp_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=CONTOUR_SPEC,
        )
        _drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=_mp_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=IRIS_SPEC,
        )
    return frame


def overlay_asset(frame, asset_img, cx, ty, face_width, angle, scale=1.3):
    hat_w = int(face_width * scale)
    hat_h = int(hat_w * asset_img.shape[0] / asset_img.shape[1])

    asset_resized = cv2.resize(asset_img, (hat_w, hat_h))

    M = cv2.getRotationMatrix2D((hat_w // 2, hat_h // 2), angle, 1.0)
    asset_rotated = cv2.warpAffine(asset_resized, M, (hat_w, hat_h))

    x1 = cx - hat_w // 2
    y1 = ty - hat_h
    x2 = x1 + hat_w
    y2 = y1 + hat_h

    x1c, y1c = max(x1, 0), max(y1, 0)
    x2c, y2c = min(x2, frame.shape[1]), min(y2, frame.shape[0])

    if x1c >= x2c or y1c >= y2c:
        return frame

    ax1 = x1c - x1
    ay1 = y1c - y1
    ax2 = ax1 + (x2c - x1c)
    ay2 = ay1 + (y2c - y1c)

    alpha = asset_rotated[ay1:ay2, ax1:ax2, 3:4] / 255.0
    roi = frame[y1c:y2c, x1c:x2c]
    rgb = asset_rotated[ay1:ay2, ax1:ax2, :3]

    frame[y1c:y2c, x1c:x2c] = (alpha * rgb + (1 - alpha) * roi).astype(np.uint8)
    return frame


def apply_hat(frame, face_landmarks, asset_img, scale=1.3):
    h, w = frame.shape[:2]

    left  = face_landmarks.landmark[234]
    right = face_landmarks.landmark[454]
    top   = face_landmarks.landmark[10]

    lx, ly = int(left.x * w),  int(left.y * h)
    rx, ry = int(right.x * w), int(right.y * h)
    tx, ty = int(top.x * w),   int(top.y * h)

    face_width = int(np.linalg.norm([rx - lx, ry - ly]))
    angle = -np.degrees(np.arctan2(ry - ly, rx - lx))
    cx = (lx + rx) // 2

    return overlay_asset(frame, asset_img, cx, ty, face_width, angle, scale)


def apply_sunglasses(frame, face_landmarks, asset_img, scale=1.1):
    h, w = frame.shape[:2]

    left  = face_landmarks.landmark[33]
    right = face_landmarks.landmark[263]

    lx, ly = int(left.x * w),  int(left.y * h)
    rx, ry = int(right.x * w), int(right.y * h)

    eye_width = int(np.linalg.norm([rx - lx, ry - ly]))
    glasses_w = int(eye_width * scale)
    glasses_h = int(glasses_w * asset_img.shape[0] / asset_img.shape[1])

    angle = -np.degrees(np.arctan2(ry - ly, rx - lx))
    cx = (lx + rx) // 2
    cy = (ly + ry) // 2

    asset_resized = cv2.resize(asset_img, (glasses_w, glasses_h))
    M = cv2.getRotationMatrix2D((glasses_w // 2, glasses_h // 2), angle, 1.0)
    asset_rotated = cv2.warpAffine(asset_resized, M, (glasses_w, glasses_h))

    x1 = cx - glasses_w // 2
    y1 = cy - glasses_h // 2
    x2 = x1 + glasses_w
    y2 = y1 + glasses_h

    x1c, y1c = max(x1, 0), max(y1, 0)
    x2c, y2c = min(x2, frame.shape[1]), min(y2, frame.shape[0])

    if x1c >= x2c or y1c >= y2c:
        return frame

    ax1 = x1c - x1
    ay1 = y1c - y1
    ax2 = ax1 + (x2c - x1c)
    ay2 = ay1 + (y2c - y1c)

    alpha = asset_rotated[ay1:ay2, ax1:ax2, 3:4] / 255.0
    roi = frame[y1c:y2c, x1c:x2c]
    rgb = asset_rotated[ay1:ay2, ax1:ax2, :3]

    frame[y1c:y2c, x1c:x2c] = (alpha * rgb + (1 - alpha) * roi).astype(np.uint8)
    return frame
