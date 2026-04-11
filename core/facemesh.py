"""
FaceMesh dispatcher — reads detector.toml and returns the right backend.

Supported backends (set via detector.toml):
  haar       — OpenCV Haar Cascade, no extra deps (default)
  mediapipe  — MediaPipe FaceMesh 468 points, pip install mediapipe

Usage (unchanged from before):
    from core.facemesh import FaceMesh
    fm = FaceMesh()
    result = fm.process(rgb_frame)
"""

import os


def _read_config() -> dict:
    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "detector.toml")
    )
    cfg = {"backend": "haar"}
    if not os.path.exists(config_path):
        return cfg
    try:
        with open(config_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("["):
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key in cfg:
                    cfg[key] = val
    except Exception:
        pass
    return cfg


def FaceMesh(max_faces: int = 2, min_detect: float = 0.5, min_track: float = 0.5):
    """Factory — returns an instance of the configured backend."""
    cfg = _read_config()
    backend = cfg["backend"].lower()

    if backend == "mediapipe":
        from core.backends.mediapipe_backend import FaceMesh as _FM
        return _FM(max_faces=max_faces, min_detect=min_detect, min_track=min_track)

    # default: haar
    from core.backends.haar import FaceMesh as _FM
    return _FM(max_faces=max_faces)
