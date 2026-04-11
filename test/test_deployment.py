"""
Pre-deployment checklist for Streamlit Cloud.

Run this before every push to catch anything that would break on the cloud.

    python -m pytest test/test_deployment.py -v

Every test has a clear label so you know exactly what failed and why.
"""

import os
import sys
import importlib
import re

import cv2
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

VIDEO_PATH = os.path.join(os.path.dirname(__file__), "video", "WIN_20260216_18_00_23_Pro.mp4")


# ── 1. Config files ───────────────────────────────────────────────────────────

class TestConfigFiles:
    def test_runtime_txt_exists(self):
        assert os.path.exists(os.path.join(ROOT, "runtime.txt")), \
            "runtime.txt missing — Streamlit Cloud needs this to pick Python version"

    def test_runtime_txt_specifies_python_311(self):
        with open(os.path.join(ROOT, "runtime.txt")) as f:
            content = f.read().strip()
        assert content == "python-3.11", \
            f"runtime.txt should be 'python-3.11' (got '{content}'). " \
            "mediapipe 0.10.x is only guaranteed on Python 3.9–3.11."

    def test_requirements_txt_exists(self):
        assert os.path.exists(os.path.join(ROOT, "requirements.txt"))

    def test_requirements_txt_has_mediapipe(self):
        with open(os.path.join(ROOT, "requirements.txt")) as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        mp_lines = [l for l in lines if l.startswith("mediapipe")]
        assert mp_lines, \
            "mediapipe is missing or still commented out in requirements.txt"

    def test_requirements_txt_mediapipe_version_pinned(self):
        with open(os.path.join(ROOT, "requirements.txt")) as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        mp_line = next((l for l in lines if l.startswith("mediapipe")), None)
        assert mp_line and (">" in mp_line or "=" in mp_line), \
            f"mediapipe line '{mp_line}' has no version constraint — pin it for reproducibility"

    def test_requirements_txt_av_upper_bound(self):
        """av>=13 breaks streamlit-webrtc <0.64 — must be capped."""
        with open(os.path.join(ROOT, "requirements.txt")) as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        av_line = next((l for l in lines if re.match(r"^av[><=]", l)), None)
        assert av_line, "av is missing from requirements.txt"
        assert "<13" in av_line or "<12" in av_line, \
            f"av line '{av_line}' has no upper bound — av>=13 breaks streamlit-webrtc <0.64"

    def test_requirements_txt_has_all_core_packages(self):
        required = {
            "streamlit", "streamlit-webrtc", "opencv-contrib-python-headless",
            "numpy", "pillow", "av", "mediapipe",
        }
        with open(os.path.join(ROOT, "requirements.txt")) as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        found = set()
        for line in lines:
            pkg = re.split(r"[><=!]", line)[0].strip().lower()
            found.add(pkg)
        missing = required - found
        assert not missing, f"Missing from requirements.txt: {missing}"

    def test_packages_txt_exists(self):
        assert os.path.exists(os.path.join(ROOT, "packages.txt")), \
            "packages.txt missing — system libs (libgl1 etc.) won't be installed on Streamlit Cloud"

    def test_packages_txt_has_libgl1(self):
        with open(os.path.join(ROOT, "packages.txt")) as f:
            pkgs = [l.strip() for l in f if l.strip()]
        assert "libgl1" in pkgs, \
            "libgl1 missing from packages.txt — OpenCV will crash on Streamlit Cloud without it"

    def test_packages_txt_has_libglib(self):
        with open(os.path.join(ROOT, "packages.txt")) as f:
            pkgs = [l.strip() for l in f if l.strip()]
        has_glib = any("libglib" in p for p in pkgs)
        assert has_glib, \
            "libglib missing from packages.txt — mediapipe requires it on Linux"

    def test_packages_txt_has_libgomp(self):
        with open(os.path.join(ROOT, "packages.txt")) as f:
            pkgs = [l.strip() for l in f if l.strip()]
        assert "libgomp1" in pkgs, \
            "libgomp1 missing from packages.txt — mediapipe uses OpenMP on Linux"

    def test_detector_toml_exists(self):
        assert os.path.exists(os.path.join(ROOT, "detector.toml")), \
            "detector.toml missing — app will silently fall back to haar on cloud"

    def test_detector_toml_backend_is_mediapipe(self):
        with open(os.path.join(ROOT, "detector.toml")) as f:
            content = f.read()
        match = re.search(r'backend\s*=\s*["\']?(\w+)["\']?', content)
        assert match, "Cannot find backend = ... in detector.toml"
        backend = match.group(1)
        assert backend == "mediapipe", \
            f"detector.toml backend is '{backend}' — set it to 'mediapipe' for deployment"


# ── 2. Asset files ────────────────────────────────────────────────────────────

class TestAssets:
    ASSETS = [
        "assets/wizard-hat.png",
        "assets/churos-mustache.png",
        "assets/milky-pour.gif",
    ]

    @pytest.mark.parametrize("asset", ASSETS)
    def test_asset_exists(self, asset):
        path = os.path.join(ROOT, asset)
        assert os.path.exists(path), \
            f"Asset file missing: {asset} — app will crash on startup"

    @pytest.mark.parametrize("asset", [a for a in ASSETS if a.endswith(".png")])
    def test_png_asset_is_valid_image(self, asset):
        path = os.path.join(ROOT, asset)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        assert img is not None, f"{asset} could not be read by OpenCV"
        assert img.ndim == 3, f"{asset} is not a 3-channel image"

    def test_gif_asset_has_frames(self):
        from PIL import Image
        path = os.path.join(ROOT, "assets/milky-pour.gif")
        gif = Image.open(path)
        frames = 0
        try:
            while True:
                frames += 1
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass
        assert frames > 0, "milky-pour.gif has no frames"


# ── 3. Python imports ─────────────────────────────────────────────────────────

class TestImports:
    @pytest.mark.parametrize("pkg", [
        "streamlit", "cv2", "numpy", "PIL", "av", "mediapipe",
    ])
    def test_package_importable(self, pkg):
        mod = importlib.import_module(pkg)
        assert mod is not None, f"Failed to import {pkg}"

    def test_streamlit_webrtc_importable(self):
        try:
            import streamlit_webrtc
        except ImportError:
            pytest.skip(
                "streamlit_webrtc not installed in this env — "
                "it MUST be installed on Streamlit Cloud via requirements.txt"
            )

    def test_mediapipe_version(self):
        import mediapipe as mp
        version = tuple(int(x) for x in mp.__version__.split(".")[:2])
        assert version >= (0, 10), \
            f"mediapipe {mp.__version__} is too old — need >=0.10.0"

    def test_numpy_version_below_2(self):
        version = tuple(int(x) for x in np.__version__.split(".")[:2])
        assert version < (2, 0), \
            f"numpy {np.__version__} — mediapipe 0.10.x is not compatible with numpy 2.x"

    def test_opencv_headless_not_full(self):
        """Streamlit Cloud needs headless — full opencv brings display deps that crash."""
        from importlib.metadata import packages_distributions
        installed = {k.lower() for k in packages_distributions()}
        non_headless = {"opencv-python", "opencv-contrib-python"}
        headless = {"opencv-python-headless", "opencv-contrib-python-headless"}
        has_non_headless = bool(non_headless & installed)
        has_headless = bool(headless & installed)
        assert not has_non_headless or has_headless, (
            "A non-headless OpenCV is installed without a headless counterpart. "
            "Use opencv-contrib-python-headless in requirements.txt."
        )


# ── 4. Core modules ───────────────────────────────────────────────────────────

class TestCoreModules:
    def test_facemesh_module_importable(self):
        from core.facemesh import FaceMesh
        assert callable(FaceMesh)

    def test_renderer_module_importable(self):
        from core import renderer
        assert callable(renderer.apply_filter)

    def test_processor_module_importable(self):
        # processor imports streamlit_webrtc — skip gracefully if not installed
        try:
            from core.processor import FaceFilterProcessor
        except ImportError as e:
            if "streamlit_webrtc" in str(e):
                pytest.skip("streamlit_webrtc not installed in this env")
            raise

    def test_haar_backend_importable(self):
        from core.backends.haar import FaceMesh
        assert callable(FaceMesh)

    def test_mediapipe_backend_importable(self):
        from core.backends.mediapipe_backend import FaceMesh
        assert callable(FaceMesh)


# ── 5. MediaPipe pipeline ─────────────────────────────────────────────────────

class TestMediaPipePipeline:
    @pytest.fixture(scope="class")
    def first_frame(self):
        cap = cv2.VideoCapture(VIDEO_PATH)
        assert cap.isOpened(), f"Cannot open video: {VIDEO_PATH}"
        ret, bgr = cap.read()
        cap.release()
        assert ret, "Could not read first frame"
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def test_dispatcher_returns_mediapipe_backend(self):
        from core.facemesh import FaceMesh
        from core.backends.mediapipe_backend import FaceMesh as MPFM
        fm = FaceMesh()
        assert isinstance(fm, MPFM), \
            f"Dispatcher returned {type(fm).__name__} instead of MediaPipe — " \
            "check detector.toml backend = mediapipe"

    def test_mediapipe_detects_face_in_first_frame(self, first_frame):
        from core.facemesh import FaceMesh
        fm = FaceMesh()
        result = fm.process(first_frame)
        assert result.face_landmarks, \
            "MediaPipe found no face in the first video frame — " \
            "check that mediapipe is correctly installed"
        fm.close()

    def test_all_landmark_indices_present(self, first_frame):
        from core.facemesh import FaceMesh
        fm = FaceMesh()
        result = fm.process(first_frame)
        if not result.face_landmarks:
            pytest.skip("No face detected")
        face = result.face_landmarks[0]
        for idx in [10, 234, 454, 61, 291, 164]:
            lm = face[idx]
            assert 0.0 <= lm.x <= 1.0
            assert 0.0 <= lm.y <= 1.0
        fm.close()

    def test_full_render_pipeline(self, first_frame):
        """Dispatcher → MediaPipe → renderer → output frame."""
        from core.facemesh import FaceMesh
        from core.renderer import apply_filter

        bgr = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
        hat = cv2.imread(os.path.join(ROOT, "assets/wizard-hat.png"), cv2.IMREAD_UNCHANGED)
        mustache = cv2.imread(os.path.join(ROOT, "assets/churos-mustache.png"), cv2.IMREAD_UNCHANGED)

        fm = FaceMesh()
        result = fm.process(first_frame)
        if not result.face_landmarks:
            pytest.skip("No face detected")

        assets = {
            "hat": hat,
            "mustache": mustache,
            "milky": [hat],
        }
        out = apply_filter(bgr.copy(), result.face_landmarks[0], "hat", assets, 0, {})
        assert out.shape == bgr.shape
        assert out.dtype == bgr.dtype
        assert not np.array_equal(out, bgr), "Renderer produced no visible change"
        fm.close()
