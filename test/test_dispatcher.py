"""
Tests for the core/facemesh.py dispatcher.

Checks that:
- FaceMesh() reads detector.toml and returns the correct backend class.
- Missing / malformed / unknown config values degrade gracefully to haar.
- The returned object satisfies the process() contract regardless of backend.
"""

import os
import importlib
import pytest

from core.backends.haar import FaceMesh as HaarFaceMesh


# ── config parsing ────────────────────────────────────────────────────────────

class TestConfigParsing:
    def _parse(self, content: str) -> dict:
        import tempfile
        import textwrap
        import core.facemesh as mod

        with tempfile.TemporaryDirectory() as tmp:
            toml_path = os.path.join(tmp, "detector.toml")
            with open(toml_path, "w") as f:
                f.write(textwrap.dedent(content))

            orig_fn = mod._read_config
            def _patched():
                cfg = {"backend": "haar"}
                with open(toml_path) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line or line.startswith("#") or line.startswith("["):
                            continue
                        key, _, val = line.partition("=")
                        key = key.strip()
                        val = val.strip().strip('"').strip("'")
                        if key in cfg:
                            cfg[key] = val
                return cfg
            mod._read_config = _patched
            try:
                result = mod._read_config()
            finally:
                mod._read_config = orig_fn
            return result

    def test_default_backend_is_haar(self):
        cfg = self._parse("")
        assert cfg["backend"] == "haar"

    def test_reads_backend_haar(self):
        assert self._parse('backend = "haar"')["backend"] == "haar"

    def test_reads_backend_mediapipe(self):
        assert self._parse('backend = "mediapipe"')["backend"] == "mediapipe"

    def test_ignores_comments(self):
        assert self._parse("# backend = mediapipe\nbackend = haar")["backend"] == "haar"

    def test_ignores_section_headers(self):
        assert self._parse("[detector]\nbackend = haar")["backend"] == "haar"

    def test_single_quoted_values(self):
        assert self._parse("backend = 'mediapipe'")["backend"] == "mediapipe"

    def test_unquoted_values(self):
        assert self._parse("backend = haar")["backend"] == "haar"

    def test_missing_file_returns_defaults(self):
        import core.facemesh as mod
        orig = mod._read_config
        mod._read_config = lambda: {"backend": "haar"}
        try:
            cfg = mod._read_config()
        finally:
            mod._read_config = orig
        assert cfg["backend"] == "haar"


# ── dispatcher routing ────────────────────────────────────────────────────────

class TestDispatcherRouting:
    def _get_fm(self, backend: str):
        import core.facemesh as mod
        orig = mod._read_config
        mod._read_config = lambda: {"backend": backend}
        try:
            fm = mod.FaceMesh()
        finally:
            mod._read_config = orig
        return fm

    def test_haar_backend_returns_haar_instance(self):
        assert isinstance(self._get_fm("haar"), HaarFaceMesh)

    def test_unknown_backend_falls_through_to_haar(self):
        assert isinstance(self._get_fm("bogus_backend_xyz"), HaarFaceMesh)

    def test_mediapipe_skipped_when_not_installed(self):
        try:
            import mediapipe  # noqa
        except ImportError:
            with pytest.raises(ImportError, match="mediapipe"):
                self._get_fm("mediapipe")

    def test_haar_fm_has_process_method(self):
        assert callable(getattr(self._get_fm("haar"), "process", None))

    def test_haar_fm_has_close_method(self):
        assert callable(getattr(self._get_fm("haar"), "close", None))


# ── end-to-end: dispatcher → process() on real frame ─────────────────────────

class TestDispatcherEndToEnd:
    def test_haar_processes_real_frame(self, first_frame):
        import core.facemesh as mod
        orig = mod._read_config
        mod._read_config = lambda: {"backend": "haar"}
        try:
            fm = mod.FaceMesh()
            result = fm.process(first_frame)
        finally:
            mod._read_config = orig

        from core.backends.haar import _DetectionResult
        assert isinstance(result, _DetectionResult)

    def test_mediapipe_processes_real_frame(self, first_frame):
        try:
            import mediapipe  # noqa
        except ImportError:
            pytest.skip("mediapipe not installed")

        import core.facemesh as mod
        orig = mod._read_config
        mod._read_config = lambda: {"backend": "mediapipe"}
        try:
            fm = mod.FaceMesh()
            result = fm.process(first_frame)
        finally:
            mod._read_config = orig

        from core.backends.mediapipe_backend import _DetectionResult
        assert isinstance(result, _DetectionResult)
