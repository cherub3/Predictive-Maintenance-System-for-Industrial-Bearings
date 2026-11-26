"""Phase 8 dashboard tests.

Each test is self-contained: no shared module state, and the Streamlit cache on
validate_and_load_manifest is cleared before use so test order cannot matter.
"""

import hashlib

import pytest
from unittest.mock import patch

from src.dashboard.data_loader import (
    verify_manifest,
    validate_and_load_manifest,
)
from src.dashboard.state import get_visible_pages


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _write(path, text="payload"):
    path.write_text(text, encoding="utf-8")
    return path


def test_verify_manifest_success(tmp_path):
    """A manifest whose recorded hashes match the files on disk verifies cleanly."""
    a = _write(tmp_path / "a.md", "alpha")
    b = _write(tmp_path / "b.parquet", "beta")
    manifest = {
        "artifacts": [
            {"path": "a.md", "sha256": _sha256(a), "missing": False},
            {"path": "b.parquet", "sha256": _sha256(b), "missing": False},
        ]
    }
    assert verify_manifest(manifest, base_dir=str(tmp_path)) == []


def test_verify_manifest_detects_tampering(tmp_path):
    """A changed artifact (hash mismatch) is reported — the fail-closed trigger."""
    a = _write(tmp_path / "a.md", "alpha")
    manifest = {"artifacts": [{"path": "a.md", "sha256": _sha256(a), "missing": False}]}
    a.write_text("TAMPERED", encoding="utf-8")  # mutate after hashing
    problems = verify_manifest(manifest, base_dir=str(tmp_path))
    assert len(problems) == 1 and "Hash mismatch" in problems[0]


def test_verify_manifest_detects_missing_file(tmp_path):
    """A recorded artifact absent from disk is reported."""
    manifest = {"artifacts": [{"path": "gone.md", "sha256": "deadbeef", "missing": False}]}
    problems = verify_manifest(manifest, base_dir=str(tmp_path))
    assert len(problems) == 1 and "missing on disk" in problems[0]


def test_verify_manifest_rejects_manifest_without_hashes():
    """A manifest lacking an artifacts list cannot be verified → fail closed."""
    assert verify_manifest({"experiment_id": "x"}) != []


def test_validate_and_load_manifest_fails_closed_when_artifacts_missing():
    """Missing required artifact paths raise FileNotFoundError (Degraded Mode)."""
    validate_and_load_manifest.clear()
    with patch("src.dashboard.data_loader.os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            validate_and_load_manifest()
    validate_and_load_manifest.clear()


def test_persona_visibility_executive():
    """Executive sees the landing page but not ML diagnostics."""
    class DummyState:
        current_persona = "Executive"

    with patch("src.dashboard.state.st.session_state", DummyState()):
        pages = get_visible_pages()
    assert "Landing Dashboard" in pages
    assert "Model Diagnostics" not in pages


def test_persona_visibility_ml_engineer():
    """The ML engineer sees model diagnostics but not the executive landing page."""
    class DummyState:
        current_persona = "Machine Learning Engineer"

    with patch("src.dashboard.state.st.session_state", DummyState()):
        pages = get_visible_pages()
    assert "Model Diagnostics" in pages
    assert "Landing Dashboard" not in pages
