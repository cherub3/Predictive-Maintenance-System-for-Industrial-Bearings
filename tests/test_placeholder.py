"""Placeholder tests for verifying basic workspace setup, config parsing, and utilities.

Serves as the bootstrap validation suite for Pytest execution.
"""

from pathlib import Path
import tempfile
import pytest

from src.config import load_config
from src.constants import DEFAULT_CONFIG_PATH
from src.utils import get_file_sha256, ensure_directory_exists


def test_config_loading():
    """Verify that configuration parsing succeeds and returns expected keys."""
    config = load_config(DEFAULT_CONFIG_PATH)
    assert isinstance(config, dict)
    assert "project" in config
    assert "logging" in config
    assert "paths" in config
    assert config["project"]["target_company"] == "industrial bearing fleet"


def test_sha256_utility():
    """Verify that the checksum utility computes SHA256 hashes accurately."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as temp_file:
        temp_file.write("test_content_for_checksum_verification")
        temp_file_path = Path(temp_file.name)

    try:
        expected_hash = "35db7f00d7f4b74032e89e51e74ec66607d5a5aeb6b75221e32ae05024bd1a2b"
        computed_hash = get_file_sha256(temp_file_path)
        assert computed_hash == expected_hash
    finally:
        temp_file_path.unlink()


def test_ensure_directory_exists():
    """Verify that the directory helper creates a folder and returns the Path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        parent_path = Path(temp_dir)
        target_sub_path = parent_path / "new_subdir" / "nested_subdir"
        
        assert not target_sub_path.exists()
        resolved_path = ensure_directory_exists(target_sub_path)
        
        assert target_sub_path.exists()
        assert resolved_path == target_sub_path
