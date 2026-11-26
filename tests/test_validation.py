"""Tests for Phase 2 Data Validation and Quality Audit module.

Verifies that the validator logic captures completeness, signal integrity,
and generates the appropriate risk frameworks.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.config import load_config
from src.constants import DEFAULT_CONFIG_PATH
from src.data_validation.validator import DataValidator


@pytest.fixture(scope="module")
def config():
    """Module-level fixture to load config parameters."""
    return load_config(DEFAULT_CONFIG_PATH)


def test_validator_initialization_and_chronology(config):
    """Test that the validator initializes and runs basic chronology logic."""
    processed_dir = Path(config["paths"]["processed_data_dir"])
    parquet_path = processed_dir / "metadata.parquet"

    # We use a dummy logger
    class DummyLogger:
        def info(self, msg): pass
        def warning(self, msg): pass
        def error(self, msg): pass
        def debug(self, msg): pass

    logger = DummyLogger()

    assert parquet_path.exists(), "Metadata parquet is required for validation tests"

    df = pd.read_parquet(parquet_path)
    validator = DataValidator(df, config, logger)

    # Run chron checks
    validator.check_completeness_and_chronology()

    # NASA IMS Set 1 has documented chronology anomalies (the acquisition cadence
    # shifts from 5 to 10 minutes and there are large gaps), so the chronology check
    # MUST surface findings. Assert real content, not merely that a list exists.
    assert len(validator.findings) > 0, "chronology check should flag known IMS Set 1 gaps"
    # This method is the only one run, and every finding it adds also increments the
    # Chronology Violations counter, so the two must agree exactly.
    assert validator.stats["Chronology Violations"] == len(validator.findings)
    for finding in validator.findings:
        assert {"Observation", "Severity", "Evidence"}.issubset(finding.keys())


def test_validator_risk_register(config):
    """Test that ML risks are successfully injected into the register."""
    # We use a dummy logger
    class DummyLogger:
        def info(self, msg): pass
        def warning(self, msg): pass
        def error(self, msg): pass
        def debug(self, msg): pass

    df = pd.DataFrame(columns=["dataset_id", "timestamp", "parser_status", "num_columns", "signal_length"])
    validator = DataValidator(df, config, DummyLogger())

    validator.define_ml_risks()
    assert len(validator.findings) >= 2, "Expected at least 2 default ML risks logged."

    observations = [r["Observation"] for r in validator.findings]
    assert any("Temporal Target Leakage Potential" in obs for obs in observations)
    assert any("Bearing Independence Leakage Potential" in obs for obs in observations)
