"""Unit tests for validating the generated metadata and pipeline consistency.

Ensures that files are fully accounted for, parsed successfully, and match expectations,
across every registered dataset (Set 1/2/3).
"""

from pathlib import Path
import duckdb
import pandas as pd
import pytest

from src.config import get_all_dataset_ids, get_dataset_config, load_config
from src.constants import DEFAULT_CONFIG_PATH

EXPECTED_TOTAL_FILES = 9463  # set1 (2156) + set2 (984) + set3 (6323, after excluding 1 dead file)


@pytest.fixture(scope="module")
def config():
    """Module-level fixture to load config parameters."""
    return load_config(DEFAULT_CONFIG_PATH)


def test_metadata_parquet_exists_and_valid(config):
    """Assert that the metadata.parquet is present and structurally correct."""
    processed_dir = Path(config["paths"]["processed_data_dir"])
    parquet_path = processed_dir / "metadata.parquet"

    assert parquet_path.exists(), f"Metadata Parquet file not found at {parquet_path}"

    df = pd.read_parquet(parquet_path)
    assert len(df) == EXPECTED_TOTAL_FILES, f"Expected {EXPECTED_TOTAL_FILES} records, got {len(df)}"

    # Assert all expected columns exist
    expected_cols = [
        "dataset_id", "experiment_name", "bearings_present", "channels_present",
        "timestamp", "filename", "absolute_path", "relative_path", "signal_length",
        "num_columns", "expected_channels_per_file", "sampling_frequency_hz",
        "file_sha256", "parser_status", "extraction_time", "notes"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Column '{col}' missing from parquet metadata schema"

    # Assert data characteristics, globally and per-dataset
    assert (df["parser_status"] == "Success").all(), "Some files failed ingestion parsing"
    assert (df["signal_length"] == 20480).all(), "Some files do not contain exactly 20480 samples"
    assert (df["sampling_frequency_hz"] == 20000.0).all(), "Incorrect sampling rate recorded"
    assert set(df["dataset_id"].unique()) == set(get_all_dataset_ids(config))

    for dataset_id in get_all_dataset_ids(config):
        dataset_cfg = get_dataset_config(config, dataset_id)
        sub = df[df["dataset_id"] == dataset_id]
        assert not sub.empty, f"No metadata rows found for dataset '{dataset_id}'"
        assert (sub["num_columns"] == dataset_cfg["expected_channels_per_file"]).all(), (
            f"Dataset '{dataset_id}' has files with unexpected channel counts"
        )


def test_metadata_duckdb_integrity(config):
    """Verify that metadata.duckdb matches the parquet dataset records."""
    processed_dir = Path(config["paths"]["processed_data_dir"])
    duckdb_path = processed_dir / "metadata.duckdb"

    assert duckdb_path.exists(), f"DuckDB database not found at {duckdb_path}"

    con = duckdb.connect(str(duckdb_path))
    try:
        res = con.execute("SELECT COUNT(*) FROM metadata").fetchone()
        assert res[0] == EXPECTED_TOTAL_FILES, (
            f"DuckDB metadata record count mismatch: expected {EXPECTED_TOTAL_FILES}, got {res[0]}"
        )

        status_counts = con.execute(
            "SELECT parser_status, COUNT(*) FROM metadata GROUP BY parser_status"
        ).fetchall()

        assert len(status_counts) == 1
        assert status_counts[0][0] == "Success"
        assert status_counts[0][1] == EXPECTED_TOTAL_FILES
    finally:
        con.close()


def test_metadata_chronological_ordering_within_each_dataset(config):
    """Ensure metadata records are strictly chronological within each dataset.

    Datasets are independent, non-overlapping test runs (different physical rigs
    of the same rig setup, different calendar periods) — a global monotonicity
    check across dataset boundaries is not a meaningful invariant, only
    within-dataset chronology is.
    """
    processed_dir = Path(config["paths"]["processed_data_dir"])
    parquet_path = processed_dir / "metadata.parquet"

    df = pd.read_parquet(parquet_path)

    for dataset_id in get_all_dataset_ids(config):
        sub = df[df["dataset_id"] == dataset_id]
        timestamps = pd.to_datetime(sub["timestamp"])
        assert timestamps.is_monotonic_increasing, (
            f"Metadata timestamps for dataset '{dataset_id}' are not in chronological order"
        )
