"""Metadata extractor for the NASA IMS Bearing Dataset.

Discovers files across every registered dataset (Set 1/2/3), validates their
shape/sample length, parses timestamps from filenames, and creates structured
Parquet, DuckDB, and CSV catalog files spanning all datasets.
"""

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import duckdb
import pandas as pd

from src.config import get_all_dataset_ids, get_dataset_config, load_config
from src.constants import PROJECT_ROOT
from src.logger import setup_logger
from src.utils import get_file_sha256, ensure_directory_exists


def parse_timestamp_from_filename(filename: str) -> datetime.datetime:
    """Parses timestamp from the NASA IMS file name format.

    Format example: '2003.10.22.12.06.24' -> YYYY.MM.DD.HH.MM.SS

    Args:
        filename: Name of the file.

    Returns:
        datetime.datetime: Parsed timestamp.
    """
    try:
        return datetime.datetime.strptime(filename.strip(), "%Y.%m.%d.%H.%M.%S")
    except ValueError as exc:
        raise ValueError(
            f"Filename '{filename}' does not match expected pattern YYYY.MM.DD.HH.MM.SS"
        ) from exc


def analyze_vibration_file(file_path: Path) -> Tuple[int, int, str]:
    """Inspects an ASCII vibration file to count samples and verify columns.

    Args:
        file_path: Path to the ASCII file.

    Returns:
        Tuple[int, int, str]: (signal_length, num_channels, status_message)
    """
    try:
        # Read file and count lines/columns
        line_count = 0
        col_count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                line_count += 1
                if line_count == 1:
                    col_count = len(line.split())
        
        if line_count == 0:
            return 0, 0, "Empty File"
        
        return line_count, col_count, "Success"
    except Exception as exc:
        return 0, 0, f"Error: {str(exc)}"


def build_dataset_notes(dataset_cfg: Dict[str, Any]) -> str:
    """Builds a human-readable failure-mode / channel-mapping note for one dataset.

    Derived entirely from the `datasets` registry so it stays correct for any
    dataset's bearing count, failure modes, and channel_axes without a code change.
    """
    bearings = dataset_cfg.get("bearings", {})

    failure_parts = []
    for bearing_id, bearing_cfg in sorted(bearings.items()):
        if bearing_cfg.get("status") == "rtf":
            failure_mode = bearing_cfg.get("failure_mode", "unknown").replace("_", " ")
            failure_parts.append(f"{bearing_id} developed {failure_mode} defect (run-to-failure)")
    failure_summary = (
        "Failure mode: " + "; ".join(failure_parts) + "."
        if failure_parts else "Failure mode: no run-to-failure bearing in this dataset (fully censored)."
    )

    channel_parts = []
    for bearing_id, bearing_cfg in sorted(bearings.items()):
        axes = bearing_cfg.get("channel_axes", {})
        axis_str = ", ".join(f"ch{ch}:{axis}" for ch, axis in sorted(axes.items()))
        channel_parts.append(f"{bearing_id}({axis_str})")
    channel_summary = "Channel Mapping: " + ", ".join(channel_parts) + "."

    return f"{failure_summary} {channel_summary}"


def _parse_run_window(dataset_cfg: Dict[str, Any]) -> Optional[Tuple[datetime.datetime, datetime.datetime]]:
    """Parses a dataset's `run_window` dates into inclusive (start, end) datetime bounds."""
    run_window = dataset_cfg.get("run_window")
    if not run_window:
        return None
    start_dt = datetime.datetime.strptime(run_window["start"], "%Y-%m-%d")
    end_dt = datetime.datetime.strptime(run_window["end"], "%Y-%m-%d") + datetime.timedelta(
        days=1
    ) - datetime.timedelta(seconds=1)
    return start_dt, end_dt


def extract_metadata_for_dataset(
    dataset_cfg: Dict[str, Any],
    sampling_rate_hz: float,
    extraction_time: str,
    logger: Any,
) -> List[Dict[str, Any]]:
    """Discovers and catalogs raw vibration files for a single registered dataset.

    Args:
        dataset_cfg: Combined registry block from `get_dataset_config` (raw_dir,
            run_window, expected_channels_per_file, bearings, etc).
        sampling_rate_hz: Sampling rate from `datasets.defaults`.
        extraction_time: Shared UTC extraction timestamp for this pipeline run.
        logger: Pipeline logger instance.

    Returns:
        List[Dict[str, Any]]: One record per included raw file.
    """
    dataset_id = dataset_cfg["dataset_id"]
    raw_dir = PROJECT_ROOT / dataset_cfg["raw_dir"]
    expected_channels = dataset_cfg.get("expected_channels_per_file")
    excluded_files: Set[str] = set(dataset_cfg.get("excluded_files", []))
    window = _parse_run_window(dataset_cfg)
    notes = build_dataset_notes(dataset_cfg)

    bearings = dataset_cfg.get("bearings", {})
    bearings_present = ",".join(sorted(bearings.keys()))
    channels_present = ",".join(
        str(ch) for b in bearings.values() for ch in sorted(b.get("channels", []))
    )

    logger.info(f"[{dataset_id}] Scanning raw files in: {raw_dir.resolve()}")

    all_files = sorted([
        f for f in raw_dir.iterdir()
        if f.is_file() and not f.name.startswith(".")
    ])

    raw_files = []
    skipped_excluded = 0
    skipped_out_of_window = 0
    for f in all_files:
        if f.name in excluded_files:
            skipped_excluded += 1
            continue
        if window is not None:
            try:
                ts = parse_timestamp_from_filename(f.name)
                if ts < window[0] or ts > window[1]:
                    skipped_out_of_window += 1
                    continue
            except ValueError:
                pass  # unparseable names are kept and flagged as Failed below
        raw_files.append(f)

    if skipped_excluded:
        logger.info(f"[{dataset_id}] Skipped {skipped_excluded} explicitly excluded file(s).")
    if skipped_out_of_window:
        logger.info(
            f"[{dataset_id}] Skipped {skipped_out_of_window} file(s) outside the "
            f"configured run_window."
        )
    logger.info(f"[{dataset_id}] Discovered {len(raw_files)} raw vibration files to process.")

    records: List[Dict[str, Any]] = []

    for i, file_path in enumerate(raw_files, start=1):
        filename = file_path.name

        if i % 500 == 0 or i == len(raw_files):
            logger.info(f"[{dataset_id}] Processing progress: {i}/{len(raw_files)} files...")

        try:
            timestamp = parse_timestamp_from_filename(filename)
            ts_str = timestamp.isoformat()
            parser_status = "Success"
            err_msg = ""
        except Exception as exc:
            ts_str = ""
            parser_status = "Failed"
            err_msg = f"Timestamp Parse Error: {exc}"
            logger.error(f"[{dataset_id}] Error parsing timestamp for {filename}: {exc}")

        sig_len = 0
        num_cols = 0
        file_sha256 = ""

        if parser_status == "Success":
            sig_len, num_cols, parse_res = analyze_vibration_file(file_path)
            if parse_res != "Success":
                parser_status = "Failed"
                err_msg = parse_res
                logger.error(f"[{dataset_id}] Error reading structure of {filename}: {parse_res}")
            elif expected_channels is not None and num_cols != expected_channels:
                parser_status = "Failed"
                err_msg = (
                    f"Channel count mismatch: expected {expected_channels}, found {num_cols}"
                )
                logger.error(f"[{dataset_id}] {err_msg} in {filename}")
            else:
                try:
                    file_sha256 = get_file_sha256(file_path)
                except Exception as exc:
                    file_sha256 = ""
                    logger.warning(f"[{dataset_id}] Could not compute checksum for {filename}: {exc}")

        record_notes = notes
        if err_msg:
            record_notes += f" | Status details: {err_msg}"

        records.append({
            "dataset_id": dataset_id,
            "experiment_name": dataset_cfg.get("display_name", dataset_id),
            "bearings_present": bearings_present,
            "channels_present": channels_present,
            "timestamp": ts_str,
            "filename": filename,
            "absolute_path": str(file_path.resolve()),
            "relative_path": str(file_path.relative_to(PROJECT_ROOT)),
            "signal_length": sig_len,
            "num_columns": num_cols,
            "expected_channels_per_file": expected_channels,
            "sampling_frequency_hz": sampling_rate_hz,
            "file_sha256": file_sha256,
            "parser_status": parser_status,
            "extraction_time": extraction_time,
            "notes": record_notes,
        })

    return records


def extract_metadata(
    config: Dict[str, Any], processed_dir: Path, report_dir: Path, logger: Any
) -> pd.DataFrame:
    """Discovers raw vibration files across every registered dataset and saves results.

    Args:
        config: Loaded project configuration (must contain the `datasets` registry).
        processed_dir: Path to processed outputs directory.
        report_dir: Path to report directory.
        logger: Pipeline logger instance.

    Returns:
        pd.DataFrame: Consolidated metadata dataframe spanning all datasets.
    """
    dataset_ids = get_all_dataset_ids(config)
    sampling_rate_hz = config.get("datasets", {}).get("defaults", {}).get(
        "sampling_rate_hz", 20000.0
    )
    extraction_time = datetime.datetime.utcnow().isoformat()

    logger.info(f"Registered datasets: {dataset_ids}")

    records: List[Dict[str, Any]] = []
    for dataset_id in dataset_ids:
        dataset_cfg = get_dataset_config(config, dataset_id)
        records.extend(
            extract_metadata_for_dataset(dataset_cfg, sampling_rate_hz, extraction_time, logger)
        )

    df = pd.DataFrame(records)
    
    # Save CSV Inventory
    csv_path = report_dir / "dataset_inventory.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Successfully saved inventory CSV to: {csv_path.resolve()}")
    
    # Save Parquet Metadata
    parquet_path = processed_dir / "metadata.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info(f"Successfully saved metadata Parquet to: {parquet_path.resolve()}")
    
    # Save DuckDB database file
    duckdb_path = processed_dir / "metadata.duckdb"
    if duckdb_path.exists():
        duckdb_path.unlink() # Fresh recreation
        
    con = duckdb.connect(str(duckdb_path))
    con.execute("CREATE TABLE metadata AS SELECT * FROM df")
    con.close()
    logger.info(f"Successfully saved DuckDB database to: {duckdb_path.resolve()}")
    
    return df


def generate_statistics_and_docs(
    df: pd.DataFrame, config: Dict[str, Any], report_dir: Path, logger: Any
) -> None:
    """Calculates per-dataset statistics and writes summary reports.

    Chronological intervals are computed **within each dataset_id group only** —
    the three datasets are non-overlapping, unrelated test runs, so a global sort
    across all rows would produce a spurious multi-day "interval" at each dataset
    boundary instead of a real chronology signal.

    Args:
        df: Consolidated metadata DataFrame spanning all datasets.
        config: Loaded project configuration (for the datasets registry).
        report_dir: Path to reports directory.
        logger: Pipeline logger instance.
    """
    logger.info("Computing dataset summary statistics and generating reports...")

    dataset_ids = get_all_dataset_ids(config)
    total_files = len(df)
    successful_parses = len(df[df["parser_status"] == "Success"])
    failed_parses = len(df[df["parser_status"] == "Failed"])

    per_dataset_sections = []
    for dataset_id in dataset_ids:
        dataset_cfg = get_dataset_config(config, dataset_id)
        sub = df[df["dataset_id"] == dataset_id]
        sub_success = sub[sub["parser_status"] == "Success"].copy()
        sub_success["dt"] = pd.to_datetime(sub_success["timestamp"])
        sub_success = sub_success.sort_values("dt")
        sub_success["interval_mins"] = sub_success["dt"].diff().dt.total_seconds() / 60.0

        n_bearings = len(dataset_cfg.get("bearings", {}))
        rtf_bearings = [
            b for b, cfg in dataset_cfg.get("bearings", {}).items() if cfg.get("status") == "rtf"
        ]

        if not sub_success.empty:
            avg_interval = sub_success["interval_mins"].mean()
            min_interval = sub_success["interval_mins"].min()
            max_interval = sub_success["interval_mins"].max()
            avg_len = sub_success["signal_length"].mean()
            min_len = sub_success["signal_length"].min()
            max_len = sub_success["signal_length"].max()
        else:
            avg_interval = min_interval = max_interval = float("nan")
            avg_len = min_len = max_len = float("nan")

        per_dataset_sections.append(f"""### {dataset_cfg.get('display_name', dataset_id)} (`{dataset_id}`)
- **Bearings**: {n_bearings} ({dataset_cfg.get('channel_layout')})
- **Run-to-failure bearings**: {', '.join(rtf_bearings) if rtf_bearings else 'none (fully censored)'}
- **Files discovered**: {len(sub)} (successful: {len(sub_success)}, failed: {len(sub) - len(sub_success)})
- **Signal length**: avg {avg_len:.1f}, min {min_len}, max {max_len} samples
- **File interval**: avg {avg_interval:.2f} min, min {min_interval:.2f} min, max {max_interval:.2f} min
""")

    summary_content = f"""# NASA IMS Bearing Dataset Summary

## Dataset Characteristics
- **Total Datasets**: {len(dataset_ids)} ({', '.join(dataset_ids)})
- **Total Files Discovered**: {total_files}
- **Successfully Parsed Files**: {successful_parses}
- **Failed Files**: {failed_parses}
- **Sampling Frequency**: {config.get('datasets', {}).get('defaults', {}).get('sampling_rate_hz', 20000.0):.0f} Hz
- **Vibration snapshot length**: 1.024 seconds (20,480 samples @ 20kHz)

## Per-Dataset Breakdown

{''.join(per_dataset_sections)}
"""

    summary_path = report_dir / "dataset_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_content)
    logger.info(f"Saved dataset summary to: {summary_path.resolve()}")

    # Write folder_structure.md
    structure_content = """# Folder Hierarchy and Structure

This document outlines the organization of the workspace directory.

```text
bearing_predictive_maintenance/
├── config/
│   └── config.yaml           # Global YAML configuration (datasets registry: set1/set2/set3)
├── src/
│   ├── config.py             # Config parser + datasets registry helpers
│   ├── constants.py          # Constants & exit codes
│   ├── logger.py             # Custom logging infrastructure
│   ├── utils.py              # File utilities
│   ├── main.py               # Application orchestrator
│   └── data_ingestion/       # Data ingestion & metadata extraction layer
├── data/
│   ├── raw/
│   │   ├── set1/             # NASA IMS Set 1 (2,156 files, 8 channels)
│   │   ├── set2/             # NASA IMS Set 2 (984 files, 4 channels)
│   │   └── set3/             # NASA IMS Set 3 (6,323 usable files, 4 channels)
│   ├── processed/            # Serialized metadata tables (Parquet, DuckDB)
│   └── external/             # Empty placeholder
├── reports/
│   ├── dataset_summary.md    # Summary statistics report (per-dataset breakdown)
│   ├── folder_structure.md   # Current document
│   ├── metadata_schema.md    # Metadata schema description
│   └── dataset_inventory.csv # Flat catalog inventory CSV (all datasets)
└── logs/
    └── pipeline.log          # Runtime log file
```
"""
    structure_path = report_dir / "folder_structure.md"
    with open(structure_path, "w", encoding="utf-8") as f:
        f.write(structure_content)
    logger.info(f"Saved folder structure to: {structure_path.resolve()}")

    # Write metadata_schema.md
    schema_content = """# Metadata Schema Specification

The metadata database tables (`metadata.parquet` and `metadata.duckdb`) and catalog (`dataset_inventory.csv`) use the following schema. One row per raw vibration file, spanning all registered datasets (Set 1/2/3).

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `dataset_id` | String | Registry key of the source dataset (e.g. "set1", "set2", "set3"). |
| `experiment_name` | String | Display name of the dataset (e.g. "NASA IMS Set 1"). |
| `bearings_present` | String | Comma-separated list of bearings tracked in this dataset. |
| `channels_present` | String | Comma-separated list of channels recorded in this dataset. |
| `timestamp` | String (ISO) | Extracted date-time stamp from the file name. |
| `filename` | String | Base name of the ASCII data file. |
| `absolute_path` | String | Absolute local system path to the file. |
| `relative_path` | String | Workspace relative path to the file. |
| `signal_length` | Integer | Total sample lines (rows of vibration values). |
| `num_columns` | Integer | Column count found in the file. |
| `expected_channels_per_file` | Integer | Column count declared by the dataset's registry entry; a mismatch marks the file `parser_status="Failed"`. |
| `sampling_frequency_hz` | Float | Data acquisition sampling rate. |
| `file_sha256` | String | Calculated SHA256 checksum of raw data. |
| `parser_status` | String | Parser result indicator (`Success` or `Failed`). |
| `extraction_time` | String (ISO) | UTC timestamp when processing occurred. |
| `notes` | String | Channel mappings and failure descriptors, derived from the dataset registry. |
"""
    schema_path = report_dir / "metadata_schema.md"
    with open(schema_path, "w", encoding="utf-8") as f:
        f.write(schema_content)
    logger.info(f"Saved metadata schema to: {schema_path.resolve()}")


def main() -> None:
    """Main execution block to load configurations and trigger metadata parsing."""
    # Load configuration
    config = load_config()
    
    # Configure logger
    logger = setup_logger(name="ingestion", config=config)
    logger.info("Initializing Data Ingestion & Metadata Extraction Phase 1...")
    
    paths = config.get("paths", {})
    processed_dir = Path(paths.get("processed_data_dir", "data/processed"))
    report_dir = Path(paths.get("report_dir", "reports"))

    # Ensure outputs directories exist
    ensure_directory_exists(processed_dir)
    ensure_directory_exists(report_dir)

    try:
        df = extract_metadata(config, processed_dir, report_dir, logger)
        generate_statistics_and_docs(df, config, report_dir, logger)
        logger.info("Data Ingestion & Metadata Extraction Phase 1 complete successfully.")
    except Exception as exc:
        logger.exception(f"Unhandled exception during metadata extraction: {exc}")
        raise exc


if __name__ == "__main__":
    main()
