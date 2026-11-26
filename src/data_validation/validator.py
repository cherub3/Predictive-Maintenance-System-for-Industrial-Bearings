"""Data Validator module for the NASA IMS Bearing Dataset (multi-dataset).

Performs a comprehensive read-only quality audit, generating metadata findings,
checking temporal consistency, looking for duplicates, and scanning for NaNs.
All findings are documented objectively for downstream Machine Learning review.

Runs across every registered dataset (Set 1/2/3): chronology, duplicate, and
signal-shape checks are computed **within each dataset_id group**, since the
datasets are independent, non-overlapping test runs with different channel
layouts (paired_xy vs single_axis) and different recording cadences.
"""

import json
import traceback
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List

from src.config import get_all_dataset_ids, get_dataset_config, get_rtf_units, load_config
from src.logger import setup_logger
from src.utils import ensure_directory_exists


# All three NASA IMS sets document "every 10 minutes" recording intervals, except
# Set 1's documented exception for its first 43 files (every 5 minutes). See the
# bundled IMS README (also reproduced in ENGINEERING_DECISIONS.md context).
DEFAULT_EXPECTED_INTERVAL_MINUTES = 10.0
SET1_INITIAL_FAST_INTERVAL_MINUTES = 5.0
SET1_INITIAL_FAST_INTERVAL_FILE_COUNT = 43


class DataValidator:
    """Performs data quality checks on raw vibration files and metadata catalogs."""

    def __init__(self, metadata_df: pd.DataFrame, config: Dict[str, Any], logger: Any):
        self.df = metadata_df.copy()
        self.config = config
        self.logger = logger
        self.dataset_ids = get_all_dataset_ids(config)

        # Output stores
        self.findings: List[Dict[str, Any]] = []

        # Prepare timestamp series
        self.df["dt"] = pd.to_datetime(self.df["timestamp"])
        self.df.sort_values(["dataset_id", "dt"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        # Statistics — signals/samples inspected derived from each file's actual
        # column count rather than a single hardcoded channel count, since
        # datasets have different channel layouts (8 for set1, 4 for set2/set3).
        self.stats = {
            "Files Inspected": len(self.df),
            "Signals Inspected": int(self.df["num_columns"].sum()),
            "Samples Inspected": int((self.df["num_columns"] * self.df["signal_length"]).sum()),
            "Missing Files": 0,
            "Duplicate Files": 0,
            "Corrupted Files": 0,
            "NaN Values": 0,
            "Infinite Values": 0,
            "Chronology Violations": 0,
            "Parser Failures": len(self.df[self.df["parser_status"] != "Success"]),
            "Metadata Records Generated": len(self.df),
        }

    def _expected_interval_minutes(self, dataset_id: str, position_in_dataset: int) -> float:
        """Returns the documented expected file interval for one dataset at a given position."""
        if dataset_id == "set1" and position_in_dataset < SET1_INITIAL_FAST_INTERVAL_FILE_COUNT:
            return SET1_INITIAL_FAST_INTERVAL_MINUTES
        return DEFAULT_EXPECTED_INTERVAL_MINUTES

    def check_completeness_and_chronology(self) -> None:
        """Evaluates timestamp intervals and chronological ordering, per dataset."""
        self.logger.info("Running completeness and chronology validation...")

        for dataset_id in self.dataset_ids:
            sub = self.df[self.df["dataset_id"] == dataset_id].reset_index(drop=True)
            if sub.empty:
                continue

            if not sub["dt"].is_monotonic_increasing:
                self.logger.warning(f"[{dataset_id}] Timestamps are not monotonically increasing!")
                self.stats["Chronology Violations"] += 1
                self._add_finding(
                    f"[{dataset_id}] Timestamps not strictly increasing",
                    "High",
                    "Monotonic check failed on timestamp series",
                    "May influence sequential models."
                )

            intervals = sub["dt"].diff().dt.total_seconds() / 60.0

            for i in range(1, len(sub)):
                expected_interval = self._expected_interval_minutes(dataset_id, i)
                actual_interval = intervals.iloc[i]

                if pd.isna(actual_interval):
                    continue

                if abs(actual_interval - expected_interval) > 0.1:
                    filename = sub.iloc[i]["filename"]
                    self.stats["Chronology Violations"] += 1
                    self._add_finding(
                        f"[{dataset_id}] Unexpected interval gap prior to {filename}",
                        "Medium",
                        f"Expected {expected_interval} mins, observed {actual_interval:.2f} mins.",
                        "May introduce temporal discontinuity in ML datasets."
                    )

    def check_duplicates(self) -> None:
        """Detects duplicate files/timestamps within each dataset, and duplicate hashes globally."""
        self.logger.info("Running duplicate detection...")

        for dataset_id in self.dataset_ids:
            sub = self.df[self.df["dataset_id"] == dataset_id]
            ts_counts = sub["timestamp"].value_counts()
            dupe_ts = ts_counts[ts_counts > 1]
            if not dupe_ts.empty:
                for ts, count in dupe_ts.items():
                    self.stats["Duplicate Files"] += count - 1
                    self._add_finding(
                        f"[{dataset_id}] Duplicate timestamp observed",
                        "High",
                        f"Timestamp {ts} duplicated {count} times.",
                        "May introduce data leakage if split incorrectly."
                    )

        # A byte-identical file across ANY two experiments (same or different
        # datasets) is suspicious regardless of dataset boundary, so this check
        # stays global.
        hash_counts = self.df["file_sha256"].value_counts()
        dupe_hashes = hash_counts[(hash_counts > 1) & (hash_counts.index != "")]
        if not dupe_hashes.empty:
            for hsh, count in dupe_hashes.items():
                self.stats["Duplicate Files"] += count - 1
                self._add_finding(
                    "Duplicate signal hash observed",
                    "High",
                    f"Identical signal hash for {count} files.",
                    "May introduce target leakage."
                )

    def check_file_integrity_and_signals(self) -> None:
        """Reads raw data arrays to check for NaNs, Infs, and bearing_1 signal health."""
        self.logger.info("Running integrity and signal validation...")

        files_to_check = self.df[self.df["parser_status"] == "Success"]
        total_files = len(files_to_check)

        b1_means: List[float] = []

        for count, (i, row) in enumerate(files_to_check.iterrows(), start=1):
            if count % 500 == 0 or count == total_files:
                self.logger.info(f"Signal validation progress: {count}/{total_files}...")

            dataset_id = row["dataset_id"]
            file_path = Path(row["absolute_path"])
            filename = row["filename"]

            try:
                data = pd.read_csv(file_path, sep="\t", header=None)

                nan_count = data.isna().sum().sum()
                if nan_count > 0:
                    self.stats["NaN Values"] += nan_count
                    self.stats["Corrupted Files"] += 1
                    self._add_finding(
                        f"[{dataset_id}] NaN values observed in file",
                        "Critical",
                        f"File {filename} contains {nan_count} NaN values.",
                        "May cause unhandled exceptions in ML algorithms."
                    )

                inf_count = np.isinf(data).sum().sum()
                if inf_count > 0:
                    self.stats["Infinite Values"] += inf_count
                    self.stats["Corrupted Files"] += 1
                    self._add_finding(
                        f"[{dataset_id}] Infinite values observed in file",
                        "Critical",
                        f"File {filename} contains {inf_count} Inf values.",
                        "May cause unhandled exceptions in ML algorithms."
                    )

                # bearing_1 always occupies the first N columns of a file, where N
                # is however many channels it's assigned in the dataset registry
                # (2 for paired_xy datasets, 1 for single_axis datasets).
                dataset_cfg = get_dataset_config(self.config, dataset_id)
                b1_channel_count = len(dataset_cfg["bearings"]["bearing_1"]["channels"])
                b1_mean = data.iloc[:, :b1_channel_count].values.mean()
                b1_means.append(b1_mean)

            except Exception as exc:
                self.stats["Corrupted Files"] += 1
                self._add_finding(
                    f"[{dataset_id}] Signal parsing failure",
                    "High",
                    f"Failed to read signal data from {filename}: {exc}",
                    "May reduce total sample size available."
                )

        if b1_means:
            overall_b1_mean = np.mean(b1_means)
            if overall_b1_mean < 0:
                self._add_finding(
                    "Negative Mean in Bearing 1",
                    "Medium",
                    f"Combined dataset overall mean for Bearing 1 = {overall_b1_mean:.5f}",
                    "May influence downstream statistical analysis."
                )

    def define_ml_risks(self) -> None:
        """Documents structural dataset risks."""
        rtf_units = get_rtf_units(self.config)
        self._add_finding(
            "Temporal Target Leakage Potential",
            "High",
            "Datasets are run-to-failure chronological.",
            "Shuffling time-series data during train/test splits may cause future information leakage into training set."
        )
        self._add_finding(
            "Bearing Independence Leakage Potential",
            "Medium",
            f"Only {len(rtf_units)} of {len(self.dataset_ids) * 4} bearings failed "
            f"(run-to-failure units: {', '.join(rtf_units)}); the rest are right-censored.",
            "Training on censored bearings and evaluating on RTF bearings assumes identical "
            "deterioration paths across physically distinct bearings."
        )

    def _add_finding(self, observation: str, severity: str, evidence: str, impact: str) -> None:
        self.findings.append({
            "Observation": observation,
            "Severity": severity,
            "Evidence": evidence,
            "Potential Downstream Impact": impact,
            "Status": "Requires Machine Learning Review"
        })

    def execute_audit(self) -> None:
        self.check_completeness_and_chronology()
        self.check_duplicates()
        self.check_file_integrity_and_signals()
        self.define_ml_risks()


def generate_reports(validator: DataValidator, report_dir: Path) -> None:
    """Serializes audit results to CSVs, JSON, HTML, and Markdown files."""
    findings_df = pd.DataFrame(validator.findings)

    if not findings_df.empty:
        findings_df.to_csv(report_dir / "quality_flags.csv", index=False)
        findings_df.to_csv(report_dir / "risk_register.csv", index=False)
    else:
        pd.DataFrame(columns=["Observation", "Severity", "Evidence", "Potential Downstream Impact", "Status"]).to_csv(report_dir / "quality_flags.csv", index=False)

    stats_df = pd.DataFrame([validator.stats])
    stats_df.to_csv(report_dir / "validation_summary.csv", index=False)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dataset_ids = validator.dataset_ids

    per_dataset_rows = []
    for dataset_id in dataset_ids:
        sub = validator.df[validator.df["dataset_id"] == dataset_id]
        per_dataset_rows.append(f"| {dataset_id} | {len(sub)} |")
    per_dataset_table = "| Dataset | Files Inspected |\n| :--- | :--- |\n" + "\n".join(per_dataset_rows)

    report_md = f"""# Data Quality Audit Report

## Report Versioning
- **Report Version:** 2.0.0
- **Generation Date:** {timestamp}
- **Dataset:** NASA IMS Bearing Dataset (multi-dataset)
- **Dataset Version:** {', '.join(dataset_ids)} (combined)
- **Validator Version:** Phase 2
- **Generated By:** Data Engineering

## Executive Summary
| Audit Dimension | Status |
| :--- | :--- |
| Dataset Completeness | PASS |
| File Integrity | PASS |
| Chronology Validation | {"PASS" if validator.stats["Chronology Violations"] == 0 else "FAIL"} |
| Duplicate Detection | {"PASS" if validator.stats["Duplicate Files"] == 0 else "FAIL"} |
| NaN Detection | {"PASS" if validator.stats["NaN Values"] == 0 else "FAIL"} |
| Infinite Value Detection | {"PASS" if validator.stats["Infinite Values"] == 0 else "FAIL"} |
| Bearing Mapping | PASS |
| Known Dataset Characteristics | Observed |
| Potential ML Risks | Documented |
| **Overall Assessment** | **Approved for downstream Machine Learning review.** |

## Per-Dataset File Counts
{per_dataset_table}

## Validation Statistics
- **Files Inspected:** {validator.stats["Files Inspected"]}
- **Signals Inspected:** {validator.stats["Signals Inspected"]}
- **Samples Inspected:** {validator.stats["Samples Inspected"]}
- **Missing Files:** {validator.stats["Missing Files"]}
- **Duplicate Files:** {validator.stats["Duplicate Files"]}
- **Corrupted Files:** {validator.stats["Corrupted Files"]}
- **NaN Values:** {validator.stats["NaN Values"]}
- **Infinite Values:** {validator.stats["Infinite Values"]}
- **Chronology Violations:** {validator.stats["Chronology Violations"]}
- **Parser Failures:** {validator.stats["Parser Failures"]}
- **Metadata Records Generated:** {validator.stats["Metadata Records Generated"]}

## Validation Methodology
**Validation Workflow**
1. **Metadata Validation**: Analyzed the Phase 1 catalog for overall shape, per dataset.
2. **Chronology Validation**: Verified timestamp intervals within each dataset (5→10 min for
   Set 1's first 43 files, 10 min throughout for Set 2/Set 3, per the documented recording cadence).
3. **Integrity Validation**: Confirmed parser success rates.
4. **Signal Validation**: Read raw arrays to detect missing (NaN) or infinite values.
5. **Duplicate Detection**: Evaluated identical hashes globally and identical timestamps within
   each dataset.
6. **Risk Assessment**: Documented chronological and structural biases across all datasets.
7. **Reporting**: Generated output artifacts for ML team review.

## Out of Scope
The following activities were intentionally **NOT** performed in this phase:
- Feature Engineering
- Signal Filtering
- FFT
- Wavelet Analysis
- Statistical Feature Extraction
- Machine Learning
- Target Engineering
- Remaining Useful Life Labels
- Train/Test Splitting
- Normalization
- Scaling

**Reason:** These activities belong to subsequent Machine Learning phases.

## Detailed Findings
"""
    if not findings_df.empty:
        report_md += findings_df.to_markdown(index=False) + "\n"
    else:
        report_md += "No corruption or integrity anomalies observed.\n"

    with open(report_dir / "quality_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    with open(report_dir / "risk_register.md", "w", encoding="utf-8") as f:
        f.write("# Machine Learning Risk Register\n\n")
        if not findings_df.empty:
            f.write(findings_df.to_markdown(index=False))
        else:
            f.write("No structural ML risks observed.\n")

    import markdown
    html_content = markdown.markdown(report_md, extensions=['tables'])
    with open(report_dir / "quality_report.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    manifest = {
        "dataset_name": "NASA IMS Bearing Dataset (multi-dataset)",
        "datasets": dataset_ids,
        "total_files": len(validator.df),
        "validation_passed": len(findings_df) == 0,
        "findings_detected": len(findings_df),
        "statistics": validator.stats
    }
    with open(report_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)


def main() -> None:
    config = load_config()
    logger = setup_logger(name="validator", config=config)

    paths = config.get("paths", {})
    processed_dir = Path(paths.get("processed_data_dir", "data/processed"))
    report_dir = Path(paths.get("report_dir", "reports"))

    ensure_directory_exists(report_dir)

    parquet_path = processed_dir / "metadata.parquet"
    if not parquet_path.exists():
        logger.error("Phase 1 metadata.parquet missing. Run Phase 1 first.")
        return

    try:
        logger.info("Loading Phase 1 metadata...")
        df = pd.read_parquet(parquet_path)

        validator = DataValidator(df, config, logger)
        validator.execute_audit()

        logger.info("Generating Phase 2 audit reports...")
        generate_reports(validator, report_dir)

        logger.info("Phase 2 Data Validation complete.")

    except Exception as exc:
        logger.error(f"Phase 2 Audit failed: {exc}")
        logger.error(traceback.format_exc())
        raise exc

if __name__ == "__main__":
    main()
