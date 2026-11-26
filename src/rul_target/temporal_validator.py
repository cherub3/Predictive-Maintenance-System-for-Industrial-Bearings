"""Temporal integrity validator for the Phase 4 feature store.

Enforces Phase 5 Architecture §8: chronology preservation, time ordering,
timestamp validity, and sequence integrity. Must pass before any label
generation begins.

All checks are fatal on violation — a non-monotonic or corrupted timeline
cannot produce trustworthy RUL labels.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd

from src.config import get_all_dataset_ids, get_censored_units, get_dataset_config, get_rtf_units


@dataclass
class TemporalValidationResult:
    """Container for temporal validation outcomes."""

    passed: bool
    fatals: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    per_unit_summary: Dict[str, Dict] = field(default_factory=dict)


def validate_temporal_integrity(
    feature_store: pd.DataFrame,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> TemporalValidationResult:
    """Validate all temporal-integrity invariants of the feature store.

    Checks (all fatal on failure unless noted):
    1. timestamp column dtype is datetime64[ns]
    2. No NaT values in timestamp column
    3. All timestamps within each dataset's own frozen run window (checked
       per dataset_id, since each registered dataset has a different window)
    4. Strictly increasing timestamps within each (unit_id, channel)
    5. Expected unit IDs present (qualified dataset_id::bearing_id)

    Args:
        feature_store: Phase 4 feature store DataFrame, with a `unit_id` column.
        config: Full project configuration dictionary.
        logger: Pipeline logger.

    Returns:
        TemporalValidationResult with detailed findings.
    """
    result = TemporalValidationResult(passed=True)

    logger.info("Temporal integrity validation started")

    # Check 1: timestamp dtype
    if not pd.api.types.is_datetime64_any_dtype(feature_store["timestamp"]):
        result.fatals.append(
            f"timestamp dtype is {feature_store['timestamp'].dtype}; expected datetime64[ns]"
        )
        result.passed = False
    else:
        result.info.append("timestamp dtype: datetime64[ns] ✓")

    # Check 2: NaT values
    nat_count = feature_store["timestamp"].isna().sum()
    if nat_count > 0:
        result.fatals.append(f"timestamp column contains {nat_count} NaT values")
        result.passed = False
    else:
        result.info.append("NaT in timestamp: 0 ✓")

    # Check 3: run window, evaluated PER DATASET — each registered dataset has its
    # own documented run window (see config.yaml datasets.sets.*.run_window); a
    # single global window would reject every row from every dataset but one.
    if not result.fatals:  # Only if timestamps are valid datetimes
        total_out_of_range = 0
        for dataset_id in get_all_dataset_ids(config):
            dataset_cfg = get_dataset_config(config, dataset_id)
            run_window = dataset_cfg.get("run_window", {})
            window_start = pd.Timestamp(run_window["start"])
            window_end = pd.Timestamp(run_window["end"]) + pd.Timedelta(days=1)

            dataset_rows = feature_store[feature_store["dataset_id"] == dataset_id]
            out_of_range = (
                (dataset_rows["timestamp"] < window_start) |
                (dataset_rows["timestamp"] >= window_end)
            ).sum()
            if out_of_range > 0:
                result.fatals.append(
                    f"[{dataset_id}] {out_of_range} timestamps outside frozen run window "
                    f"[{window_start.date()} → {window_end.date()})"
                )
                result.passed = False
                total_out_of_range += out_of_range

        if total_out_of_range == 0:
            result.info.append("All timestamps within each dataset's own run window ✓")

    # Check 4: strictly increasing within each (unit_id, channel)
    # Guard: .diff() on non-datetime dtype raises TypeError and masks the real cause.
    # Skip entirely when Check 1 (dtype) has already failed.
    if not pd.api.types.is_datetime64_any_dtype(feature_store["timestamp"]):
        result.info.append(
            "Check 4 (monotonicity) skipped: timestamp dtype is not datetime64 "
            "(see FATAL above — fix the dtype first)"
        )
    else:
        groups = feature_store.groupby(["unit_id", "channel"], sort=True)
        for (unit_id, channel), group in groups:
            group_sorted = group.sort_values("timestamp")
            diffs = group_sorted["timestamp"].diff()
            # Drop the first NaT diff
            non_positive = (diffs.dropna() <= pd.Timedelta(0)).sum()
            if non_positive > 0:
                result.fatals.append(
                    f"Non-strictly-increasing timestamps in unit={unit_id} "
                    f"channel={channel}: {non_positive} violations"
                )
                result.passed = False

            # Per-unit summary
            key = f"{unit_id}_ch{channel}"
            result.per_unit_summary[key] = {
                "row_count": len(group),
                "min_ts": str(group_sorted["timestamp"].iloc[0]),
                "max_ts": str(group_sorted["timestamp"].iloc[-1]),
                "monotonic_ok": non_positive == 0,
            }

        if all(v["monotonic_ok"] for v in result.per_unit_summary.values()):
            result.info.append("Timestamps strictly increasing within all (unit_id, channel) groups ✓")

    # Check 5: expected unit IDs (qualified dataset_id::bearing_id)
    rtf_units = get_rtf_units(config)
    censored_units = get_censored_units(config)
    expected_units = set(rtf_units + censored_units)
    actual_units = set(feature_store["unit_id"].unique())
    missing = expected_units - actual_units
    unexpected = actual_units - expected_units
    if missing:
        result.fatals.append(f"Missing expected unit IDs: {sorted(missing)}")
        result.passed = False
    if unexpected:
        result.warnings.append(f"Unexpected unit IDs in feature store: {sorted(unexpected)}")
    if not missing:
        result.info.append(f"All expected unit IDs present: {sorted(expected_units)} ✓")

    severity = "PASSED" if result.passed else "FAILED"
    logger.info(f"Temporal integrity validation: {severity}")
    for fatal in result.fatals:
        logger.error(f"  FATAL: {fatal}")
    for warning in result.warnings:
        logger.warning(f"  WARNING: {warning}")
    for info in result.info:
        logger.info(f"  INFO: {info}")

    return result
