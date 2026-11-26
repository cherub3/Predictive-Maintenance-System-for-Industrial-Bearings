"""Failure event indexer for run-to-failure bearing units.

Computes t_fail (last valid acquisition timestamp) for each RTF unit and
classifies censored units. Per architecture §7: t_fail = max(timestamp)
of that unit's valid rows. Cross-checks against Phase 3 health evidence
at warning level only — Phase 5 owns the label; Phase 3 is informational.

Units are identified by their qualified unit_id (f"{dataset_id}::{bearing_id}"),
never by bare bearing_id — the same bearing_id string (e.g. "bearing_3") refers
to physically unrelated bearings in different datasets.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config import get_censored_units, get_rtf_units


@dataclass
class UnitFailureIndex:
    """Failure index record for one bearing unit."""

    unit_id: str
    is_censored: bool
    t_fail: Optional[pd.Timestamp]  # None for censored units
    total_life_minutes: Optional[float]  # None for censored units
    total_cycles: Optional[int]  # None for censored units


def build_failure_index(
    feature_store: pd.DataFrame,
    config: Dict[str, Any],
    logger: logging.Logger,
    health_indicators_path: Optional[Path] = None,
) -> Dict[str, UnitFailureIndex]:
    """Compute failure event indices for all bearing units.

    For RTF units: t_fail = max(timestamp) of that unit's rows.
    For censored units: t_fail = None, is_censored = True.

    If health_indicators_path is provided, cross-checks t_fail against
    the end of the kurtosis/RMS runaway signal. Mismatches are warnings only.

    Args:
        feature_store: Phase 4 feature store DataFrame, with a `unit_id` column.
        config: Full project configuration dictionary.
        logger: Pipeline logger.
        health_indicators_path: Optional path to Phase 3 health indicators
            for cross-validation of failure terminus.

    Returns:
        Dict mapping unit_id to UnitFailureIndex.
    """
    rtf_units: List[str] = get_rtf_units(config)
    censored_units: List[str] = get_censored_units(config)

    logger.info(f"Building failure index — RTF units: {rtf_units}, censored: {censored_units}")

    failure_index: Dict[str, UnitFailureIndex] = {}
    all_units = rtf_units + censored_units

    for unit_id in all_units:
        unit_rows = feature_store[feature_store["unit_id"] == unit_id]

        if unit_rows.empty:
            if unit_id in rtf_units:
                raise ValueError(
                    f"RTF unit '{unit_id}' has no rows in the feature store. "
                    "Cannot compute t_fail for a missing RTF unit. "
                    "Ensure Phase 4 ran successfully and unit_id values match the datasets registry."
                )
            logger.warning(f"  {unit_id}: no rows in feature store; skipping censored unit")
            continue

        if unit_id in censored_units:
            failure_index[unit_id] = UnitFailureIndex(
                unit_id=unit_id,
                is_censored=True,
                t_fail=None,
                total_life_minutes=None,
                total_cycles=None,
            )
            logger.info(f"  {unit_id}: RIGHT-CENSORED (survived test; no failure event)")

        elif unit_id in rtf_units:
            # t_fail = max timestamp of unit's rows (architecture §6.3, §7)
            # Per-channel timestamps are all identical for the same file, so
            # t_fail is the same regardless of which channel we use.
            unique_ts = unit_rows["timestamp"].sort_values()
            t_start = unique_ts.iloc[0]
            t_fail = unique_ts.iloc[-1]
            total_life_minutes = (t_fail - t_start).total_seconds() / 60.0
            # Total cycles = number of unique acquisition timestamps for this unit
            total_cycles = unique_ts.nunique()

            failure_index[unit_id] = UnitFailureIndex(
                unit_id=unit_id,
                is_censored=False,
                t_fail=t_fail,
                total_life_minutes=total_life_minutes,
                total_cycles=total_cycles,
            )
            logger.info(
                f"  {unit_id}: RTF — t_fail={t_fail}, "
                f"total_life={total_life_minutes:.1f} min, {total_cycles} cycles"
            )

    # Cross-check against Phase 3 health indicators (warning only)
    if health_indicators_path and health_indicators_path.exists():
        _cross_check_health_terminus(failure_index, health_indicators_path, logger)

    return failure_index


def _cross_check_health_terminus(
    failure_index: Dict[str, UnitFailureIndex],
    health_indicators_path: Path,
    logger: logging.Logger,
) -> None:
    """Warning-level cross-check: does t_fail coincide with health signal end?

    Phase 3 is authoritative on health; Phase 5 owns the label. Mismatches
    raise a warning and are recorded but do not stop the pipeline.

    Args:
        failure_index: Dict of UnitFailureIndex records.
        health_indicators_path: Path to Phase 3 health indicators parquet.
        logger: Pipeline logger.
    """
    try:
        health_df = pd.read_parquet(health_indicators_path)
        health_df["timestamp"] = pd.to_datetime(health_df["timestamp"])
        health_df["unit_id"] = health_df["dataset_id"] + "::" + health_df["bearing_id"]
    except Exception as exc:
        logger.warning(f"Could not load health indicators for cross-check: {exc}")
        return

    for unit_id, idx in failure_index.items():
        if idx.is_censored or idx.t_fail is None:
            continue

        unit_health = health_df[health_df["unit_id"] == unit_id]
        if unit_health.empty:
            logger.warning(f"No Phase 3 health rows found for {unit_id}; skipping cross-check")
            continue

        health_max_ts = unit_health["timestamp"].max()
        if health_max_ts != idx.t_fail:
            logger.warning(
                f"t_fail mismatch for {unit_id}: "
                f"Phase 5 t_fail={idx.t_fail}, Phase 3 health max={health_max_ts}. "
                "Phase 5 label is authoritative; recording for human review."
            )
        else:
            logger.info(f"  {unit_id}: t_fail cross-check OK ({idx.t_fail}) ✓")
