"""RUL label generator for the Phase 5 training dataset.

Implements the approved RUL definition (Architecture §6–7):
  RUL(t) = (t_fail − t) in minutes,   RUL(t_fail) = 0

Labels are generated only for RTF units (one per registered dataset with a
documented run-to-failure bearing). Censored units receive is_censored=True
and NULL labels — fabricated labels are never assigned.

Output grain: (unit_id, channel, timestamp) — identical to the Phase 4
feature store grain, so no row multiplication occurs. bearing_id and
dataset_id are retained as descriptive columns.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.rul_target.failure_indexer import UnitFailureIndex


def generate_rul_labels(
    feature_store: pd.DataFrame,
    failure_index: Dict[str, UnitFailureIndex],
    config: Dict[str, Any],
    logger: logging.Logger,
) -> pd.DataFrame:
    """Generate per-row RUL labels for all bearing units.

    For RTF units: rul_minutes = (t_fail - t).total_seconds() / 60
                   rul_cycles  = index distance from failure row (descending)
    For censored:  rul_minutes = NaN, rul_cycles = NaN, is_censored = True

    Optional rul_frac = rul_minutes / total_life_minutes, config-gated.

    Args:
        feature_store: Phase 4 feature store DataFrame.
        failure_index: Dict of UnitFailureIndex from failure_indexer.
        config: Full project configuration dictionary.
        logger: Pipeline logger.

    Returns:
        DataFrame with grain (unit_id, channel, timestamp) and label columns
        (plus descriptive dataset_id/bearing_id columns).
    """
    rul_cfg = config.get("rul_target", {})
    emit_rul_cycles: bool = rul_cfg.get("emit_rul_cycles", True)
    emit_rul_frac: bool = rul_cfg.get("emit_rul_frac", False)
    knee_cfg = rul_cfg.get("knee_capping", {})
    knee_enabled: bool = knee_cfg.get("enabled", False)

    logger.info("Generating RUL labels")

    label_rows: List[pd.DataFrame] = []

    for unit_id, idx in failure_index.items():
        unit_rows = feature_store[feature_store["unit_id"] == unit_id].copy()

        if unit_rows.empty:
            logger.warning(f"No rows for {unit_id}; skipping label generation")
            continue

        # Preserve grain columns (unit_id primary; dataset_id/bearing_id descriptive) + label columns
        label_df = unit_rows[["unit_id", "dataset_id", "bearing_id", "channel", "timestamp"]].copy()
        label_df = label_df.sort_values(["channel", "timestamp"]).reset_index(drop=True)

        if idx.is_censored:
            label_df["rul_minutes"] = np.nan
            label_df["is_censored"] = True
            if emit_rul_cycles:
                label_df["rul_cycles"] = np.nan
            if emit_rul_frac:
                label_df["rul_frac"] = np.nan
            logger.info(f"  {unit_id}: censored — all labels NULL ({len(label_df)} rows)")

        else:
            t_fail: pd.Timestamp = idx.t_fail
            total_life_minutes: float = idx.total_life_minutes

            # Primary label: elapsed minutes to failure (per-channel, per-timestamp)
            label_df["rul_minutes"] = label_df["timestamp"].apply(
                lambda t: (t_fail - t).total_seconds() / 60.0
            )
            label_df["is_censored"] = False

            if emit_rul_cycles:
                # Remaining cycles = number of unique future timestamps for this unit
                # Compute per-channel: descending count of rows after this timestamp
                # rul_cycles is the same value for both channels at the same timestamp
                # because both channels have the same acquisition schedule.
                unique_timestamps = sorted(label_df["timestamp"].unique())
                ts_to_remaining = {
                    ts: len(unique_timestamps) - i - 1
                    for i, ts in enumerate(unique_timestamps)
                }
                label_df["rul_cycles"] = label_df["timestamp"].map(ts_to_remaining)

            if emit_rul_frac:
                # Per-unit normalization only — never pool total_life across units
                if total_life_minutes > 0:
                    label_df["rul_frac"] = label_df["rul_minutes"] / total_life_minutes
                else:
                    label_df["rul_frac"] = np.nan
                    logger.warning(f"  {unit_id}: total_life_minutes=0; rul_frac set to NaN")

            if knee_enabled:
                label_df = _apply_knee_capping(label_df, knee_cfg, total_life_minutes, logger)

            # Assert no negative RUL — would indicate timestamps after t_fail in feature_store
            negative_count = (label_df["rul_minutes"] < -1e-9).sum()
            if negative_count > 0:
                raise ValueError(
                    f"RTF unit '{unit_id}': {negative_count} rows have rul_minutes < 0. "
                    "This means the feature store contains timestamps after t_fail. "
                    f"t_fail={t_fail}. Check Phase 4 feature store for out-of-range rows."
                )

            min_rul = label_df["rul_minutes"].min()
            max_rul = label_df["rul_minutes"].max()
            logger.info(
                f"  {unit_id}: {len(label_df)} rows labelled — "
                f"rul_minutes range [{min_rul:.1f}, {max_rul:.1f}]"
            )

        label_rows.append(label_df)

    if not label_rows:
        raise ValueError("No label rows generated — check failure_index and feature_store contents")

    labels = pd.concat(label_rows, ignore_index=True)
    labels = labels.sort_values(["unit_id", "channel", "timestamp"]).reset_index(drop=True)

    logger.info(
        f"Label generation complete: {len(labels)} total rows "
        f"({labels['is_censored'].sum()} censored, "
        f"{(~labels['is_censored']).sum()} labelled)"
    )
    return labels


def _apply_knee_capping(
    label_df: pd.DataFrame,
    knee_cfg: Dict,
    total_life_minutes: float,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Apply optional knee-capping: flat RUL until r_early_minutes, linear thereafter.

    Early life is assigned a constant RUL equal to r_early_minutes. This
    reflects uncertainty about when meaningful degradation begins. Only active
    when knee_capping.enabled = true in config.

    Args:
        label_df: DataFrame with rul_minutes column already computed.
        knee_cfg: Knee-capping sub-config dict.
        total_life_minutes: Total observed life of this unit.
        logger: Pipeline logger.

    Returns:
        label_df with rul_minutes capped where appropriate.
    """
    r_early: Optional[float] = knee_cfg.get("r_early_minutes")
    if r_early is None:
        logger.warning("knee_capping.enabled=true but r_early_minutes is null; skipping")
        return label_df

    # Cap RUL at r_early for rows where raw rul_minutes > r_early
    cap_mask = label_df["rul_minutes"] > r_early
    label_df.loc[cap_mask, "rul_minutes"] = float(r_early)
    logger.info(
        f"  Knee-capping applied: {cap_mask.sum()} rows capped at {r_early} minutes"
    )
    return label_df


def save_rul_labels(
    labels: pd.DataFrame,
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """Atomically write RUL labels to parquet.

    Args:
        labels: Labels DataFrame.
        output_path: Final destination path.
        logger: Pipeline logger.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".parquet.tmp")
    try:
        labels.to_parquet(tmp_path, index=False, compression="snappy")
        tmp_path.replace(output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    logger.info(f"RUL labels saved: {output_path} ({len(labels)} rows)")
