"""Training dataset builder — merges features, labels, and fold assignments.

Runs ONLY after leakage_validator reports all_passed=True. Joins the Phase 4
feature store with Phase 5 RUL labels and fold assignments to produce the
final training_dataset.parquet.

Output grain: (unit_id, channel, timestamp) — RTF units only.
Censored units are excluded from training_dataset.parquet (no labels).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.config import get_rtf_units


def build_training_dataset(
    feature_store: pd.DataFrame,
    rul_labels: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> pd.DataFrame:
    """Join features + labels + fold assignments into the training dataset.

    Only RTF (non-censored) rows are included. Censored rows exist in the
    feature store and label DataFrame but are excluded here — they carry
    no ground-truth RUL.

    Merge strategy (both left-join on grain):
      1. RTF feature rows ← left join → rul_labels (on unit_id, channel, timestamp)
      2. result ← left join → fold_assignments (on unit_id, channel, timestamp)

    Args:
        feature_store: Phase 4 feature store DataFrame (all units, all channels),
            with a `unit_id` column.
        rul_labels: Labels DataFrame from label_generator.
        fold_assignments: Fold assignment DataFrame from split_generator (RTF only).
        config: Full project configuration dictionary.
        logger: Pipeline logger.

    Returns:
        training_dataset DataFrame — RTF rows only, features + labels + fold cols.
    """
    rtf_units: List[str] = get_rtf_units(config)

    logger.info("Building training dataset — RTF units: %s", rtf_units)

    # 1. Restrict feature store to RTF units only
    rtf_features = feature_store[feature_store["unit_id"].isin(rtf_units)].copy()
    logger.info("  RTF feature rows: %d", len(rtf_features))

    # 2. Restrict labels to RTF units only (sanity — label_generator also excludes these)
    rtf_labels = rul_labels[rul_labels["unit_id"].isin(rtf_units)].copy()
    # Drop is_censored from labels — it's always False for RTF; we'll confirm with an assert
    if rtf_labels["is_censored"].any():
        raise ValueError("rul_labels contains is_censored=True rows for RTF units — logic error")
    logger.info("  RTF label rows: %d", len(rtf_labels))

    grain = ["unit_id", "channel", "timestamp"]
    # dataset_id/bearing_id are descriptive columns already present in rtf_features
    # (from the feature store) — exclude them from the labels/fold_assignments
    # merges to avoid _x/_y column-name collisions.
    descriptive_cols = {"dataset_id", "bearing_id"}

    # 3. Merge features with labels
    label_cols = [c for c in rtf_labels.columns if c not in grain + ["is_censored"] and c not in descriptive_cols]
    merged = rtf_features.merge(
        rtf_labels[grain + label_cols],
        on=grain,
        how="left",
    )

    unmatched_labels = merged["rul_minutes"].isna().sum()
    if unmatched_labels > 0:
        raise ValueError(
            f"{unmatched_labels} RTF feature rows have no matching label after merge — "
            "timestamp alignment between feature store and label DataFrame is broken"
        )

    # 4. Merge with fold assignments
    fold_cols = [c for c in fold_assignments.columns if c not in grain and c not in descriptive_cols]
    merged = merged.merge(
        fold_assignments[grain + fold_cols],
        on=grain,
        how="left",
    )

    fold_role_cols = [c for c in fold_cols if c.endswith("_role")]
    for role_col in fold_role_cols:
        unmatched = merged[role_col].isna().sum()
        if unmatched > 0:
            raise ValueError(
                f"{unmatched} RTF rows have no {role_col} assignment after merge — "
                "fold_assignments and feature store are misaligned"
            )

    # 5. Enforce monotonically non-increasing RUL within each (unit_id, channel)
    _assert_rul_monotone(merged, logger)

    merged = merged.sort_values(grain).reset_index(drop=True)

    logger.info(
        "Training dataset built: %d rows, %d columns — units: %s",
        len(merged),
        len(merged.columns),
        sorted(merged["unit_id"].unique()),
    )
    return merged


def _assert_rul_monotone(dataset: pd.DataFrame, logger: logging.Logger) -> None:
    """Assert RUL is monotonically non-increasing within each (unit_id, channel).

    RUL(t) = t_fail - t → as t increases toward t_fail, RUL decreases.
    Any increase in RUL over time indicates a label or timestamp ordering bug.

    Args:
        dataset: Training dataset DataFrame.
        logger: Pipeline logger.

    Raises:
        ValueError: If RUL is not monotonically non-increasing for any group.
    """
    violations = []
    for (unit_id, channel), group in dataset.groupby(["unit_id", "channel"]):
        group_sorted = group.sort_values("timestamp")
        rul_diffs = group_sorted["rul_minutes"].diff().dropna()
        # RUL must be non-increasing: diffs should be <= 0
        increases = (rul_diffs > 1e-9).sum()
        if increases > 0:
            violations.append(
                f"{unit_id} ch{channel}: {increases} rul_minutes increases "
                f"(max increase: {rul_diffs[rul_diffs > 0].max():.4f} min)"
            )

    if violations:
        raise ValueError(
            f"RUL monotonicity violated in training dataset — "
            + "; ".join(violations)
        )

    logger.info("  RUL monotonically non-increasing: verified ✓")


def save_training_dataset(
    dataset: pd.DataFrame,
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """Atomically write training dataset to parquet.

    Args:
        dataset: Training dataset DataFrame.
        output_path: Final destination path.
        logger: Pipeline logger.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".parquet.tmp")
    try:
        dataset.to_parquet(tmp_path, index=False, compression="snappy")
        tmp_path.replace(output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    logger.info("Training dataset saved: %s (%d rows)", output_path, len(dataset))
