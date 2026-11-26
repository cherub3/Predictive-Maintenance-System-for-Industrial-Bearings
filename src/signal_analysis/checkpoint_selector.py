"""Tier 2 checkpoint selection from health indicator time series.

Selects a representative set of acquisition files for spectral analysis,
covering early, mid, and late degradation stages, plus kurtosis-spike events.
Selection is deterministic and reproducible given fixed health indicator data.

Runs independently **per dataset**: each registered dataset (Set 1/2/3) is a
separate, non-overlapping run-to-failure experiment with its own life-timeline,
so "early/mid/late stage" only means something within one dataset's own run —
mixing datasets into one combined chronological timeline would let, e.g., all
of Set 1 (Oct-Nov 2003) count as "early" relative to Set 3 (Mar-Apr 2004)
purely by calendar date, which has no physical meaning.

Terminology:
  early_fraction / late_fraction define WINDOWS on the timeline (e.g. the first
  5% and last 10% of the run), not counts of files to select. A small, bounded
  number of representative checkpoints (governed by n_checkpoints) is then
  sampled from within each window, so total selected files never exceed the
  configured n_checkpoints budget **per dataset**.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.config import get_all_dataset_ids


def _select_checkpoints_for_dataset(
    health_df: pd.DataFrame,
    n_checkpoints: int,
    early_frac: float,
    late_frac: float,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Selects checkpoint files representing key degradation stages within one dataset.

    Strategy (all counts bounded by n_checkpoints):
      1. Early stage: evenly-spaced samples from the first early_fraction
         window of the chronological timeline.
      2. Late stage: evenly-spaced samples from the last late_fraction
         window of the chronological timeline.
      3. Kurtosis peaks: files with top-N kurtosis values (run-to-failure channels).
      4. Mid stage: evenly-spaced samples filling remaining slots.

    All selection is performed on the dataset's run-to-failure (is_rtf) channels.

    Args:
        health_df: Health indicators DataFrame for a single dataset_id.
        n_checkpoints: Checkpoint budget for this dataset.
        early_frac: Fraction of the timeline counted as "early".
        late_frac: Fraction of the timeline counted as "late".
        logger: Pipeline logger.

    Returns:
        pd.DataFrame: Checkpoint selection with columns:
            file_id, filename, timestamp, mean_kurtosis_failure_bearings,
            selection_reason, checkpoint_rank.
            Total rows never exceed n_checkpoints.
    """
    # Work at file level: aggregate mean kurtosis across run-to-failure channels
    failure_channels = health_df[health_df["is_rtf"]]
    file_kurtosis = (
        failure_channels.groupby(["file_id", "filename", "timestamp"])["kurtosis"]
        .mean()
        .reset_index()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    total_files = len(file_kurtosis)
    if total_files == 0:
        logger.warning("No run-to-failure channels found for this dataset; skipping checkpoint selection.")
        return pd.DataFrame(
            columns=["file_id", "filename", "timestamp", "mean_kurtosis_failure_bearings", "selection_reason", "checkpoint_rank"]
        )

    # Timeline WINDOWS (how much of the run counts as "early"/"late")
    early_window_size = max(1, int(total_files * early_frac))
    late_window_size = max(1, int(total_files * late_frac))

    # Checkpoint SLOT BUDGET per category (bounded by n_checkpoints)
    n_early_slots = max(1, round(n_checkpoints * 0.25))
    n_late_slots = max(1, round(n_checkpoints * 0.25))
    n_peak_slots = max(1, round(n_checkpoints * 0.20))
    n_mid_slots = max(
        1, n_checkpoints - n_early_slots - n_late_slots - n_peak_slots
    )

    selected_ids: Dict[int, str] = {}

    def _sample_evenly(df_slice: pd.DataFrame, count: int) -> List[int]:
        """Selects up to `count` evenly-spaced file_ids from a DataFrame slice."""
        if df_slice.empty or count <= 0:
            return []
        n = len(df_slice)
        if n <= count:
            return df_slice["file_id"].tolist()
        indices = np.linspace(0, n - 1, count, dtype=int)
        return df_slice.iloc[indices]["file_id"].tolist()

    # 1. Early stage: sample from the first early_window_size files
    early_window = file_kurtosis.head(early_window_size)
    for file_id in _sample_evenly(early_window, n_early_slots):
        selected_ids[file_id] = "early_stage"

    # 2. Late stage: sample from the last late_window_size files
    late_window = file_kurtosis.tail(late_window_size)
    for file_id in _sample_evenly(late_window, n_late_slots):
        if file_id not in selected_ids:
            selected_ids[file_id] = "late_stage"

    # 3. Kurtosis peak events (top-N kurtosis values across the full run)
    top_kurt = file_kurtosis.nlargest(n_peak_slots * 3, "kurtosis")
    peak_count = 0
    for file_id in top_kurt["file_id"]:
        if file_id not in selected_ids:
            selected_ids[file_id] = "kurtosis_peak"
            peak_count += 1
        if peak_count >= n_peak_slots:
            break

    # 4. Mid-stage: evenly-spaced samples from the remaining middle region
    mid_region = file_kurtosis.iloc[early_window_size : total_files - late_window_size]
    for file_id in _sample_evenly(mid_region, n_mid_slots):
        if file_id not in selected_ids:
            selected_ids[file_id] = "mid_stage"

    logger.info(f"Selected {len(selected_ids)} checkpoint files from {total_files} total files")

    # Build result DataFrame, ordered chronologically
    rows = []
    id_timestamp_pairs = [
        (fid, file_kurtosis.loc[file_kurtosis["file_id"] == fid, "timestamp"].values[0])
        for fid in selected_ids
    ]
    for rank, (fid, _ts) in enumerate(
        sorted(id_timestamp_pairs, key=lambda x: x[1]), start=1
    ):
        row_data = file_kurtosis[file_kurtosis["file_id"] == fid].iloc[0]
        rows.append(
            {
                "file_id": fid,
                "filename": row_data["filename"],
                "timestamp": row_data["timestamp"],
                "mean_kurtosis_failure_bearings": row_data["kurtosis"],
                "selection_reason": selected_ids[fid],
                "checkpoint_rank": rank,
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        reason_counts = result["selection_reason"].value_counts().to_dict()
        logger.info(f"Checkpoint breakdown: {reason_counts}")

    return result


def select_checkpoints(
    health_df: pd.DataFrame,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> pd.DataFrame:
    """Selects checkpoint files across every registered dataset, independently.

    Args:
        health_df: Full health indicators DataFrame spanning all datasets, with
            a `dataset_id` column and an `is_rtf` column.
        config: Full project configuration dictionary.
        logger: Pipeline logger.

    Returns:
        pd.DataFrame: Combined checkpoint selection with a `dataset_id` column
            added, plus the per-dataset columns from `_select_checkpoints_for_dataset`.
            `checkpoint_rank` is per-dataset (each dataset's own chronological rank),
            not a global rank across datasets.
    """
    cfg = config["signal_analysis"]["checkpoint_selection"]
    n_checkpoints = int(cfg["n_checkpoints"])
    early_frac = float(cfg["early_fraction"])
    late_frac = float(cfg["late_fraction"])

    per_dataset_results = []
    for dataset_id in get_all_dataset_ids(config):
        sub = health_df[health_df["dataset_id"] == dataset_id]
        logger.info(f"[{dataset_id}] Selecting checkpoints")
        result = _select_checkpoints_for_dataset(sub, n_checkpoints, early_frac, late_frac, logger)
        if not result.empty:
            result.insert(1, "dataset_id", dataset_id)
        per_dataset_results.append(result)

    combined = pd.concat(per_dataset_results, ignore_index=True) if per_dataset_results else pd.DataFrame()
    logger.info(f"Selected {len(combined)} checkpoint files total across {len(get_all_dataset_ids(config))} datasets")
    return combined
