"""Cross-channel features describing relationships between X and Y axes.

Features in this module (cross_* family) operationalize relationships between
the two radial measurement axes (X and Y) on the same bearing. These are
conditionally included based on Phase 3's channel comparison findings.

Structurally inapplicable to single_axis datasets (Set 2/3 have no Y channel
at all) — those datasets legitimately contribute zero rows to this feature,
not a zero-valued ratio, since there is nothing to compute a ratio from.

Naming Convention:
  cross_xy_energy_ratio - Ratio of X-axis RMS to Y-axis RMS
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

from src.config import get_all_dataset_ids, get_channel_bearing_map, get_dataset_config


def compute_xy_energy_ratio(
    td_features_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Compute X/Y channel energy ratio per bearing/file, per dataset.

    Args:
        td_features_df: Time-domain features with td_rms per (dataset_id,
            bearing_id, channel, timestamp).
        config: Full project configuration dictionary (for the datasets
            registry's channel_axes, rather than assuming odd/even channel
            numbering).

    Returns:
        DataFrame with dataset_id, bearing_id, timestamp, cross_xy_energy_ratio.
        Only paired_xy datasets contribute rows; single_axis datasets have no
        Y channel and are skipped entirely (not zero-filled).
    """
    results = []

    for dataset_id in get_all_dataset_ids(config):
        dataset_cfg = get_dataset_config(config, dataset_id)
        if dataset_cfg.get("channel_layout") != "paired_xy":
            continue

        channel_map = get_channel_bearing_map(config, dataset_id)
        x_channels_by_bearing: Dict[str, list] = {}
        y_channels_by_bearing: Dict[str, list] = {}
        for channel, info in channel_map.items():
            target = x_channels_by_bearing if info["axis"] == "X" else y_channels_by_bearing
            target.setdefault(info["bearing_id"], []).append(channel)

        dataset_df = td_features_df[td_features_df["dataset_id"] == dataset_id]

        for bearing_id in dataset_df["bearing_id"].unique():
            x_channels = x_channels_by_bearing.get(bearing_id, [])
            y_channels = y_channels_by_bearing.get(bearing_id, [])
            if not x_channels or not y_channels:
                continue

            bearing_df = dataset_df[dataset_df["bearing_id"] == bearing_id]

            for timestamp, ts_group in bearing_df.groupby("timestamp"):
                x_rows = ts_group[ts_group["channel"].isin(x_channels)]
                y_rows = ts_group[ts_group["channel"].isin(y_channels)]

                if len(x_rows) > 0 and len(y_rows) > 0:
                    x_rms = x_rows["td_rms"].mean()
                    y_rms = y_rows["td_rms"].mean()

                    ratio = x_rms / y_rms if y_rms > 0 else np.nan

                    results.append({
                        "dataset_id": dataset_id,
                        "bearing_id": bearing_id,
                        "timestamp": timestamp,
                        "cross_xy_energy_ratio": ratio,
                    })

    return pd.DataFrame(results)


def build_cross_channel_features(
    td_features_df: pd.DataFrame,
    gates: dict,
    config: dict,
    logger=None,
) -> pd.DataFrame:
    """Build cross-channel feature table.

    Args:
        td_features_df: Time-domain features table
        gates: Feature gate decisions
        config: Full project configuration dictionary
        logger: Logger instance

    Returns:
        DataFrame with cross-channel features
    """
    if logger:
        logger.info("Building cross-channel features")

    if not gates.get("xy_energy_ratio", False):
        if logger:
            logger.debug("Cross-channel features gated OFF per Phase 3 findings")
        return pd.DataFrame()  # Return empty; skip this feature family

    xy_ratio_df = compute_xy_energy_ratio(td_features_df, config)

    if logger:
        logger.info(f"Cross-channel features built: {len(xy_ratio_df)} records")

    return xy_ratio_df


def validate_cross_channel_features(
    df: pd.DataFrame, logger=None
) -> Tuple[bool, str]:
    """Validate cross-channel feature table."""
    if df.empty:
        msg = "Cross-channel features table is empty (feature gated off)"
        if logger:
            logger.debug(msg)
        return True, msg  # Not an error if gated off

    if "cross_xy_energy_ratio" not in df.columns:
        msg = "Missing cross_xy_energy_ratio"
        if logger:
            logger.error(msg)
        return False, msg

    if logger:
        logger.info("Cross-channel features validation passed")
    return True, "Cross-channel features valid"
