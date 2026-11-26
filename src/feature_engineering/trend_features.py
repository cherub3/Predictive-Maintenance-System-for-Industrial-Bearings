"""Trend/derived features capturing how base features change over time.

Features in this module (trend_* family) capture the direction and speed of
change in base indicators. Critical implementation detail: trend features are
computed per-bearing with explicit handling of Phase 2's documented chronology
gaps. Windows that span gaps exceeding a threshold are nulled out rather than
computed, preventing artificial slope inflation.

Naming Convention:
  trend_rms_slope        - Rate of RMS change over rolling window
  trend_kurtosis_slope   - Rate of Kurtosis change over rolling window
  trend_rms_variance     - Variance (instability) of RMS, conditional
"""

import numpy as np
import pandas as pd
from typing import Tuple
from scipy import stats


def compute_chronology_gaps(
    timestamps: pd.Series,
    gap_threshold_minutes: float = 30.0,
) -> pd.Series:
    """Identify chronology gaps exceeding threshold.

    Args:
        timestamps: Sorted timestamp series
        gap_threshold_minutes: Maximum allowed gap in minutes

    Returns:
        Boolean series True where a gap exceeds threshold
    """
    time_diffs = timestamps.diff().dt.total_seconds() / 60.0  # Convert to minutes
    gaps_exceeded = time_diffs > gap_threshold_minutes
    return gaps_exceeded


def compute_rolling_slope(
    series: pd.Series,
    timestamps: pd.Series,
    window_size: int,
    gap_threshold_minutes: float = 30.0,
) -> pd.Series:
    """Compute rolling linear regression slope with gap awareness.

    Args:
        series: Feature values (e.g., RMS over time)
        timestamps: Corresponding timestamps
        window_size: Number of points in rolling window
        gap_threshold_minutes: Threshold for chronology gap detection

    Returns:
        Series of slopes (nans for windows spanning gaps or start of series)
    """
    if len(series) < window_size:
        return pd.Series(np.nan, index=series.index)

    slopes = np.full(len(series), np.nan)
    gaps_exceeded = compute_chronology_gaps(timestamps, gap_threshold_minutes)

    for i in range(window_size - 1, len(series)):
        window_start = i - window_size + 1
        window_indices = range(window_start, i + 1)

        # Check if any gap in this window exceeds threshold
        if gaps_exceeded.iloc[window_start : i + 1].any():
            # Window spans a documented gap; null out slope
            slopes[i] = np.nan
        else:
            # Compute slope using elapsed time in minutes as x-axis.
            # Sample indices must not be used because NASA IMS Set 1 has variable
            # acquisition intervals (5-min early, 10-min late); index-based slopes
            # would have inconsistent time units across the run.
            ts_window = timestamps.iloc[window_start : i + 1]
            x = (ts_window - ts_window.iloc[0]).dt.total_seconds().values / 60.0
            y = series.iloc[window_indices].values

            if np.all(np.isfinite(y)):
                slope, _, _, _, _ = stats.linregress(x, y)
                slopes[i] = slope

    return pd.Series(slopes, index=series.index)


def compute_rolling_variance(
    series: pd.Series,
    timestamps: pd.Series,
    window_size: int,
    gap_threshold_minutes: float = 30.0,
) -> pd.Series:
    """Compute rolling variance with gap awareness.

    Args:
        series: Feature values
        timestamps: Corresponding timestamps
        window_size: Number of points in rolling window
        gap_threshold_minutes: Threshold for chronology gap detection

    Returns:
        Series of variances (nans for windows spanning gaps)
    """
    if len(series) < window_size:
        return pd.Series(np.nan, index=series.index)

    variances = np.full(len(series), np.nan)
    gaps_exceeded = compute_chronology_gaps(timestamps, gap_threshold_minutes)

    for i in range(window_size - 1, len(series)):
        window_start = i - window_size + 1

        if gaps_exceeded.iloc[window_start : i + 1].any():
            variances[i] = np.nan
        else:
            window_values = series.iloc[window_start : i + 1].values
            if np.all(np.isfinite(window_values)):
                variances[i] = np.var(window_values, ddof=1)

    return pd.Series(variances, index=series.index)


def build_trend_features(
    td_features_df: pd.DataFrame,
    gates: dict,
    window_size: int = 5,
    gap_threshold_minutes: float = 30.0,
    logger=None,
) -> pd.DataFrame:
    """Build trend/derived feature table.

    Args:
        td_features_df: Time-domain features table with timestamp/bearing_id/channel
        gates: Feature gate decisions
        window_size: Rolling window size in number of files
        gap_threshold_minutes: Chronology gap threshold
        logger: Logger instance

    Returns:
        DataFrame with columns: [timestamp, bearing_id, channel, trend_rms_slope,
                                  trend_kurtosis_slope, trend_rms_variance (conditional)]
    """
    if logger:
        logger.info("Building trend/derived features with chronology gap handling")

    results = []

    # Group by (dataset_id, bearing_id, channel) to compute rolling windows
    # independently. dataset_id is required: bearing_id/channel repeat across
    # datasets (e.g. channel 1 exists in set1, set2, AND set3), so grouping
    # without it would merge unrelated experiments' rows into one rolling
    # window under a shared sort order. Today's non-overlapping dataset date
    # ranges mean the existing chronology-gap check happens to null out any
    # window that would span a dataset boundary, but that's incidental
    # protection, not a substitute for an explicit, correct grouping key.
    for (dataset_id, bearing_id, channel), group in td_features_df.groupby(["dataset_id", "bearing_id", "channel"]):
        group = group.sort_values("timestamp").reset_index(drop=True)

        # Compute rolling slopes
        rms_slope = compute_rolling_slope(
            group["td_rms"],
            group["timestamp"],
            window_size,
            gap_threshold_minutes,
        )

        kurtosis_slope = compute_rolling_slope(
            group["td_kurtosis"],
            group["timestamp"],
            window_size,
            gap_threshold_minutes,
        )

        # Build output for this group
        group_result = group[["dataset_id", "timestamp", "bearing_id", "channel"]].copy()
        group_result["trend_rms_slope"] = rms_slope.values
        group_result["trend_kurtosis_slope"] = kurtosis_slope.values

        # Conditionally include rolling variance
        if gates.get("rolling_variance", False):
            rms_variance = compute_rolling_variance(
                group["td_rms"],
                group["timestamp"],
                window_size,
                gap_threshold_minutes,
            )
            group_result["trend_rms_variance"] = rms_variance.values
            if logger:
                logger.debug(f"Included trend_rms_variance (gated ON) for B{bearing_id}C{channel}")
        else:
            if logger:
                logger.debug(f"Skipped trend_rms_variance (gated OFF) for B{bearing_id}C{channel}")

        results.append(group_result)

    output_df = pd.concat(results, ignore_index=True)

    if logger:
        null_counts = output_df[["trend_rms_slope", "trend_kurtosis_slope"]].isna().sum()
        logger.info(f"Trend features built; NaN counts: {null_counts.to_dict()}")

    return output_df


def validate_trend_features(df: pd.DataFrame, logger=None) -> Tuple[bool, str]:
    """Validate trend feature table."""
    required_features = ["trend_rms_slope", "trend_kurtosis_slope"]
    missing = [f for f in required_features if f not in df.columns]

    if missing:
        msg = f"Missing trend features: {missing}"
        if logger:
            logger.error(msg)
        return False, msg

    if logger:
        logger.info("Trend features validation passed")
    return True, "Trend features valid"
