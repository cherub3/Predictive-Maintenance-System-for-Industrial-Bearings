"""Time-domain statistical features for bearing condition monitoring.

Features in this module (td_* family) describe the shape and energy of raw
vibration waveforms. These are directly reused from Phase 3's scalar indicator
computation (signal_health_indicators.parquet) rather than recomputed.

Naming Convention:
  td_rms          - Root Mean Square (overall vibration energy)
  td_kurtosis     - Kurtosis (impulsiveness / sharpness)
  td_crest_factor - Crest Factor (peak-to-RMS ratio)
  td_skewness     - Skewness (distribution asymmetry, conditional)
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Tuple


def load_time_domain_features(
    phase3_indicators_path: Path,
    gates: dict,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Load time-domain features from Phase 3's pre-computed indicators.

    This function reuses Phase 3's signal_health_indicators.parquet rather than
    recomputing indicators, as per the Phase 4 design specification. This ensures
    perfect consistency between Phase 3's exploratory analysis and Phase 4's
    feature table.

    Args:
        phase3_indicators_path: Path to data/processed/signal_health_indicators.parquet
        gates: Feature gate decisions (to exclude skewness if gated off)
        logger: Logger instance

    Returns:
        DataFrame with columns: [timestamp, bearing_id, channel, td_rms, td_kurtosis,
                                 td_crest_factor, td_skewness (conditional)]

    Raises:
        FileNotFoundError: If Phase 3 indicators have not been computed
        ValueError: If Phase 3 indicators table is malformed
    """
    logger.info(f"Loading time-domain features from {phase3_indicators_path}")

    if not phase3_indicators_path.exists():
        error_msg = (
            f"Phase 3 indicators not found at {phase3_indicators_path}. "
            "Phase 3 (Signal Understanding & Exploratory Signal Analysis) must be "
            "completed and signal_health_indicators.parquet generated before Phase 4 "
            "feature engineering can proceed."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Load Phase 3's scalar indicators
    try:
        df = pd.read_parquet(phase3_indicators_path)
        logger.info(f"Loaded {len(df)} rows from Phase 3 indicators")
    except Exception as e:
        logger.error(f"Failed to load Phase 3 indicators: {e}")
        raise

    # Validate required columns exist
    required_columns = ["dataset_id", "timestamp", "bearing_id", "channel", "rms", "kurtosis", "crest_factor"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        error_msg = f"Phase 3 indicators missing columns: {missing}. Table is malformed."
        logger.error(error_msg)
        raise ValueError(error_msg)

    # dataset_id is carried through as a merge key: bearing_id/channel repeat across
    # datasets (e.g. channel 1 exists in set1, set2, AND set3), so every downstream
    # feature family must merge on (dataset_id, timestamp, bearing_id, channel), not
    # just the latter three, to avoid conflating physically unrelated bearings.
    output_df = df[["dataset_id", "timestamp", "bearing_id", "channel"]].copy()

    # Always include these (ungated) time-domain features
    output_df["td_rms"] = df["rms"]
    logger.debug(f"Included td_rms (count: {output_df['td_rms'].notna().sum()})")

    output_df["td_kurtosis"] = df["kurtosis"]
    logger.debug(f"Included td_kurtosis (count: {output_df['td_kurtosis'].notna().sum()})")

    output_df["td_crest_factor"] = df["crest_factor"]
    logger.debug(f"Included td_crest_factor (count: {output_df['td_crest_factor'].notna().sum()})")

    # Conditionally include skewness based on Phase 3 findings
    if gates.get("skewness", False):
        if "skewness" in df.columns:
            output_df["td_skewness"] = df["skewness"]
            logger.info(f"Included td_skewness (gated ON; count: {output_df['td_skewness'].notna().sum()})")
        else:
            logger.warning("Skewness gated ON but not present in Phase 3 indicators; skipping")
    else:
        logger.debug("Skewness gated OFF per Phase 3 findings")

    # Note: Peak-to-Peak is explicitly excluded per design (redundant with RMS + Crest Factor)
    if "peak_to_peak" in df.columns:
        logger.debug("Peak-to-Peak present in Phase 3 indicators but intentionally excluded (redundant)")

    logger.info(f"Time-domain features loaded: {list(output_df.columns)}")
    return output_df


def validate_time_domain_features(df: pd.DataFrame, logger: logging.Logger) -> Tuple[bool, str]:
    """Validate time-domain feature table integrity.

    Args:
        df: DataFrame with time-domain features
        logger: Logger instance

    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    required_features = ["td_rms", "td_kurtosis", "td_crest_factor"]

    missing_required = [f for f in required_features if f not in df.columns]
    if missing_required:
        msg = f"Missing required time-domain features: {missing_required}"
        logger.error(msg)
        return False, msg

    # Check for null values (should be none given Phase 2's 100% parse success)
    null_counts = df[required_features].isna().sum().to_dict()
    if any(count > 0 for count in null_counts.values()):
        msg = f"Unexpected nulls in time-domain features: {null_counts}"
        logger.warning(msg)

    logger.info("Time-domain features validation passed")
    return True, "Time-domain features valid"
