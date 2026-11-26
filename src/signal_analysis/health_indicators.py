"""Tier 1 time-domain health indicator computation for all acquisition files.

Computes RMS, Kurtosis, Crest Factor, and Skewness for every (file, channel)
combination across every registered dataset's acquisition files (9,463 files x
their respective channel counts: 8 for set1, 4 for set2/set3).

Physical meanings:
  RMS           -- Energy content; rises monotonically with broad bearing wear.
  Kurtosis      -- Impulsiveness; sensitive to early-stage defect impacts.
                   Note: rolls back toward Gaussian at late-stage damage.
  Crest Factor  -- Peak-to-RMS ratio; detects localised impulsive events.
  Skewness      -- Distributional asymmetry; conditionally informative.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew

from src.config import get_all_dataset_ids, get_channel_bearing_map, get_dataset_config
from src.signal_analysis.signal_loader import load_signal_file


def _compute_indicators(waveform: np.ndarray) -> Dict[str, float]:
    """Computes scalar health indicators for a single channel waveform.

    Args:
        waveform: 1-D array of vibration samples.

    Returns:
        Dict with keys: rms, kurtosis, crest_factor, skewness, peak_amplitude, mean.
    """
    n = len(waveform)
    if n == 0:
        return {
            "rms": float("nan"),
            "kurtosis": float("nan"),
            "crest_factor": float("nan"),
            "skewness": float("nan"),
            "peak_amplitude": float("nan"),
            "mean": float("nan"),
        }

    rms = float(np.sqrt(np.mean(waveform**2)))
    # scipy kurtosis uses Fisher definition (excess kurtosis); bias=False for unbiased
    kurt = float(scipy_kurtosis(waveform, fisher=True, bias=False))
    peak = float(np.max(np.abs(waveform)))
    crest = peak / rms if rms > 0.0 else float("nan")
    skewness = float(scipy_skew(waveform, bias=False))
    mean = float(np.mean(waveform))

    return {
        "rms": rms,
        "kurtosis": kurt,
        "crest_factor": crest,
        "skewness": skewness,
        "peak_amplitude": peak,
        "mean": mean,
    }


def compute_health_indicators(
    metadata_df: pd.DataFrame,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> pd.DataFrame:
    """Computes time-domain health indicators for all files and channels (Tier 1).

    Iterates every registered dataset's acquisition files (using each file's
    already-resolved `absolute_path` from Phase 1 metadata, so no raw_dir
    reconstruction or per-dataset directory juggling is needed here), loads
    each signal, and computes scalar health indicators for every channel using
    that dataset's own channel/bearing map and expected channel count.

    Args:
        metadata_df: Phase 1 metadata DataFrame with 'dataset_id', 'filename',
            'timestamp', 'absolute_path', 'expected_channels_per_file', and
            'parser_status' columns.
        config: Full project configuration dictionary.
        logger: Pipeline logger.

    Returns:
        pd.DataFrame: Health indicators with columns:
            file_id, dataset_id, timestamp, filename, bearing_id, channel, axis,
            is_rtf, rms, kurtosis, crest_factor, skewness, peak_amplitude, mean,
            load_status.
    """
    expected_samples = int(config["datasets"]["defaults"]["samples_per_file"])
    dataset_ids = get_all_dataset_ids(config)

    # Filter to successfully parsed files, sorted chronologically WITHIN each
    # dataset — the datasets are independent, non-overlapping test runs, so a
    # global chronological sort across dataset boundaries is not meaningful.
    valid_meta = metadata_df[metadata_df["parser_status"] == "Success"].copy()
    valid_meta["_dt"] = pd.to_datetime(valid_meta["timestamp"])
    valid_meta = valid_meta.sort_values(["dataset_id", "_dt"])

    total_files = len(valid_meta)
    logger.info(f"Computing health indicators for {total_files} files across {len(dataset_ids)} datasets")

    records: List[Dict[str, Any]] = []
    processed = 0

    for dataset_id in dataset_ids:
        dataset_cfg = get_dataset_config(config, dataset_id)
        rtf_bearing_ids = {
            b for b, cfg in dataset_cfg.get("bearings", {}).items() if cfg.get("status") == "rtf"
        }
        channel_map = get_channel_bearing_map(config, dataset_id)
        expected_channels = int(dataset_cfg["expected_channels_per_file"])

        dataset_rows = valid_meta[valid_meta["dataset_id"] == dataset_id]

        for file_idx, row in dataset_rows.iterrows():
            processed += 1
            if processed % 500 == 0 or processed == total_files:
                logger.info(f"  Progress: {processed}/{total_files} files processed")

            file_path = Path(row["absolute_path"])
            data, status = load_signal_file(
                file_path,
                expected_samples=expected_samples,
                expected_channels=expected_channels,
                logger=logger,
            )

            for ch_1indexed in range(1, expected_channels + 1):
                channel_info = channel_map[ch_1indexed]
                bearing_id = channel_info["bearing_id"]
                axis = channel_info["axis"]

                if data is not None:
                    waveform = data[:, ch_1indexed - 1]
                    indicators = _compute_indicators(waveform)
                    load_status = "OK"
                else:
                    indicators = {
                        "rms": float("nan"),
                        "kurtosis": float("nan"),
                        "crest_factor": float("nan"),
                        "skewness": float("nan"),
                        "peak_amplitude": float("nan"),
                        "mean": float("nan"),
                    }
                    load_status = status

                records.append(
                    {
                        "file_id": file_idx,
                        "dataset_id": dataset_id,
                        "timestamp": row["timestamp"],
                        "filename": row["filename"],
                        "bearing_id": bearing_id,
                        "channel": ch_1indexed,
                        "axis": axis,
                        "is_rtf": bearing_id in rtf_bearing_ids,
                        "rms": indicators["rms"],
                        "kurtosis": indicators["kurtosis"],
                        "crest_factor": indicators["crest_factor"],
                        "skewness": indicators["skewness"],
                        "peak_amplitude": indicators["peak_amplitude"],
                        "mean": indicators["mean"],
                        "load_status": load_status,
                    }
                )

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    logger.info(
        f"Health indicators computed: {len(df)} rows, "
        f"{df['load_status'].eq('OK').sum()} successful channel reads"
    )
    return df


def validate_health_indicators(
    df: pd.DataFrame, expected_rows: int, logger: logging.Logger
) -> bool:
    """Validates structural correctness of the health indicators DataFrame.

    Args:
        df: Health indicators DataFrame.
        expected_rows: Expected number of rows (files x channels).
        logger: Pipeline logger.

    Returns:
        bool: True if all validation checks pass.
    """
    required_cols = {
        "file_id", "dataset_id", "timestamp", "filename", "bearing_id", "channel",
        "axis", "is_rtf", "rms", "kurtosis", "crest_factor", "skewness",
        "peak_amplitude", "mean", "load_status",
    }
    missing = required_cols - set(df.columns)
    if missing:
        logger.error(f"Health indicators missing required columns: {missing}")
        return False

    if len(df) != expected_rows:
        logger.error(f"Row count mismatch: got {len(df)}, expected {expected_rows}")
        return False

    ok_rows = df[df["load_status"] == "OK"]
    null_rms = ok_rows["rms"].isna().sum()
    if null_rms > 0:
        logger.warning(f"{null_rms} null RMS values in successfully loaded rows")

    bearings_found = set(df["bearing_id"].unique())
    expected_bearings = {"bearing_1", "bearing_2", "bearing_3", "bearing_4"}
    if bearings_found != expected_bearings:
        logger.error(f"Bearing ID mismatch: found {bearings_found}")
        return False

    logger.info("Health indicator validation passed")
    return True
