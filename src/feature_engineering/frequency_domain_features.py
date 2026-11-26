"""Frequency-domain features for bearing vibration analysis.

Features in this module (fd_* family) describe where in the frequency spectrum
vibration energy is concentrated. These are computed from raw signals via FFT,
without requiring verified bearing geometry (geometry-agnostic).

Naming Convention:
  fd_band_energy_*       - Spectral energy in defined frequency bands
  fd_spectral_centroid   - Energy-weighted mean frequency
  fd_spectral_entropy    - Shannon entropy of normalized power spectrum
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal as scipy_signal
from typing import List, Tuple

from src.config import get_all_dataset_ids, get_channel_bearing_map, get_dataset_config


def compute_fft_spectrum(
    waveform: np.ndarray,
    sampling_rate: float,
    window: str = "hann",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the single-sided FFT magnitude spectrum of a vibration waveform.

    The transform is computed over the ENTIRE waveform. Earlier revisions
    truncated the signal to the first 2048 samples, discarding ~90% of every
    20,480-sample acquisition and inflating the frequency-bin width tenfold; the
    full-length transform gives a bin resolution of sampling_rate / len(waveform)
    (0.977 Hz for a 20,480-sample record at 20 kHz).

    Args:
        waveform: 1-D array of time-domain samples (full record; never truncated).
        sampling_rate: Sampling frequency in Hz.
        window: Window function name (e.g., "hann", "hamming").

    Returns:
        Tuple of (frequencies in Hz, single-sided amplitude spectrum). The
        spectrum has len(waveform) // 2 + 1 bins (numpy rfft convention).
    """
    n = len(waveform)
    win = scipy_signal.get_window(window, n)
    windowed = waveform * win
    spectrum = np.fft.rfft(windowed)
    # Single-sided amplitude, normalized by the window's coherent gain (sum of
    # window samples) so amplitudes are comparable across records.
    magnitude = np.abs(spectrum) * 2.0 / np.sum(win)
    frequencies = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    return frequencies, magnitude


def compute_band_energy(
    frequencies: np.ndarray,
    magnitude: np.ndarray,
    band_low_hz: float,
    band_high_hz: float,
) -> float:
    """Compute total energy in a frequency band."""
    mask = (frequencies >= band_low_hz) & (frequencies < band_high_hz)
    energy = np.sum(magnitude[mask] ** 2)
    return float(energy)


def compute_spectral_centroid(
    frequencies: np.ndarray,
    magnitude: np.ndarray,
) -> float:
    """Compute energy-weighted mean frequency."""
    power = magnitude ** 2
    power_sum = np.sum(power)
    if power_sum == 0:
        return 0.0
    centroid = np.sum(frequencies * power) / power_sum
    return float(centroid)


def compute_spectral_entropy(magnitude: np.ndarray, base: float = 2.0) -> float:
    """Compute Shannon entropy of normalized power spectrum."""
    power = magnitude ** 2
    power_sum = np.sum(power)
    if power_sum == 0:
        return 0.0
    p = power / power_sum
    p_nonzero = p[p > 0]
    entropy = -np.sum(p_nonzero * np.log(p_nonzero) / np.log(base))
    return float(entropy)


def build_frequency_domain_features(
    metadata_df: pd.DataFrame,
    full_config: dict,
    frequency_bands: List[Tuple[float, float]],
    sampling_rate: float,
    config: dict,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Build frequency-domain feature table for all files and channels, per dataset.

    Args:
        metadata_df: Phase 1 metadata DataFrame spanning all datasets, with
            `dataset_id` and `absolute_path` columns.
        full_config: Full project configuration dictionary (for the datasets
            registry — NOT just the `feature_engineering` sub-block, which is
            passed separately as `config` for the fft_window/samples_per_file/
            spectral_entropy_base settings, matching the existing call site).
        frequency_bands: List of (low_hz, high_hz) band boundaries.
        sampling_rate: Sampling frequency in Hz (same across all datasets).
        config: `feature_engineering` config sub-block.
        logger: Logger instance.
    """
    logger.info("Building frequency-domain features for all files")

    window_function = config.get("fft_window", "hann")
    expected_samples = config.get("samples_per_file", 20480)
    entropy_base = config.get("spectral_entropy_base", 2.0)

    results = []
    files_processed = 0
    files_failed = 0

    for dataset_id in get_all_dataset_ids(full_config):
        dataset_cfg = get_dataset_config(full_config, dataset_id)
        channel_map = get_channel_bearing_map(full_config, dataset_id)
        expected_channels = dataset_cfg["expected_channels_per_file"]

        for idx, row in metadata_df[metadata_df["dataset_id"] == dataset_id].iterrows():
            try:
                file_path = Path(row["absolute_path"])

                if not file_path.exists():
                    logger.warning(f"File not found: {file_path}")
                    files_failed += 1
                    continue

                raw_data = pd.read_csv(file_path, sep="\t", header=None)
                raw_array = raw_data.values

                if raw_array.shape != (expected_samples, expected_channels):
                    logger.warning(
                        f"[{dataset_id}] Unexpected shape in {row['filename']}: "
                        f"{raw_array.shape}, expected ({expected_samples}, {expected_channels})"
                    )
                    files_failed += 1
                    continue

                for channel_idx in range(expected_channels):
                    signal_channel = raw_array[:, channel_idx]
                    # Guard against any future re-introduction of signal truncation:
                    # the FFT must always see the full acquisition.
                    assert signal_channel.shape[0] == expected_samples, (
                        f"Expected {expected_samples} samples, got "
                        f"{signal_channel.shape[0]} in {row['filename']} ch{channel_idx + 1}"
                    )
                    freqs, magnitude = compute_fft_spectrum(
                        signal_channel, sampling_rate, window_function
                    )

                    channel_num = channel_idx + 1
                    bearing_id = channel_map[channel_num]["bearing_id"]

                    feature_row = {
                        "dataset_id": dataset_id,
                        "timestamp": row["timestamp"],
                        "bearing_id": bearing_id,
                        "channel": channel_num,
                        "file_id": idx,
                        "file_sha256": row.get("file_sha256", ""),
                    }

                    for band_low, band_high in frequency_bands:
                        band_energy = compute_band_energy(freqs, magnitude, band_low, band_high)
                        band_name = f"fd_band_energy_{int(band_low)}_{int(band_high)}"
                        feature_row[band_name] = band_energy

                    centroid = compute_spectral_centroid(freqs, magnitude)
                    feature_row["fd_spectral_centroid"] = centroid

                    entropy = compute_spectral_entropy(magnitude, entropy_base)
                    feature_row["fd_spectral_entropy"] = entropy

                    results.append(feature_row)

                files_processed += 1

            except Exception as e:
                logger.warning(f"[{dataset_id}] Error processing file: {e}")
                files_failed += 1
                continue

    logger.info(f"Frequency-domain features: {files_processed} processed, {files_failed} failed")
    output_df = pd.DataFrame(results)
    return output_df


def validate_frequency_domain_features(df: pd.DataFrame, logger: logging.Logger) -> Tuple[bool, str]:
    """Validate frequency-domain feature table integrity."""
    required_features = ["fd_spectral_centroid", "fd_spectral_entropy"]
    band_features = [col for col in df.columns if col.startswith("fd_band_energy_")]

    if not band_features:
        msg = "No frequency band energy features found"
        logger.error(msg)
        return False, msg

    missing_required = [f for f in required_features if f not in df.columns]
    if missing_required:
        msg = f"Missing features: {missing_required}"
        logger.error(msg)
        return False, msg

    logger.info(f"Frequency-domain features validation passed")
    return True, "Frequency-domain features valid"
