"""Envelope/defect-frequency features for bearing diagnostics.

Features in this module (env_* family) directly target bearing characteristic
defect frequencies (BPFO, BPFI, BSF). These features are conditionally included
ONLY if bearing geometry has been verified from an authoritative source.

Naming Convention:
  env_bpfo_energy - Outer race defect frequency energy
  env_bpfi_energy - Inner race defect frequency energy
  env_bsf_energy  - Rolling element defect frequency energy

IMPORTANT: If bearing geometry verification failed in Phase 3, this entire
feature family is excluded per the Bearing Geometry Safety Rule.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt, hilbert
from typing import Optional, Tuple

from src.config import get_all_dataset_ids, get_channel_bearing_map, get_dataset_config


def compute_envelope_spectrum(
    waveform: np.ndarray,
    sampling_rate: float,
    band_low_hz: float = 5000.0,
    band_high_hz: float = 9000.0,
    filter_order: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the envelope spectrum using bandpass-then-Hilbert demodulation.

    Physically correct bearing envelope analysis. Bearing impacts are impulsive
    and excite a high-frequency structural resonance; the defect-frequency
    modulation lives in the ENVELOPE of that resonance band, not in the raw
    broadband signal. The previous revision applied the Hilbert transform to the
    raw, un-filtered, DC-laden signal and truncated to 2048 samples, so the
    resulting "envelope" was dominated by low-frequency shaft dynamics and the
    defect modulation was buried — the env_* features were effectively noise.

    Pipeline:
      1. Remove the DC offset (zero-mean the signal).
      2. Bandpass-filter the high-frequency resonance band (default 5-9 kHz;
         Nyquist is 10 kHz at 20 kHz sampling) with a zero-phase Butterworth.
      3. Take the analytic signal (Hilbert) and its magnitude = the envelope.
      4. Remove the envelope's DC term (a large positive mean that would
         otherwise dominate the low-frequency defect bins via spectral leakage).
      5. FFT the envelope over the FULL record (no truncation) for a fine
         (~0.977 Hz) bin resolution around the defect frequencies.

    Args:
        waveform: 1-D array of time-domain samples (full record; never truncated).
        sampling_rate: Sampling frequency in Hz.
        band_low_hz: Lower edge of the resonance bandpass.
        band_high_hz: Upper edge of the resonance bandpass (must be < Nyquist).
        filter_order: Butterworth filter order.

    Returns:
        Tuple of (frequencies in Hz, envelope amplitude spectrum).
    """
    # Step 1: DC removal on the raw signal.
    x = np.asarray(waveform, dtype=float)
    x = x - np.mean(x)

    # Step 2: Zero-phase bandpass around the structural resonance band.
    nyquist = sampling_rate / 2.0
    high_edge = min(band_high_hz, nyquist * 0.99)
    b, a = butter(filter_order, [band_low_hz / nyquist, high_edge / nyquist], btype="band")
    filtered = filtfilt(b, a, x)

    # Step 3: Envelope via the analytic signal.
    envelope = np.abs(hilbert(filtered))

    # Step 4: Remove the envelope DC so it cannot swamp the low defect bins.
    envelope = envelope - np.mean(envelope)

    # Step 5: Single-sided envelope amplitude spectrum over the full record.
    n = len(envelope)
    spectrum = np.abs(np.fft.rfft(envelope)) * 2.0 / n
    frequencies = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    return frequencies, spectrum


def extract_defect_frequency_energy(
    frequencies: np.ndarray,
    magnitude: np.ndarray,
    defect_frequency_hz: float,
    tolerance_hz: float = 5.0,
) -> float:
    """Extract envelope-spectrum energy in a narrow band around a defect frequency.

    Uses an ABSOLUTE ±tolerance_hz band (default ±5 Hz). With the corrected
    full-length envelope spectrum the bin resolution is ~0.977 Hz, so a ±5 Hz
    band captures the defect peak and its immediate skirt (~10 bins) without
    bleeding into neighbouring defect orders.

    Args:
        frequencies: Frequency bins from the envelope spectrum.
        magnitude: Amplitude spectrum from the envelope spectrum.
        defect_frequency_hz: Target defect frequency (BPFO / BPFI / BSF).
        tolerance_hz: Half-width of the integration band in Hz.

    Returns:
        Scalar energy value in the defect-frequency band.
    """
    low_freq = max(0.0, defect_frequency_hz - tolerance_hz)
    high_freq = defect_frequency_hz + tolerance_hz

    mask = (frequencies >= low_freq) & (frequencies < high_freq)
    energy = np.sum(magnitude[mask] ** 2)
    return float(energy)


def build_envelope_features(
    metadata_df: pd.DataFrame,
    bearing_geometry_module: Optional[object],
    sampling_rate: float,
    config: dict,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Build envelope/defect-frequency feature table, per dataset.

    Args:
        metadata_df: Metadata table spanning all datasets, with `dataset_id`
            and `absolute_path` columns.
        bearing_geometry_module: Imported Phase 3 bearing_geometry module (or None if not verified)
        sampling_rate: Sampling frequency (same across all datasets)
        config: Full project configuration dictionary
        logger: Logger instance

    Returns:
        DataFrame with columns: [timestamp, bearing_id, channel, env_bpfo_energy, env_bpfi_energy, env_bsf_energy]

    Raises:
        ValueError: If bearing_geometry_module is None (geometry not verified)
    """
    if bearing_geometry_module is None:
        error_msg = (
            "Envelope features require verified bearing geometry, but bearing_geometry.py "
            "was not successfully verified in Phase 3. Excluding envelope features entirely "
            "per Bearing Geometry Safety Rule."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info("Building envelope/defect-frequency features")

    # Get defect frequencies from Phase 3's bearing_geometry module.
    # load_bearing_geometry reads geometry from the full project config;
    # compute_defect_frequencies takes the BearingGeometry NamedTuple and
    # returns a DefectFrequencies NamedTuple with .bpfo_hz / .bpfi_hz / .bsf_hz.
    try:
        geometry = bearing_geometry_module.load_bearing_geometry(config)
        defect_frequencies = bearing_geometry_module.compute_defect_frequencies(geometry)
        logger.info(f"Defect frequencies: {defect_frequencies}")
    except Exception as e:
        error_msg = f"Failed to retrieve defect frequencies: {e}"
        logger.error(error_msg)
        raise

    bpfo = defect_frequencies.bpfo_hz
    bpfi = defect_frequencies.bpfi_hz
    bsf = defect_frequencies.bsf_hz

    results = []
    files_processed = 0
    files_failed = 0
    fe_config = config.get("feature_engineering", config)
    expected_samples = fe_config.get("samples_per_file", 20480)
    band_low_hz = fe_config.get("envelope_band_low_hz", 5000.0)
    band_high_hz = fe_config.get("envelope_band_high_hz", 9000.0)

    for dataset_id in get_all_dataset_ids(config):
        dataset_cfg = get_dataset_config(config, dataset_id)
        channel_map = get_channel_bearing_map(config, dataset_id)
        expected_channels = dataset_cfg["expected_channels_per_file"]

        for idx, row in metadata_df[metadata_df["dataset_id"] == dataset_id].iterrows():
            try:
                file_path = Path(row["absolute_path"])

                if not file_path.exists():
                    files_failed += 1
                    continue

                raw_data = pd.read_csv(file_path, sep="\t", header=None)
                raw_array = raw_data.values

                if raw_array.shape != (expected_samples, expected_channels):
                    files_failed += 1
                    continue

                for channel_idx in range(expected_channels):
                    signal_channel = raw_array[:, channel_idx]
                    channel_num = channel_idx + 1
                    bearing_id = channel_map[channel_num]["bearing_id"]

                    try:
                        assert signal_channel.shape[0] == expected_samples, (
                            f"Expected {expected_samples} samples, got "
                            f"{signal_channel.shape[0]} in envelope path"
                        )
                        freqs, magnitude = compute_envelope_spectrum(
                            signal_channel, sampling_rate, band_low_hz, band_high_hz
                        )

                        bpfo_energy = extract_defect_frequency_energy(freqs, magnitude, bpfo)
                        bpfi_energy = extract_defect_frequency_energy(freqs, magnitude, bpfi)
                        bsf_energy = extract_defect_frequency_energy(freqs, magnitude, bsf)

                        feature_row = {
                            "dataset_id": dataset_id,
                            "timestamp": row["timestamp"],
                            "bearing_id": bearing_id,
                            "channel": channel_num,
                            "file_id": idx,
                            "env_bpfo_energy": bpfo_energy,
                            "env_bpfi_energy": bpfi_energy,
                            "env_bsf_energy": bsf_energy,
                        }

                        results.append(feature_row)

                    except Exception as e:
                        logger.debug(f"[{dataset_id}] Error processing channel {channel_num}: {e}")
                        continue

                files_processed += 1

            except Exception as e:
                logger.debug(f"[{dataset_id}] Error processing file: {e}")
                files_failed += 1
                continue

    logger.info(f"Envelope features: {files_processed} processed, {files_failed} failed")
    output_df = pd.DataFrame(results)
    return output_df


def validate_envelope_features(df: pd.DataFrame, logger: logging.Logger) -> Tuple[bool, str]:
    """Validate envelope feature table."""
    required_features = ["env_bpfo_energy", "env_bpfi_energy", "env_bsf_energy"]
    missing = [f for f in required_features if f not in df.columns]
    
    if missing:
        msg = f"Missing envelope features: {missing}"
        logger.error(msg)
        return False, msg

    logger.info("Envelope features validation passed")
    return True, "Envelope features valid"
