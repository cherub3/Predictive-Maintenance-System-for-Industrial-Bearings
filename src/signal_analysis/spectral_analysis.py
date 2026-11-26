"""Tier 2 spectral analysis for selected checkpoint files.

Computes FFT-based spectral features for a selected subset of acquisition
files at key degradation stages. Used to characterize spectral evolution
and validate frequency-domain feature design for Phase 4.

This module does NOT produce Phase 4 features. It produces engineering
observations about spectral behaviour across the run-to-failure timeline.
"""

from typing import Any, Dict, List, Tuple

import numpy as np


def compute_fft_spectrum(
    waveform: np.ndarray,
    sampling_rate: float,
    window: str = "hann",
    n_fft: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes one-sided magnitude FFT spectrum with windowing.

    Args:
        waveform: 1-D vibration signal array.
        sampling_rate: Sampling frequency in Hz.
        window: Window function name ('hann', 'hamming', 'rectangular').
        n_fft: FFT size. Truncates or zero-pads the waveform to this length.

    Returns:
        Tuple of (frequencies_hz, magnitude_spectrum) arrays.
        Both are one-sided (length n_fft // 2 + 1).
    """
    n = min(len(waveform), n_fft)
    segment = waveform[:n].copy()

    if window == "hann":
        win = np.hanning(n)
    elif window == "hamming":
        win = np.hamming(n)
    else:
        win = np.ones(n)

    # Zero-pad to n_fft if segment shorter than n_fft
    padded = np.zeros(n_fft)
    padded[:n] = segment * win

    fft_raw = np.fft.rfft(padded)
    magnitude = np.abs(fft_raw) / n_fft  # Amplitude-normalised

    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sampling_rate)
    return frequencies, magnitude


def compute_band_energy(
    frequencies: np.ndarray,
    magnitude: np.ndarray,
    band_low_hz: float,
    band_high_hz: float,
) -> float:
    """Computes total spectral energy within a frequency band.

    Energy is defined as the sum of squared magnitudes within the band,
    consistent with Parseval's theorem applied to the windowed FFT.

    Args:
        frequencies: Frequency axis array (Hz).
        magnitude: Magnitude spectrum array, same length as frequencies.
        band_low_hz: Lower bound of the frequency band (inclusive).
        band_high_hz: Upper bound of the frequency band (inclusive).

    Returns:
        float: Total energy in the band (sum of magnitude^2).
    """
    mask = (frequencies >= band_low_hz) & (frequencies <= band_high_hz)
    return float(np.sum(magnitude[mask] ** 2))


def compute_spectral_centroid(
    frequencies: np.ndarray,
    magnitude: np.ndarray,
) -> float:
    """Computes energy-weighted mean frequency (spectral centroid).

    Args:
        frequencies: Frequency axis array (Hz).
        magnitude: Magnitude spectrum array.

    Returns:
        float: Spectral centroid in Hz. Returns 0.0 if magnitude sum is zero.
    """
    total_energy = np.sum(magnitude**2)
    if total_energy == 0.0:
        return 0.0
    return float(np.sum(frequencies * magnitude**2) / total_energy)


def compute_spectral_entropy(
    magnitude: np.ndarray,
    base: int = 2,
) -> float:
    """Computes Shannon entropy of the normalised power spectrum.

    A broad, flat spectrum yields high entropy; narrow tonal content yields
    low entropy. Rising entropy late in degradation indicates broadband noise.

    Args:
        magnitude: Magnitude spectrum array (non-negative).
        base: Logarithm base (2 = bits, math.e = nats).

    Returns:
        float: Spectral entropy. Returns 0.0 if power sum is zero.
    """
    power = magnitude**2
    total = np.sum(power)
    if total == 0.0:
        return 0.0
    p_norm = power / total
    # Avoid log(0)
    p_nonzero = p_norm[p_norm > 0]
    if base == 2:
        return float(-np.sum(p_nonzero * np.log2(p_nonzero)))
    return float(-np.sum(p_nonzero * np.log(p_nonzero)))


def analyse_checkpoint_spectrum(
    waveform: np.ndarray,
    sampling_rate: float,
    frequency_bands: List[List[float]],
    window: str = "hann",
    n_fft: int = 2048,
    entropy_base: int = 2,
) -> Dict[str, Any]:
    """Computes full spectral feature set for a single checkpoint waveform.

    Args:
        waveform: 1-D channel waveform.
        sampling_rate: Sampling frequency in Hz.
        frequency_bands: List of [low_hz, high_hz] band definitions.
        window: FFT window function name.
        n_fft: FFT size.
        entropy_base: Logarithm base for entropy computation.

    Returns:
        Dict with keys:
            frequencies, magnitude, band_energies (list),
            spectral_centroid, spectral_entropy.
    """
    frequencies, magnitude = compute_fft_spectrum(
        waveform, sampling_rate, window=window, n_fft=n_fft
    )

    band_energies = [
        compute_band_energy(frequencies, magnitude, float(b[0]), float(b[1]))
        for b in frequency_bands
    ]
    centroid = compute_spectral_centroid(frequencies, magnitude)
    entropy = compute_spectral_entropy(magnitude, base=entropy_base)

    return {
        "frequencies": frequencies,
        "magnitude": magnitude,
        "band_energies": band_energies,
        "spectral_centroid": centroid,
        "spectral_entropy": entropy,
    }
