"""Raw vibration signal loader for NASA IMS Bearing Dataset (Set 1).

Loads a single ASCII acquisition file into a NumPy array, with validation
of expected shape (20,480 samples x 8 channels). Designed for Tier 1 and
Tier 2 processing in Phase 3.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


EXPECTED_SAMPLES = 20480
EXPECTED_CHANNELS = 8


def load_signal_file(
    file_path: Path,
    expected_samples: int = EXPECTED_SAMPLES,
    expected_channels: int = EXPECTED_CHANNELS,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Optional[np.ndarray], str]:
    """Loads an ASCII vibration file into a (samples x channels) NumPy array.

    Args:
        file_path: Path to the ASCII data file.
        expected_samples: Expected row count (20,480 for NASA IMS Set 1).
        expected_channels: Expected column count (8 for Set 1).
        logger: Optional logger for warnings.

    Returns:
        Tuple of (array or None, status string).
        Array shape is (expected_samples, expected_channels) on success.
    """
    try:
        data = np.loadtxt(file_path)
    except Exception as exc:
        msg = f"Failed to read {file_path.name}: {exc}"
        if logger:
            logger.warning(msg)
        return None, msg

    if data.ndim != 2:
        msg = f"Unexpected shape {data.shape} in {file_path.name}"
        if logger:
            logger.warning(msg)
        return None, msg

    rows, cols = data.shape
    if rows != expected_samples or cols != expected_channels:
        msg = (
            f"Shape mismatch in {file_path.name}: got ({rows}, {cols}), "
            f"expected ({expected_samples}, {expected_channels})"
        )
        if logger:
            logger.warning(msg)
        return None, msg

    return data, "OK"


def extract_channel(data: np.ndarray, channel: int) -> np.ndarray:
    """Extracts a single channel waveform (0-indexed column) from the signal array.

    Args:
        data: Array of shape (samples, channels).
        channel: 0-indexed channel number.

    Returns:
        np.ndarray: 1-D waveform array of length samples.
    """
    return data[:, channel]
