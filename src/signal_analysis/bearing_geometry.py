"""Rexnord ZA-2115 bearing geometry and defect frequency computation.

Computes Ball Pass Frequency Outer race (BPFO), Ball Pass Frequency Inner race
(BPFI), and Ball Spin Frequency (BSF) from physical bearing geometry parameters.
All geometry parameters are loaded from configuration; nothing is hardcoded.
"""

import math
from typing import Any, Dict, NamedTuple


class BearingGeometry(NamedTuple):
    """Physical geometry parameters for a rolling-element bearing."""

    num_rolling_elements: int
    ball_diameter_inches: float
    pitch_diameter_inches: float
    contact_angle_degrees: float
    shaft_speed_rpm: float


class DefectFrequencies(NamedTuple):
    """Characteristic defect frequencies in Hz for a rolling-element bearing."""

    bpfo_hz: float  # Ball Pass Frequency Outer race
    bpfi_hz: float  # Ball Pass Frequency Inner race
    bsf_hz: float   # Ball Spin Frequency
    shaft_frequency_hz: float


def load_bearing_geometry(config: Dict[str, Any]) -> BearingGeometry:
    """Loads Rexnord ZA-2115 geometry from the datasets registry defaults.

    The physical rig (bearing type, shaft speed) is identical across every
    registered dataset (Set 1/2/3), so geometry is a single shared constant
    sourced from `datasets.defaults`, not a per-dataset value.

    Args:
        config: Full project configuration dictionary.

    Returns:
        BearingGeometry: Populated named-tuple of physical parameters.

    Raises:
        KeyError: If required geometry keys are absent from configuration.
    """
    defaults = config["datasets"]["defaults"]
    geom_cfg = defaults["bearing_geometry"]

    return BearingGeometry(
        num_rolling_elements=int(geom_cfg["num_rolling_elements"]),
        ball_diameter_inches=float(geom_cfg["ball_diameter_inches"]),
        pitch_diameter_inches=float(geom_cfg["pitch_diameter_inches"]),
        contact_angle_degrees=float(geom_cfg["contact_angle_degrees"]),
        shaft_speed_rpm=float(defaults["shaft_speed_rpm"]),
    )


def compute_defect_frequencies(geometry: BearingGeometry) -> DefectFrequencies:
    """Computes characteristic defect frequencies from bearing geometry.

    Implements standard rolling-element bearing defect frequency formulas:
      BPFO = (N/2) * fr * (1 - (Bd/Pd) * cos(alpha))
      BPFI = (N/2) * fr * (1 + (Bd/Pd) * cos(alpha))
      BSF  = (Pd / (2*Bd)) * fr * (1 - ((Bd/Pd) * cos(alpha))^2)

    where:
      N     = number of rolling elements
      fr    = shaft rotational frequency (Hz)
      Bd    = ball diameter
      Pd    = pitch diameter
      alpha = contact angle (radians)

    Args:
        geometry: Physical geometry parameters for the bearing.

    Returns:
        DefectFrequencies: Computed characteristic frequencies in Hz.
    """
    n = geometry.num_rolling_elements
    bd = geometry.ball_diameter_inches
    pd = geometry.pitch_diameter_inches
    alpha_rad = math.radians(geometry.contact_angle_degrees)
    fr = geometry.shaft_speed_rpm / 60.0  # Hz

    cos_alpha = math.cos(alpha_rad)
    ratio = (bd / pd) * cos_alpha

    # Outer race: fewer ball contacts per shaft revolution → (1 - ratio)
    # Inner race: more ball contacts per shaft revolution → (1 + ratio)
    bpfo = (n / 2.0) * fr * (1.0 - ratio)
    bpfi = (n / 2.0) * fr * (1.0 + ratio)
    bsf = (pd / (2.0 * bd)) * fr * (1.0 - ratio**2)

    return DefectFrequencies(
        bpfo_hz=round(bpfo, 4),
        bpfi_hz=round(bpfi, 4),
        bsf_hz=round(bsf, 4),
        shaft_frequency_hz=round(fr, 4),
    )


BEARING_TYPE = "Rexnord ZA-2115 Double-Row"
