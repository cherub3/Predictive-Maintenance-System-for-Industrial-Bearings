"""Confidence estimator — derives categorical decision confidence.

Architecture §7.5 and §11. Three factors applied in sequence:
  1. Model-type modifier (from config)
  2. n=2 penalty (mandatory when n2_caveat is True)
  3. Boundary-proximity downgrade (when RUL is near a threshold)

Confidence is categorical (HIGH / MEDIUM / LOW) — not a calibrated probability.
Architecture §11a explains why numerical probabilities are not scientifically
supportable for n=2 training corpora.
"""

import logging
from enum import Enum
from typing import Any, Dict



class ConfidenceLevel(str, Enum):
    """Three-tier confidence scale. Architecture §7.5."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# Ordered worst → best for downgrade arithmetic.
_LEVEL_ORDER = [ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH]


def _downgrade(level: ConfidenceLevel) -> ConfidenceLevel:
    """Reduce confidence by one grade; floor at LOW."""
    idx = _LEVEL_ORDER.index(level)
    return _LEVEL_ORDER[max(0, idx - 1)]


def _apply_model_type_modifier(
    model_type: str,
    modifier_map: Dict[str, str],
) -> ConfidenceLevel:
    """Return the base confidence for this model type.

    Unknown model types default to MEDIUM (conservative).
    """
    raw = modifier_map.get(model_type, "MEDIUM")
    try:
        return ConfidenceLevel(raw)
    except ValueError:
        return ConfidenceLevel.MEDIUM


def _apply_n2_penalty(level: ConfidenceLevel, n2_caveat: bool) -> ConfidenceLevel:
    """Downgrade by one grade when n2_caveat is True. Architecture §7.5a Factor 2."""
    if n2_caveat:
        return _downgrade(level)
    return level


def _apply_boundary_proximity(
    level: ConfidenceLevel,
    rul_minutes: float,
    thresholds: Dict[str, Any],
    tolerance_minutes: float,
) -> ConfidenceLevel:
    """Downgrade by one grade when RUL is within ±tolerance of any threshold.

    Architecture §7.5a Factor 3.
    """
    boundary_values = [
        float(thresholds["critical_minutes"]),
        float(thresholds["high_minutes"]),
        float(thresholds["medium_minutes"]),
        float(thresholds["low_minutes"]),
    ]
    for bv in boundary_values:
        if abs(rul_minutes - bv) <= tolerance_minutes:
            return _downgrade(level)
    return level


def estimate_confidence(
    rul_minutes: float,
    model_type: str,
    n2_caveat: bool,
    thresholds: Dict[str, Any],
    config: Dict[str, Any],
    logger: logging.Logger,
) -> ConfidenceLevel:
    """Compute the confidence level for one decision row.

    Applies factors in the fixed order defined by architecture §7.5a:
      1. Model-type modifier
      2. n=2 penalty
      3. Boundary-proximity downgrade
    Floor at LOW.

    Args:
        rul_minutes: The aggregated (min across channels) predicted RUL.
        model_type: Model type string from Phase 6 contract.
        n2_caveat: Whether the n=2 caveat is declared.
        thresholds: Validated threshold dict.
        config: Full project configuration dict.
        logger: Pipeline logger.

    Returns:
        ConfidenceLevel enum member.
    """
    de_cfg = config.get("decision_engine", {})
    conf_cfg = de_cfg.get("confidence", {})
    modifier_map = conf_cfg.get("model_type_modifiers", {})
    tolerance = float(conf_cfg.get("boundary_proximity_tolerance_minutes", 0.0))

    level = _apply_model_type_modifier(model_type, modifier_map)
    level = _apply_n2_penalty(level, n2_caveat)
    level = _apply_boundary_proximity(level, rul_minutes, thresholds, tolerance)

    logger.debug(
        "Confidence estimate: rul=%.0f model_type=%s n2=%s → %s",
        rul_minutes, model_type, n2_caveat, level.value,
    )
    return level
