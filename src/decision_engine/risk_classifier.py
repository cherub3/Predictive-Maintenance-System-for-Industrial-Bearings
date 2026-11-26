"""Risk classifier — maps predicted RUL (minutes) to a five-tier RiskCategory.

Architecture §7.3 and §9. No hardcoded threshold values; all thresholds are
read from configuration. Threshold ordering is validated at pipeline entry
(§9c, §9d); raises ValueError on ordering violation.
"""

import logging
from enum import Enum
from typing import Any, Dict


class RiskCategory(str, Enum):
    """Five ordered risk tiers (highest to lowest severity). Architecture §9a."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    HEALTHY = "HEALTHY"


def validate_thresholds(thresholds: Dict[str, Any], logger: logging.Logger) -> None:
    """Enforce ordering constraint: 0 < critical < high < medium < low.

    Args:
        thresholds: Mapping with keys critical_minutes, high_minutes,
            medium_minutes, low_minutes.
        logger: Pipeline logger.

    Raises:
        ValueError: If any threshold is None/null or ordering is violated.
    """
    keys = ["critical_minutes", "high_minutes", "medium_minutes", "low_minutes"]
    for k in keys:
        if thresholds.get(k) is None:
            raise ValueError(
                f"Threshold '{k}' is null in configuration. "
                "Domain-expert-validated values are required before the "
                "decision engine can run (architecture §9b, §9d)."
            )

    c = float(thresholds["critical_minutes"])
    h = float(thresholds["high_minutes"])
    m = float(thresholds["medium_minutes"])
    lo = float(thresholds["low_minutes"])

    if not (0 < c < h < m < lo):
        raise ValueError(
            f"Threshold ordering violated: expected 0 < critical({c}) < "
            f"high({h}) < medium({m}) < low({lo}). "
            "Fix config/config.yaml::decision_engine.risk_thresholds."
        )
    logger.info(
        "Threshold ordering validated: critical=%.0f high=%.0f medium=%.0f low=%.0f",
        c, h, m, lo,
    )


def classify_risk(rul_minutes: float, thresholds: Dict[str, Any]) -> RiskCategory:
    """Map a RUL value (minutes) to a RiskCategory.

    Pure mapping — no side effects. Architecture §9a.

    Args:
        rul_minutes: Predicted RUL in minutes. Must be non-negative.
        thresholds: Validated threshold dict (call validate_thresholds first).

    Returns:
        RiskCategory enum member.
    """
    c = float(thresholds["critical_minutes"])
    h = float(thresholds["high_minutes"])
    m = float(thresholds["medium_minutes"])
    lo = float(thresholds["low_minutes"])

    if rul_minutes <= 0.0:
        # BR-05 safety floor — zero/negative RUL is always CRITICAL.
        return RiskCategory.CRITICAL
    elif rul_minutes <= c:
        return RiskCategory.CRITICAL
    elif rul_minutes <= h:
        return RiskCategory.HIGH
    elif rul_minutes <= m:
        return RiskCategory.MEDIUM
    elif rul_minutes <= lo:
        return RiskCategory.LOW
    else:
        return RiskCategory.HEALTHY
