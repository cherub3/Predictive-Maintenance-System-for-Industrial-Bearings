"""Priority calculator — maps (RiskCategory, ConfidenceLevel) → int 1–5.

Architecture §7.6. Priority matrix is loaded from configuration and therefore
overridable without code changes. CRITICAL is always Priority 1 regardless of
confidence (architecture §12a.2 Tier 1 safety floor).
"""

import logging
from typing import Any, Dict

from src.decision_engine.confidence_estimator import ConfidenceLevel
from src.decision_engine.risk_classifier import RiskCategory


def load_priority_matrix(config: Dict[str, Any], logger: logging.Logger) -> Dict[str, Dict[str, int]]:
    """Load and validate the priority matrix from configuration.

    Args:
        config: Full project configuration dict.
        logger: Pipeline logger.

    Returns:
        Nested dict: {risk_category_str: {confidence_level_str: priority_int}}.

    Raises:
        ValueError: If a required cell is missing or out of range.
    """
    raw = config.get("decision_engine", {}).get("priority_matrix", {})
    matrix: Dict[str, Dict[str, int]] = {}

    for cat in RiskCategory:
        if cat.value not in raw:
            raise ValueError(
                f"Priority matrix missing risk category '{cat.value}'. "
                "All five categories must be present in "
                "config::decision_engine.priority_matrix."
            )
        matrix[cat.value] = {}
        for conf in ConfidenceLevel:
            if conf.value not in raw[cat.value]:
                raise ValueError(
                    f"Priority matrix missing confidence level '{conf.value}' "
                    f"for category '{cat.value}'."
                )
            val = int(raw[cat.value][conf.value])
            if val not in range(1, 6):
                raise ValueError(
                    f"Priority value {val} for [{cat.value}][{conf.value}] "
                    "is not in {{1,2,3,4,5}}."
                )
            matrix[cat.value][conf.value] = val

    # Safety floor: CRITICAL must always be 1 (architecture §12a.2 Tier 1).
    for conf in ConfidenceLevel:
        if matrix["CRITICAL"][conf.value] != 1:
            raise ValueError(
                f"Priority matrix violation: CRITICAL/{conf.value} must be 1 "
                "(architecture §12a.2 Tier 1 non-overridable safety floor)."
            )

    logger.info("Priority matrix loaded and validated.")
    return matrix


def compute_priority(
    risk_category: RiskCategory,
    confidence_level: ConfidenceLevel,
    priority_matrix: Dict[str, Dict[str, int]],
) -> int:
    """Return the priority score for a given risk/confidence pair.

    Args:
        risk_category: Classified risk tier.
        confidence_level: Estimated confidence level.
        priority_matrix: Loaded matrix (from load_priority_matrix).

    Returns:
        Integer in {1, 2, 3, 4, 5} — 1 = highest urgency.
    """
    return priority_matrix[risk_category.value][confidence_level.value]
