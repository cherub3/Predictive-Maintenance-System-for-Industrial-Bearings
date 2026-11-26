"""Maintenance policy — maps RiskCategory to a MaintenanceAction.

Architecture §7.4 and §10. One-to-one mapping loaded from configuration;
no hardcoded values in source. Mapping is total (every category has an action)
and functional (each category maps to exactly one action).
"""

import logging
from enum import Enum
from typing import Any, Dict

from src.decision_engine.risk_classifier import RiskCategory


class MaintenanceAction(str, Enum):
    """Five maintenance actions ordered by urgency. Architecture §10c."""

    IMMEDIATE_SHUTDOWN = "IMMEDIATE_SHUTDOWN"
    URGENT_MAINTENANCE = "URGENT_MAINTENANCE"
    SCHEDULE_MAINTENANCE = "SCHEDULE_MAINTENANCE"
    ROUTINE_MONITORING = "ROUTINE_MONITORING"
    NORMAL_OPERATION = "NORMAL_OPERATION"


def load_policy_map(config: Dict[str, Any], logger: logging.Logger) -> Dict[RiskCategory, MaintenanceAction]:
    """Load the risk→action mapping from configuration.

    Args:
        config: Full project configuration dict.
        logger: Pipeline logger.

    Returns:
        Dict mapping each RiskCategory to a MaintenanceAction.

    Raises:
        ValueError: If a category is missing from the policy map or contains
            an invalid action name.
    """
    raw = config.get("decision_engine", {}).get("policy_map", {})
    policy: Dict[RiskCategory, MaintenanceAction] = {}

    for cat in RiskCategory:
        if cat.value not in raw:
            raise ValueError(
                f"Risk category '{cat.value}' has no entry in "
                "config::decision_engine.policy_map. "
                "All five categories must be mapped."
            )
        raw_action = raw[cat.value]
        try:
            policy[cat] = MaintenanceAction(raw_action)
        except ValueError:
            raise ValueError(
                f"Unknown maintenance action '{raw_action}' for category "
                f"'{cat.value}'. Valid actions: "
                f"{[a.value for a in MaintenanceAction]}"
            )

    logger.info("Policy map loaded: %s", {k.value: v.value for k, v in policy.items()})
    return policy


def get_maintenance_action(
    risk_category: RiskCategory,
    policy_map: Dict[RiskCategory, MaintenanceAction],
) -> MaintenanceAction:
    """Return the maintenance action for a given risk category.

    Args:
        risk_category: Classified risk tier.
        policy_map: Loaded policy mapping (from load_policy_map).

    Returns:
        MaintenanceAction for this risk category.
    """
    return policy_map[risk_category]
