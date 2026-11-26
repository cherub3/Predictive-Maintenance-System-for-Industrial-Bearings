"""Recommendation assembler — builds Recommendation dataclass instances.

Architecture §7.7. Pure assembler: calls no decision logic. Every field is
provided by caller; this module only packages them into the canonical struct
and converts the list to a DataFrame for persistence.
"""

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from src.decision_engine.confidence_estimator import ConfidenceLevel
from src.decision_engine.maintenance_policy import MaintenanceAction
from src.decision_engine.risk_classifier import RiskCategory


@dataclass
class Recommendation:
    """One bearing-level decision at a single timestamp. Architecture §6a."""

    bearing_id: str
    timestamp: pd.Timestamp
    min_predicted_rul_minutes: float
    mean_predicted_rul_minutes: float
    channel_count: int
    risk_category: RiskCategory
    maintenance_action: MaintenanceAction
    confidence_level: ConfidenceLevel
    priority_score: int
    is_business_rule_override: bool
    override_reason: Optional[str]
    explanation_text: str
    model_type: str
    n2_caveat: bool
    rule_version: str
    config_fingerprint: str
    experiment_id: str  # Phase 6 experiment ID for upstream traceability


def assemble_recommendation(
    bearing_id: str,
    timestamp: pd.Timestamp,
    min_rul: float,
    mean_rul: float,
    channel_count: int,
    risk_category: RiskCategory,
    maintenance_action: MaintenanceAction,
    confidence_level: ConfidenceLevel,
    priority_score: int,
    explanation_text: str,
    model_type: str,
    n2_caveat: bool,
    rule_version: str,
    config_fingerprint: str,
    experiment_id: str,
    override_reason: Optional[str] = None,
) -> Recommendation:
    """Assemble all decision components into a Recommendation.

    Args:
        bearing_id: Bearing identifier.
        timestamp: Data file timestamp for this row.
        min_rul: Minimum predicted RUL across channels (conservative).
        mean_rul: Mean predicted RUL across channels (informational).
        channel_count: Number of channels that contributed to aggregation.
        risk_category: Classified risk tier.
        maintenance_action: Recommended action (may be post-business-rule).
        confidence_level: Estimated decision confidence.
        priority_score: Integer 1–5 (1 = highest urgency).
        explanation_text: Human-readable justification.
        model_type: Phase 6 model type (propagated from contract).
        n2_caveat: Whether n=2 caveat applies (always True per Phase 6 contract).
        rule_version: Business rule set version string.
        config_fingerprint: SHA-256 of decision_engine config block.
        experiment_id: Phase 6 experiment ID (upstream traceability).
        override_reason: Non-null when a business rule altered the base decision.

    Returns:
        Recommendation dataclass instance.
    """
    return Recommendation(
        bearing_id=bearing_id,
        timestamp=timestamp,
        min_predicted_rul_minutes=min_rul,
        mean_predicted_rul_minutes=mean_rul,
        channel_count=channel_count,
        risk_category=risk_category,
        maintenance_action=maintenance_action,
        confidence_level=confidence_level,
        priority_score=priority_score,
        is_business_rule_override=override_reason is not None,
        override_reason=override_reason,
        explanation_text=explanation_text,
        model_type=model_type,
        n2_caveat=n2_caveat,
        rule_version=rule_version,
        config_fingerprint=config_fingerprint,
        experiment_id=experiment_id,
    )


def recommendations_to_dataframe(recommendations: List[Recommendation]) -> pd.DataFrame:
    """Convert a list of Recommendation objects to a pandas DataFrame.

    Enum values are serialised as their string `.value` for parquet compatibility.

    Args:
        recommendations: List of assembled Recommendation instances.

    Returns:
        DataFrame with one row per recommendation, columns matching §6a schema.
    """
    rows = []
    for r in recommendations:
        rows.append(
            {
                "bearing_id": r.bearing_id,
                "timestamp": r.timestamp,
                "min_predicted_rul_minutes": r.min_predicted_rul_minutes,
                "mean_predicted_rul_minutes": r.mean_predicted_rul_minutes,
                "channel_count": r.channel_count,
                "risk_category": r.risk_category.value,
                "maintenance_action": r.maintenance_action.value,
                "confidence_level": r.confidence_level.value,
                "priority_score": r.priority_score,
                "is_business_rule_override": r.is_business_rule_override,
                "override_reason": r.override_reason,
                "explanation_text": r.explanation_text,
                "model_type": r.model_type,
                "n2_caveat": r.n2_caveat,
                "rule_version": r.rule_version,
                "config_fingerprint": r.config_fingerprint,
                "experiment_id": r.experiment_id,
            }
        )
    return pd.DataFrame(rows)
