"""Business rules — BR-01 through BR-05 applied in fixed precedence order.

Architecture §7.9, §12a. Rules execute in the documented order and authority
hierarchy:
  Tier 1 (non-overridable decision-altering): BR-01, BR-05
  Tier 2 (decision-substituting, absent-data path): BR-02
  Tier 3 (modifying/annotating, non-decision-altering): BR-03, BR-04

Business rule version is pinned and must match config::decision_engine.rule_version.
Any modification to rule logic must increment RULE_VERSION.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.decision_engine.confidence_estimator import ConfidenceLevel, _downgrade
from src.decision_engine.maintenance_policy import MaintenanceAction
from src.decision_engine.risk_classifier import RiskCategory

RULE_VERSION = "1.0"

# Constant-predictor annotation text (BR-04). Architecture §7.9a.
_BR04_ANNOTATION = (
    "CONSTANT PREDICTOR NOTE (BR-04): All predictions for this bearing are "
    "constant (same value at every timestamp) because the selected model is a "
    "baseline predictor with no feature sensitivity. The time series of "
    "decisions for this bearing is therefore uniform. Decisions reflect the "
    "training-distribution central tendency, not the bearing's evolving "
    "degradation state."
)


@dataclass
class BusinessRuleResult:
    """Output of apply_business_rules for one recommendation row."""

    risk_category: RiskCategory
    maintenance_action: MaintenanceAction
    confidence_level: ConfidenceLevel
    override_reason: Optional[str]
    br04_annotation: Optional[str]
    rules_applied: List[str]


def _apply_br01(
    risk: RiskCategory,
    action: MaintenanceAction,
    policy_map: Dict[RiskCategory, MaintenanceAction],
) -> tuple:
    """BR-01: Safety floor — CRITICAL is always IMMEDIATE_SHUTDOWN.

    Non-overridable. Architecture §7.9a, §12a.2 Tier 1.

    Returns:
        (risk, action, override_reason_or_None, fired: bool)
    """
    if risk == RiskCategory.CRITICAL and action != MaintenanceAction.IMMEDIATE_SHUTDOWN:
        return (
            risk,
            MaintenanceAction.IMMEDIATE_SHUTDOWN,
            "BR-01 safety floor: CRITICAL risk always maps to IMMEDIATE_SHUTDOWN "
            "(non-overridable, architecture §12a.2 Tier 1).",
            True,
        )
    return risk, action, None, False


def _apply_br05(
    rul_minutes: float,
    risk: RiskCategory,
    action: MaintenanceAction,
    policy_map: Dict[RiskCategory, MaintenanceAction],
) -> tuple:
    """BR-05: Zero-RUL floor — RUL ≤ 0 is always CRITICAL.

    Non-overridable. Architecture §7.9a, §12a.2 Tier 1.

    Returns:
        (risk, action, override_reason_or_None, fired: bool)
    """
    if rul_minutes <= 0.0 and risk != RiskCategory.CRITICAL:
        new_action = MaintenanceAction.IMMEDIATE_SHUTDOWN
        return (
            RiskCategory.CRITICAL,
            new_action,
            f"BR-05 zero-RUL floor: predicted RUL={rul_minutes:.1f} ≤ 0 → "
            "forced CRITICAL / IMMEDIATE_SHUTDOWN (non-overridable).",
            True,
        )
    return risk, action, None, False


def _apply_br03(
    confidence: ConfidenceLevel,
    rul_spread: float,
    threshold_minutes: float,
) -> tuple:
    """BR-03: High channel disagreement → confidence downgrade + annotation.

    Tier 3 — modifies confidence_level only. Architecture §7.9a.

    Returns:
        (confidence, annotation_or_None, fired: bool)
    """
    if rul_spread > threshold_minutes:
        downgraded = _downgrade(confidence)
        annotation = (
            f"BR-03 channel disagreement: inter-channel RUL spread "
            f"({rul_spread:.0f} min) exceeds threshold ({threshold_minutes:.0f} min). "
            "Confidence downgraded by one grade due to physical uncertainty."
        )
        return downgraded, annotation, True
    return confidence, None, False


def _apply_br04(model_type: str) -> Optional[str]:
    """BR-04: Constant-predictor annotation for mean/median models.

    Tier 3 — annotation only. Architecture §7.9a. Extended rationale in §7.9a.

    Returns:
        Annotation string if applicable, else None.
    """
    if model_type in ("mean", "median"):
        return _BR04_ANNOTATION
    return None


def apply_business_rules(
    rul_minutes: float,
    risk_category: RiskCategory,
    maintenance_action: MaintenanceAction,
    confidence_level: ConfidenceLevel,
    rul_spread: float,
    model_type: str,
    policy_map: Dict[RiskCategory, MaintenanceAction],
    config: Dict[str, Any],
    logger: logging.Logger,
) -> BusinessRuleResult:
    """Apply BR-01 through BR-05 in fixed precedence order.

    Execution order per architecture §12a.1:
      1. BR-01 (Tier 1 — safety floor)
      2. BR-05 (Tier 1 — zero-RUL floor, defensive redundancy to BR-01)
      3. BR-02 not applicable here (only fires when no prediction exists at all)
      4. BR-03 (Tier 3 — confidence modifier)
      5. BR-04 (Tier 3 — annotation)

    BR-02 is handled separately in decision_engine.py when a bearing-timestamp
    pair has no prediction row.

    Args:
        rul_minutes: Aggregated minimum RUL across channels.
        risk_category: Base risk classification.
        maintenance_action: Base maintenance action.
        confidence_level: Estimated confidence level (pre-business-rule).
        rul_spread: Max − min RUL across channels.
        model_type: From Phase 6 contract.
        policy_map: Loaded policy mapping.
        config: Full project configuration dict.
        logger: Pipeline logger.

    Returns:
        BusinessRuleResult with potentially modified fields and rule trace.
    """
    de_cfg = config.get("decision_engine", {})
    br_cfg = de_cfg.get("business_rules", {})
    spread_threshold = float(br_cfg.get("channel_rul_spread_threshold_minutes", 1e9))

    rules_applied: List[str] = []
    override_parts: List[str] = []
    br03_annotation: Optional[str] = None

    # --- Tier 1: BR-01 Safety floor ---
    risk_category, maintenance_action, br01_reason, br01_fired = _apply_br01(
        risk_category, maintenance_action, policy_map
    )
    if br01_fired:
        rules_applied.append("BR-01")
        override_parts.append(br01_reason)
        logger.debug("BR-01 fired for RUL=%.0f", rul_minutes)

    # --- Tier 1: BR-05 Zero-RUL floor (defensive) ---
    risk_category, maintenance_action, br05_reason, br05_fired = _apply_br05(
        rul_minutes, risk_category, maintenance_action, policy_map
    )
    if br05_fired:
        rules_applied.append("BR-05")
        override_parts.append(br05_reason)
        logger.debug("BR-05 fired for RUL=%.1f", rul_minutes)

    # --- Tier 3: BR-03 Channel disagreement ---
    confidence_level, br03_annotation_raw, br03_fired = _apply_br03(
        confidence_level, rul_spread, spread_threshold
    )
    if br03_fired:
        rules_applied.append("BR-03")
        br03_annotation = br03_annotation_raw
        logger.debug(
            "BR-03 fired: spread=%.0f > threshold=%.0f, confidence downgraded",
            rul_spread, spread_threshold,
        )

    # --- Tier 3: BR-04 Constant-predictor annotation ---
    br04_annotation = _apply_br04(model_type)
    if br04_annotation:
        rules_applied.append("BR-04")
        logger.debug("BR-04 fired: model_type=%s", model_type)

    # Combine BR-03 and BR-04 annotations into a single annotation field.
    combined_annotation = None
    parts = [p for p in [br03_annotation, br04_annotation] if p]
    if parts:
        combined_annotation = " | ".join(parts)

    override_reason = "; ".join(override_parts) if override_parts else None

    return BusinessRuleResult(
        risk_category=risk_category,
        maintenance_action=maintenance_action,
        confidence_level=confidence_level,
        override_reason=override_reason,
        br04_annotation=combined_annotation,
        rules_applied=rules_applied,
    )


def apply_missing_prediction_fallback(
    bearing_id: str,
    timestamp: Any,
    model_type: str,
    n2_caveat: bool,
    n2_caveat_text: str,
    mean_mae_minutes: float,
    experiment_id: str,
    rule_version: str,
    config_fingerprint: str,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> dict:
    """BR-02: Substitute row when a bearing-timestamp has no prediction.

    Architecture §7.9a BR-02. Returns a dict of fields for Recommendation
    construction — the caller assembles the full Recommendation.

    Args:
        bearing_id: Bearing ID with missing prediction.
        timestamp: Timestamp with missing prediction.
        model_type: From Phase 6 contract.
        n2_caveat: From Phase 6 contract.
        n2_caveat_text: From Phase 6 contract.
        mean_mae_minutes: From Phase 6 contract.
        experiment_id: Phase 6 experiment ID.
        rule_version: Business rule version.
        config_fingerprint: Phase 7 config fingerprint.
        config: Full project configuration dict.
        logger: Pipeline logger.

    Returns:
        Dict with all fields needed for Recommendation construction.
    """
    br_cfg = config.get("decision_engine", {}).get("business_rules", {})
    default_risk_str = br_cfg.get("missing_prediction_default_risk", "HIGH")
    risk = RiskCategory(default_risk_str)
    policy_map_raw = config.get("decision_engine", {}).get("policy_map", {})
    action = MaintenanceAction(policy_map_raw.get(risk.value, "URGENT_MAINTENANCE"))

    override_reason = (
        f"BR-02 missing-prediction fallback: no prediction row found for "
        f"{bearing_id} at {timestamp}. Conservative {risk.value} risk assigned "
        "(architecture §7.9a BR-02)."
    )
    logger.warning("BR-02 fired: no prediction for %s @ %s", bearing_id, timestamp)

    return {
        "risk_category": risk,
        "maintenance_action": action,
        "confidence_level": ConfidenceLevel.LOW,
        "override_reason": override_reason,
        "min_predicted_rul_minutes": float("nan"),
        "mean_predicted_rul_minutes": float("nan"),
        "channel_count": 0,
    }
