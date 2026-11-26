"""Decision explainer — generates human-readable justification for each recommendation.

Architecture §7.8 and §13. Every explanation must satisfy the seven-question
checklist defined in §13d. The explanation references Phase 6 contract fields
(model_type, n2_caveat_text, mean_mae_minutes, experiment_id) as read-only
evidence — it does not re-run SHAP and does not import the shap library.
"""

from typing import Any, List, Optional

from src.decision_engine.confidence_estimator import ConfidenceLevel
from src.decision_engine.maintenance_policy import MaintenanceAction
from src.decision_engine.risk_classifier import RiskCategory


def _format_rul_human(rul_minutes: float) -> str:
    """Convert minutes to a human-readable hours/days string."""
    hours = rul_minutes / 60.0
    days = rul_minutes / 1440.0
    return f"{rul_minutes:.0f} minutes ({hours:.1f} hours / {days:.2f} days)"


def _build_model_selection_note(model_type: str) -> str:
    """Return the mandatory model-selection prose note. Architecture §7.8."""
    if model_type in ("mean", "median"):
        return (
            f"Baseline model ('{model_type}') selected via demotion fallback — "
            "no non-baseline candidate individually cleared the Phase 6 promotion "
            "criteria on this n=2 dataset. This is a constant predictor: "
            "predictions reflect training-distribution central tendency, "
            "not feature-driven estimation of the bearing's evolving degradation state."
        )
    return (
        f"Model type '{model_type}' was promoted through Phase 6 promotion "
        "criteria and selected as the best model per the Phase 6 architecture §8.3 hierarchy."
    )


def _build_confidence_basis(
    confidence_level: ConfidenceLevel,
    model_type: str,
    n2_caveat: bool,
    near_boundary: bool,
) -> str:
    """Summarise the three-factor confidence derivation."""
    factors = []
    if model_type in ("mean", "median"):
        factors.append(f"model-type modifier: '{model_type}' → base LOW (constant predictor)")
    else:
        factors.append(f"model-type modifier: '{model_type}' → base {confidence_level.value}")
    if n2_caveat:
        factors.append("n=2 penalty: mandatory downgrade applied")
    if near_boundary:
        factors.append("boundary proximity: RUL near a decision threshold — additional downgrade applied")
    return "; ".join(factors) if factors else f"confidence level: {confidence_level.value}"


def _build_action_rationale(maintenance_action: MaintenanceAction) -> str:
    """Return the operational meaning of the recommended action. Architecture §10c."""
    rationales = {
        MaintenanceAction.IMMEDIATE_SHUTDOWN: (
            "Stop the equipment now. Do not await the next maintenance window. "
            "Failure is imminent — continuing operation risks catastrophic bearing seizure."
        ),
        MaintenanceAction.URGENT_MAINTENANCE: (
            "Complete bearing replacement at the next available maintenance window, "
            "bringing it forward if necessary. Bearing is in the danger zone."
        ),
        MaintenanceAction.SCHEDULE_MAINTENANCE: (
            "Add bearing replacement to the near-term maintenance schedule. "
            "Do not skip the next planned inspection. Degradation has been detected."
        ),
        MaintenanceAction.ROUTINE_MONITORING: (
            "Confirm bearing health at the next scheduled inspection. "
            "No immediate action required. Early degradation signal detected."
        ),
        MaintenanceAction.NORMAL_OPERATION: (
            "Continue standard operating schedule. "
            "No specific maintenance action needed at this time."
        ),
    }
    return rationales.get(maintenance_action, maintenance_action.value)


def _build_shap_evidence(
    model_type: str,
    top_features: List[str],
    max_features: int,
) -> str:
    """Build SHAP feature evidence prose — suppressed for constant predictors."""
    if model_type in ("mean", "median"):
        return (
            "Feature evidence suppressed: the selected model is a constant predictor "
            "with no feature sensitivity. Top permutation-ranked features from Phase 6 "
            "are not meaningful for this model type."
        )
    if not top_features:
        return "Feature importance: not available (see Phase 6 feature_importance_report.md)."
    features_listed = ", ".join(top_features[:max_features])
    return (
        f"Top Phase 6 permutation-ranked features (static reference, not re-computed): "
        f"{features_listed}."
    )


def generate_explanation(
    bearing_id: str,
    timestamp: Any,
    min_rul: float,
    mean_rul: float,
    channel_count: int,
    rul_spread: float,
    risk_category: RiskCategory,
    maintenance_action: MaintenanceAction,
    confidence_level: ConfidenceLevel,
    priority_score: int,
    model_type: str,
    n2_caveat: bool,
    n2_caveat_text: str,
    n_rtf_units: int,
    mean_mae_minutes: float,
    experiment_id: str,
    override_reason: Optional[str],
    top_features: List[str],
    near_boundary: bool,
    max_top_features: int,
    rule_version: str,
    config_fingerprint: str,
) -> str:
    """Generate a structured explanation satisfying the §13d seven-question checklist.

    Questions answered:
      1. What prediction produced this decision? (Risk Assessment block)
      2. Which business rules fired? (Business Rules block)
      3. Which rule determined the final recommendation? (Business Rules block)
      4. Which evidence was used? (Evidence block)
      5. Which assumptions were applied? (Limitations block)
      6. Which model produced the prediction? (Model Evidence block)
      7. Which experiment generated the model? (Model Evidence block)

    Args:
        bearing_id: Bearing identifier.
        timestamp: Data file timestamp.
        min_rul: Minimum RUL across channels (conservative).
        mean_rul: Mean RUL across channels (informational).
        channel_count: Number of channels contributing.
        rul_spread: Max−min RUL across channels (inter-channel spread).
        risk_category: Classified risk tier.
        maintenance_action: Final recommended action.
        confidence_level: Decision confidence level.
        priority_score: Urgency score 1–5.
        model_type: From Phase 6 contract.
        n2_caveat: Always True per Phase 6 contract.
        n2_caveat_text: Full n2 caveat prose from Phase 6 contract.
        n_rtf_units: Number of run-to-failure units backing the LOBO folds (Phase 6 contract).
        mean_mae_minutes: Phase 6 evaluation mean MAE.
        experiment_id: Phase 6 experiment ID.
        override_reason: Non-null if a business rule changed the base decision.
        top_features: Top permutation-ranked features from Phase 6 contract.
        near_boundary: Whether RUL was near a decision-threshold boundary.
        max_top_features: Maximum number of features to list.
        rule_version: Business rule set version.
        config_fingerprint: SHA-256 of Phase 7 decision_engine config.

    Returns:
        Multi-line explanation string.
    """
    model_note = _build_model_selection_note(model_type)
    confidence_basis = _build_confidence_basis(
        confidence_level, model_type, n2_caveat, near_boundary
    )
    action_rationale = _build_action_rationale(maintenance_action)
    feature_evidence = _build_shap_evidence(model_type, top_features, max_top_features)

    rules_fired = "None (base risk→action mapping applied without modification)"
    governing_rule = "Base decision pipeline (risk_classifier → maintenance_policy)"
    if override_reason:
        rules_fired = override_reason
        governing_rule = override_reason

    lines = [
        f"Bearing {bearing_id} — {maintenance_action.value} ({timestamp})",
        "",
        "── Risk Assessment ─────────────────────────────────────────────",
        f"  Predicted RUL (minimum across {channel_count} channels): {_format_rul_human(min_rul)}",
        f"  Predicted RUL (mean across {channel_count} channels):    {_format_rul_human(mean_rul)}",
        f"  Inter-channel RUL spread: {rul_spread:.0f} minutes",
        f"  Risk Category: {risk_category.value}",
        f"  Maintenance Action: {maintenance_action.value}",
        f"  Priority: {priority_score} / 5  (1 = highest urgency)",
        f"  Confidence Level: {confidence_level.value}",
        "",
        "── Business Rules ──────────────────────────────────────────────",
        f"  Rules fired: {rules_fired}",
        f"  Governing rule: {governing_rule}",
        "",
        "── Model Evidence ──────────────────────────────────────────────",
        f"  Selected Model: {model_type}",
        f"  Model Selection: {model_note}",
        f"  Phase 6 Experiment: {experiment_id}",
        f"  Evaluation MAE: {mean_mae_minutes:.0f} minutes (mean across {n_rtf_units} LOBO folds)",
        f"  {feature_evidence}",
        "",
        "── Confidence Basis ────────────────────────────────────────────",
        f"  {confidence_basis}",
        "",
        "── Statistical Limitations ─────────────────────────────────────",
        f"  n={n_rtf_units} Caveat: {n2_caveat_text}",
        "",
        "── Recommended Action ──────────────────────────────────────────",
        f"  {action_rationale}",
        "",
        "── Governance ──────────────────────────────────────────────────",
        f"  Rule Version: {rule_version}",
        f"  Config Fingerprint: {config_fingerprint}",
    ]

    if override_reason:
        lines += [
            "",
            "── Override Note ────────────────────────────────────────────────",
            f"  {override_reason}",
        ]

    return "\n".join(lines)
