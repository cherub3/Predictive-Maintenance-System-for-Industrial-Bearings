"""Phase 7 → Phase 8 contract generator.

Architecture §7.11 and §21. Produces phase7_decision_contract.yaml — the sole
machine-readable interface Phase 8 may consume from Phase 7. Contract includes
the complete artifact enumeration per §21c and all consumer_expectations per §21a.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from src.decision_engine.business_rules import RULE_VERSION
from src.decision_engine.contract_loader import ContractValidationResult


def build_phase8_contract(
    recommendations_df: pd.DataFrame,
    contract_validation: ContractValidationResult,
    manifest: Dict[str, Any],
    experiment_id: str,
    config_fingerprint: str,
    git_commit: str,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Build the complete Phase 7 → Phase 8 contract payload. Architecture §21.

    Args:
        recommendations_df: Final recommendations DataFrame.
        contract_validation: Validated Phase 6 contract metadata.
        manifest: Built decision manifest.
        experiment_id: Phase 7 experiment ID.
        config_fingerprint: SHA-256 of decision_engine config.
        git_commit: Git commit SHA or 'unknown'.
        config: Full project configuration dict.
        logger: Pipeline logger.

    Returns:
        Dict representing the phase7_decision_contract.yaml payload.
    """
    de_cfg = config.get("decision_engine", {})
    output_paths = de_cfg.get("output_paths", {})

    risk_counts = (
        recommendations_df["risk_category"].value_counts().to_dict()
        if not recommendations_df.empty
        else {}
    )

    contract = {
        "schema_version": "1.0",
        "producer": {
            "phase": "7",
            "module": "decision_engine",
            "experiment_id": experiment_id,
            "config_fingerprint": config_fingerprint,
            "rule_version": RULE_VERSION,
            "git_commit": git_commit,
        },
        "upstream_lineage": {
            "phase6_experiment_id": contract_validation.experiment_id,
            "phase6_config_fingerprint": contract_validation.phase6_config_fingerprint,
            "phase6_model_type": contract_validation.model_type,
            "phase6_model_fingerprint": contract_validation.phase6_model_fingerprint,
            "phase6_n2_caveat": contract_validation.n2_caveat,
        },
        "decision_schema": {
            "grain": ["bearing_id", "timestamp"],
            "risk_categories": ["CRITICAL", "HIGH", "MEDIUM", "LOW", "HEALTHY"],
            "maintenance_actions": [
                "IMMEDIATE_SHUTDOWN",
                "URGENT_MAINTENANCE",
                "SCHEDULE_MAINTENANCE",
                "ROUTINE_MONITORING",
                "NORMAL_OPERATION",
            ],
            "confidence_levels": ["HIGH", "MEDIUM", "LOW"],
            "priority_range": [1, 5],
            "fields": [
                "bearing_id",
                "timestamp",
                "min_predicted_rul_minutes",
                "mean_predicted_rul_minutes",
                "channel_count",
                "risk_category",
                "maintenance_action",
                "confidence_level",
                "priority_score",
                "is_business_rule_override",
                "override_reason",
                "explanation_text",
                "model_type",
                "n2_caveat",
                "rule_version",
                "config_fingerprint",
                "experiment_id",
            ],
        },
        # §21c: explicit enumeration of all transferred artifacts
        "artifact_paths": {
            "recommendations": output_paths.get("recommendations", "reports/recommendations.parquet"),
            "decision_report": output_paths.get("decision_report", "reports/decision_report.md"),
            "decision_manifest": output_paths.get("decision_manifest", "reports/decision_manifest.json"),
            "decision_validation_report": output_paths.get(
                "decision_validation_report", "reports/decision_validation_report.md"
            ),
        },
        "performance_context": {
            "phase6_mean_mae_minutes": contract_validation.mean_mae_minutes,
            "phase6_stability_ratio": contract_validation.stability_ratio,
            "n2_caveat": contract_validation.n2_caveat,
            "n2_caveat_text": contract_validation.n2_caveat_text,
            "model_type": contract_validation.model_type,
            "model_selection_outcome": (
                "demotion_fallback"
                if contract_validation.model_type in ("mean", "median")
                else "promoted"
            ),
        },
        "decision_summary": {
            "total_recommendations": len(recommendations_df),
            "bearing_count": recommendations_df["bearing_id"].nunique()
            if not recommendations_df.empty
            else 0,
            "risk_category_counts": risk_counts,
        },
        "statistical_limitations": {
            "n_rtf_units": contract_validation.n_rtf_units,
            "confidence_type": "categorical_only",
            "probability_estimates": False,
            "extrapolation_permitted": False,
        },
        # §21a consumer_expectations — all 8 declared expectations
        "consumer_expectations": {
            "must_honour_n2_caveat": True,
            "must_display_confidence_level_with_every_recommendation": True,
            "must_display_model_type_and_selection_outcome": True,
            "must_not_aggregate_across_rule_versions": True,
            "must_not_compare_experiments_with_different_config_fingerprints": True,
            "must_honour_business_rule_override_flags": True,
            "must_not_use_shap_outputs_for_model_modification": True,
            "apply_domain_expert_validated_thresholds_before_operational_use": True,
        },
    }

    logger.info(
        "Phase 7→8 contract built: experiment_id=%s, n_recommendations=%d",
        experiment_id,
        len(recommendations_df),
    )
    return contract


def save_phase8_contract(
    contract: Dict[str, Any], output_path: Path, logger: logging.Logger
) -> None:
    """Atomically write phase7_decision_contract.yaml. Architecture §21.

    Args:
        contract: Contract payload dict.
        output_path: Target file path.
        logger: Pipeline logger.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".yaml.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            yaml.dump(contract, f, default_flow_style=False, allow_unicode=True, sort_keys=True)
        tmp_path.replace(output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    logger.info("Phase 7→8 contract saved: %s", output_path)
