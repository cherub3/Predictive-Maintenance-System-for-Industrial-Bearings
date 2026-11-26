"""Phase 6 → Phase 7 contract generator (architecture §17).

Produces `phase6_model_contract.yaml` — the only machine-readable interface
Phase 7 may consume. Named `contract_generator.py` to match the sibling
naming convention established in `src/rul_target/contract_generator.py`
(the architecture document calls this module `phase7_contract`; this is a
file-naming convention match only, not a responsibility change — see
Implementation Notes in PHASE_6_IMPLEMENTATION_REPORT.md).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from src.config import get_dataset_for_unit
from src.model_dev.evaluation import EvaluationResult
from src.model_dev.feature_importance import FeatureImportanceResult


def build_phase7_contract(
    experiment_id: str,
    config_fingerprint: str,
    git_commit: str,
    selected_model: str,
    registry_info: Dict[str, Any],
    feature_columns: List[str],
    evaluation_result: EvaluationResult,
    importance_result: FeatureImportanceResult,
    phase5_contract: Dict[str, Any],
    config: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Build the complete Phase 6 → Phase 7 contract payload (architecture §17).

    Args:
        experiment_id: Governance experiment ID.
        config_fingerprint: SHA-256 of the model_dev config.
        git_commit: Short git commit SHA.
        selected_model: The model chosen by model_selection.
        registry_info: Output of `model_registry.register_model`.
        feature_columns: Feature column names (from the Phase 5 contract).
        evaluation_result: Full evaluation results.
        importance_result: Feature importance results.
        phase5_contract: Parsed Phase 5 contract (for upstream lineage).
        config: Full project configuration dictionary.
        logger: Pipeline logger.

    Returns:
        Dict representing the phase6_model_contract.yaml payload.
    """
    model_dev_cfg = config.get("model_dev", {})
    selected_eval = evaluation_result.model_evaluations[selected_model]

    # Derived from the Phase 5 contract rather than hardcoded — the RTF unit
    # population, grain, and n_rtf_units all vary with the datasets registry
    # (e.g. n=2 with Set 1 only, n=4 once Set 2/3 are included).
    rtf_units: List[str] = phase5_contract.get("artifacts", {}).get("training_dataset", {}).get("units", [])
    grain: List[str] = phase5_contract.get("columns", {}).get("grain", ["unit_id", "channel", "timestamp"])
    n_rtf_units: int = phase5_contract.get("statistical_limitations", {}).get("n_rtf_units", len(rtf_units))
    n2_caveat_text: str = phase5_contract.get("statistical_limitations", {}).get(
        "caveat",
        f"With only {n_rtf_units} RTF unit(s), LOBO CV produces {n_rtf_units} evaluation "
        "data point(s). Performance estimates cannot carry classical confidence intervals. "
        "Results are methodology validation, not fleet-level guarantees.",
    )
    failure_modes = sorted({
        get_dataset_for_unit(config, unit_id).get("failure_mode")
        for unit_id in rtf_units
        if get_dataset_for_unit(config, unit_id).get("failure_mode")
    })

    per_fold_performance = {
        fold_id: {
            "test_bearing": fm.test_bearing,
            "mae": fm.metrics.mae,
            "rmse": fm.metrics.rmse,
            "r2": fm.metrics.r2,
            "near_failure_mae": fm.metrics.near_failure_mae,
            "signed_error": fm.metrics.signed_error,
            "late_prediction_rate": fm.metrics.late_prediction_rate,
        }
        for fold_id, fm in selected_eval.fold_metrics.items()
    }

    mean_signed_error = float(np.mean([fm.metrics.signed_error for fm in selected_eval.fold_metrics.values()]))
    bias_direction = "over_predicts_late_warning" if mean_signed_error > 0 else "under_predicts_conservative"

    top_features_perm = []
    perm_df = importance_result.permutation_importance.get(selected_model)
    if perm_df is not None and not perm_df.empty:
        top_features_perm = perm_df.head(10)["feature"].tolist()

    contract = {
        "schema_version": "1.0",
        "producer": {
            "phase": "6",
            "module": "model_dev",
            "git_commit": git_commit,
            "config_fingerprint": config_fingerprint,
        },
        "experiment": {
            "experiment_id": experiment_id,
            "random_seed": model_dev_cfg.get("random_seed", 42),
            "config_fingerprint": config_fingerprint,
        },
        "model": {
            "type": selected_model,
            "path": registry_info["refit_model_path"],
            "fold_models_path": registry_info["model_path"],
            "model_fingerprint": registry_info["refit_model_fingerprint"],
            "training_population": {"rtf_units": rtf_units},
            "preprocessing": {
                "scaler": model_dev_cfg.get("preprocessing", {}).get("scaler", "standard"),
                "imputer": model_dev_cfg.get("preprocessing", {}).get("imputer", "median"),
                "fit_scope": "train_fold_only",
            },
        },
        "prediction_schema": {
            "input_features": feature_columns,
            "output": "predicted_rul_minutes",
            "output_dtype": "float",
            "output_constraint": "non_negative",
            "grain": grain,
            "units": "minutes",
        },
        "performance": {
            "mean_mae_minutes": selected_eval.mean_mae,
            "mae_spread_minutes": selected_eval.mae_spread,
            "stability_ratio": selected_eval.stability_ratio,
            "per_fold": per_fold_performance,
            "n2_caveat": True,
            "n2_caveat_text": n2_caveat_text,
        },
        "feature_importance": {
            "top_permutation_ranked_features": top_features_perm,
        },
        "error_analysis_summary": {
            "bias_direction": bias_direction,
            "mean_signed_error_minutes": mean_signed_error,
        },
        "consumer_expectations": {
            "feed_features_in_exact_contract_schema": True,
            "must_not_retrain_or_retune": True,
            "treat_performance_as_n2_estimate_not_fleet_sla": True,
            "apply_decision_thresholds_on_top_of_raw_predictions": True,
            "honour_late_prediction_risk_direction": True,
            "must_not_use_shap_outputs_for_model_modification": True,
        },
        "limitations": {
            "n_rtf_units": n_rtf_units,
            "operating_regime": "constant",
            "failure_modes": failure_modes,
            "extrapolation_permitted": False,
        },
        "versioning": {
            "model_version": "1.0",
            "experiment_id": experiment_id,
            "upstream_dataset_version": phase5_contract.get("dataset_version"),
            "upstream_contract_schema_version": phase5_contract.get("contract_version"),
        },
    }
    logger.info("Phase 6 contract built: selected_model=%s, mean_mae=%.2f", selected_model, selected_eval.mean_mae)
    return contract


def save_phase7_contract(contract: Dict[str, Any], output_path: Path, logger: logging.Logger) -> None:
    """Atomically write `phase6_model_contract.yaml`."""
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
    logger.info("Phase 6 contract saved: %s", output_path)
