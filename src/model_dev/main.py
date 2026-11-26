"""Phase 6: Model Development & Evaluation — Pipeline Orchestrator.

Execution order (architecture §13):
  contract_loader (entry gate) → data_loader → cross_validation (LOBO)
  → evaluation → feature_importance → shap_analysis
  → model_selection (promotion + hierarchy) → refit-for-handoff
  → model_registry → prediction_generator
  → experiment_tracker / model_manifest / contract_generator
  → validation → implementation_report
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict

from src.config import load_config
from src.constants import DEFAULT_CONFIG_PATH, EXIT_ERROR_CONFIG, EXIT_ERROR_MODEL_DEV, EXIT_SUCCESS, PROJECT_ROOT
from src.logger import setup_logger

from src.model_dev.contract_generator import build_phase7_contract, save_phase7_contract
from src.model_dev.contract_loader import get_feature_columns, load_phase5_contract, validate_phase5_contract
from src.model_dev.cross_validation import run_lobo_cv
from src.model_dev.data_loader import assert_unit_disjoint, load_fold_assignments, load_training_dataset
from src.model_dev.evaluation import evaluate_cv_results, generate_evaluation_report, save_evaluation_report
from src.model_dev.experiment_tracker import (
    build_experiment_record,
    compute_config_fingerprint,
    generate_experiment_id,
    save_experiment_log,
)
from src.model_dev.feature_importance import (
    compute_feature_importance,
    generate_feature_importance_report,
    save_feature_importance_report,
)
from src.model_dev.model_manifest import build_model_manifest, get_git_commit, save_model_manifest
from src.model_dev.model_registry import register_model
from src.model_dev.model_selection import (
    apply_promotion_criteria,
    generate_model_selection_report,
    refit_best_model,
    save_model_selection_report,
    select_best_model,
)
from src.model_dev.prediction_generator import generate_predictions, save_predictions
from src.model_dev.shap_analysis import compute_all_shap
from src.model_dev.validation import generate_model_validation_report, run_model_validation, save_model_validation_report


def _resolve_output_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """Resolve all Phase 6 output paths from config."""
    model_dev_cfg = config.get("model_dev", {})
    path_cfg = model_dev_cfg.get("output_paths", {})

    defaults = {
        "model_dir": "models",
        "model_predictions": "reports/model_predictions.parquet",
        "model_contract": "reports/phase6_model_contract.yaml",
        "evaluation_report": "reports/evaluation_report.md",
        "model_selection_report": "reports/model_selection_report.md",
        "feature_importance_report": "reports/feature_importance_report.md",
        "experiment_log": "reports/experiment_log.json",
        "model_manifest": "reports/model_manifest.json",
        "model_validation_report": "reports/model_validation_report.md",
        "implementation_report": "reports/PHASE_6_IMPLEMENTATION_REPORT.md",
    }

    paths: Dict[str, Path] = {}
    for key, default in defaults.items():
        raw = path_cfg.get(key, default)
        p = Path(raw)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        paths[key] = p
    return paths


def _generate_implementation_report(
    start_time: float,
    output_paths: Dict[str, Path],
    experiment_id: str,
    selected_model: str,
    evaluation_result,
    selection_result,
    validation_result,
    logger,
) -> None:
    """Write PHASE_6_IMPLEMENTATION_REPORT.md — implementation only, no architecture discussion."""
    elapsed = time.time() - start_time
    status = "COMPLETE ✓" if validation_result.all_passed else "FAILED ✗"

    lines = [
        "# Phase 6 — Implementation Report",
        "",
        f"**Status:** {status}",
        f"**Elapsed time:** {elapsed:.1f}s",
        f"**Experiment ID:** {experiment_id}",
        f"**Selected model:** {selected_model}",
        "",
        "---",
        "",
        "## Modules Implemented",
        "",
        "| Module | Responsibility |",
        "|--------|----------------|",
        "| contract_loader | Load + validate Phase 5 contract; fail-closed on mismatch |",
        "| data_loader | Load training dataset + fold assignments; re-assert unit-disjointness |",
        "| preprocessing | In-fold imputation + scaling (fit on train fold only) |",
        "| baseline_models | Mean, median, linear regression baselines |",
        "| model_trainer | Random Forest, XGBoost candidate training |",
        "| hyperparameter_search | Optional within-train-unit-slice grid search (disabled by default) |",
        "| cross_validation | LOBO orchestration across all enabled models |",
        "| metrics | MAE, RMSE, R², near-failure MAE, signed error, late-prediction rate |",
        "| evaluation | Per-fold + summary metrics, error analysis, evaluation_report.md |",
        "| feature_importance | Permutation + native importance |",
        "| shap_analysis | SHAP global + local (tree models only, interpretation-only) |",
        "| model_selection | Promotion criteria + best-model hierarchy + refit-for-handoff |",
        "| model_registry | Model persistence + fingerprinting |",
        "| prediction_generator | Grain-aligned held-out predictions |",
        "| experiment_tracker | Config fingerprint, experiment ID, experiment_log.json |",
        "| model_manifest | Provenance + environment capture |",
        "| contract_generator | Phase 6 → Phase 7 machine-readable contract |",
        "| validation | Final artifact validation gate |",
        "",
        "---",
        "",
        "## Training Workflow",
        "",
        "1. Load and validate the Phase 5 contract (fail-closed entry gate).",
        "2. Load `training_dataset.parquet` and `fold_assignments.parquet`; re-assert unit-disjointness.",
        "3. For each LOBO fold: fit preprocessing on the training fold only, train every enabled "
        "baseline and candidate model, collect train-fold and test-fold predictions.",
        "4. Compute the full metric bundle (MAE, RMSE, R², near-failure MAE, signed error, "
        "overfitting ratio) per model per fold.",
        "",
        "## Evaluation Workflow",
        "",
        "1. Aggregate per-fold summaries (mean MAE, spread, stability ratio) per model, one fold per RTF unit.",
        "2. Apply the 5-rule promotion gate at every rung of the model progression.",
        "3. Apply the 5-priority best-model selection hierarchy among hard-gate survivors.",
        "4. Refit the selected model type on all RTF data for the deployment artifact.",
        "5. Compute permutation + native + SHAP feature importance for the evaluated models.",
        "",
        "---",
        "",
        "## Experiment Summary",
        "",
    ]

    for model_name in evaluation_result.model_order:
        ev = evaluation_result.model_evaluations[model_name]
        lines.append(f"- **{model_name}**: mean MAE={ev.mean_mae:.2f} min, spread={ev.mae_spread:.2f} min, stability_ratio={ev.stability_ratio:.2f}")

    lines += [
        "",
        f"**Selected model:** `{selection_result.selected_model}` "
        f"(resolved at priority {selection_result.resolved_at_priority}: {selection_result.resolved_by})",
        "",
        "---",
        "",
        "## Artifacts Generated",
        "",
    ]

    for key, path in output_paths.items():
        if key in ("implementation_report", "model_dir"):
            continue
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        symbol = "✓" if exists else "✗"
        lines.append(f"- [{symbol}] `{path.name}` — {path} ({size:,} bytes)")

    lines += [
        "",
        "---",
        "",
        "## Validation Summary",
        "",
        f"**FATAL checks failed:** {validation_result.fatal_count}",
        f"**Warnings:** {validation_result.warning_count}",
        "",
    ]
    for check in validation_result.checks:
        symbol = "✓" if check.passed else "✗"
        lines.append(f"- [{check.severity}] {symbol} {check.name}: {check.detail}")

    lines += [
        "",
        "---",
        "",
        "## Known Limitations",
        "",
        "- Only 2 RTF units exist. LOBO CV produces exactly 2 evaluation data points; no classical "
        "confidence interval applies to any metric in this report.",
        "- Hyperparameter search is disabled by default (architecture §9); the reported model uses "
        "fixed, config-declared hyperparameters.",
        "",
        "## Implementation Notes",
        "",
        "- **Known Phase 5 contract/dataset gap:** `phase5_dataset_contract.yaml` declares "
        "`columns.metadata: [is_censored]`, but the physical `training_dataset.parquet` (RTF-only "
        "by construction) does not carry this column — it is always False for RTF rows and was "
        "dropped in Phase 5's `dataset_builder.py`. Phase 5 is frozen and cannot be corrected here; "
        "`contract_loader.validate_phase5_contract` treats this specific, documented gap as a "
        "WARNING rather than a FATAL, since it does not affect model training (Phase 6 never reads "
        "`is_censored`).",
        "- **Module naming:** the architecture document names the Phase 6→7 contract module "
        "`phase7_contract`; this implementation names the file `contract_generator.py` to match the "
        "sibling naming convention already established in `src/rul_target/contract_generator.py`. "
        "Same responsibility, no behavior change.",
        "- **Artifact list:** artifacts strictly follow the frozen architecture §5 table "
        "(`model.pkl`, `refit_model.pkl`, `model_predictions.parquet`, `phase6_model_contract.yaml`, "
        "`evaluation_report.md`, `model_selection_report.md`, `feature_importance_report.md`, "
        "`experiment_log.json`, `model_manifest.json`, `model_validation_report.md`). Illustrative "
        "artifact names mentioned in the implementation instruction but not present in the frozen "
        "architecture's authoritative artifact table (e.g. `baseline_results.json`, "
        "`cross_validation_results.parquet`, `feature_importance.parquet`, `shap_values.parquet`, "
        "`model_card.md`, `experiment_manifest.json`) were not introduced as separate files — their "
        "underlying data is folded into the artifacts above, per 'DO NOT add new deliverables'.",
        "- `near_failure_threshold_minutes` (1440, i.e. last 24h of life) and the promotion/stability/"
        "overfitting numeric thresholds are Phase 6 implementation defaults operationalizing the "
        "architecture's qualitative design (§7b, §8.3, §10.5), which specified the mechanism but not "
        "exact numeric values. Documented here per 'if any ambiguity exists, document it'.",
        "",
        "---",
        "_Generated by Phase 6 pipeline. Do not edit manually._",
        "_Phase 7 may begin only after a formal Phase 6 code review._",
    ]

    report_md = "\n".join(lines)
    output_path = output_paths["implementation_report"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".md.tmp")
    try:
        tmp_path.write_text(report_md, encoding="utf-8")
        tmp_path.replace(output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    logger.info("Implementation report saved: %s", output_path)


def run_model_dev_pipeline() -> int:
    """Execute the complete Phase 6 Model Development & Evaluation pipeline."""
    start_time = time.time()

    try:
        config = load_config(DEFAULT_CONFIG_PATH)
    except Exception as exc:
        print(f"Configuration load failed: {exc}")
        return EXIT_ERROR_CONFIG

    logger = setup_logger(name="model_dev", config=config)
    logger.info("=" * 80)
    logger.info("PHASE 6: MODEL DEVELOPMENT & EVALUATION")
    logger.info("=" * 80)

    output_paths = _resolve_output_paths(config)
    config_fingerprint = compute_config_fingerprint(config)
    seed = config.get("model_dev", {}).get("random_seed", 42)
    experiment_id = generate_experiment_id(config_fingerprint, seed)
    git_commit = get_git_commit()
    logger.info("Experiment ID: %s | Config fingerprint: %s", experiment_id, config_fingerprint)

    try:
        # ──────────────────────────────────────────────────────────────
        # Step 1 — Load and validate the Phase 5 contract (entry gate)
        # ──────────────────────────────────────────────────────────────
        logger.info("Step 1: Loading Phase 5 contract")
        phase5_contract = load_phase5_contract(config, PROJECT_ROOT, logger)

        logger.info("Step 2: Loading Phase 5 training dataset + fold assignments")
        training_dataset = load_training_dataset(config, PROJECT_ROOT, logger)
        fold_assignments = load_fold_assignments(config, PROJECT_ROOT, logger)

        logger.info("Step 3: Validating Phase 5 contract against actual dataset (fail-closed gate)")
        contract_validation = validate_phase5_contract(phase5_contract, training_dataset, fold_assignments, logger)
        if not contract_validation.passed:
            logger.error("Phase 5 contract validation FAILED — aborting Phase 6")
            return EXIT_ERROR_MODEL_DEV

        feature_columns = get_feature_columns(phase5_contract)
        fold_role_cols = [c for c in training_dataset.columns if c.endswith("_role")]
        assert_unit_disjoint(fold_assignments, fold_role_cols, logger)

        # ──────────────────────────────────────────────────────────────
        # Step 4 — LOBO cross-validation (baselines + candidates)
        # ──────────────────────────────────────────────────────────────
        logger.info("Step 4: Running LOBO cross-validation")
        cv_results = run_lobo_cv(training_dataset, fold_assignments, feature_columns, config, logger)

        # ──────────────────────────────────────────────────────────────
        # Step 5 — Evaluation
        # ──────────────────────────────────────────────────────────────
        logger.info("Step 5: Evaluating cross-validation results")
        evaluation_result = evaluate_cv_results(cv_results, config, logger)
        evaluation_report_md = generate_evaluation_report(evaluation_result, config, logger)
        save_evaluation_report(evaluation_report_md, output_paths["evaluation_report"], logger)

        # ──────────────────────────────────────────────────────────────
        # Step 6 — Feature importance + SHAP (interpretation only)
        # ──────────────────────────────────────────────────────────────
        logger.info("Step 6: Computing feature importance (permutation + native)")
        importance_result = compute_feature_importance(cv_results, training_dataset, config, logger)

        logger.info("Step 7: Computing SHAP (tree models only, interpretation only)")
        shap_results = compute_all_shap(cv_results, training_dataset, config, logger)

        importance_report_md = generate_feature_importance_report(importance_result, shap_results, config, logger)
        save_feature_importance_report(importance_report_md, output_paths["feature_importance_report"], logger)

        # ──────────────────────────────────────────────────────────────
        # Step 8 — Model selection (promotion + hierarchy) — never sees SHAP
        # ──────────────────────────────────────────────────────────────
        logger.info("Step 8: Applying model promotion criteria")
        promotion_ledger = apply_promotion_criteria(evaluation_result, config, logger)

        logger.info("Step 9: Selecting best model (hierarchy)")
        selection_result = select_best_model(evaluation_result, promotion_ledger, config, logger)
        selection_report_md = generate_model_selection_report(evaluation_result, promotion_ledger, selection_result, config, logger)
        save_model_selection_report(selection_report_md, output_paths["model_selection_report"], logger)

        # ──────────────────────────────────────────────────────────────
        # Step 10 — Refit-for-handoff + model registry
        # ──────────────────────────────────────────────────────────────
        logger.info("Step 10: Refitting selected model on all RTF data")
        refit_model, refit_preprocessor = refit_best_model(
            selection_result.selected_model, training_dataset, feature_columns, cv_results, config, logger
        )

        logger.info("Step 11: Registering model artifacts")
        registry_info = register_model(
            selection_result.selected_model, cv_results, refit_model, refit_preprocessor,
            experiment_id, config, PROJECT_ROOT, logger,
        )

        # ──────────────────────────────────────────────────────────────
        # Step 12 — Predictions
        # ──────────────────────────────────────────────────────────────
        logger.info("Step 12: Generating held-out predictions")
        predictions = generate_predictions(cv_results, selection_result.selected_model, logger)
        save_predictions(predictions, output_paths["model_predictions"], logger)

        # ──────────────────────────────────────────────────────────────
        # Step 13 — Experiment tracking, model manifest, Phase 7 contract
        # ──────────────────────────────────────────────────────────────
        logger.info("Step 13: Building experiment record")
        hp_search_summary = {
            model_name: {
                fold_id: {
                    "enabled": fr.hp_record.enabled if fr.hp_record else False,
                    "selected_params": fr.hp_record.selected_params if fr.hp_record else {},
                }
                for fold_id, fr in cv_results.results[model_name].items()
            }
            for model_name in cv_results.model_order if model_name in ("random_forest", "xgboost")
        }
        evaluation_summary = {
            name: {"mean_mae": ev.mean_mae, "mae_spread": ev.mae_spread, "stability_ratio": ev.stability_ratio}
            for name, ev in evaluation_result.model_evaluations.items()
        }
        experiment_record = build_experiment_record(
            experiment_id, config_fingerprint, seed, git_commit,
            phase5_contract.get("dataset_version", "unknown"),
            phase5_contract.get("config_fingerprint", "unknown"),
            evaluation_summary, selection_result.selected_model, hp_search_summary, logger,
        )
        save_experiment_log(experiment_record, output_paths["experiment_log"], logger)

        logger.info("Step 14: Building model manifest")
        manifest = build_model_manifest(
            experiment_id, config_fingerprint, seed, selection_result.selected_model,
            registry_info["model_fingerprint"], registry_info["refit_model_fingerprint"],
            evaluation_result, selection_result, phase5_contract, config, logger,
        )
        save_model_manifest(manifest, output_paths["model_manifest"], logger)

        logger.info("Step 15: Building Phase 6 -> Phase 7 contract")
        phase7_contract = build_phase7_contract(
            experiment_id, config_fingerprint, git_commit, selection_result.selected_model,
            registry_info, feature_columns, evaluation_result, importance_result,
            phase5_contract, config, logger,
        )
        save_phase7_contract(phase7_contract, output_paths["model_contract"], logger)

        # ──────────────────────────────────────────────────────────────
        # Step 16 — Final validation
        # ──────────────────────────────────────────────────────────────
        logger.info("Step 16: Final artifact validation")
        validation_result = run_model_validation(
            contract_validation, cv_results, evaluation_result, selection_result,
            predictions, registry_info, phase7_contract, experiment_record, config, logger,
        )
        validation_report_md = generate_model_validation_report(validation_result, logger)
        save_model_validation_report(validation_report_md, output_paths["model_validation_report"], logger)

        # ──────────────────────────────────────────────────────────────
        # Step 17 — Implementation report
        # ──────────────────────────────────────────────────────────────
        logger.info("Step 17: Writing implementation report")
        _generate_implementation_report(
            start_time, output_paths, experiment_id, selection_result.selected_model,
            evaluation_result, selection_result, validation_result, logger,
        )

        elapsed = time.time() - start_time
        if not validation_result.all_passed:
            logger.error("Phase 6 FAILED: %d fatal validation error(s). Elapsed: %.1fs", validation_result.fatal_count, elapsed)
            return EXIT_ERROR_MODEL_DEV

        logger.info("=" * 80)
        logger.info("PHASE 6 COMPLETE — %.1fs", elapsed)
        logger.info("Selected model: %s", selection_result.selected_model)
        logger.info("All artifacts written. Await Phase 6 formal code review.")
        logger.info("Do NOT begin Phase 7 until authorized.")
        logger.info("=" * 80)
        return EXIT_SUCCESS

    except Exception as exc:
        logger.exception("Unhandled exception in Phase 6 pipeline: %s", exc)
        return EXIT_ERROR_MODEL_DEV


if __name__ == "__main__":
    sys.exit(run_model_dev_pipeline())
