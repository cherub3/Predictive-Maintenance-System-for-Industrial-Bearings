"""Phase 7: Maintenance Decision Engine — Pipeline Orchestrator.

Execution order (architecture §7.14):
  Step  1 — Load project configuration
  Step  2 — Compute config fingerprint; generate experiment ID
  Step  3 — Load and validate Phase 6 contract (entry gate)
  Step  4 — Load Phase 6 predictions
  Step  5 — Load Phase 6 manifest (upstream lineage)
  Step  6 — Validate risk threshold ordering
  Step  7 — Aggregate channels to bearing-level (min + mean RUL)
  Step  8 — Run decision pipeline (risk / policy / confidence / priority)
  Step  9 — Apply business rules
  Step 10 — Generate decision explanations
  Step 11 — Assemble recommendation dataframe
  Step 12 — Run final validation (10 checks)
  Step 13 — Generate and save decision report
  Step 14 — Generate and save Phase 7→8 contract
  Step 15 — Generate and save decision manifest + decision log

Steps 8–10 are integrated within decision_engine.run_decision_pipeline.
Business rules (Step 9) are applied inside the decision pipeline loop.
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict

from src.config import load_config
from src.constants import (
    DEFAULT_CONFIG_PATH,
    EXIT_ERROR_CONFIG,
    EXIT_ERROR_DECISION_ENGINE,
    EXIT_SUCCESS,
    PROJECT_ROOT,
)
from src.logger import setup_logger

from src.decision_engine.business_rules import RULE_VERSION
from src.decision_engine.contract_generator import build_phase8_contract, save_phase8_contract
from src.decision_engine.contract_loader import (
    load_phase6_contract,
    load_phase6_manifest,
    load_phase6_predictions,
    validate_phase6_contract,
)
from src.decision_engine.decision_engine import aggregate_channels, run_decision_pipeline
from src.decision_engine.decision_manifest import (
    build_artifact_hashes,
    build_decision_manifest,
    capture_environment,
    get_git_commit,
    save_decision_manifest,
)
from src.decision_engine.decision_report import generate_decision_report, save_decision_report
from src.decision_engine.experiment_tracker import (
    build_experiment_record,
    compute_config_fingerprint,
    generate_experiment_id,
    save_experiment_log,
)
from src.decision_engine.maintenance_policy import load_policy_map
from src.decision_engine.priority_calculator import load_priority_matrix
from src.decision_engine.recommendation_engine import recommendations_to_dataframe
from src.decision_engine.risk_classifier import validate_thresholds
from src.decision_engine.validation import (
    generate_decision_validation_report,
    run_decision_validation,
    save_decision_validation_report,
)


def _resolve_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """Resolve all Phase 7 I/O paths from config. Architecture §7.14."""
    de_cfg = config.get("decision_engine", {})
    inp = de_cfg.get("input_paths", {})
    out = de_cfg.get("output_paths", {})

    defaults_in = {
        "phase6_contract": "reports/phase6_model_contract.yaml",
        "phase6_predictions": "reports/model_predictions.parquet",
        "phase6_manifest": "reports/model_manifest.json",
    }
    defaults_out = {
        "recommendations": "reports/recommendations.parquet",
        "decision_report": "reports/decision_report.md",
        "phase8_contract": "reports/phase7_decision_contract.yaml",
        "decision_manifest": "reports/decision_manifest.json",
        "decision_log": "reports/decision_log.json",
        "decision_validation_report": "reports/decision_validation_report.md",
        "implementation_report": "reports/PHASE_7_IMPLEMENTATION_REPORT.md",
    }

    paths: Dict[str, Path] = {}
    for k, default in defaults_in.items():
        paths[k] = PROJECT_ROOT / inp.get(k, default)
    for k, default in defaults_out.items():
        paths[k] = PROJECT_ROOT / out.get(k, default)
    return paths


def run_phase7(config_path: Path = DEFAULT_CONFIG_PATH) -> int:
    """Execute the full Phase 7 Maintenance Decision Engine pipeline.

    Args:
        config_path: Path to project config.yaml.

    Returns:
        Exit code (0 = success, 80 = decision engine error).
    """
    # ── Step 1: Load configuration ───────────────────────────────────────────
    try:
        config = load_config(config_path)
    except Exception as exc:
        print(f"FATAL: Failed to load configuration from '{config_path}': {exc}", file=sys.stderr)
        return EXIT_ERROR_CONFIG

    logger = setup_logger(name="phase7.decision_engine", config=config)
    logger.info("=" * 70)
    logger.info("PHASE 7 — MAINTENANCE DECISION ENGINE")
    logger.info("=" * 70)
    t0 = time.time()

    try:
        paths = _resolve_paths(config)
        de_cfg = config.get("decision_engine", {})

        # ── Step 2: Config fingerprint + experiment ID ───────────────────────
        logger.info("Step 2/15: Computing config fingerprint")
        config_fingerprint = compute_config_fingerprint(config)
        experiment_id = generate_experiment_id(config_fingerprint, RULE_VERSION)
        logger.info("Experiment ID: %s", experiment_id)
        logger.info("Config fingerprint: %s", config_fingerprint)

        # ── Step 3: Load and validate Phase 6 contract (entry gate) ─────────
        logger.info("Step 3/15: Loading and validating Phase 6 contract")
        phase6_contract_raw = load_phase6_contract(paths["phase6_contract"], logger)

        # Load predictions to validate grain columns exist
        # (predictions are formally loaded in Step 4; we need them for DV col-check here)
        _pred_for_entry = load_phase6_predictions(paths["phase6_predictions"], logger)
        contract_validation = validate_phase6_contract(
            phase6_contract_raw, _pred_for_entry, logger
        )
        if not contract_validation.passed:
            logger.error("Phase 6 contract entry gate FAILED — aborting Phase 7")
            return EXIT_ERROR_DECISION_ENGINE

        # ── Step 4: Load Phase 6 predictions ────────────────────────────────
        logger.info("Step 4/15: Phase 6 predictions already loaded (%d rows)", len(_pred_for_entry))
        predictions_df = _pred_for_entry  # reuse from Step 3

        # ── Step 5: Load Phase 6 manifest (upstream lineage) ────────────────
        logger.info("Step 5/15: Loading Phase 6 manifest")
        # Loaded for its validation side effect (fails if the upstream manifest is
        # absent/malformed); lineage fields are taken from the Phase 6 contract.
        load_phase6_manifest(paths["phase6_manifest"], logger)

        # ── Step 6: Validate risk threshold ordering ─────────────────────────
        logger.info("Step 6/15: Validating risk thresholds")
        thresholds = de_cfg.get("risk_thresholds", {})
        validate_thresholds(thresholds, logger)

        # Load policy map and priority matrix (validated at load time)
        policy_map = load_policy_map(config, logger)
        priority_matrix = load_priority_matrix(config, logger)

        # ── Step 7: Channel aggregation ──────────────────────────────────────
        logger.info("Step 7/15: Aggregating channels to bearing-level")
        bearing_df = aggregate_channels(predictions_df, logger)

        # ── Steps 8–10: Decision pipeline (risk / confidence / priority /
        #                business rules / explanations) ─────────────────────
        logger.info("Steps 8-10/15: Running decision pipeline")
        # Use the Phase 6 contract's authoritative bearing list (training_population.rtf_units)
        # so that bearings absent from predictions trigger BR-02 (architecture §7.9a).
        contract_expected_bearings = contract_validation.expected_bearings or sorted(
            predictions_df["unit_id"].unique().tolist()
        )
        recommendations = run_decision_pipeline(
            bearing_df=bearing_df,
            contract_validation=contract_validation,
            thresholds=thresholds,
            policy_map=policy_map,
            priority_matrix=priority_matrix,
            config=config,
            config_fingerprint=config_fingerprint,
            logger=logger,
            expected_bearings=contract_expected_bearings,
        )

        # ── Step 11: Assemble recommendations DataFrame ──────────────────────
        logger.info("Step 11/15: Assembling recommendations DataFrame")
        recommendations_df = recommendations_to_dataframe(recommendations)
        logger.info(
            "Recommendations: %d rows, bearings=%s, risk categories=%s",
            len(recommendations_df),
            sorted(recommendations_df["bearing_id"].unique().tolist()),
            recommendations_df["risk_category"].value_counts().to_dict(),
        )

        # Save recommendations.parquet (atomic write)
        paths["recommendations"].parent.mkdir(parents=True, exist_ok=True)
        tmp_rec = paths["recommendations"].with_suffix(".parquet.tmp")
        recommendations_df.to_parquet(tmp_rec, index=False)
        tmp_rec.replace(paths["recommendations"])
        logger.info("Recommendations saved: %s", paths["recommendations"])

        # ── Step 12: Final validation ────────────────────────────────────────
        logger.info("Step 12/15: Running final validation (11 checks)")

        # Build override counts for manifest
        override_counts: Dict[str, int] = {"BR-01": 0, "BR-02": 0, "BR-03": 0, "BR-04": 0, "BR-05": 0}
        for rec in recommendations:
            if rec.override_reason:
                for rule in ["BR-01", "BR-02", "BR-03", "BR-04", "BR-05"]:
                    if rule in rec.override_reason:
                        override_counts[rule] += 1

        # Build preliminary manifest for DV9
        git_commit = get_git_commit(logger)
        environment = capture_environment(logger)
        risk_counts = recommendations_df["risk_category"].value_counts().to_dict()

        manifest = build_decision_manifest(
            experiment_id=experiment_id,
            config_fingerprint=config_fingerprint,
            rule_version=RULE_VERSION,
            git_commit=git_commit,
            environment=environment,
            phase6_experiment_id=contract_validation.experiment_id,
            phase6_config_fingerprint=contract_validation.phase6_config_fingerprint,
            phase6_model_type=contract_validation.model_type,
            phase6_model_fingerprint=contract_validation.phase6_model_fingerprint,
            n2_caveat=contract_validation.n2_caveat,
            n_recommendations=len(recommendations_df),
            n_bearings=recommendations_df["bearing_id"].nunique(),
            risk_counts=risk_counts,
            override_counts=override_counts,
            logger=logger,
            n_rtf_units=contract_validation.n_rtf_units,
        )

        validation_result = run_decision_validation(
            contract_validation=contract_validation,
            recommendations_df=recommendations_df,
            manifest=manifest,
            expected_bearings=contract_expected_bearings,
            config=config,
            logger=logger,
        )

        val_report = generate_decision_validation_report(validation_result, logger)
        save_decision_validation_report(val_report, paths["decision_validation_report"], logger)

        if not validation_result.all_passed:
            logger.error(
                "Phase 7 validation FAILED (%d fatal check(s)). Aborting.",
                validation_result.fatal_count,
            )
            return EXIT_ERROR_DECISION_ENGINE

        # ── Step 13: Decision report ─────────────────────────────────────────
        logger.info("Step 13/15: Generating decision report")
        decision_report = generate_decision_report(
            recommendations_df=recommendations_df,
            contract_validation=contract_validation,
            experiment_id=experiment_id,
            config_fingerprint=config_fingerprint,
            rule_version=RULE_VERSION,
            config=config,
            logger=logger,
        )
        save_decision_report(decision_report, paths["decision_report"], logger)

        # ── Step 14: Phase 7→8 contract ──────────────────────────────────────
        logger.info("Step 14/15: Generating Phase 7→8 contract")
        phase8_contract = build_phase8_contract(
            recommendations_df=recommendations_df,
            contract_validation=contract_validation,
            manifest=manifest,
            experiment_id=experiment_id,
            config_fingerprint=config_fingerprint,
            git_commit=git_commit,
            config=config,
            logger=logger,
        )
        save_phase8_contract(phase8_contract, paths["phase8_contract"], logger)

        # ── Step 15: Decision manifest + decision log ─────────────────────────
        logger.info("Step 15/15: Saving decision manifest and experiment log")
        # Hash every dashboard-consumed artifact LAST, after all other files are
        # written, so the recorded hashes reflect their final on-disk state and the
        # dashboard can fail closed on any mismatch.
        manifest["artifacts"] = build_artifact_hashes(logger)
        save_decision_manifest(manifest, paths["decision_manifest"], logger)

        experiment_record = build_experiment_record(
            experiment_id=experiment_id,
            config_fingerprint=config_fingerprint,
            rule_version=RULE_VERSION,
            phase6_experiment_id=contract_validation.experiment_id,
            phase6_config_fingerprint=contract_validation.phase6_config_fingerprint,
            n_recommendations=len(recommendations_df),
            risk_counts=risk_counts,
            override_counts=override_counts,
        )
        save_experiment_log(experiment_record, paths["decision_log"], logger)

        # ── Pipeline complete ─────────────────────────────────────────────────
        elapsed = time.time() - t0
        logger.info("=" * 70)
        logger.info("PHASE 7 COMPLETE — %.1f seconds", elapsed)
        logger.info("  Experiment ID:     %s", experiment_id)
        logger.info("  Config fingerprint: %s", config_fingerprint)
        logger.info("  Recommendations:   %d rows", len(recommendations_df))
        logger.info(
            "  Validation:        PASSED (0 fatal, %d warning)",
            validation_result.warning_count,
        )
        logger.info("=" * 70)
        return EXIT_SUCCESS

    except Exception as exc:
        logger.exception("Unhandled exception in Phase 7 pipeline: %s", exc)
        return EXIT_ERROR_DECISION_ENGINE


if __name__ == "__main__":
    sys.exit(run_phase7())
