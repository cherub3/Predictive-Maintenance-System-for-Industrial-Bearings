"""Phase 5: RUL Target Construction & Leakage Prevention — Pipeline Orchestrator.

Execution order (architecture §11):
  temporal_validator → failure_indexer → label_generator → split_generator
  → leakage_validator (GATE) → dataset_builder
  → label_dictionary / dataset_manifest / contract_generator
  → validation → implementation_report
"""

import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.config import get_censored_units, get_rtf_units, load_config, make_unit_id
from src.constants import (
    DEFAULT_CONFIG_PATH,
    EXIT_ERROR_CONFIG,
    EXIT_ERROR_RUL_TARGET,
    EXIT_SUCCESS,
    PROJECT_ROOT,
)
from src.logger import setup_logger
from src.rul_target.contract_generator import build_phase6_contract, save_phase6_contract
from src.rul_target.dataset_builder import build_training_dataset, save_training_dataset
from src.rul_target.dataset_manifest import build_dataset_manifest, save_dataset_manifest
from src.rul_target.failure_indexer import build_failure_index
from src.rul_target.label_dictionary import build_label_dictionary, save_label_dictionary
from src.rul_target.label_generator import generate_rul_labels, save_rul_labels
from src.rul_target.leakage_validator import (
    generate_leakage_report,
    run_leakage_checks,
    save_leakage_report,
)
from src.rul_target.split_generator import (
    generate_fold_assignments,
    generate_split_summary,
    save_fold_assignments,
    save_split_summary,
)
from src.rul_target.temporal_validator import validate_temporal_integrity
from src.rul_target.validation import (
    FinalValidationResult,
    generate_validation_report,
    run_final_validation,
    save_validation_report,
)


def _compute_config_fingerprint(config: Dict[str, Any]) -> str:
    """SHA-256 of the serialized config — reuses Phase 3/4 pattern."""
    serialized = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _resolve_output_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """Resolve all Phase 5 output paths from config."""
    rul_cfg = config.get("rul_target", {})
    path_cfg = rul_cfg.get("output_paths", {})

    defaults = {
        "rul_labels": "data/processed/rul_labels.parquet",
        "training_dataset": "data/processed/training_dataset.parquet",
        "fold_assignments": "data/processed/splits/fold_assignments.parquet",
        "dataset_manifest": "reports/dataset_manifest.json",
        "label_dictionary": "reports/label_dictionary.json",
        "dataset_contract": "reports/phase5_dataset_contract.yaml",
        "leakage_report": "reports/leakage_report.md",
        "split_summary": "reports/split_summary.md",
        "validation_report": "reports/validation_report.md",
        "implementation_report": "reports/PHASE_5_IMPLEMENTATION_REPORT.md",
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
    config: Dict[str, Any],
    validation_result: FinalValidationResult,
    logger: logging.Logger,
) -> None:
    """Write PHASE_5_IMPLEMENTATION_REPORT.md."""
    rul_cfg = config.get("rul_target", {})
    rtf_units = get_rtf_units(config)
    censored_units = get_censored_units(config)
    elapsed = time.time() - start_time
    status = "COMPLETE ✓" if validation_result.all_passed else "FAILED ✗"

    lines = [
        "# Phase 5 — Implementation Report",
        "",
        f"**Status:** {status}",
        f"**Elapsed time:** {elapsed:.1f}s",
        f"**Dataset version:** {rul_cfg.get('dataset_version', '1.0')}",
        f"**Label version:** {rul_cfg.get('label_version', '1.0')}",
        "",
        "---",
        "",
        "## Artifacts Produced",
        "",
    ]

    for key, path in output_paths.items():
        if key == "implementation_report":
            # This report is still being written; it cannot report on itself.
            continue
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        symbol = "✓" if exists else "✗"
        lines.append(f"- [{symbol}] `{path.name}` — {path} ({size:,} bytes)")

    lines += [
        "",
        "---",
        "",
        "## Pipeline Summary",
        "",
        "| Step | Module | Description |",
        "|------|--------|-------------|",
        "| 1 | temporal_validator | Validates timestamp integrity of Phase 4 feature store |",
        "| 2 | failure_indexer | Computes t_fail = max(timestamp) per RTF unit |",
        "| 3 | label_generator | Generates RUL labels (minutes, optional cycles/frac) |",
        f"| 4 | split_generator | LOBO CV fold assignments — one fold per RTF unit "
        f"({len(rtf_units)} folds for {len(rtf_units)} RTF units: {', '.join(rtf_units)}) |",
        "| 5 | leakage_validator | GATE: L-01 through L-09 checks; halts on any FATAL |",
        "| 6 | dataset_builder | Joins features + labels + fold assignments |",
        "| 7 | label_dictionary | Emits label schema JSON |",
        "| 8 | dataset_manifest | Emits provenance + fingerprint JSON |",
        "| 9 | contract_generator | Emits Phase 5 → Phase 6 YAML contract |",
        "| 10 | validation | Aggregate final artifact validation |",
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
        f"- Only {len(rtf_units)} RTF units exist ({', '.join(rtf_units)}). LOBO CV therefore",
        f"  produces exactly {len(rtf_units)} folds / {len(rtf_units)} evaluation data points.",
        "  No classical confidence interval applies to any downstream performance metric",
        "  (see split_summary.md).",
        f"- Censored units ({', '.join(censored_units)}) carry no ground-truth RUL and are",
        "  excluded from the training dataset and all fold assignments by design.",
        "",
        "## Implementation Notes",
        "",
        "- The frozen architecture defines split artifacts as `fold_assignments.parquet`",
        "  (LOBO CV fold-role columns) plus `split_summary.md`, not separate",
        "  `train_split.parquet` / `validation_split.parquet` / `test_split.parquet` files.",
        "  This implementation follows the frozen architecture's artifact list exactly;",
        "  no additional split files were introduced.",
        "- `config.rul_target.split.val_from_train_early_fraction` is present in config",
        "  but unused by this implementation — the frozen architecture does not carve a",
        "  validation set out of the training fold for Phase 5. Any such carve-out is a",
        "  Phase 6 concern (e.g. early-stopping validation slices within a train fold).",
        "",
        "---",
        "_Generated by Phase 5 pipeline. Do not edit manually._",
        "_Phase 6 may begin only after a formal Phase 5 code review._",
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


def run_rul_target_pipeline() -> int:
    """Execute the complete Phase 5 RUL target construction pipeline."""
    start_time = time.time()

    try:
        config = load_config(DEFAULT_CONFIG_PATH)
    except Exception as exc:
        print(f"Configuration load failed: {exc}")
        return EXIT_ERROR_CONFIG

    logger = setup_logger(name="rul_target", config=config)
    logger.info("=" * 80)
    logger.info("PHASE 5: RUL TARGET CONSTRUCTION & LEAKAGE PREVENTION")
    logger.info("=" * 80)

    output_paths = _resolve_output_paths(config)
    config_fingerprint = _compute_config_fingerprint(config)
    logger.info("Config fingerprint: %s", config_fingerprint)

    try:
        # ──────────────────────────────────────────────────────────────
        # Step 1 — Load Phase 4 feature store
        # ──────────────────────────────────────────────────────────────
        logger.info("Step 1: Loading Phase 4 feature store")
        feature_store_path = PROJECT_ROOT / config["paths"]["processed_data_dir"] / "feature_store.parquet"
        if not feature_store_path.exists():
            logger.error("Feature store not found: %s", feature_store_path)
            logger.error("Run Phase 4 (feature engineering) first.")
            return EXIT_ERROR_RUL_TARGET

        feature_store = pd.read_parquet(feature_store_path)
        feature_store["timestamp"] = pd.to_datetime(feature_store["timestamp"])

        # unit_id is the qualified, globally-unique bearing identifier
        # (f"{dataset_id}::{bearing_id}") used as the grain key throughout Phase 5.
        # bearing_id alone is NOT unique across datasets (e.g. "bearing_3" exists in
        # both set1 and set3, referring to physically unrelated bearings) — every
        # leakage-critical grouping/filtering/disjointness check in this phase must
        # key off unit_id, not bearing_id, or it will silently conflate unrelated
        # bearings that happen to share a name.
        feature_store["unit_id"] = feature_store["dataset_id"] + "::" + feature_store["bearing_id"]

        logger.info("  Feature store loaded: %d rows, %d columns", len(feature_store), len(feature_store.columns))

        # ──────────────────────────────────────────────────────────────
        # Step 2 — Temporal integrity validation
        # ──────────────────────────────────────────────────────────────
        logger.info("Step 2: Temporal integrity validation")
        temporal_result = validate_temporal_integrity(feature_store, config, logger)
        if not temporal_result.passed:
            logger.error("Temporal validation FAILED — aborting Phase 5")
            return EXIT_ERROR_RUL_TARGET

        # ──────────────────────────────────────────────────────────────
        # Step 3 — Build failure index
        # ──────────────────────────────────────────────────────────────
        logger.info("Step 3: Building failure index (t_fail per RTF unit)")
        failure_index = build_failure_index(feature_store, config, logger)

        # ──────────────────────────────────────────────────────────────
        # Step 4 — Generate RUL labels
        # ──────────────────────────────────────────────────────────────
        logger.info("Step 4: Generating RUL labels")
        rul_labels = generate_rul_labels(feature_store, failure_index, config, logger)
        save_rul_labels(rul_labels, output_paths["rul_labels"], logger)

        # ──────────────────────────────────────────────────────────────
        # Step 5 — Generate fold assignments (LOBO CV)
        # ──────────────────────────────────────────────────────────────
        logger.info("Step 5: Generating LOBO fold assignments")
        fold_assignments = generate_fold_assignments(rul_labels, config, logger)
        save_fold_assignments(fold_assignments, output_paths["fold_assignments"], logger)
        split_summary_md = generate_split_summary(fold_assignments, config, logger)
        save_split_summary(split_summary_md, output_paths["split_summary"], logger)

        # ──────────────────────────────────────────────────────────────
        # Step 6 — LEAKAGE VALIDATION GATE (must pass before dataset_builder)
        # ──────────────────────────────────────────────────────────────
        logger.info("Step 6: Leakage validation GATE (L-01 through L-09)")
        leakage_report = run_leakage_checks(
            feature_store, rul_labels, fold_assignments, config, logger
        )
        leakage_report_md = generate_leakage_report(leakage_report, config, logger)
        save_leakage_report(leakage_report_md, output_paths["leakage_report"], logger)

        if not leakage_report.all_passed:
            logger.error(
                "LEAKAGE GATE FAILED: %d FATAL check(s). "
                "dataset_builder will NOT run. See leakage_report.md",
                leakage_report.fatal_count,
            )
            return EXIT_ERROR_RUL_TARGET

        logger.info("Leakage gate PASSED — proceeding to dataset construction")

        # ──────────────────────────────────────────────────────────────
        # Step 7 — Build training dataset
        # ──────────────────────────────────────────────────────────────
        logger.info("Step 7: Building training dataset (feature + label join)")
        training_dataset = build_training_dataset(
            feature_store, rul_labels, fold_assignments, config, logger
        )
        save_training_dataset(training_dataset, output_paths["training_dataset"], logger)

        # ──────────────────────────────────────────────────────────────
        # Step 8 — Label dictionary, dataset manifest, Phase 6 contract
        # ──────────────────────────────────────────────────────────────
        logger.info("Step 8: Generating label dictionary")
        label_dict = build_label_dictionary(rul_labels, config, logger)
        save_label_dictionary(label_dict, output_paths["label_dictionary"], logger)

        logger.info("Step 9: Generating dataset manifest")
        manifest = build_dataset_manifest(
            training_dataset,
            rul_labels,
            fold_assignments,
            config,
            output_paths,
            config_fingerprint,
            logger,
        )
        save_dataset_manifest(manifest, output_paths["dataset_manifest"], logger)

        logger.info("Step 10: Generating Phase 5 → Phase 6 contract")
        contract = build_phase6_contract(
            training_dataset,
            rul_labels,
            fold_assignments,
            config,
            config_fingerprint,
            logger,
        )
        save_phase6_contract(contract, output_paths["dataset_contract"], logger)

        # ──────────────────────────────────────────────────────────────
        # Step 11 — Final validation
        # ──────────────────────────────────────────────────────────────
        logger.info("Step 11: Final artifact validation")
        validation_result = run_final_validation(
            training_dataset,
            rul_labels,
            fold_assignments,
            config,
            output_paths,
            logger,
        )
        validation_report_md = generate_validation_report(validation_result, config, logger)
        save_validation_report(validation_report_md, output_paths["validation_report"], logger)

        # ──────────────────────────────────────────────────────────────
        # Step 12 — Implementation report
        # ──────────────────────────────────────────────────────────────
        logger.info("Step 12: Writing implementation report")
        _generate_implementation_report(
            start_time, output_paths, config, validation_result, logger
        )

        elapsed = time.time() - start_time
        if not validation_result.all_passed:
            logger.error(
                "Phase 5 FAILED: %d fatal validation error(s). "
                "Elapsed: %.1fs",
                validation_result.fatal_count,
                elapsed,
            )
            return EXIT_ERROR_RUL_TARGET

        logger.info("=" * 80)
        logger.info("PHASE 5 COMPLETE — %.1fs", elapsed)
        logger.info("All artifacts written. Await Phase 5 formal code review.")
        logger.info("Do NOT begin Phase 6 until authorized.")
        logger.info("=" * 80)
        return EXIT_SUCCESS

    except Exception as exc:
        logger.exception("Unhandled exception in Phase 5 pipeline: %s", exc)
        return EXIT_ERROR_RUL_TARGET


if __name__ == "__main__":
    sys.exit(run_rul_target_pipeline())
