"""Phase 5 contract loader and entry-gate validator.

Loads `reports/phase5_dataset_contract.yaml` — the single authoritative,
machine-readable interface Phase 5 hands to Phase 6 — and validates it
against the actual `training_dataset.parquet` / `fold_assignments.parquet`
on disk. Phase 6 must fail closed if the declared interface does not match
reality (architecture §4, §19).

Feature columns, label columns, fold-role columns, and grain columns are
read from this contract at runtime and never re-declared in Phase 6 config,
preventing drift between phases (architecture §18).
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml


@dataclass
class ContractValidationResult:
    """Outcome of validating the Phase 5 contract against the actual dataset."""

    passed: bool
    fatals: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)


def load_phase5_contract(config: Dict[str, Any], project_root: Path, logger: logging.Logger) -> Dict[str, Any]:
    """Load `phase5_dataset_contract.yaml`.

    Args:
        config: Full project configuration dictionary.
        project_root: Absolute path to the project root.
        logger: Pipeline logger.

    Returns:
        Parsed contract dictionary.

    Raises:
        FileNotFoundError: If the contract file does not exist.
    """
    model_dev_cfg = config.get("model_dev", {})
    input_paths = model_dev_cfg.get("input_paths", {})
    contract_path = project_root / input_paths.get(
        "phase5_contract", "reports/phase5_dataset_contract.yaml"
    )

    if not contract_path.exists():
        raise FileNotFoundError(
            f"Phase 5 dataset contract not found: {contract_path}. "
            "Run Phase 5 (RUL target construction) first."
        )

    with contract_path.open("r", encoding="utf-8") as f:
        contract = yaml.safe_load(f)

    logger.info("Phase 5 contract loaded: %s (schema version %s)", contract_path, contract.get("contract_version"))
    return contract


def get_feature_columns(contract: Dict[str, Any]) -> List[str]:
    """Return the authoritative feature column list declared by the Phase 5 contract."""
    return list(contract.get("columns", {}).get("features", []))


def get_label_columns(contract: Dict[str, Any]) -> List[str]:
    """Return the label column list declared by the Phase 5 contract."""
    return list(contract.get("columns", {}).get("labels", []))


def get_fold_role_columns(contract: Dict[str, Any]) -> List[str]:
    """Return the fold-role column list declared by the Phase 5 contract."""
    return list(contract.get("columns", {}).get("fold_roles", []))


def get_grain_columns(contract: Dict[str, Any]) -> List[str]:
    """Return the grain (key) column list declared by the Phase 5 contract."""
    return list(contract.get("columns", {}).get("grain", []))


def get_metadata_columns(contract: Dict[str, Any]) -> List[str]:
    """Return the metadata column list declared by the Phase 5 contract (e.g. is_censored)."""
    return list(contract.get("columns", {}).get("metadata", []))


def get_rtf_units(contract: Dict[str, Any]) -> List[str]:
    """Return the RTF (run-to-failure) unit list from the Phase 5 contract."""
    return list(contract.get("artifacts", {}).get("training_dataset", {}).get("units", []))


def get_folds(contract: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    """Return the fold → {train_units, test_units} mapping declared by the contract."""
    return dict(contract.get("artifacts", {}).get("fold_assignments", {}).get("folds", {}))


def validate_phase5_contract(
    contract: Dict[str, Any],
    training_dataset: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    logger: logging.Logger,
) -> ContractValidationResult:
    """Fail-closed validation of the Phase 5 contract against the actual dataset on disk.

    Known, documented exception: the frozen Phase 5 contract (v1.0, produced by the
    Phase 5 implementation fix-set) declares `is_censored` under `columns.metadata`,
    but the physical `training_dataset.parquet` (RTF-only by construction) does not
    carry this column — it was dropped in `dataset_builder.py` because it is always
    False for RTF rows. This is a pre-existing artifact of the frozen (unmodifiable)
    Phase 5 deliverable, not a Phase 6 defect. It is treated as a WARNING, not FATAL,
    and is documented in the Phase 6 Implementation Notes.

    Args:
        contract: Parsed Phase 5 contract.
        training_dataset: Loaded `training_dataset.parquet`.
        fold_assignments: Loaded `fold_assignments.parquet`.
        logger: Pipeline logger.

    Returns:
        ContractValidationResult with fatals/warnings/info.
    """
    result = ContractValidationResult(passed=True)

    if "contract_version" not in contract:
        result.fatals.append("Contract missing 'contract_version' field")
        result.passed = False

    dataset_cols = set(training_dataset.columns)

    # Grain columns — FATAL if missing (structural join key)
    grain_cols = get_grain_columns(contract)
    missing_grain = [c for c in grain_cols if c not in dataset_cols]
    if missing_grain:
        result.fatals.append(f"Grain columns declared but missing from training_dataset: {missing_grain}")
        result.passed = False
    else:
        result.info.append(f"Grain columns present: {grain_cols} ✓")

    # Feature columns — FATAL if any declared feature is missing (the model cannot train)
    feature_cols = get_feature_columns(contract)
    missing_features = [c for c in feature_cols if c not in dataset_cols]
    if missing_features:
        result.fatals.append(f"Feature columns declared but missing from training_dataset: {missing_features}")
        result.passed = False
    else:
        result.info.append(f"All {len(feature_cols)} declared feature columns present ✓")

    # Label columns — FATAL if the primary label (rul_minutes) is missing.
    # rul_cycles is a declared secondary label; its absence is a WARNING only.
    label_cols = get_label_columns(contract)
    if "rul_minutes" in label_cols and "rul_minutes" not in dataset_cols:
        result.fatals.append("Primary label 'rul_minutes' declared but missing from training_dataset")
        result.passed = False
    missing_secondary_labels = [c for c in label_cols if c != "rul_minutes" and c not in dataset_cols]
    if missing_secondary_labels:
        result.warnings.append(f"Secondary label columns declared but missing: {missing_secondary_labels}")
    if "rul_minutes" in dataset_cols:
        result.info.append("Primary label 'rul_minutes' present ✓")

    # Fold-role columns — FATAL if missing (cannot run LOBO without them)
    fold_role_cols = get_fold_role_columns(contract)
    missing_fold_roles = [c for c in fold_role_cols if c not in dataset_cols]
    if missing_fold_roles:
        result.fatals.append(f"Fold-role columns declared but missing from training_dataset: {missing_fold_roles}")
        result.passed = False
    else:
        result.info.append(f"Fold-role columns present: {fold_role_cols} ✓")

    # Metadata columns — documented exception (see docstring): WARNING, not FATAL.
    metadata_cols = get_metadata_columns(contract)
    missing_metadata = [c for c in metadata_cols if c not in dataset_cols]
    if missing_metadata:
        result.warnings.append(
            f"Metadata columns declared in contract but absent from training_dataset: {missing_metadata}. "
            "KNOWN ISSUE: is_censored is documented in the Phase 5 contract's columns.metadata "
            "section (added by the Phase 5 fix-set) but was never added to dataset_builder.py's "
            "output columns (training_dataset.parquet is RTF-only, so is_censored is always False "
            "and was intentionally dropped from the physical file). Phase 5 is frozen; this gap "
            "cannot be corrected here. Documented in PHASE_6_IMPLEMENTATION_REPORT.md."
        )

    # Row count sanity check — FATAL if actual rows differ from contract's declared count
    declared_rows = contract.get("artifacts", {}).get("training_dataset", {}).get("total_rows")
    if declared_rows is not None and len(training_dataset) != declared_rows:
        result.fatals.append(
            f"Row count mismatch: contract declares {declared_rows}, "
            f"actual training_dataset has {len(training_dataset)} rows"
        )
        result.passed = False
    else:
        result.info.append(f"Row count matches contract: {len(training_dataset)} ✓")

    # Fold assignments must cover the same fold-role columns
    for col in fold_role_cols:
        if col not in fold_assignments.columns:
            result.fatals.append(f"fold_assignments.parquet missing declared fold-role column: {col}")
            result.passed = False

    severity = "PASSED" if result.passed else "FAILED"
    logger.info("Phase 5 contract validation: %s", severity)
    for fatal in result.fatals:
        logger.error("  FATAL: %s", fatal)
    for warning in result.warnings:
        logger.warning("  WARNING: %s", warning)
    for info in result.info:
        logger.info("  INFO: %s", info)

    return result
