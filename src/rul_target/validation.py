"""Phase 5 final validation — aggregates all rules, generates validation_report.md.

Runs after all artifacts have been produced. Validates:
1. All required output files exist and are non-empty
2. training_dataset row count matches expected (RTF rows only)
3. Label column completeness for RTF units
4. Fold assignment coverage (all RTF rows have fold roles)
5. RUL range bounds: rul_minutes >= 0, no negative values
6. Monotonicity invariant (per unit per channel)
7. Artifact consistency: training_dataset row count == fold_assignments row count

Exits with EXIT_ERROR_RUL_TARGET if any fatal check fails.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.config import get_rtf_units


@dataclass
class ValidationCheck:
    """Result of a single final validation check."""

    name: str
    passed: bool
    severity: str  # "FATAL" | "WARNING" | "INFO"
    detail: str


@dataclass
class FinalValidationResult:
    """Aggregate result of all final validation checks."""

    all_passed: bool
    checks: List[ValidationCheck] = field(default_factory=list)

    @property
    def fatal_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == "FATAL")

    @property
    def warning_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == "WARNING")


def run_final_validation(
    training_dataset: pd.DataFrame,
    rul_labels: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    config: Dict[str, Any],
    output_paths: Dict[str, Path],
    logger: logging.Logger,
) -> FinalValidationResult:
    """Execute all final validation checks.

    Args:
        training_dataset: The joined RTF training dataset.
        rul_labels: Labels DataFrame (all units).
        fold_assignments: Fold assignments DataFrame (RTF units).
        config: Full project configuration dictionary.
        output_paths: Dict mapping artifact names to their resolved Path objects.
        logger: Pipeline logger.

    Returns:
        FinalValidationResult. Caller must exit on fatal failures.
    """
    logger.info("=" * 60)
    logger.info("FINAL VALIDATION — Phase 5 artifacts")
    logger.info("=" * 60)

    rtf_units: List[str] = get_rtf_units(config)

    checks: List[ValidationCheck] = []

    checks.extend(_check_output_files_exist(output_paths, logger))
    checks.extend(_check_row_counts(training_dataset, fold_assignments, rtf_units, logger))
    checks.extend(_check_label_completeness(training_dataset, rtf_units, logger))
    checks.extend(_check_fold_coverage(training_dataset, rtf_units, logger))
    checks.extend(_check_rul_non_negative(training_dataset, logger))
    checks.extend(_check_rul_monotone(training_dataset, logger))

    fatals_fired = [c for c in checks if not c.passed and c.severity == "FATAL"]
    result = FinalValidationResult(
        all_passed=len(fatals_fired) == 0,
        checks=checks,
    )

    status = "PASSED ✓" if result.all_passed else "FAILED ✗"
    logger.info(f"Final validation: {status}")
    for c in checks:
        symbol = "✓" if c.passed else "✗"
        level = logger.info if c.passed else logger.error
        level(f"  [{c.severity}] {symbol} {c.name}: {c.detail}")

    return result


def _check_output_files_exist(
    output_paths: Dict[str, Path], logger: logging.Logger
) -> List[ValidationCheck]:
    """Check all expected output artifacts exist and are non-empty."""
    checks = []
    required = [
        "rul_labels", "training_dataset", "fold_assignments",
        "dataset_manifest", "label_dictionary", "dataset_contract",
        "leakage_report", "split_summary",
    ]
    for key in required:
        path = output_paths.get(key)
        if path is None:
            checks.append(ValidationCheck(
                name=f"artifact_exists:{key}", passed=False, severity="FATAL",
                detail=f"Path not configured for artifact '{key}'"
            ))
        elif not path.exists():
            checks.append(ValidationCheck(
                name=f"artifact_exists:{key}", passed=False, severity="FATAL",
                detail=f"Expected artifact not found: {path}"
            ))
        elif path.stat().st_size == 0:
            checks.append(ValidationCheck(
                name=f"artifact_non_empty:{key}", passed=False, severity="FATAL",
                detail=f"Artifact is empty (0 bytes): {path}"
            ))
        else:
            checks.append(ValidationCheck(
                name=f"artifact_exists:{key}", passed=True, severity="INFO",
                detail=f"{path.name} exists ({path.stat().st_size:,} bytes) ✓"
            ))
    return checks


def _check_row_counts(
    training_dataset: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    rtf_units: List[str],
    logger: logging.Logger,
) -> List[ValidationCheck]:
    """Validate training_dataset and fold_assignments have consistent row counts."""
    checks = []

    td_count = len(training_dataset)
    fa_count = len(fold_assignments)

    if td_count != fa_count:
        checks.append(ValidationCheck(
            name="row_count_consistency",
            passed=False, severity="FATAL",
            detail=(
                f"training_dataset has {td_count} rows but fold_assignments has {fa_count} rows. "
                "These must be equal."
            )
        ))
    else:
        checks.append(ValidationCheck(
            name="row_count_consistency",
            passed=True, severity="INFO",
            detail=f"training_dataset and fold_assignments both have {td_count} rows ✓"
        ))

    # Each RTF unit must be present
    missing_units = [u for u in rtf_units if u not in training_dataset["unit_id"].values]
    if missing_units:
        checks.append(ValidationCheck(
            name="rtf_unit_presence",
            passed=False, severity="FATAL",
            detail=f"RTF units missing from training_dataset: {missing_units}"
        ))
    else:
        checks.append(ValidationCheck(
            name="rtf_unit_presence",
            passed=True, severity="INFO",
            detail=f"All RTF units present: {rtf_units} ✓"
        ))

    return checks


def _check_label_completeness(
    training_dataset: pd.DataFrame,
    rtf_units: List[str],
    logger: logging.Logger,
) -> List[ValidationCheck]:
    """Validate no NaN rul_minutes in training dataset."""
    checks = []
    null_count = training_dataset["rul_minutes"].isna().sum()
    if null_count > 0:
        checks.append(ValidationCheck(
            name="label_completeness",
            passed=False, severity="FATAL",
            detail=f"{null_count} rows have NaN rul_minutes in training_dataset (RTF only expected)"
        ))
    else:
        checks.append(ValidationCheck(
            name="label_completeness",
            passed=True, severity="INFO",
            detail="All rul_minutes values are non-null in training_dataset ✓"
        ))
    return checks


def _check_fold_coverage(
    training_dataset: pd.DataFrame,
    rtf_units: List[str],
    logger: logging.Logger,
) -> List[ValidationCheck]:
    """Validate all rows have fold role assignments for every EXPECTED fold.

    Expected fold count is derived from the number of registered RTF units
    (one LOBO fold per RTF unit) rather than inferred purely from whichever
    fold_N_role columns happen to be present — otherwise a fold column that
    went missing entirely would be invisible to this check.
    """
    checks = []
    expected_fold_ids = [f"fold_{i}" for i in range(1, len(rtf_units) + 1)]
    for fold_id in expected_fold_ids:
        col = f"{fold_id}_role"
        if col not in training_dataset.columns:
            checks.append(ValidationCheck(
                name=f"fold_column_present:{col}",
                passed=False, severity="FATAL",
                detail=f"Column '{col}' missing from training_dataset "
                       f"(expected one fold per RTF unit: {len(rtf_units)} unit(s))"
            ))
            continue
        null_count = training_dataset[col].isna().sum()
        if null_count > 0:
            checks.append(ValidationCheck(
                name=f"fold_assignment_complete:{col}",
                passed=False, severity="FATAL",
                detail=f"{null_count} rows have NaN {col}"
            ))
        else:
            checks.append(ValidationCheck(
                name=f"fold_assignment_complete:{col}",
                passed=True, severity="INFO",
                detail=f"All rows have {col} assignments ✓"
            ))
    return checks


def _check_rul_non_negative(
    training_dataset: pd.DataFrame,
    logger: logging.Logger,
) -> List[ValidationCheck]:
    """Validate rul_minutes is >= 0 for all rows."""
    negative_count = (training_dataset["rul_minutes"] < -1e-9).sum()
    if negative_count > 0:
        return [ValidationCheck(
            name="rul_non_negative",
            passed=False, severity="FATAL",
            detail=f"{negative_count} rows have rul_minutes < 0"
        )]
    return [ValidationCheck(
        name="rul_non_negative",
        passed=True, severity="INFO",
        detail="All rul_minutes >= 0 ✓"
    )]


def _check_rul_monotone(
    training_dataset: pd.DataFrame,
    logger: logging.Logger,
) -> List[ValidationCheck]:
    """Validate RUL is monotonically non-increasing per (unit_id, channel)."""
    violations = []
    for (uid, ch), group in training_dataset.groupby(["unit_id", "channel"]):
        sorted_group = group.sort_values("timestamp")
        diffs = sorted_group["rul_minutes"].diff().dropna()
        n_up = (diffs > 1e-9).sum()
        if n_up > 0:
            violations.append(f"{uid}/ch{ch}: {n_up} increases")

    if violations:
        return [ValidationCheck(
            name="rul_monotone",
            passed=False, severity="FATAL",
            detail=f"RUL increases detected: {'; '.join(violations)}"
        )]
    return [ValidationCheck(
        name="rul_monotone",
        passed=True, severity="INFO",
        detail="RUL monotonically non-increasing across all (unit_id, channel) groups ✓"
    )]


def generate_validation_report(
    result: FinalValidationResult,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> str:
    """Generate validation_report.md documenting all final validation results."""
    status = "PASSED ✓" if result.all_passed else "FAILED ✗"
    lines = [
        "# Phase 5 — Final Validation Report",
        "",
        f"**Overall status:** {status}",
        f"**FATAL checks failed:** {result.fatal_count}",
        f"**Warnings:** {result.warning_count}",
        f"**Total checks:** {len(result.checks)}",
        "",
        "---",
        "",
        "## Validation Results",
        "",
        "| Check | Severity | Result | Detail |",
        "|-------|----------|--------|--------|",
    ]

    for check in result.checks:
        result_str = "PASS ✓" if check.passed else "FAIL ✗"
        detail = check.detail.replace("|", "\\|")
        lines.append(
            f"| {check.name} | {check.severity} | {result_str} | {detail} |"
        )

    lines += [
        "",
        "---",
        "_Generated by Phase 5 validation. Do not edit manually._",
    ]

    return "\n".join(lines)


def save_validation_report(
    report_md: str, output_path: Path, logger: logging.Logger
) -> None:
    """Atomically write validation report markdown."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".md.tmp")
    try:
        tmp_path.write_text(report_md, encoding="utf-8")
        tmp_path.replace(output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    logger.info("Validation report saved: %s", output_path)
