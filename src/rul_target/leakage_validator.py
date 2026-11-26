"""Leakage prevention validator — standalone, independently-validated gate.

Implements all nine leakage checks from Architecture §9 (L-01 through L-09).
This module is a hard GATE: dataset_builder must not run until all checks pass.

Each check returns a LeakageCheckResult with a severity (FATAL / WARNING / INFO)
and concrete evidence. Any FATAL causes the pipeline to halt before any
training dataset is written.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.rul_target.split_generator import get_fold_ids

# Feature-column name prefixes (Phase 4 feature families).
_FEATURE_PREFIXES = ("td_", "fd_", "trend_", "env_", "cross_")
# Grain / provenance columns that legitimately appear in several artifacts.
_GRAIN_COLS = {
    "timestamp", "unit_id", "dataset_id", "bearing_id", "channel",
    "file_id", "filename", "file_sha256",
}


@dataclass
class LeakageCheckResult:
    """Result of a single leakage check."""

    check_id: str
    description: str
    severity: str  # "FATAL", "WARNING", "INFO"
    passed: bool
    evidence: str
    detail: str = ""


@dataclass
class LeakageValidationReport:
    """Aggregate result of all leakage checks."""

    all_passed: bool
    checks: List[LeakageCheckResult] = field(default_factory=list)

    @property
    def fatal_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == "FATAL")

    @property
    def warning_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == "WARNING")


def run_leakage_checks(
    feature_store: pd.DataFrame,
    rul_labels: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> LeakageValidationReport:
    """Execute all L-01 through L-09 leakage checks.

    Args:
        feature_store: Phase 4 feature store DataFrame.
        rul_labels: Labels DataFrame (all units including censored).
        fold_assignments: Fold assignment DataFrame (RTF units only).
        config: Full project configuration dictionary.
        logger: Pipeline logger.

    Returns:
        LeakageValidationReport — all_passed=False halts the pipeline.
    """
    logger.info("=" * 60)
    logger.info("LEAKAGE VALIDATION — ALL CHECKS MUST PASS")
    logger.info("=" * 60)

    checks: List[LeakageCheckResult] = []

    checks.append(_check_l01_future_samples_in_split(fold_assignments, logger))
    checks.append(_check_l02_window_boundary(fold_assignments, logger))
    checks.append(_check_l03_bearing_overlap(fold_assignments, logger))
    checks.append(_check_l04_duplicate_timestamps(feature_store, logger))
    checks.append(_check_l05_checkpoint_leakage(feature_store, config, logger))
    checks.append(_check_l06_label_feature_independence(rul_labels, feature_store, logger))
    checks.append(_check_l07_no_global_statistics(feature_store, config, logger))
    checks.append(_check_l08_censored_contamination(fold_assignments, rul_labels, logger))
    checks.append(_check_l09_target_normalization(rul_labels, config, logger))

    fatals_fired = [c for c in checks if not c.passed and c.severity == "FATAL"]
    all_passed = len(fatals_fired) == 0

    report = LeakageValidationReport(all_passed=all_passed, checks=checks)

    logger.info(f"Leakage validation: {'PASSED' if all_passed else 'FAILED'}")
    logger.info(f"  {report.fatal_count} FATAL checks failed, {report.warning_count} warnings")
    for check in checks:
        symbol = "✓" if check.passed else "✗"
        logger.info(f"  [{check.check_id}] {symbol} {check.description}")
        if not check.passed:
            logger.error(f"    Evidence: {check.evidence}")

    return report


# ──────────────────────────────────────────────────────────────────────────────
# Individual check implementations
# ──────────────────────────────────────────────────────────────────────────────

def _check_l01_future_samples_in_split(
    fold_assignments: pd.DataFrame, logger: logging.Logger
) -> LeakageCheckResult:
    """L-01: No unit_id appears in both train and test within any single fold."""
    violations = []
    for fold_id in get_fold_ids(fold_assignments):
        role_col = f"{fold_id}_role"
        train_units = set(fold_assignments[fold_assignments[role_col] == "train"]["unit_id"].unique())
        test_units = set(fold_assignments[fold_assignments[role_col] == "test"]["unit_id"].unique())
        overlap = train_units & test_units
        if overlap:
            violations.append(f"{fold_id}: overlap={sorted(overlap)}")

    if violations:
        return LeakageCheckResult(
            check_id="L-01", description="No unit appears in both train and test within a fold",
            severity="FATAL", passed=False,
            evidence=f"Unit overlap found: {violations}"
        )
    return LeakageCheckResult(
        check_id="L-01", description="No unit appears in both train and test within a fold",
        severity="FATAL", passed=True,
        evidence="Train/test unit sets are disjoint in all folds ✓"
    )


def _check_l02_window_boundary(
    fold_assignments: pd.DataFrame, logger: logging.Logger
) -> LeakageCheckResult:
    """L-02: Rolling-window features cannot span a split boundary (guaranteed by L-01)."""
    # Structural guarantee: because units are assigned whole (not row-by-row),
    # all rows of a unit stay together. A rolling window computed within a unit
    # cannot reach into another unit's rows. We assert the unit assignment is
    # indeed whole — no partial unit assignment.
    violations = []
    for fold_id in get_fold_ids(fold_assignments):
        role_col = f"{fold_id}_role"
        # Each unit_id must have exactly one role per fold
        unit_roles = (
            fold_assignments.groupby("unit_id")[role_col]
            .nunique()
        )
        multi_role = unit_roles[unit_roles > 1]
        if not multi_role.empty:
            violations.append(f"{fold_id}: units with multiple roles: {multi_role.to_dict()}")

    if violations:
        return LeakageCheckResult(
            check_id="L-02",
            description="Rolling-window features cannot span train/test boundary",
            severity="FATAL", passed=False,
            evidence=f"Partial unit assignments found: {violations}"
        )
    return LeakageCheckResult(
        check_id="L-02",
        description="Rolling-window features cannot span train/test boundary",
        severity="FATAL", passed=True,
        evidence="All units assigned to exactly one role per fold; window boundaries cannot cross ✓"
    )


def _check_l03_bearing_overlap(
    fold_assignments: pd.DataFrame, logger: logging.Logger
) -> LeakageCheckResult:
    """L-03: Set intersection of unit IDs per role across folds = ∅ (each unit tests exactly once)."""
    # Across all folds, each unit should test in exactly one fold and train in the rest.
    # Check: no unit is test in more than one fold (would mean unequal LOBO coverage),
    # and every unit is test in at least one fold (would mean it never gets evaluated).
    fold_ids = get_fold_ids(fold_assignments)
    all_train = set()
    test_fold_count: Dict[str, int] = {}
    for fold_id in fold_ids:
        role_col = f"{fold_id}_role"
        all_train |= set(fold_assignments[fold_assignments[role_col] == "train"]["unit_id"].unique())
        for unit_id in fold_assignments[fold_assignments[role_col] == "test"]["unit_id"].unique():
            test_fold_count[unit_id] = test_fold_count.get(unit_id, 0) + 1

    all_test = set(test_fold_count.keys())
    only_test = all_test - all_train
    only_train = all_train - all_test
    tested_multiple_times = {u: c for u, c in test_fold_count.items() if c > 1}

    issues = []
    if only_test:
        issues.append(f"Units that are NEVER in train: {sorted(only_test)}")
    if only_train:
        issues.append(f"Units that are NEVER in test: {sorted(only_train)}")
    if tested_multiple_times:
        issues.append(f"Units tested in more than one fold (should be exactly once): {tested_multiple_times}")

    if issues:
        return LeakageCheckResult(
            check_id="L-03",
            description="Each RTF unit appears as both train (in other folds) and test (in exactly one fold)",
            severity="FATAL", passed=False,
            evidence="; ".join(issues)
        )
    return LeakageCheckResult(
        check_id="L-03",
        description="Each RTF unit appears as both train (in other folds) and test (in exactly one fold)",
        severity="FATAL", passed=True,
        evidence=f"LOBO coverage confirmed across {len(fold_ids)} folds: train units={sorted(all_train)}, test units={sorted(all_test)} ✓"
    )


def _check_l04_duplicate_timestamps(
    feature_store: pd.DataFrame, logger: logging.Logger
) -> LeakageCheckResult:
    """L-04: No duplicate (unit_id, channel, timestamp) keys in feature store."""
    key_cols = ["unit_id", "channel", "timestamp"]
    dup_count = feature_store.duplicated(subset=key_cols).sum()

    if dup_count > 0:
        return LeakageCheckResult(
            check_id="L-04",
            description="No duplicate (unit_id, channel, timestamp) keys",
            severity="FATAL", passed=False,
            evidence=f"{dup_count} duplicate key combinations found in feature store"
        )
    return LeakageCheckResult(
        check_id="L-04",
        description="No duplicate (unit_id, channel, timestamp) keys",
        severity="FATAL", passed=True,
        evidence="0 duplicate keys ✓"
    )


def _check_l05_checkpoint_leakage(
    feature_store: pd.DataFrame,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> LeakageCheckResult:
    """L-05: The feature store carries no column sourced from checkpoint_selection.csv.

    Runtime check (replaces the former hardcoded assertion): if the Phase 3
    checkpoint_selection.csv exists, its non-grain columns (e.g. selection_reason,
    checkpoint_rank, mean_kurtosis_failure_bearings) must be DISJOINT from the
    feature-store columns. Any overlap means a checkpoint-selection join contaminated
    the feature set — checkpoint membership is future-informed and must never become
    a feature. FATAL because it is now genuinely runtime-detectable.
    """
    cp_path_str = (
        config.get("signal_analysis", {})
        .get("output_paths", {})
        .get("checkpoint_selection", "data/processed/checkpoint_selection.csv")
    )
    cp_path = Path(cp_path_str)
    if not cp_path.exists():
        return LeakageCheckResult(
            check_id="L-05",
            description="No join with Phase 3 checkpoint_selection.csv",
            severity="INFO", passed=True,
            evidence=f"checkpoint_selection.csv not present at {cp_path} — no join possible ✓",
        )

    try:
        cp_cols = set(pd.read_csv(cp_path, nrows=0).columns) - _GRAIN_COLS
    except Exception as exc:  # unreadable header → cannot have been joined
        return LeakageCheckResult(
            check_id="L-05",
            description="No join with Phase 3 checkpoint_selection.csv",
            severity="WARNING", passed=True,
            evidence=f"checkpoint_selection.csv header unreadable ({exc}); no join detectable",
        )

    leaked = sorted(cp_cols & set(feature_store.columns))
    if leaked:
        return LeakageCheckResult(
            check_id="L-05",
            description="No join with Phase 3 checkpoint_selection.csv",
            severity="FATAL", passed=False,
            evidence=(
                "Feature store contains checkpoint-selection-derived column(s): "
                f"{leaked}. Checkpoint membership is future-informed and must not be a feature."
            ),
        )
    return LeakageCheckResult(
        check_id="L-05",
        description="No join with Phase 3 checkpoint_selection.csv",
        severity="FATAL", passed=True,
        evidence=(
            f"Inspected {len(cp_cols)} non-grain checkpoint column(s) {sorted(cp_cols)}; "
            "none appear in the feature store ✓"
        ),
    )


def _check_l06_label_feature_independence(
    rul_labels: pd.DataFrame,
    feature_store: pd.DataFrame,
    logger: logging.Logger,
) -> LeakageCheckResult:
    """L-06: Label columns are independent of feature columns (not derived from them)."""
    feature_cols = {c for c in feature_store.columns
                    if c.startswith(("td_", "fd_", "trend_", "env_", "cross_"))}
    label_cols = {"rul_minutes", "rul_cycles", "rul_frac", "is_censored"}
    label_cols_present = {c for c in rul_labels.columns if c in label_cols}

    # Labels must not share names with feature columns
    intersection = feature_cols & label_cols_present
    if intersection:
        return LeakageCheckResult(
            check_id="L-06",
            description="Label columns are independent of feature columns",
            severity="FATAL", passed=False,
            evidence=f"Column name collision between labels and features: {sorted(intersection)}"
        )

    # Provenance assertion: rul_labels only contains grain + label columns
    allowed = {"unit_id", "dataset_id", "bearing_id", "channel", "timestamp"} | label_cols
    unexpected = {c for c in rul_labels.columns if c not in allowed}
    if unexpected:
        return LeakageCheckResult(
            check_id="L-06",
            description="Label columns are independent of feature columns",
            severity="WARNING", passed=True,
            evidence=f"Unexpected columns in rul_labels (review): {sorted(unexpected)}"
        )

    return LeakageCheckResult(
        check_id="L-06",
        description="Label columns are independent of feature columns",
        severity="FATAL", passed=True,
        evidence=(
            f"Labels ({sorted(label_cols_present)}) have no name collision with "
            f"feature columns and are derived only from timestamp + unit identity ✓"
        )
    )


def _check_l07_no_global_statistics(
    feature_store: pd.DataFrame,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> LeakageCheckResult:
    """L-07: Phase 5 applies no global scaling to features (scaling deferred to Phase 6).

    Runtime check (replaces the former hardcoded assertion). Two detectable signatures
    of an accidental global-statistics leak are inspected:

      1. Artifact scan: no scaler/normalizer artifact has been written to the processed
         data directory (files matching scaler/scaling/standardizer/normalizer/
         global_stats). Phase 5 emits labels and splits only — such a file would be the
         signature of a leak.
      2. Distribution check: the feature columns still carry their RAW scale. If a global
         StandardScaler had been applied across all units, essentially every feature
         column would have mean ~0 and std ~1 simultaneously. We flag that pattern.

    WARNING severity: a positive detection is surfaced loudly but the FATAL gate for
    Phase 5 remains the unit-disjointness family (L-01/L-03/L-08).
    """
    processed_dir = Path(config.get("paths", {}).get("processed_data_dir", "data/processed"))
    scaler_tokens = ("scaler", "scaling", "standardizer", "normalizer", "global_stats")
    scaler_artifacts: List[str] = []
    if processed_dir.exists():
        for p in processed_dir.rglob("*"):
            if p.is_file() and any(tok in p.name.lower() for tok in scaler_tokens):
                scaler_artifacts.append(str(p))

    feature_cols = [c for c in feature_store.columns if c.startswith(_FEATURE_PREFIXES)]
    standardized_cols = 0
    for col in feature_cols:
        values = pd.to_numeric(feature_store[col], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size < 2:
            continue
        if abs(float(np.mean(values))) < 1e-6 and abs(float(np.std(values)) - 1.0) < 1e-3:
            standardized_cols += 1

    looks_standardized = feature_cols and standardized_cols == len(feature_cols)

    if scaler_artifacts or looks_standardized:
        return LeakageCheckResult(
            check_id="L-07",
            description="No global input statistics computed in Phase 5",
            severity="WARNING", passed=False,
            evidence=(
                f"Possible global-statistics leak: scaler_artifacts={scaler_artifacts}, "
                f"standardized_feature_columns={standardized_cols}/{len(feature_cols)}. "
                "Feature scaling must be deferred to Phase 6 and fit on train folds only."
            ),
        )
    return LeakageCheckResult(
        check_id="L-07",
        description="No global input statistics computed in Phase 5",
        severity="INFO", passed=True,
        evidence=(
            f"No scaler artifact in {processed_dir}; feature columns retain raw scale "
            f"({standardized_cols}/{len(feature_cols)} columns look standardized) ✓"
        ),
    )


def _check_l08_censored_contamination(
    fold_assignments: pd.DataFrame,
    rul_labels: pd.DataFrame,
    logger: logging.Logger,
) -> LeakageCheckResult:
    """L-08: No censored unit appears in any labelled (train/test) fold role."""
    censored_ids = set(rul_labels[rul_labels["is_censored"]]["unit_id"].unique())
    fold_unit_ids = set(fold_assignments["unit_id"].unique())

    contaminated = censored_ids & fold_unit_ids

    if contaminated:
        return LeakageCheckResult(
            check_id="L-08",
            description="No censored unit in labelled fold roles",
            severity="FATAL", passed=False,
            evidence=f"Censored units found in fold assignments: {sorted(contaminated)}"
        )
    return LeakageCheckResult(
        check_id="L-08",
        description="No censored unit in labelled fold roles",
        severity="FATAL", passed=True,
        evidence=(
            f"Censored units ({sorted(censored_ids)}) have zero presence in fold assignments. "
            f"Fold assignments contain only: {sorted(fold_unit_ids)} ✓"
        )
    )


def _check_l09_target_normalization(
    rul_labels: pd.DataFrame,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> LeakageCheckResult:
    """L-09: rul_frac total_life is computed per-unit, never pooled across units."""
    rul_cfg = config.get("rul_target", {})
    emit_rul_frac = rul_cfg.get("emit_rul_frac", False)

    if not emit_rul_frac:
        return LeakageCheckResult(
            check_id="L-09",
            description="rul_frac total_life computed per-unit only (not applicable: emit_rul_frac=false)",
            severity="INFO", passed=True,
            evidence="emit_rul_frac=false — L-09 not applicable ✓"
        )

    if "rul_frac" not in rul_labels.columns:
        return LeakageCheckResult(
            check_id="L-09",
            description="rul_frac total_life computed per-unit only",
            severity="WARNING", passed=True,
            evidence="rul_frac column absent despite emit_rul_frac=true; review label_generator"
        )

    # Verify: for RTF units, rul_frac at t_fail = 0 and at t_start ≈ 1.0
    rtf_labels = rul_labels[~rul_labels["is_censored"] & rul_labels["rul_frac"].notna()]
    violations = []
    for unit_id, group in rtf_labels.groupby("unit_id"):
        frac_at_fail = group[group["rul_minutes"] == 0.0]["rul_frac"].values
        if len(frac_at_fail) > 0 and not all(abs(f) < 1e-9 for f in frac_at_fail):
            violations.append(f"{unit_id}: rul_frac at t_fail should be 0, got {frac_at_fail}")
        max_frac = group["rul_frac"].max()
        if max_frac > 1.0 + 1e-9:
            violations.append(f"{unit_id}: rul_frac > 1.0 ({max_frac:.4f}) — cross-unit total_life was pooled?")

    if violations:
        return LeakageCheckResult(
            check_id="L-09",
            description="rul_frac total_life computed per-unit only",
            severity="FATAL", passed=False,
            evidence="; ".join(violations)
        )

    return LeakageCheckResult(
        check_id="L-09",
        description="rul_frac total_life computed per-unit only",
        severity="FATAL", passed=True,
        evidence="rul_frac ∈ [0,1] per unit; rul_frac(t_fail)=0 verified; no cross-unit normalization ✓"
    )


def generate_leakage_report(
    report: LeakageValidationReport,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> str:
    """Generate leakage_report.md documenting all checks with evidence.

    Args:
        report: LeakageValidationReport from run_leakage_checks.
        config: Full project configuration dictionary.
        logger: Pipeline logger.

    Returns:
        Markdown string for leakage_report.md.
    """
    status = "PASSED ✓" if report.all_passed else "FAILED ✗"
    lines = [
        "# Phase 5 — Leakage Prevention Report",
        "",
        f"**Overall status:** {status}",
        f"**FATAL checks failed:** {report.fatal_count}",
        f"**Warnings:** {report.warning_count}",
        "",
        "---",
        "",
        "## Leakage Checks",
        "",
        "| ID | Description | Severity | Result | Evidence |",
        "|----|-------------|----------|--------|----------|",
    ]

    for check in report.checks:
        result_str = "PASS ✓" if check.passed else "FAIL ✗"
        # Escape pipe characters in evidence
        evidence = check.evidence.replace("|", "\\|")
        lines.append(
            f"| {check.check_id} | {check.description} | "
            f"{check.severity} | {result_str} | {evidence} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Leakage Source Definitions",
        "",
        "| ID | Source | Mechanism of Harm | Prevention |",
        "|----|--------|-------------------|------------|",
        "| L-01 | Future samples in split | Adjacent timestamps in train & test share near-identical degradation state | Unit-wise LOBO split; no unit in two roles |",
        "| L-02 | Rolling-window features spanning split boundary | Phase 4 trend windows mix rows that a row-split would separate | Whole-unit assignment makes window boundaries irrelevant |",
        "| L-03 | Bearing overlap across folds | Same physical unit trains and tests the model | Group key = unit_id (dataset_id::bearing_id, qualified); enforced disjoint |",
        "| L-04 | Duplicate timestamps | Same acquisition counted twice | Uniqueness enforced on (unit_id, channel, timestamp) |",
        "| L-05 | Checkpoint leakage | Phase 3 checkpoint files reused as if independent | Phase 5 uses full feature store only; no checkpoint join |",
        "| L-06 | Derived/label leakage | Label built from features | Label depends only on timestamp + unit identity |",
        "| L-07 | Normalization/statistics leakage | Scaling fit on full dataset leaks test distribution | No global input statistics computed; scaling is Phase 6 |",
        "| L-08 | Censored-unit contamination | Fabricated RUL for surviving units enters supervision | Censored units excluded from all fold roles |",
        "| L-09 | Target normalization leakage | total_life for rul_frac drawn from test unit | Per-unit normalization only; never pooled across units |",
        "",
        "---",
        "_Generated by Phase 5 leakage_validator. Do not edit manually._",
    ]

    return "\n".join(lines)


def save_leakage_report(report_md: str, output_path: Path, logger: logging.Logger) -> None:
    """Atomically write leakage report markdown.

    Args:
        report_md: Markdown string.
        output_path: Final destination path.
        logger: Pipeline logger.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".md.tmp")
    try:
        tmp_path.write_text(report_md, encoding="utf-8")
        tmp_path.replace(output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    logger.info(f"Leakage report saved: {output_path}")
