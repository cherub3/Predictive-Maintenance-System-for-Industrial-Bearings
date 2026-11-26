"""Phase 7 final validation — 10 checks covering artifact integrity and decision
consistency. Architecture §16 and §16c.

Mirrors the ValidationCheck / FinalValidationResult pattern from
src/rul_target/validation.py and src/model_dev/validation.py.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.decision_engine.contract_loader import ContractValidationResult


@dataclass
class ValidationCheck:
    """Result of a single validation check."""

    name: str
    passed: bool
    severity: str  # "FATAL" | "WARNING" | "INFO"
    detail: str


@dataclass
class FinalValidationResult:
    """Aggregate result of all Phase 7 validation checks."""

    all_passed: bool
    checks: List[ValidationCheck] = field(default_factory=list)

    @property
    def fatal_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == "FATAL")

    @property
    def warning_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == "WARNING")


# Expected risk category values (architecture §16a DV5)
_VALID_RISK_CATEGORIES = {"CRITICAL", "HIGH", "MEDIUM", "LOW", "HEALTHY"}

# Expected maintenance actions for cross-referencing
_VALID_ACTIONS = {
    "IMMEDIATE_SHUTDOWN",
    "URGENT_MAINTENANCE",
    "SCHEDULE_MAINTENANCE",
    "ROUTINE_MONITORING",
    "NORMAL_OPERATION",
}

# Required CRITICAL → IMMEDIATE_SHUTDOWN mapping (architecture §12a.2 Tier 1)
_CRITICAL_ACTION = "IMMEDIATE_SHUTDOWN"
_CRITICAL_PRIORITY = 1

# Risk → expected action (base policy, may differ if BR-01 overrides)
_BASE_POLICY = {
    "CRITICAL": "IMMEDIATE_SHUTDOWN",
    "HIGH": "URGENT_MAINTENANCE",
    "MEDIUM": "SCHEDULE_MAINTENANCE",
    "LOW": "ROUTINE_MONITORING",
    "HEALTHY": "NORMAL_OPERATION",
}


def run_decision_validation(
    contract_validation: ContractValidationResult,
    recommendations_df: pd.DataFrame,
    manifest: Dict[str, Any],
    expected_bearings: List[str],
    config: Dict[str, Any],
    logger: logging.Logger,
) -> FinalValidationResult:
    """Run all 10 Phase 7 validation checks. Architecture §16a and §16c.

    Args:
        contract_validation: Phase 6 contract entry-gate result.
        recommendations_df: Final recommendations DataFrame.
        manifest: Built decision manifest.
        expected_bearings: Bearings expected in the output (from Phase 6).
        config: Full project configuration dict.
        logger: Pipeline logger.

    Returns:
        FinalValidationResult. Caller must exit on fatal failures.
    """
    logger.info("=" * 60)
    logger.info("FINAL VALIDATION — Phase 7 artifacts")
    logger.info("=" * 60)

    checks: List[ValidationCheck] = []

    # DV1 — Phase 6 contract entry gate (FATAL)
    checks.append(ValidationCheck(
        name="phase6_contract_entry_gate",
        passed=contract_validation.passed,
        severity="FATAL",
        detail=(
            f"{len(contract_validation.fatals)} fatal contract error(s): "
            f"{contract_validation.fatals}"
            if not contract_validation.passed
            else "Phase 6 contract entry gate passed ✓"
        ),
    ))

    # DV2 — All expected bearings present in output (FATAL)
    if recommendations_df.empty:
        actual_bearings = set()
    else:
        actual_bearings = set(recommendations_df["bearing_id"].unique())
    missing_bearings = set(expected_bearings) - actual_bearings
    checks.append(ValidationCheck(
        name="all_bearings_present",
        passed=len(missing_bearings) == 0,
        severity="FATAL",
        detail=(
            f"Missing bearings in output: {sorted(missing_bearings)}"
            if missing_bearings
            else f"All expected bearings present: {sorted(actual_bearings)} ✓"
        ),
    ))

    # DV3 — No NaN in min_predicted_rul_minutes (FATAL)
    if not recommendations_df.empty:
        # NaN rows are BR-02 fallback rows (is_business_rule_override=True) and are
        # allowed to have NaN min_rul; only non-override rows are checked for NaN.
        non_br02 = recommendations_df[~recommendations_df["is_business_rule_override"]]
        has_nan = non_br02["min_predicted_rul_minutes"].isna().any()
    else:
        has_nan = False
    checks.append(ValidationCheck(
        name="no_nan_rul_outside_fallback",
        passed=not has_nan,
        severity="FATAL",
        detail=(
            "NaN values found in min_predicted_rul_minutes for non-fallback rows"
            if has_nan
            else "No unexpected NaN in min_predicted_rul_minutes ✓"
        ),
    ))

    # DV4 — No negative min_predicted_rul_minutes outside fallback (FATAL)
    if not recommendations_df.empty:
        non_br02_rul = recommendations_df.loc[
            ~recommendations_df["is_business_rule_override"], "min_predicted_rul_minutes"
        ].dropna()
        has_negative = (non_br02_rul < 0).any()
    else:
        has_negative = False
    checks.append(ValidationCheck(
        name="no_negative_rul",
        passed=not has_negative,
        severity="FATAL",
        detail=(
            "Negative values found in min_predicted_rul_minutes"
            if has_negative
            else "All min_predicted_rul_minutes are non-negative ✓"
        ),
    ))

    # DV5 — All risk_category values valid (FATAL)
    if not recommendations_df.empty:
        invalid_cats = set(recommendations_df["risk_category"].unique()) - _VALID_RISK_CATEGORIES
    else:
        invalid_cats = set()
    checks.append(ValidationCheck(
        name="valid_risk_categories",
        passed=len(invalid_cats) == 0,
        severity="FATAL",
        detail=(
            f"Invalid risk categories found: {invalid_cats}"
            if invalid_cats
            else f"All risk categories valid: {sorted(recommendations_df['risk_category'].unique())} ✓"
            if not recommendations_df.empty
            else "No recommendations to check ✓"
        ),
    ))

    # DV6 — All priority_score values in {1,2,3,4,5} (FATAL)
    if not recommendations_df.empty:
        invalid_priorities = set(recommendations_df["priority_score"].unique()) - {1, 2, 3, 4, 5}
    else:
        invalid_priorities = set()
    checks.append(ValidationCheck(
        name="valid_priority_scores",
        passed=len(invalid_priorities) == 0,
        severity="FATAL",
        detail=(
            f"Invalid priority scores found: {invalid_priorities}"
            if invalid_priorities
            else "All priority scores are in {1,2,3,4,5} ✓"
        ),
    ))

    # DV7 — n2_caveat == True in every row (WARNING)
    if not recommendations_df.empty:
        n2_violations = (~recommendations_df["n2_caveat"].astype(bool)).sum()
    else:
        n2_violations = 0
    checks.append(ValidationCheck(
        name="n2_caveat_propagated",
        passed=n2_violations == 0,
        severity="WARNING",
        detail=(
            f"n2_caveat is False/missing in {n2_violations} row(s)"
            if n2_violations
            else "n2_caveat=True in all recommendation rows ✓"
        ),
    ))

    # DV8 — CRITICAL rows all have priority_score == 1 (WARNING → FATAL per §16c)
    if not recommendations_df.empty:
        critical_rows = recommendations_df[recommendations_df["risk_category"] == "CRITICAL"]
        critical_not_p1 = (critical_rows["priority_score"] != 1).sum()
    else:
        critical_not_p1 = 0
    checks.append(ValidationCheck(
        name="critical_rows_priority_1",
        passed=critical_not_p1 == 0,
        severity="FATAL",  # §16c: CRITICAL/priority mismatch is FATAL (safety violation)
        detail=(
            f"{critical_not_p1} CRITICAL row(s) have priority_score != 1 "
            "(architecture §12a.2 Tier 1 safety floor violated)"
            if critical_not_p1
            else "All CRITICAL rows have priority_score == 1 ✓"
        ),
    ))

    # DV9 — Governance fields in manifest (WARNING)
    has_experiment_id = bool(manifest.get("experiment_id"))
    has_fingerprint = bool(manifest.get("config_fingerprint"))
    has_rule_version = bool(manifest.get("rule_version"))
    governance_ok = has_experiment_id and has_fingerprint and has_rule_version
    checks.append(ValidationCheck(
        name="governance_fields_in_manifest",
        passed=governance_ok,
        severity="WARNING",
        detail=(
            f"Manifest missing fields: experiment_id={has_experiment_id}, "
            f"config_fingerprint={has_fingerprint}, rule_version={has_rule_version}"
            if not governance_ok
            else "Manifest has all required governance fields ✓"
        ),
    ))

    # DV10 — Channel count consistency (INFO)
    if not recommendations_df.empty:
        min_channels = recommendations_df["channel_count"].min()
        max_channels = recommendations_df["channel_count"].max()
        channel_consistent = min_channels == max_channels
    else:
        channel_consistent = True
        min_channels = max_channels = 0
    checks.append(ValidationCheck(
        name="channel_count_consistency",
        passed=True,  # INFO: never blocks
        severity="INFO",
        detail=(
            f"Channel counts vary: min={min_channels}, max={max_channels} "
            "(inter-channel differences noted; BR-03 would handle large spread)"
            if not channel_consistent
            else f"Channel count consistent: {min_channels} per bearing-timestamp ✓"
        ),
    ))

    # DV11 — Contradictory recommendation check (FATAL per §16c)
    # Every (risk_category, maintenance_action) pair must match _BASE_POLICY.
    # Any deviation is a structural safety violation (§16c).
    if not recommendations_df.empty:
        expected_actions = recommendations_df["risk_category"].map(_BASE_POLICY)
        contradictory_mask = recommendations_df["maintenance_action"] != expected_actions
        contradictory_count = int(contradictory_mask.sum())
    else:
        contradictory_count = 0
    checks.append(ValidationCheck(
        name="no_contradictory_recommendations",
        passed=contradictory_count == 0,
        severity="FATAL",
        detail=(
            f"{contradictory_count} row(s) have risk_category/maintenance_action "
            "combinations inconsistent with the declared policy map "
            "(architecture §16c — structural safety violation)"
            if contradictory_count > 0
            else "All risk_category/maintenance_action pairs are policy-consistent ✓"
        ),
    ))

    fatal_count = sum(1 for c in checks if not c.passed and c.severity == "FATAL")
    all_passed = fatal_count == 0

    status = "PASSED" if all_passed else "FAILED"
    logger.info(
        "Phase 7 validation: %s (%d fatal, %d warning)",
        status,
        fatal_count,
        sum(1 for c in checks if not c.passed and c.severity == "WARNING"),
    )
    for c in checks:
        symbol = "✓" if c.passed else "✗"
        logger.info("  [%s] %s %s: %s", c.severity, symbol, c.name, c.detail)

    return FinalValidationResult(all_passed=all_passed, checks=checks)


def generate_decision_validation_report(result: FinalValidationResult, logger: logging.Logger) -> str:
    """Render decision_validation_report.md. Architecture §16."""
    lines = [
        "# Phase 7 — Decision Validation Report",
        "",
        f"**Status:** {'PASSED ✓' if result.all_passed else 'FAILED ✗'}",
        f"**Fatal checks failed:** {result.fatal_count}",
        f"**Warnings:** {result.warning_count}",
        "",
        "| Severity | Status | Check | Detail |",
        "|----------|--------|-------|--------|",
    ]
    for c in result.checks:
        symbol = "PASS" if c.passed else "FAIL"
        lines.append(f"| {c.severity} | {symbol} | {c.name} | {c.detail} |")
    lines += ["", "---", "", "_Generated by Phase 7 pipeline. Do not edit manually._"]

    report_md = "\n".join(lines)
    logger.info("Decision validation report generated")
    return report_md


def save_decision_validation_report(
    report_md: str, output_path: Path, logger: logging.Logger
) -> None:
    """Atomically write decision_validation_report.md. Architecture §16."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".md.tmp")
    try:
        tmp_path.write_text(report_md, encoding="utf-8")
        tmp_path.replace(output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    logger.info("Decision validation report saved: %s", output_path)
