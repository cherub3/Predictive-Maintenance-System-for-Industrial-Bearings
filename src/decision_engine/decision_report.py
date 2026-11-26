"""Decision report generator — produces decision_report.md.

Architecture §6. Human-readable summary of all decisions, including the
mandatory Statistical Limitations header that must not be removed by Phase 8
(architecture §22 IN-09).
"""

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.decision_engine.contract_loader import ContractValidationResult


def generate_decision_report(
    recommendations_df: pd.DataFrame,
    contract_validation: ContractValidationResult,
    experiment_id: str,
    config_fingerprint: str,
    rule_version: str,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> str:
    """Render decision_report.md. Architecture §6.

    The report has a mandatory Statistical Limitations section before the
    decision tables — this section must not be removed or reduced (§22 IN-09).

    Args:
        recommendations_df: Final recommendations DataFrame.
        contract_validation: Validated Phase 6 contract metadata.
        experiment_id: Phase 7 experiment ID.
        config_fingerprint: SHA-256 of decision_engine config.
        rule_version: Business rule version.
        config: Full project configuration dict.
        logger: Pipeline logger.

    Returns:
        Markdown report string.
    """
    de_cfg = config.get("decision_engine", {})
    thresholds = de_cfg.get("risk_thresholds", {})

    lines = [
        "# Phase 7 — Maintenance Decision Report",
        "",
        f"**Experiment ID:** `{experiment_id}`",
        f"**Config Fingerprint:** `{config_fingerprint}`",
        f"**Rule Version:** `{rule_version}`",
        f"**Phase 6 Model Type:** `{contract_validation.model_type}`",
        f"**Phase 6 Experiment:** `{contract_validation.experiment_id}`",
        "",
        "---",
        "",
        "## ⚠ Statistical Limitations",
        "",
        "> **This section must not be removed or reduced in Phase 8 display.**",
        "> Architecture §22 IN-09.",
        "",
        f"- **n={contract_validation.n_rtf_units} caveat:** {contract_validation.n2_caveat_text}",
        f"- **Confidence type:** Categorical only (HIGH / MEDIUM / LOW). "
        f"Numerical probability estimates are not scientifically supportable for "
        f"n={contract_validation.n_rtf_units} corpora.",
        f"- **Model type:** `{contract_validation.model_type}` "
        + (
            "(demotion fallback — constant predictor, no feature sensitivity)"
            if contract_validation.model_type in ("mean", "median")
            else ""
        ),
        f"- **Phase 6 MAE:** {contract_validation.mean_mae_minutes:.0f} minutes "
        f"({contract_validation.mean_mae_minutes / 60:.1f} hours)",
        "- **Threshold values:** Engineering placeholders — require domain-expert validation "
        "before operational deployment (architecture §9b, §9d).",
        "",
        "---",
        "",
        "## Decision Summary",
        "",
    ]

    if recommendations_df.empty:
        lines.append("_No recommendations generated._")
    else:
        risk_counts = recommendations_df["risk_category"].value_counts()
        action_counts = recommendations_df["maintenance_action"].value_counts()
        override_count = recommendations_df["is_business_rule_override"].sum()

        lines += [
            f"**Total recommendations:** {len(recommendations_df)}",
            f"**Bearings:** {sorted(recommendations_df['bearing_id'].unique())}",
            f"**Timestamps per bearing:** "
            + str(recommendations_df.groupby("bearing_id").size().to_dict()),
            f"**Business rule overrides:** {override_count}",
            "",
            "### Risk Category Distribution",
            "",
            "| Risk Category | Count |",
            "|---------------|-------|",
        ]
        for cat, count in risk_counts.items():
            lines.append(f"| {cat} | {count} |")

        lines += [
            "",
            "### Maintenance Action Distribution",
            "",
            "| Maintenance Action | Count |",
            "|--------------------|-------|",
        ]
        for action, count in action_counts.items():
            lines.append(f"| {action} | {count} |")

        lines += [
            "",
            "## Risk Thresholds (Engineering Placeholders)",
            "",
            "| Category | Threshold (minutes) | Hours | Days |",
            "|----------|---------------------|-------|------|",
        ]
        for name, label in [
            ("critical_minutes", "CRITICAL"),
            ("high_minutes", "HIGH"),
            ("medium_minutes", "MEDIUM"),
            ("low_minutes", "LOW"),
        ]:
            val = thresholds.get(name)
            if val is not None:
                h = float(val) / 60.0
                d = float(val) / 1440.0
                lines.append(f"| {label} | {val} | {h:.1f} | {d:.2f} |")
            else:
                lines.append(f"| {label} | *null* | — | — |")

        # Per-bearing summary
        lines += [
            "",
            "## Per-Bearing Decision Summary",
            "",
        ]
        for bearing_id, grp in recommendations_df.groupby("bearing_id"):
            rc = grp["risk_category"].value_counts()
            conf = grp["confidence_level"].iloc[0]
            prio = grp["priority_score"].iloc[0]
            min_rul = grp["min_predicted_rul_minutes"].iloc[0]
            lines += [
                f"### {bearing_id}",
                "",
                f"- **Predicted RUL (constant):** {min_rul:.0f} minutes "
                f"({min_rul/60:.1f} hours / {min_rul/1440:.2f} days)",
                f"- **Risk Category:** {rc.index[0]} ({len(grp)} timestamps)",
                f"- **Confidence:** {conf}",
                f"- **Priority:** {prio}/5",
                "",
            ]

    lines += [
        "---",
        "",
        "_Generated by Phase 7 Maintenance Decision Engine. Do not edit manually._",
        "_Architecture §22 IN-09: the Statistical Limitations section must not be "
        "removed in Phase 8._",
    ]
    report = "\n".join(lines)
    logger.info("Decision report generated (%d lines)", len(lines))
    return report


def save_decision_report(report_md: str, output_path: Path, logger: logging.Logger) -> None:
    """Atomically write decision_report.md."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".md.tmp")
    try:
        tmp_path.write_text(report_md, encoding="utf-8")
        tmp_path.replace(output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    logger.info("Decision report saved: %s", output_path)
