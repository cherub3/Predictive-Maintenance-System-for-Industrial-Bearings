"""Aggregate LOBO metrics per model, per fold, and generate the evaluation report
including the mandatory error-analysis decomposition (architecture §10, §10.5, §10.6).
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.model_dev.cross_validation import CVResults, FoldResult
from src.model_dev.metrics import MetricBundle, compute_all_metrics, compute_mae


@dataclass
class ModelFoldMetrics:
    """Metrics + error-analysis artifacts for one model on one fold."""

    fold_id: str
    train_bearing: str
    test_bearing: str
    metrics: MetricBundle
    channel_specific_mae: Dict[str, float]
    largest_residuals: pd.DataFrame


@dataclass
class ModelEvaluation:
    """All fold metrics for one model plus the two-fold summary."""

    model_name: str
    fold_metrics: Dict[str, ModelFoldMetrics] = field(default_factory=dict)
    mean_mae: float = float("nan")
    mae_spread: float = float("nan")
    stability_ratio: float = float("nan")  # max(fold_mae) / min(fold_mae)


@dataclass
class EvaluationResult:
    """Full evaluation across all models."""

    model_evaluations: Dict[str, ModelEvaluation] = field(default_factory=dict)
    model_order: List[str] = field(default_factory=list)


def _largest_residuals(fold_result: FoldResult, top_n: int) -> pd.DataFrame:
    """Identify the top_n held-out rows with the largest absolute residual (architecture §10.6)."""
    residuals = np.abs(fold_result.y_test_pred - fold_result.y_test_true)
    top_n = min(top_n, len(residuals))
    top_idx = np.argsort(-residuals)[:top_n]
    df = fold_result.grain_df.iloc[top_idx].copy().reset_index(drop=True)
    df["actual_rul_minutes"] = fold_result.y_test_true[top_idx]
    df["predicted_rul_minutes"] = fold_result.y_test_pred[top_idx]
    df["abs_residual_minutes"] = residuals[top_idx]
    return df


def _channel_specific_mae(fold_result: FoldResult) -> Dict[str, float]:
    """MAE broken down per sensor channel within the held-out bearing (architecture §10.6)."""
    out: Dict[str, float] = {}
    if "channel" not in fold_result.grain_df.columns:
        return out
    channels = fold_result.grain_df["channel"].to_numpy()
    for ch in np.unique(channels):
        mask = channels == ch
        out[str(ch)] = compute_mae(fold_result.y_test_true[mask], fold_result.y_test_pred[mask])
    return out


def evaluate_cv_results(cv_results: CVResults, config: Dict[str, Any], logger: logging.Logger) -> EvaluationResult:
    """Compute the full metric + error-analysis bundle for every model and fold.

    Args:
        cv_results: Output of `cross_validation.run_lobo_cv`.
        config: Full project configuration dictionary.
        logger: Pipeline logger.

    Returns:
        EvaluationResult with per-model, per-fold metrics and two-fold summaries.
    """
    top_n = config.get("model_dev", {}).get("metrics", {}).get("top_n_residuals", 10)
    evaluation = EvaluationResult(model_order=cv_results.model_order)

    for model_name in cv_results.model_order:
        fold_metrics: Dict[str, ModelFoldMetrics] = {}
        for fold_id, fold_result in cv_results.results[model_name].items():
            metrics = compute_all_metrics(
                fold_result.y_test_true, fold_result.y_test_pred,
                fold_result.y_train_true, fold_result.y_train_pred,
                config,
            )
            fold_metrics[fold_id] = ModelFoldMetrics(
                fold_id=fold_id,
                train_bearing=", ".join(fold_result.train_bearings),
                test_bearing=fold_result.test_bearings[0],
                metrics=metrics,
                channel_specific_mae=_channel_specific_mae(fold_result),
                largest_residuals=_largest_residuals(fold_result, top_n),
            )

        fold_maes = [fm.metrics.mae for fm in fold_metrics.values()]
        mean_mae = float(np.mean(fold_maes))
        spread = float(np.max(fold_maes) - np.min(fold_maes))
        stability_ratio = float(np.max(fold_maes) / np.min(fold_maes)) if np.min(fold_maes) > 0 else float("inf")

        evaluation.model_evaluations[model_name] = ModelEvaluation(
            model_name=model_name,
            fold_metrics=fold_metrics,
            mean_mae=mean_mae,
            mae_spread=spread,
            stability_ratio=stability_ratio,
        )

        logger.info(
            "Evaluation [%s]: mean MAE=%.2f, spread=%.2f, stability_ratio=%.2f",
            model_name, mean_mae, spread, stability_ratio,
        )

    return evaluation


def generate_evaluation_report(evaluation_result: EvaluationResult, config: Dict[str, Any], logger: logging.Logger) -> str:
    """Render `evaluation_report.md` (architecture §5, §10.4, §10.6).

    Every number is reported with the small-n methodology-validation caveat
    stated adjacent to it, per the honesty guardrail (§10.4). n is the number
    of LOBO folds actually run (one per RTF unit), not a hardcoded 2.
    """
    metrics_cfg = config.get("model_dev", {}).get("metrics", {})
    near_failure_threshold = metrics_cfg.get("near_failure_threshold_minutes", 1440)

    n_folds = 0
    if evaluation_result.model_order:
        n_folds = len(evaluation_result.model_evaluations[evaluation_result.model_order[0]].fold_metrics)

    lines = [
        "# Phase 6 — Evaluation Report",
        "",
        "**Protocol:** Leave-One-Bearing-Out (LOBO) CV, inherited verbatim from Phase 5 fold assignments.",
        f"**Near-failure threshold:** RUL ≤ {near_failure_threshold} minutes.",
        "",
        f"> **Honesty guardrail (architecture §10.4):** every metric below is the average/spread of "
        f"**exactly {n_folds} evaluation run{'s' if n_folds != 1 else ''} (n={n_folds} units)**. "
        "These numbers carry **no classical confidence interval** and are "
        "**methodology validation, not fleet accuracy**.",
        "",
        "---",
        "",
        f"## {n_folds}-Fold Summary (all models)",
        "",
        "| Model | Mean MAE (min) | Fold Spread (min) | Stability Ratio | Mean Signed Error | Late-Prediction Rate |",
        "|-------|---------------:|-------------------:|-----------------:|-------------------:|----------------------:|",
    ]

    for model_name in evaluation_result.model_order:
        ev = evaluation_result.model_evaluations[model_name]
        mean_signed = float(np.mean([fm.metrics.signed_error for fm in ev.fold_metrics.values()]))
        mean_late_rate = float(np.mean([fm.metrics.late_prediction_rate for fm in ev.fold_metrics.values()]))
        lines.append(
            f"| {model_name} | {ev.mean_mae:.2f} | {ev.mae_spread:.2f} | {ev.stability_ratio:.2f} | "
            f"{mean_signed:+.2f} | {mean_late_rate:.1%} |"
        )

    lines += ["", "---", "", "## Per-Fold Detail", ""]

    for model_name in evaluation_result.model_order:
        ev = evaluation_result.model_evaluations[model_name]
        lines.append(f"### {model_name}")
        lines.append("")
        lines.append(
            "| Fold | Train Bearing | Test Bearing | MAE | RMSE | R² | Near-Failure MAE | "
            "Train MAE | Overfitting Ratio |"
        )
        lines.append("|------|---------------|--------------|----:|-----:|---:|------------------:|----------:|-------------------:|")
        for fold_id, fm in sorted(ev.fold_metrics.items()):
            m = fm.metrics
            lines.append(
                f"| {fold_id} | {fm.train_bearing} | {fm.test_bearing} | {m.mae:.2f} | {m.rmse:.2f} | "
                f"{m.r2:.3f} | {m.near_failure_mae:.2f} | {m.train_mae:.2f} | {m.overfitting_ratio:.2f} |"
            )
        lines.append("")

    lines += ["---", "", "## Error Analysis (architecture §10.6)", ""]

    for model_name in evaluation_result.model_order:
        ev = evaluation_result.model_evaluations[model_name]
        lines.append(f"### {model_name}")
        lines.append("")
        for fold_id, fm in sorted(ev.fold_metrics.items()):
            bias_direction = "OVER-predicts (late warning, dangerous)" if fm.metrics.signed_error > 0 else "UNDER-predicts (early warning, conservative)"
            lines.append(f"**{fold_id}** (test bearing: {fm.test_bearing})")
            lines.append(f"- Systematic bias: {bias_direction} — mean signed error = {fm.metrics.signed_error:+.2f} min")
            lines.append(f"- Late-prediction rate: {fm.metrics.late_prediction_rate:.1%}")
            channel_str = ", ".join(f"channel {ch}: MAE={mae:.2f}" for ch, mae in sorted(fm.channel_specific_mae.items()))
            lines.append(f"- Channel-specific MAE: {channel_str}")
            lines.append(f"- Largest residuals (top {len(fm.largest_residuals)}):")
            lines.append("")
            lines.append("  | unit_id | channel | timestamp | actual RUL | predicted RUL | abs residual |")
            lines.append("  |---------|---------|-----------|-----------:|---------------:|-------------:|")
            for _, row in fm.largest_residuals.iterrows():
                ch_val = row["channel"] if "channel" in row else "-"
                unit_val = row["unit_id"] if "unit_id" in row else row.get("bearing_id", "-")
                lines.append(
                    f"  | {unit_val} | {ch_val} | {row['timestamp']} | "
                    f"{row['actual_rul_minutes']:.2f} | {row['predicted_rul_minutes']:.2f} | {row['abs_residual_minutes']:.2f} |"
                )
            lines.append("")

    lines += [
        "---",
        "",
        "_Generated by Phase 6 pipeline. Do not edit manually._",
    ]

    report_md = "\n".join(lines)
    logger.info("Evaluation report generated (%d models)", len(evaluation_result.model_order))
    return report_md


def save_evaluation_report(report_md: str, output_path: Path, logger: logging.Logger) -> None:
    """Atomically write `evaluation_report.md`."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".md.tmp")
    try:
        tmp_path.write_text(report_md, encoding="utf-8")
        tmp_path.replace(output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    logger.info("Evaluation report saved: %s", output_path)
