"""Global feature importance: permutation (primary, model-agnostic) + native
model importance (cross-check) — architecture §11.

All importance is computed on HELD-OUT folds (out-of-sample), never on
training rows, to avoid over-crediting memorized features. SHAP is handled
separately in `shap_analysis.py` and is interpretation-only (architecture §11a).
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import get_rtf_units
from src.model_dev.cross_validation import CVResults, FoldResult
from src.model_dev.preprocessing import transform_features
from src.model_dev.shap_analysis import ShapResult


@dataclass
class FeatureImportanceResult:
    """Global importance rankings per model (averaged across folds where applicable)."""

    permutation_importance: Dict[str, pd.DataFrame] = field(default_factory=dict)
    native_importance: Dict[str, pd.DataFrame] = field(default_factory=dict)


def compute_permutation_importance(
    fold_result: FoldResult,
    original_test_df: pd.DataFrame,
    feature_columns: List[str],
    seed: int,
) -> pd.DataFrame:
    """Permutation importance for one model on one held-out fold.

    Args:
        fold_result: FoldResult containing the fitted model and preprocessor.
        original_test_df: The raw (unpreprocessed) held-out fold DataFrame,
            used to re-derive X_test via the fold's fitted preprocessor.
        feature_columns: Feature column names.
        seed: Global reproducibility seed (for permutation shuffling).

    Returns:
        DataFrame with columns [feature, importance_mean, importance_std], sorted descending.
    """
    if fold_result.preprocessor is None:
        # Constant baselines (mean/median) have no feature dependency.
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])

    X_test = transform_features(original_test_df, fold_result.preprocessor)
    y_test = fold_result.y_test_true

    result = permutation_importance(
        fold_result.fitted_model, X_test, y_test,
        n_repeats=10, random_state=seed, scoring="neg_mean_absolute_error",
    )
    df = pd.DataFrame({
        "feature": feature_columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return df


def compute_native_importance(fold_result: FoldResult, feature_columns: List[str]) -> pd.DataFrame:
    """Native model importance (RF/XGBoost gain, Linear standardized coefficients).

    Args:
        fold_result: FoldResult containing the fitted model.
        feature_columns: Feature column names.

    Returns:
        DataFrame with columns [feature, importance], sorted descending. Empty
        for constant baselines (no native importance concept applies).
    """
    model = fold_result.fitted_model
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_)
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    df = pd.DataFrame({"feature": feature_columns, "importance": values})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def compute_feature_importance(
    cv_results: CVResults,
    training_dataset: pd.DataFrame,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> FeatureImportanceResult:
    """Compute global permutation + native importance for every model, averaged across folds.

    Args:
        cv_results: Output of `cross_validation.run_lobo_cv`.
        training_dataset: Full training dataset (used to re-derive raw test-fold rows).
        config: Full project configuration dictionary.
        logger: Pipeline logger.

    Returns:
        FeatureImportanceResult with per-model importance tables.
    """
    seed = config.get("model_dev", {}).get("random_seed", 42)
    feature_columns = cv_results.feature_columns
    result = FeatureImportanceResult()

    for model_name in cv_results.model_order:
        perm_frames: List[pd.DataFrame] = []
        native_frames: List[pd.DataFrame] = []

        for fold_id, fold_result in cv_results.results[model_name].items():
            test_units = fold_result.test_bearings  # qualified unit_ids, despite the field name
            test_df_raw = training_dataset[training_dataset["unit_id"].isin(test_units)]
            # Align raw rows to the same order as fold_result.grain_df (both built from the same filter)
            perm_df = compute_permutation_importance(fold_result, test_df_raw, feature_columns, seed)
            if not perm_df.empty:
                perm_frames.append(perm_df.set_index("feature")["importance_mean"])

            native_df = compute_native_importance(fold_result, feature_columns)
            if not native_df.empty:
                native_frames.append(native_df.set_index("feature")["importance"])

        if perm_frames:
            avg_perm = pd.concat(perm_frames, axis=1).mean(axis=1).sort_values(ascending=False)
            result.permutation_importance[model_name] = avg_perm.reset_index().rename(
                columns={0: "importance_mean", "index": "feature"}
            )
        if native_frames:
            avg_native = pd.concat(native_frames, axis=1).mean(axis=1).sort_values(ascending=False)
            result.native_importance[model_name] = avg_native.reset_index().rename(
                columns={0: "importance", "index": "feature"}
            )

        logger.info("Feature importance computed for %s", model_name)

    return result


def generate_feature_importance_report(
    importance_result: FeatureImportanceResult,
    shap_results: Dict[str, ShapResult],
    config: Dict[str, Any],
    logger: logging.Logger,
) -> str:
    """Render `feature_importance_report.md` overlaying permutation, native, and SHAP views.

    A feature is called a "trusted driver" only if it ranks highly across
    methods (architecture §11). Divergence between native and permutation
    importance is flagged as a caution, not silently reconciled.

    SHAP GOVERNANCE (architecture §11a): the observations in this report are
    for human review only. They are never fed back into model_selection.py —
    model_selection.py has no code path that reads this report or its inputs.
    """
    lines = [
        "# Phase 6 — Feature Importance Report",
        "",
        "> **SHAP governance (architecture §11a):** SHAP is used ONLY for model "
        "interpretation. It is computed after model selection and is never used to "
        "choose between candidate models. Observations below are for human review only.",
        "",
        "---",
        "",
    ]

    for model_name in importance_result.permutation_importance.keys() | importance_result.native_importance.keys():
        lines.append(f"## {model_name}")
        lines.append("")

        perm_df = importance_result.permutation_importance.get(model_name)
        native_df = importance_result.native_importance.get(model_name)

        if perm_df is not None and not perm_df.empty:
            lines.append("### Permutation Importance (primary, out-of-sample)")
            lines.append("")
            lines.append("| Feature | Importance (mean MAE increase) |")
            lines.append("|---------|-------------------------------:|")
            for _, row in perm_df.head(10).iterrows():
                lines.append(f"| {row['feature']} | {row['importance_mean']:.4f} |")
            lines.append("")

        if native_df is not None and not native_df.empty:
            lines.append("### Native Model Importance (cross-check)")
            lines.append("")
            lines.append("| Feature | Native Importance |")
            lines.append("|---------|-------------------:|")
            for _, row in native_df.head(10).iterrows():
                lines.append(f"| {row['feature']} | {row['importance']:.4f} |")
            lines.append("")

            if perm_df is not None and not perm_df.empty:
                top_perm = set(perm_df.head(5)["feature"])
                top_native = set(native_df.head(5)["feature"])
                overlap = top_perm & top_native
                if len(overlap) < 3:
                    lines.append(
                        f"**Caution:** top-5 permutation vs. native importance overlap is only "
                        f"{len(overlap)}/5 features — native gain may be biased toward "
                        "high-cardinality features (architecture §11)."
                    )
                    lines.append("")

        shap_result = shap_results.get(model_name)
        if shap_result is not None:
            lines.append("### SHAP Global Ranking (mean |SHAP|)")
            lines.append("")
            lines.append("| Feature | Mean |SHAP| |")
            lines.append("|---------|------------:|")
            for _, row in shap_result.global_ranking.head(10).iterrows():
                lines.append(f"| {row['feature']} | {row['mean_abs_shap']:.4f} |")
            lines.append("")

            lines.append("### SHAP Local Examples (near-failure region)")
            lines.append("")
            for ex in shap_result.local_examples[:5]:
                unit_label = ex.get("unit_id") or ex["bearing_id"]
                lines.append(
                    f"- **{unit_label}** (fold {ex['fold_id']}, channel {ex['channel']}, "
                    f"actual RUL={ex['actual_rul_minutes']:.1f} min): "
                    + ", ".join(f"{feat}={val:+.2f}" for feat, val in ex["top_shap_attributions"])
                )
            lines.append("")

    n_rtf_units = len(get_rtf_units(config))
    lines += [
        "---",
        "",
        f"**n={n_rtf_units} caveat:** importance rankings are derived from {n_rtf_units} "
        "fold(s) and are indicative of mechanism, not statistically stable rankings "
        "(architecture §11).",
        "",
        "_Generated by Phase 6 pipeline. Do not edit manually._",
    ]

    report_md = "\n".join(lines)
    logger.info("Feature importance report generated")
    return report_md


def save_feature_importance_report(report_md: str, output_path: Path, logger: logging.Logger) -> None:
    """Atomically write `feature_importance_report.md`."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".md.tmp")
    try:
        tmp_path.write_text(report_md, encoding="utf-8")
        tmp_path.replace(output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    logger.info("Feature importance report saved: %s", output_path)
