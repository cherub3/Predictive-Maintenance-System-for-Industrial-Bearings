"""SHAP interpretability — global + local, tree models only (architecture §11).

GOVERNANCE (architecture §11a): SHAP is used ONLY for model interpretation.
It is computed AFTER model_selection (see main.py pipeline order) and its
output is never passed to `model_selection.py` — this is enforced by data
flow, not convention: `select_best_model` takes only an `EvaluationResult`
as input and has no parameter through which a SHAP value could reach it.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import shap

from src.model_dev.cross_validation import CVResults
from src.model_dev.preprocessing import transform_features

TREE_MODELS = frozenset({"random_forest", "xgboost"})


@dataclass
class ShapResult:
    """Global ranking + local near-failure examples for one model (tree models only)."""

    global_ranking: pd.DataFrame = field(default_factory=pd.DataFrame)
    local_examples: List[Dict[str, Any]] = field(default_factory=list)


def _global_shap_ranking(shap_values: np.ndarray, feature_columns: List[str]) -> pd.DataFrame:
    """Mean |SHAP| ranking, descending (architecture §11)."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_columns, "mean_abs_shap": mean_abs})
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def _local_shap_examples(
    shap_values: np.ndarray,
    X_test_df: pd.DataFrame,
    y_test: np.ndarray,
    feature_columns: List[str],
    near_failure_threshold: float,
    n_examples: int,
) -> List[Dict[str, Any]]:
    """Local per-prediction SHAP attribution, drawn from the near-failure region (architecture §11)."""
    near_failure_mask = y_test <= near_failure_threshold
    candidate_idx = np.where(near_failure_mask)[0]
    if len(candidate_idx) == 0:
        candidate_idx = np.arange(len(y_test))
    selected_idx = candidate_idx[:n_examples]

    examples = []
    for idx in selected_idx:
        row_grain = X_test_df.iloc[idx]
        attribution = dict(zip(feature_columns, shap_values[idx].tolist()))
        top_features = sorted(attribution.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
        examples.append({
            "unit_id": row_grain.get("unit_id", row_grain.get("bearing_id")),
            "bearing_id": row_grain.get("bearing_id"),
            "channel": row_grain.get("channel"),
            "timestamp": str(row_grain.get("timestamp")),
            "actual_rul_minutes": float(y_test[idx]),
            "top_shap_attributions": top_features,
        })
    return examples


def compute_shap_for_model(
    cv_results: CVResults,
    training_dataset: pd.DataFrame,
    model_name: str,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> Optional[ShapResult]:
    """Compute SHAP global ranking + local near-failure examples for one model, averaged across folds.

    Args:
        cv_results: Output of `cross_validation.run_lobo_cv`.
        training_dataset: Full training dataset (used to re-derive raw test-fold rows).
        model_name: Model to analyze. Must be a tree model ('random_forest', 'xgboost');
            returns None for linear/baseline models (architecture §11: TreeExplainer only).
        config: Full project configuration dictionary.
        logger: Pipeline logger.

    Returns:
        ShapResult, or None if model_name is not a tree model.
    """
    if model_name not in TREE_MODELS:
        logger.info("SHAP skipped for %s (TreeExplainer scope is tree models only, architecture §11)", model_name)
        return None

    feature_columns = cv_results.feature_columns
    metrics_cfg = config.get("model_dev", {}).get("metrics", {})
    near_failure_threshold = metrics_cfg.get("near_failure_threshold_minutes", 1440)
    top_n = metrics_cfg.get("top_n_residuals", 10)

    ranking_frames: List[pd.DataFrame] = []
    all_local_examples: List[Dict[str, Any]] = []

    for fold_id, fold_result in cv_results.results[model_name].items():
        test_df_raw = training_dataset[training_dataset["unit_id"].isin(fold_result.test_bearings)].reset_index(drop=True)
        X_test = transform_features(test_df_raw, fold_result.preprocessor)

        explainer = shap.TreeExplainer(fold_result.fitted_model)
        shap_values = explainer.shap_values(X_test)

        ranking = _global_shap_ranking(shap_values, feature_columns)
        ranking_frames.append(ranking.set_index("feature")["mean_abs_shap"])

        local_examples = _local_shap_examples(
            shap_values, test_df_raw, fold_result.y_test_true, feature_columns,
            near_failure_threshold, top_n,
        )
        for ex in local_examples:
            ex["fold_id"] = fold_id
        all_local_examples.extend(local_examples)

        logger.info("SHAP computed for %s / %s (%d test rows)", model_name, fold_id, len(test_df_raw))

    avg_ranking = pd.concat(ranking_frames, axis=1).mean(axis=1).sort_values(ascending=False)
    global_ranking = avg_ranking.reset_index().rename(columns={0: "mean_abs_shap", "index": "feature"})

    return ShapResult(global_ranking=global_ranking, local_examples=all_local_examples)


def compute_all_shap(
    cv_results: CVResults, training_dataset: pd.DataFrame, config: Dict[str, Any], logger: logging.Logger
) -> Dict[str, ShapResult]:
    """Compute SHAP for every tree model in the evaluated progression.

    Returns:
        Dict mapping model_name -> ShapResult, only for tree models present in cv_results.
    """
    results: Dict[str, ShapResult] = {}
    for model_name in cv_results.model_order:
        if model_name in TREE_MODELS:
            shap_result = compute_shap_for_model(cv_results, training_dataset, model_name, config, logger)
            if shap_result is not None:
                results[model_name] = shap_result
    return results
