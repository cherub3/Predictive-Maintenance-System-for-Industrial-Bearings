"""LOBO cross-validation orchestrator (architecture §12, §13).

Consumes the Phase 5 `fold_assignments.parquet` verbatim — never recomputes
splits. For each fold, preprocessing is fit on the training fold only, every
enabled baseline and candidate model is trained, and both train-fold and
test-fold predictions are collected (train-fold predictions are needed for
the overfitting check, architecture §10.5).
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.model_dev.baseline_models import (
    fit_linear_baseline,
    fit_mean_baseline,
    fit_median_baseline,
    get_enabled_baselines,
)
from src.model_dev.data_loader import get_fold_bearings
from src.model_dev.hyperparameter_search import HyperparameterSearchRecord, search_hyperparameters
from src.model_dev.model_trainer import train_candidate
from src.model_dev.preprocessing import FittedPreprocessor, maybe_aggregate, prepare_fold_matrices

BASELINE_ORDER = ["mean", "median", "linear"]
CANDIDATE_ORDER = ["random_forest", "xgboost"]


@dataclass
class FoldResult:
    """Result of training and evaluating one model on one LOBO fold."""

    fold_id: str
    train_bearings: List[str]  # qualified unit_ids, despite the field name (call-site compatibility)
    test_bearings: List[str]  # qualified unit_ids, despite the field name (call-site compatibility)
    grain_df: pd.DataFrame
    y_test_true: np.ndarray
    y_test_pred: np.ndarray
    y_train_true: np.ndarray
    y_train_pred: np.ndarray
    fitted_model: Any
    preprocessor: Optional[FittedPreprocessor]
    hp_record: Optional[HyperparameterSearchRecord] = None


@dataclass
class CVResults:
    """All fold results for all evaluated models (baselines + candidates)."""

    results: Dict[str, Dict[str, FoldResult]] = field(default_factory=dict)
    feature_columns: List[str] = field(default_factory=list)
    model_order: List[str] = field(default_factory=list)
    seed: int = 42


def _get_model_order(config: Dict[str, Any]) -> List[str]:
    """Return the enabled model list in the architecture-mandated progression order (§7)."""
    baselines = get_enabled_baselines(config)
    candidates = config.get("model_dev", {}).get("candidates", {})
    order = [m for m in BASELINE_ORDER if baselines.get(m, False)]
    order += [m for m in CANDIDATE_ORDER if candidates.get(m, False)]
    return order


def run_lobo_cv(
    training_dataset: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    feature_columns: List[str],
    config: Dict[str, Any],
    logger: logging.Logger,
) -> CVResults:
    """Run the full LOBO CV loop over every enabled baseline and candidate model.

    Args:
        training_dataset: Phase 5 `training_dataset.parquet` (RTF-only).
        fold_assignments: Phase 5 `fold_assignments.parquet`.
        feature_columns: Feature column names (from the Phase 5 contract).
        config: Full project configuration dictionary.
        logger: Pipeline logger.

    Returns:
        CVResults containing every model's per-fold predictions and fitted artifacts.
    """
    model_dev_cfg = config.get("model_dev", {})
    seed = model_dev_cfg.get("random_seed", 42)
    fold_role_cols = get_fold_role_columns_from_dataset(training_dataset, fold_assignments)

    model_order = _get_model_order(config)
    logger.info("LOBO CV model progression (architecture §7): %s", model_order)

    results: Dict[str, Dict[str, FoldResult]] = {m: {} for m in model_order}

    for role_col in fold_role_cols:
        fold_id = role_col.replace("_role", "")
        train_units, test_units = get_fold_bearings(fold_assignments, role_col)
        if set(train_units) & set(test_units):
            raise ValueError(f"Unit-disjointness violated in {role_col}: overlap between train/test units")

        logger.info("Fold %s: train=%s, test=%s", fold_id, train_units, test_units)

        # Filtered by unit_id (qualified dataset_id::bearing_id), NOT bare bearing_id —
        # bearing_id alone is ambiguous across datasets (e.g. Set 1's bearing_3 and Set
        # 3's bearing_3 are both RTF units and can both be present in training_dataset).
        train_df = training_dataset[training_dataset["unit_id"].isin(train_units)].copy()
        test_df = training_dataset[training_dataset["unit_id"].isin(test_units)].copy()

        train_df = maybe_aggregate(train_df, feature_columns, config, logger)
        test_df = maybe_aggregate(test_df, feature_columns, config, logger)

        y_train = train_df["rul_minutes"].to_numpy(dtype=float)
        y_test = test_df["rul_minutes"].to_numpy(dtype=float)

        grain_cols = [c for c in ("unit_id", "dataset_id", "bearing_id", "channel", "timestamp") if c in test_df.columns]
        grain_df = test_df[grain_cols].reset_index(drop=True)

        # Preprocessing fit on the TRAIN FOLD ONLY — shared feature matrix for
        # linear/RF/XGBoost. Constant baselines do not use it.
        X_train, X_test, preprocessor = prepare_fold_matrices(train_df, test_df, feature_columns, config)

        for model_name in model_order:
            if model_name == "mean":
                predictor: Any = fit_mean_baseline(y_train)
                y_train_pred = predictor.predict(len(y_train))
                y_test_pred = predictor.predict(len(y_test))
                fold_preprocessor = None
                hp_record = None
            elif model_name == "median":
                predictor = fit_median_baseline(y_train)
                y_train_pred = predictor.predict(len(y_train))
                y_test_pred = predictor.predict(len(y_test))
                fold_preprocessor = None
                hp_record = None
            elif model_name == "linear":
                predictor = fit_linear_baseline(X_train, y_train)
                y_train_pred = predictor.predict(X_train)
                y_test_pred = predictor.predict(X_test)
                fold_preprocessor = preprocessor
                hp_record = None
            elif model_name in CANDIDATE_ORDER:
                hp_record = search_hyperparameters(model_name, train_df, feature_columns, config, seed, logger)
                predictor = train_candidate(model_name, X_train, y_train, hp_record.selected_params, seed)
                y_train_pred = predictor.predict(X_train)
                y_test_pred = predictor.predict(X_test)
                fold_preprocessor = preprocessor
            else:
                raise ValueError(f"Unrecognized model in progression: {model_name}")

            results[model_name][fold_id] = FoldResult(
                fold_id=fold_id,
                train_bearings=train_units,
                test_bearings=test_units,
                grain_df=grain_df,
                y_test_true=y_test,
                y_test_pred=np.asarray(y_test_pred, dtype=float),
                y_train_true=y_train,
                y_train_pred=np.asarray(y_train_pred, dtype=float),
                fitted_model=predictor,
                preprocessor=fold_preprocessor,
                hp_record=hp_record,
            )

            logger.info(
                "  [%s] %s: train n=%d, test n=%d — fitted ✓",
                fold_id, model_name, len(y_train), len(y_test),
            )

    return CVResults(results=results, feature_columns=feature_columns, model_order=model_order, seed=seed)


def get_fold_role_columns_from_dataset(training_dataset: pd.DataFrame, fold_assignments: pd.DataFrame) -> List[str]:
    """Return the fold-role columns actually present, sorted numerically (fold_1, fold_2,
    ..., fold_10 — not lexicographically, which would order fold_10 before fold_2).

    Defensive helper: reads from the dataset itself rather than trusting the
    contract alone, so a partially-generated fold_assignments file cannot
    silently produce fewer folds than expected without being visible in logs.
    """
    import re
    cols = [c for c in training_dataset.columns if re.fullmatch(r"fold_\d+_role", c)]
    return sorted(cols, key=lambda c: int(c.split("_")[1]))
