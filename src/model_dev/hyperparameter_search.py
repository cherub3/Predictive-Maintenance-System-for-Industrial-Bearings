"""Optional, leakage-safe, constrained hyperparameter search (architecture §9).

Disabled by default. With only two RTF units, the outer LOBO loop already
consumes both — there is no third unit to form a leakage-free inner
validation set. When enabled, the inner validation slice is carved from the
EARLY LIFE of the TRAINING bearing only (past-only), so it never touches the
held-out test bearing. This is strictly weaker evidence than the untuned
default and is reported as such.
"""

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import pandas as pd
from xgboost import XGBRegressor

from src.model_dev.metrics import compute_mae
from src.model_dev.model_trainer import TRAINER_REGISTRY
from src.model_dev.preprocessing import prepare_fold_matrices


@dataclass
class HyperparameterSearchRecord:
    """Full record of a hyperparameter search run (or a no-op if disabled)."""

    enabled: bool
    model_name: str
    grid: Dict[str, List[Any]] = field(default_factory=dict)
    selected_params: Dict[str, Any] = field(default_factory=dict)
    inner_val_mae: float = float("nan")
    n_combinations_tried: int = 0


def _split_early_life(train_df: pd.DataFrame, early_life_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the training bearing's timeline into a PAST inner-train and a FUTURE inner-val.

    Temporal validation must never let the model see data that occurs AFTER the
    validation window. The inner training set is the EARLIEST `1 - early_life_fraction`
    of the bearing's own timeline (the past); the inner validation set is the latest
    `early_life_fraction` (the future, including the near-failure region). A previous
    revision inverted this — it trained on late life and validated on early life — so
    hyperparameters were chosen from a model that had already seen the future.

    Args:
        train_df: The outer training fold's rows (a single RTF bearing).
        early_life_fraction: Fraction of the timeline (measured from the END) held
            out as the inner validation set. Named for backward compatibility; it is
            the size of the future validation slice.

    Returns:
        (inner_train_df, inner_val_df) with inner_train strictly earlier than inner_val.
    """
    ts_min = train_df["timestamp"].min()
    ts_max = train_df["timestamp"].max()
    # Cutoff placed so the LAST `early_life_fraction` of the timeline is validation.
    cutoff = ts_min + (1.0 - early_life_fraction) * (ts_max - ts_min)
    inner_train_df = train_df[train_df["timestamp"] <= cutoff].copy()
    inner_val_df = train_df[train_df["timestamp"] > cutoff].copy()

    # Guarantee no temporal overlap: training is strictly in the past of validation.
    if not inner_train_df.empty and not inner_val_df.empty:
        assert inner_train_df["timestamp"].max() < inner_val_df["timestamp"].min(), (
            "Temporal split invariant violated: inner-train must precede inner-val"
        )
    return inner_train_df, inner_val_df


def _iter_grid(grid: Dict[str, List[Any]]):
    """Yield every combination of a small, hand-declared hyperparameter grid."""
    keys = list(grid.keys())
    for combo in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, combo))


def search_hyperparameters(
    model_name: str,
    train_df: pd.DataFrame,
    feature_columns: List[str],
    config: Dict[str, Any],
    seed: int,
    logger: logging.Logger,
) -> HyperparameterSearchRecord:
    """Select hyperparameters for a candidate model.

    Default path (search disabled): returns the fixed, config-declared
    defaults from `model_dev.model_params.<model_name>` verbatim.

    Optional path (search enabled): performs a small grid search using a
    within-train-unit temporal inner split (early life = inner-train, late life = inner-val).

    Args:
        model_name: 'random_forest' or 'xgboost'.
        train_df: The OUTER training fold's rows (never includes the held-out bearing).
        feature_columns: Feature column names.
        config: Full project configuration dictionary.
        seed: Global reproducibility seed.
        logger: Pipeline logger.

    Returns:
        HyperparameterSearchRecord documenting what was selected and why.
    """
    model_dev_cfg = config.get("model_dev", {})
    hp_cfg = model_dev_cfg.get("hyperparameter_search", {})
    default_params = model_dev_cfg.get("model_params", {}).get(model_name, {})

    if not hp_cfg.get("enabled", False):
        return HyperparameterSearchRecord(
            enabled=False, model_name=model_name, selected_params=dict(default_params)
        )

    logger.warning(
        "Hyperparameter search ENABLED for %s — this is a within-train-unit-slice result "
        "and is WEAKER EVIDENCE than the untuned default (architecture §9).",
        model_name,
    )

    early_life_fraction = hp_cfg.get("early_life_fraction", 0.3)
    early_stopping_rounds = hp_cfg.get("early_stopping_rounds", 50)
    grid = hp_cfg.get("grid", {}).get(model_name, {})
    if not grid:
        logger.warning("No grid declared for %s; falling back to fixed defaults", model_name)
        return HyperparameterSearchRecord(
            enabled=True, model_name=model_name, selected_params=dict(default_params)
        )

    inner_train_df, inner_val_df = _split_early_life(train_df, early_life_fraction)
    X_inner_train, X_inner_val, _ = prepare_fold_matrices(inner_train_df, inner_val_df, feature_columns, config)
    y_inner_train = inner_train_df["rul_minutes"].to_numpy(dtype=float)
    y_inner_val = inner_val_df["rul_minutes"].to_numpy(dtype=float)

    best_params: Dict[str, Any] = dict(default_params)
    best_mae = float("inf")
    n_tried = 0

    for combo in _iter_grid(grid):
        n_tried += 1
        params = dict(default_params)
        params.update(combo)
        if model_name == "xgboost":
            model = XGBRegressor(
                random_state=seed,
                n_jobs=-1,
                objective="reg:squarederror",
                early_stopping_rounds=early_stopping_rounds,
                eval_metric="mae",
                **{k: v for k, v in params.items() if k not in ("early_stopping_rounds", "eval_metric")},
            )
            model.fit(X_inner_train, y_inner_train, eval_set=[(X_inner_val, y_inner_val)], verbose=False)
        else:
            trainer_fn = TRAINER_REGISTRY[model_name]
            model = trainer_fn(X_inner_train, y_inner_train, params, seed)

        preds = model.predict(X_inner_val)
        mae = compute_mae(y_inner_val, preds)
        if mae < best_mae:
            best_mae = mae
            best_params = params

    logger.info(
        "Hyperparameter search for %s: tried %d combinations, selected %s (inner-val MAE=%.2f)",
        model_name, n_tried, best_params, best_mae,
    )

    return HyperparameterSearchRecord(
        enabled=True,
        model_name=model_name,
        grid=grid,
        selected_params=best_params,
        inner_val_mae=float(best_mae),
        n_combinations_tried=n_tried,
    )
