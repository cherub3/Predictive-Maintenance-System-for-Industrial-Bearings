"""Candidate model trainer: Random Forest and XGBoost (architecture §8.1).

Each candidate is trained on an already-preprocessed (fit-on-train-only)
feature matrix; this module performs no data access of its own and cannot
introduce leakage — it only fits an estimator on whatever matrix it is given.
"""

from typing import Any, Dict

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def train_random_forest(
    X_train: np.ndarray, y_train: np.ndarray, params: Dict[str, Any], seed: int
) -> RandomForestRegressor:
    """Fit a Random Forest regressor with fixed, config-declared hyperparameters.

    Args:
        X_train: Preprocessed training-fold feature matrix.
        y_train: Training-fold rul_minutes values.
        params: Hyperparameters from `config.model_dev.model_params.random_forest`.
        seed: Global reproducibility seed.

    Returns:
        Fitted RandomForestRegressor.
    """
    model = RandomForestRegressor(random_state=seed, n_jobs=-1, **params)
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: np.ndarray, y_train: np.ndarray, params: Dict[str, Any], seed: int
) -> XGBRegressor:
    """Fit an XGBoost regressor with fixed, config-declared hyperparameters.

    Args:
        X_train: Preprocessed training-fold feature matrix.
        y_train: Training-fold rul_minutes values.
        params: Hyperparameters from `config.model_dev.model_params.xgboost`.
        seed: Global reproducibility seed.

    Returns:
        Fitted XGBRegressor.
    """
    model = XGBRegressor(random_state=seed, n_jobs=-1, objective="reg:squarederror", **params)
    model.fit(X_train, y_train)
    return model


TRAINER_REGISTRY = {
    "random_forest": train_random_forest,
    "xgboost": train_xgboost,
}


def train_candidate(
    model_name: str, X_train: np.ndarray, y_train: np.ndarray, params: Dict[str, Any], seed: int
) -> Any:
    """Dispatch to the correct trainer function for a candidate model name.

    Args:
        model_name: One of 'random_forest', 'xgboost'.
        X_train: Preprocessed training-fold feature matrix.
        y_train: Training-fold rul_minutes values.
        params: Hyperparameters for this model.
        seed: Global reproducibility seed.

    Returns:
        Fitted estimator.

    Raises:
        ValueError: If model_name is not a registered candidate.
    """
    if model_name not in TRAINER_REGISTRY:
        raise ValueError(
            f"Unknown candidate model '{model_name}'. In-scope candidates (architecture §8.1): "
            f"{list(TRAINER_REGISTRY.keys())}. LightGBM/CatBoost/deep learning/stacking are "
            "intentionally excluded (architecture §8.2)."
        )
    return TRAINER_REGISTRY[model_name](X_train, y_train, params, seed)
