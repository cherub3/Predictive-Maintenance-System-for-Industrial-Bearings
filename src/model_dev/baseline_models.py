"""Mandatory naive baselines, evaluated before any advanced model (architecture §7).

Every baseline is fit on the training fold's `rul_minutes` values only — the
mean/median predictors must never see the held-out bearing's rows, echoing
the same fit-on-train-only discipline as feature scaling (Phase 5 L-07).
"""

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from sklearn.linear_model import LinearRegression


@dataclass
class ConstantPredictor:
    """A predictor that always returns a fixed constant value, fit on train-fold labels only."""

    value: float

    def predict(self, n_or_X: Any) -> np.ndarray:
        """Return the fitted constant, broadcast to the requested length.

        Args:
            n_or_X: Either an integer row count or an array-like whose length
                determines the number of predictions.

        Returns:
            1D numpy array of length n filled with the fitted constant.
        """
        n = n_or_X if isinstance(n_or_X, int) else len(n_or_X)
        return np.full(n, self.value, dtype=float)


def fit_mean_baseline(y_train: np.ndarray) -> ConstantPredictor:
    """Fit the mean-RUL baseline on the training fold only (architecture §7, order 0)."""
    return ConstantPredictor(value=float(np.mean(y_train)))


def fit_median_baseline(y_train: np.ndarray) -> ConstantPredictor:
    """Fit the median-RUL baseline on the training fold only (architecture §7, order 1)."""
    return ConstantPredictor(value=float(np.median(y_train)))


def fit_linear_baseline(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """Fit the Linear Regression baseline on the training fold only (architecture §7, order 2).

    Args:
        X_train: Preprocessed (imputed + scaled) training-fold feature matrix.
        y_train: Training-fold rul_minutes values.

    Returns:
        Fitted sklearn LinearRegression.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def get_enabled_baselines(config: Dict[str, Any]) -> Dict[str, bool]:
    """Return which baselines are enabled per config (architecture §18)."""
    return dict(config.get("model_dev", {}).get("baselines", {"mean": True, "median": True, "linear": True}))
