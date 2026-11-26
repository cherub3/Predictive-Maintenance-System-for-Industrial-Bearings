"""RUL regression metrics (architecture §10.2, §10.3).

MAE is primary. RMSE, R², near-failure MAE, and signed-error/late-rate are
secondary/diagnostic. MAPE/sMAPE and classification metrics are explicitly
REJECTED (architecture §10.3) because RUL reaches exactly zero at failure,
which makes any percentage-based metric divide by (near-)zero precisely in
the safety-critical region.
"""

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REJECTED_METRICS = frozenset({"mape", "smape", "accuracy", "f1", "auc", "precision", "recall"})


def reject_disallowed_metric(name: str) -> None:
    """Raise if a metric explicitly excluded by the architecture is requested.

    Args:
        name: Lowercased metric name to check.

    Raises:
        ValueError: If `name` is in the architecturally-rejected metric set.
    """
    if name.lower() in REJECTED_METRICS:
        raise ValueError(
            f"Metric '{name}' is explicitly rejected by the Phase 6 architecture (§10.3). "
            "MAPE/sMAPE divide by (near-)zero RUL at failure; classification metrics "
            "discard the regression signal and pre-empt the Phase 7 decision policy."
        )


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error in minutes — the primary metric (architecture §10.2)."""
    return float(mean_absolute_error(y_true, y_pred))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error in minutes — secondary metric."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² — secondary diagnostic metric, interpreted cautiously at n=2."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def compute_near_failure_mae(y_true: np.ndarray, y_pred: np.ndarray, threshold_minutes: float) -> float:
    """MAE restricted to the near-failure region (RUL <= threshold) — architecture §10.2, §10.6."""
    mask = y_true <= threshold_minutes
    if not np.any(mask):
        return float("nan")
    return compute_mae(y_true[mask], y_pred[mask])


def compute_signed_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean signed error (predicted - actual). Positive = over-predicts RUL = late warning (dangerous).

    Architecture §10.2, §10.6: over-predicting RUL is more dangerous than
    under-predicting in a maintenance context (a late warning delays action).
    """
    return float(np.mean(y_pred - y_true))


def compute_late_prediction_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions where predicted RUL > actual RUL (dangerously late warning)."""
    return float(np.mean(y_pred > y_true))


@dataclass
class MetricBundle:
    """All computed metrics for one model on one fold."""

    mae: float
    rmse: float
    r2: float
    near_failure_mae: float
    signed_error: float
    late_prediction_rate: float
    train_mae: float
    overfitting_ratio: float  # train_mae / test_mae; low value => possible overfitting (architecture §10.5)


def compute_all_metrics(
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    y_train_true: np.ndarray,
    y_train_pred: np.ndarray,
    config: Dict[str, Any],
) -> MetricBundle:
    """Compute the full metric bundle for one model on one LOBO fold.

    Args:
        y_test_true: Ground-truth rul_minutes on the held-out test fold.
        y_test_pred: Predicted rul_minutes on the held-out test fold.
        y_train_true: Ground-truth rul_minutes on the training fold (for overfitting check).
        y_train_pred: Predicted rul_minutes on the training fold.
        config: Full project configuration dictionary.

    Returns:
        MetricBundle with all architecture-approved metrics.
    """
    metrics_cfg = config.get("model_dev", {}).get("metrics", {})
    near_failure_threshold = metrics_cfg.get("near_failure_threshold_minutes", 1440)

    test_mae = compute_mae(y_test_true, y_test_pred)
    train_mae = compute_mae(y_train_true, y_train_pred)
    overfitting_ratio = train_mae / test_mae if test_mae > 0 else float("nan")

    return MetricBundle(
        mae=test_mae,
        rmse=compute_rmse(y_test_true, y_test_pred),
        r2=compute_r2(y_test_true, y_test_pred),
        near_failure_mae=compute_near_failure_mae(y_test_true, y_test_pred, near_failure_threshold),
        signed_error=compute_signed_error(y_test_true, y_test_pred),
        late_prediction_rate=compute_late_prediction_rate(y_test_true, y_test_pred),
        train_mae=train_mae,
        overfitting_ratio=overfitting_ratio,
    )
