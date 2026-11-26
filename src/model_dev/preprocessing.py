"""In-fold preprocessing: imputation + scaling fit on the training fold only.

Structurally leakage-free by construction (architecture §9, §12, Phase 5
contract `consumer_expectations.scaling`): `fit_train_preprocessor` only ever
receives training-fold rows; the returned fitted transformer objects are then
applied — never re-fit — to the held-out fold via `transform_features`.

Optional channel aggregation (architecture §6, config `channel_handling`) is
applied independently within each fold's train/test slice and never mixes
rows across the train/test boundary.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


@dataclass
class FittedPreprocessor:
    """Container for a preprocessor fitted on a single training fold."""

    imputer: SimpleImputer
    scaler: StandardScaler
    feature_columns: List[str]


def aggregate_channels(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """Aggregate the two sensor channels of a bearing into one row per timestamp.

    Off by default (architecture §6). Applied independently to a single
    train/test slice — never combines rows spanning the fold boundary.

    Grouped by (unit_id, timestamp), NOT (bearing_id, timestamp) — a train or
    test slice can legitimately contain two units that share a bare bearing_id
    (e.g. Set 1's bearing_3 and Set 3's bearing_3, both RTF, both in the same
    LOBO training fold). Their timestamps happen not to overlap today, so
    grouping by bare bearing_id wouldn't currently corrupt the aggregation —
    but unit_id is the actually-correct key and doesn't rely on that coincidence.

    Args:
        df: DataFrame at (unit_id, channel, timestamp) grain.
        feature_columns: Feature column names to average across channels.

    Returns:
        DataFrame aggregated to (unit_id, timestamp) grain; label and fold
        columns (identical across channels at a timestamp) are preserved via 'first'.
    """
    non_feature_cols = [c for c in df.columns if c not in feature_columns and c != "channel"]
    agg_map = {c: "mean" for c in feature_columns}
    agg_map.update({c: "first" for c in non_feature_cols if c not in ("unit_id", "timestamp")})
    grouped = df.groupby(["unit_id", "timestamp"], as_index=False).agg(agg_map)
    return grouped


def fit_train_preprocessor(
    train_df: pd.DataFrame,
    feature_columns: List[str],
    config: Dict[str, Any],
) -> FittedPreprocessor:
    """Fit imputer + scaler on the TRAINING FOLD ONLY.

    Args:
        train_df: Training-fold rows (must never include the held-out bearing).
        feature_columns: Feature column names (from the Phase 5 contract).
        config: Full project configuration dictionary.

    Returns:
        FittedPreprocessor holding the fitted imputer and scaler.
    """
    prep_cfg = config.get("model_dev", {}).get("preprocessing", {})
    imputer_strategy = prep_cfg.get("imputer", "median")

    imputer = SimpleImputer(strategy=imputer_strategy)
    X_train_raw = train_df[feature_columns].to_numpy(dtype=float)
    X_train_imputed = imputer.fit_transform(X_train_raw)

    scaler = StandardScaler()
    scaler.fit(X_train_imputed)

    return FittedPreprocessor(imputer=imputer, scaler=scaler, feature_columns=feature_columns)


def transform_features(df: pd.DataFrame, preprocessor: FittedPreprocessor) -> np.ndarray:
    """Apply an already-fitted preprocessor to a DataFrame (train or test fold).

    Never calls `.fit()` — this function is the only way test-fold rows are
    ever transformed, guaranteeing the held-out bearing cannot influence the
    fitted imputer/scaler parameters.

    Args:
        df: DataFrame containing at least `preprocessor.feature_columns`.
        preprocessor: A FittedPreprocessor returned by `fit_train_preprocessor`.

    Returns:
        2D numpy array of imputed, scaled feature values.
    """
    X_raw = df[preprocessor.feature_columns].to_numpy(dtype=float)
    X_imputed = preprocessor.imputer.transform(X_raw)
    X_scaled = preprocessor.scaler.transform(X_imputed)
    return X_scaled


def prepare_fold_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: List[str],
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, FittedPreprocessor]:
    """Fit preprocessing on train_df only, then transform both train_df and test_df.

    Args:
        train_df: Training-fold rows.
        test_df: Held-out test-fold rows.
        feature_columns: Feature column names.
        config: Full project configuration dictionary.

    Returns:
        Tuple of (X_train, X_test, fitted_preprocessor).
    """
    preprocessor = fit_train_preprocessor(train_df, feature_columns, config)
    X_train = transform_features(train_df, preprocessor)
    X_test = transform_features(test_df, preprocessor)
    return X_train, X_test, preprocessor


def maybe_aggregate(
    df: pd.DataFrame, feature_columns: List[str], config: Dict[str, Any], logger: logging.Logger
) -> pd.DataFrame:
    """Apply channel aggregation to a single fold slice if enabled in config.

    Args:
        df: A single train or test fold slice (never a combination of both).
        feature_columns: Feature column names.
        config: Full project configuration dictionary.
        logger: Pipeline logger.

    Returns:
        df unchanged if aggregation is disabled (default); aggregated otherwise.
    """
    channel_cfg = config.get("model_dev", {}).get("channel_handling", {})
    if channel_cfg.get("aggregate", False):
        aggregated = aggregate_channels(df, feature_columns)
        logger.info("Channel aggregation applied: %d rows -> %d rows", len(df), len(aggregated))
        return aggregated
    return df
