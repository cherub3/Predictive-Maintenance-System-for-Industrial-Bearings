"""Tests for Phase 6 Model Development & Evaluation module.

Tests cover:
- Metric correctness (MAE/RMSE/R²/near-failure/signed-error) and MAPE rejection
- Baseline correctness (mean/median fit on train-fold only; unaffected by test data)
- Preprocessing leakage-freedom (fit uses train rows only; test data cannot alter fit)
- Contract loader validation (FATAL vs WARNING severities, including the documented
  is_censored metadata gap)
- LOBO cross-validation correctness (unit-disjointness, structural leakage-freedom)
- Model Promotion Criteria (§7b) and Best Model selection hierarchy (§8.3),
  including the demotion-fallback path when no candidate clears the hard gates
- Model registry fingerprinting and round-trip persistence
- Experiment tracking (fingerprint/ID determinism, log append semantics)
- Contract generation schema completeness
- Reproducibility (identical seed + config -> identical metrics)
- SHAP governance (tree-model-only scope; structural absence from model_selection)
"""

import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.constants import DEFAULT_CONFIG_PATH
from src.logger import setup_logger

from src.model_dev import metrics as metrics_mod
from src.model_dev.baseline_models import fit_linear_baseline, fit_mean_baseline, fit_median_baseline
from src.model_dev.contract_loader import (
    get_feature_columns,
    get_fold_role_columns,
    get_grain_columns,
    get_label_columns,
    validate_phase5_contract,
)
from src.model_dev.cross_validation import CANDIDATE_ORDER, FoldResult, run_lobo_cv
from src.model_dev.data_loader import assert_unit_disjoint, get_fold_bearings
from src.model_dev.evaluation import ModelEvaluation, ModelFoldMetrics, EvaluationResult, evaluate_cv_results
from src.model_dev.experiment_tracker import compute_config_fingerprint, generate_experiment_id, save_experiment_log
from src.model_dev.contract_generator import build_phase7_contract
from src.model_dev.metrics import MetricBundle, compute_all_metrics, compute_mae, compute_rmse, reject_disallowed_metric
from src.model_dev.model_registry import compute_model_fingerprint, register_model
from src.model_dev.model_selection import apply_promotion_criteria, select_best_model
from src.model_dev.model_trainer import train_candidate
from src.model_dev.prediction_generator import generate_predictions
from src.model_dev.preprocessing import fit_train_preprocessor, prepare_fold_matrices, transform_features
from src.model_dev.shap_analysis import compute_shap_for_model


FEATURES = [
    "td_rms", "td_kurtosis", "td_crest_factor",
    "fd_band_energy_0_1000", "fd_band_energy_1000_5000", "fd_band_energy_5000_10000",
    "fd_spectral_centroid", "fd_spectral_entropy",
    "trend_rms_slope", "trend_kurtosis_slope", "trend_rms_variance",
    "env_bpfo_energy", "env_bpfi_energy", "env_bsf_energy",
]


# ──────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def logger():
    return setup_logger(name="test_model_dev", config=load_config(DEFAULT_CONFIG_PATH))


@pytest.fixture(scope="module")
def base_config():
    return load_config(DEFAULT_CONFIG_PATH)


def _make_synthetic_training_dataset(signal_strength: float = 5.0, seed: int = 7, n_files: int = 24) -> pd.DataFrame:
    """Small, fast synthetic RTF training dataset matching the Phase 5 grain.

    `td_rms` is engineered to correlate with rul_minutes (scaled by
    `signal_strength`); the rest of the feature columns are pure noise. This
    lets tests dial signal strength up (models should promote) or down
    (baselines should win / demotion fallback triggers).
    """
    rng = np.random.default_rng(seed)
    rows = []
    bearings = {
        "bearing_3": {"channels": [5, 6], "start": "2003-10-22 12:06:24"},
        "bearing_4": {"channels": [7, 8], "start": "2003-10-22 12:06:24"},
    }
    for bearing_id, meta in bearings.items():
        timestamps = pd.date_range(start=meta["start"], periods=n_files, freq="10min")
        t_fail = timestamps[-1]
        for channel in meta["channels"]:
            for i, ts in enumerate(timestamps):
                rul_minutes = float((t_fail - ts).total_seconds() / 60.0)
                row = {"unit_id": f"set1::{bearing_id}", "bearing_id": bearing_id, "channel": channel, "timestamp": ts,
                       "rul_minutes": rul_minutes, "rul_cycles": float(n_files - i - 1)}
                for feat in FEATURES:
                    row[feat] = float(rng.normal(0, 1))
                # Inject correlated signal into td_rms only
                row["td_rms"] = float(rul_minutes / 1000.0 * signal_strength + rng.normal(0, 1))
                rows.append(row)

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["fold_1_role"] = df["bearing_id"].map({"bearing_3": "train", "bearing_4": "test"})
    df["fold_2_role"] = df["bearing_id"].map({"bearing_3": "test", "bearing_4": "train"})
    return df


def _make_synthetic_fold_assignments(training_dataset: pd.DataFrame) -> pd.DataFrame:
    return training_dataset[["unit_id", "bearing_id", "channel", "timestamp", "fold_1_role", "fold_2_role"]].copy()


def _make_synthetic_contract(training_dataset: pd.DataFrame) -> Dict:
    return {
        "contract_version": "1.0",
        "dataset_version": "1.0",
        "config_fingerprint": "test_fingerprint",
        "columns": {
            "grain": ["unit_id", "channel", "timestamp"],
            "features": FEATURES,
            "labels": ["rul_minutes", "rul_cycles"],
            "fold_roles": ["fold_1_role", "fold_2_role"],
            "metadata": ["is_censored"],  # deliberately absent from dataset — tests the WARNING path
        },
        "artifacts": {
            "training_dataset": {
                "total_rows": len(training_dataset),
                "total_columns": len(training_dataset.columns),
                "units": ["set1::bearing_3", "set1::bearing_4"],
            },
            "fold_assignments": {"folds": {
                "fold_1": {"train_units": ["set1::bearing_3"], "test_units": ["set1::bearing_4"]},
                "fold_2": {"train_units": ["set1::bearing_4"], "test_units": ["set1::bearing_3"]},
            }},
        },
    }


@pytest.fixture
def signal_dataset():
    """Dataset with a strong, learnable signal — models should beat baselines."""
    return _make_synthetic_training_dataset(signal_strength=8.0, seed=1)


@pytest.fixture
def noise_dataset():
    """Dataset with no learnable signal — baselines should win (demotion fallback)."""
    return _make_synthetic_training_dataset(signal_strength=0.0, seed=2)


@pytest.fixture
def small_model_dev_config(base_config):
    """A config dict with a lightweight model_dev block for fast test execution."""
    config = dict(base_config)
    config["model_dev"] = dict(base_config["model_dev"])
    config["model_dev"]["model_params"] = {
        "random_forest": {"n_estimators": 20, "max_depth": 4, "min_samples_leaf": 2},
        "xgboost": {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.1, "subsample": 1.0, "colsample_bytree": 1.0},
    }
    config["model_dev"]["metrics"] = dict(base_config["model_dev"]["metrics"])
    config["model_dev"]["metrics"]["near_failure_threshold_minutes"] = 60
    config["model_dev"]["metrics"]["top_n_residuals"] = 3
    return config


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

class TestMetrics:

    def test_mae_matches_manual_computation(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 33.0])
        expected = np.mean([2.0, 2.0, 3.0])
        assert compute_mae(y_true, y_pred) == pytest.approx(expected)

    def test_rmse_matches_manual_computation(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([3.0, 4.0])
        expected = np.sqrt(np.mean([9.0, 16.0]))
        assert compute_rmse(y_true, y_pred) == pytest.approx(expected)

    def test_near_failure_mae_restricts_to_subset(self):
        y_true = np.array([10.0, 500.0, 20.0])
        y_pred = np.array([15.0, 100.0, 25.0])
        # threshold=100 -> only rows 0 and 2 are "near failure"
        result = metrics_mod.compute_near_failure_mae(y_true, y_pred, threshold_minutes=100)
        expected = np.mean([5.0, 5.0])
        assert result == pytest.approx(expected)

    def test_near_failure_mae_nan_when_no_rows_qualify(self):
        y_true = np.array([500.0, 600.0])
        y_pred = np.array([100.0, 200.0])
        result = metrics_mod.compute_near_failure_mae(y_true, y_pred, threshold_minutes=10)
        assert np.isnan(result)

    def test_signed_error_direction(self):
        # predicted > actual => positive => over-predicts (late warning)
        y_true = np.array([100.0, 100.0])
        y_pred = np.array([150.0, 150.0])
        assert metrics_mod.compute_signed_error(y_true, y_pred) == pytest.approx(50.0)

    def test_late_prediction_rate(self):
        y_true = np.array([100.0, 100.0, 100.0])
        y_pred = np.array([150.0, 50.0, 100.0])  # 1 late (over), 1 early, 1 exact
        rate = metrics_mod.compute_late_prediction_rate(y_true, y_pred)
        assert rate == pytest.approx(1 / 3)

    def test_mape_is_rejected(self):
        with pytest.raises(ValueError, match="explicitly rejected"):
            reject_disallowed_metric("mape")

    def test_classification_metrics_rejected(self):
        with pytest.raises(ValueError):
            reject_disallowed_metric("accuracy")

    def test_valid_metric_not_rejected(self):
        reject_disallowed_metric("mae")  # must not raise

    def test_overfitting_ratio_computation(self, base_config):
        bundle = compute_all_metrics(
            y_test_true=np.array([100.0, 200.0]), y_test_pred=np.array([110.0, 190.0]),
            y_train_true=np.array([50.0, 60.0]), y_train_pred=np.array([51.0, 59.0]),
            config=base_config,
        )
        train_mae = compute_mae(np.array([50.0, 60.0]), np.array([51.0, 59.0]))
        test_mae = compute_mae(np.array([100.0, 200.0]), np.array([110.0, 190.0]))
        assert bundle.overfitting_ratio == pytest.approx(train_mae / test_mae)


# ──────────────────────────────────────────────────────────────────────────────
# Baseline models
# ──────────────────────────────────────────────────────────────────────────────

class TestBaselineModels:

    def test_mean_baseline_fit_on_train_only(self):
        y_train = np.array([10.0, 20.0, 30.0])
        predictor = fit_mean_baseline(y_train)
        assert predictor.value == pytest.approx(20.0)
        # predict() must broadcast the fitted constant regardless of "n_or_X" content
        preds = predictor.predict(5)
        assert len(preds) == 5
        assert np.all(preds == 20.0)

    def test_median_baseline_robust_to_outlier(self):
        y_train = np.array([10.0, 20.0, 1000.0])  # mean would be skewed; median should not be
        predictor = fit_median_baseline(y_train)
        assert predictor.value == pytest.approx(20.0)

    def test_constant_predictor_unaffected_by_test_data_shape(self):
        predictor = fit_mean_baseline(np.array([5.0, 15.0]))
        preds_for_test_fold = predictor.predict(np.zeros(100))  # array-like input, not int
        assert len(preds_for_test_fold) == 100
        assert np.all(preds_for_test_fold == 10.0)

    def test_linear_baseline_fits_and_predicts(self):
        X_train = np.array([[1.0], [2.0], [3.0]])
        y_train = np.array([2.0, 4.0, 6.0])
        model = fit_linear_baseline(X_train, y_train)
        preds = model.predict(np.array([[4.0]]))
        assert preds[0] == pytest.approx(8.0, abs=0.5)


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing — leakage-freedom
# ──────────────────────────────────────────────────────────────────────────────

class TestPreprocessing:

    def test_fit_uses_train_rows_only(self, small_model_dev_config):
        train_df = pd.DataFrame({f: [1.0, 2.0, 3.0] for f in FEATURES})
        preprocessor = fit_train_preprocessor(train_df, FEATURES, small_model_dev_config)
        # imputer statistics must reflect only train_df's median (2.0 for each column)
        assert np.allclose(preprocessor.imputer.statistics_, 2.0)

    def test_test_fold_values_never_influence_fit(self, small_model_dev_config):
        """Leakage test: extreme test-fold values must not change the fitted preprocessor."""
        train_df = pd.DataFrame({f: [1.0, 2.0, 3.0] for f in FEATURES})
        test_df_normal = pd.DataFrame({f: [2.5, 2.6] for f in FEATURES})
        test_df_extreme = pd.DataFrame({f: [1e9, -1e9] for f in FEATURES})

        prep_a = fit_train_preprocessor(train_df, FEATURES, small_model_dev_config)
        prep_b = fit_train_preprocessor(train_df, FEATURES, small_model_dev_config)

        # Fitting is driven only by train_df — regardless of what test data will later be used.
        assert np.allclose(prep_a.imputer.statistics_, prep_b.imputer.statistics_)
        assert np.allclose(prep_a.scaler.mean_, prep_b.scaler.mean_)

        # transform() on extreme test data must not raise or alter the fitted params
        _ = transform_features(test_df_extreme, prep_a)
        assert np.allclose(prep_a.imputer.statistics_, prep_b.imputer.statistics_)

    def test_prepare_fold_matrices_shapes(self, small_model_dev_config):
        train_df = pd.DataFrame({f: np.arange(5, dtype=float) for f in FEATURES})
        test_df = pd.DataFrame({f: np.arange(3, dtype=float) for f in FEATURES})
        X_train, X_test, preprocessor = prepare_fold_matrices(train_df, test_df, FEATURES, small_model_dev_config)
        assert X_train.shape == (5, len(FEATURES))
        assert X_test.shape == (3, len(FEATURES))


# ──────────────────────────────────────────────────────────────────────────────
# Contract loader
# ──────────────────────────────────────────────────────────────────────────────

class TestContractLoader:

    def test_get_feature_columns(self, signal_dataset):
        contract = _make_synthetic_contract(signal_dataset)
        assert get_feature_columns(contract) == FEATURES

    def test_get_label_and_fold_role_and_grain_columns(self, signal_dataset):
        contract = _make_synthetic_contract(signal_dataset)
        assert get_label_columns(contract) == ["rul_minutes", "rul_cycles"]
        assert get_fold_role_columns(contract) == ["fold_1_role", "fold_2_role"]
        assert get_grain_columns(contract) == ["unit_id", "channel", "timestamp"]

    def test_validate_passes_on_matching_dataset(self, signal_dataset, logger):
        contract = _make_synthetic_contract(signal_dataset)
        fold_assignments = _make_synthetic_fold_assignments(signal_dataset)
        result = validate_phase5_contract(contract, signal_dataset, fold_assignments, logger)
        assert result.passed is True
        assert len(result.fatals) == 0

    def test_missing_metadata_column_is_warning_not_fatal(self, signal_dataset, logger):
        """The is_censored gap (documented, pre-existing Phase 5 artifact issue) must be a WARNING."""
        contract = _make_synthetic_contract(signal_dataset)
        fold_assignments = _make_synthetic_fold_assignments(signal_dataset)
        result = validate_phase5_contract(contract, signal_dataset, fold_assignments, logger)
        assert result.passed is True  # must still PASS overall
        assert any("is_censored" in w for w in result.warnings)
        assert not any("is_censored" in f for f in result.fatals)

    def test_missing_feature_column_is_fatal(self, signal_dataset, logger):
        contract = _make_synthetic_contract(signal_dataset)
        contract["columns"]["features"] = FEATURES + ["nonexistent_feature"]
        fold_assignments = _make_synthetic_fold_assignments(signal_dataset)
        result = validate_phase5_contract(contract, signal_dataset, fold_assignments, logger)
        assert result.passed is False
        assert any("nonexistent_feature" in f for f in result.fatals)

    def test_row_count_mismatch_is_fatal(self, signal_dataset, logger):
        contract = _make_synthetic_contract(signal_dataset)
        contract["artifacts"]["training_dataset"]["total_rows"] = len(signal_dataset) + 100
        fold_assignments = _make_synthetic_fold_assignments(signal_dataset)
        result = validate_phase5_contract(contract, signal_dataset, fold_assignments, logger)
        assert result.passed is False


# ──────────────────────────────────────────────────────────────────────────────
# Data loader — unit disjointness
# ──────────────────────────────────────────────────────────────────────────────

class TestDataLoader:

    def test_get_fold_bearings(self, signal_dataset):
        fold_assignments = _make_synthetic_fold_assignments(signal_dataset)
        train_b, test_b = get_fold_bearings(fold_assignments, "fold_1_role")
        assert train_b == ["set1::bearing_3"]
        assert test_b == ["set1::bearing_4"]

    def test_assert_unit_disjoint_passes_on_clean_data(self, signal_dataset, logger):
        fold_assignments = _make_synthetic_fold_assignments(signal_dataset)
        assert_unit_disjoint(fold_assignments, ["fold_1_role", "fold_2_role"], logger)  # must not raise

    def test_assert_unit_disjoint_raises_on_overlap(self, signal_dataset, logger):
        corrupted = _make_synthetic_fold_assignments(signal_dataset).copy()
        # Force bearing_3 to also appear as "test" in fold_1 (in addition to train) -> overlap
        idx = corrupted[corrupted["bearing_id"] == "bearing_3"].index[:2]
        corrupted.loc[idx, "fold_1_role"] = "test"
        with pytest.raises(ValueError, match="Unit-disjointness violation"):
            assert_unit_disjoint(corrupted, ["fold_1_role", "fold_2_role"], logger)


# ──────────────────────────────────────────────────────────────────────────────
# Cross-validation — LOBO correctness + structural leakage-freedom
# ──────────────────────────────────────────────────────────────────────────────

class TestCrossValidation:

    def test_model_progression_order(self, signal_dataset, small_model_dev_config, logger):
        fold_assignments = _make_synthetic_fold_assignments(signal_dataset)
        cv_results = run_lobo_cv(signal_dataset, fold_assignments, FEATURES, small_model_dev_config, logger)
        assert cv_results.model_order == ["mean", "median", "linear", "random_forest", "xgboost"]

    def test_folds_are_unit_disjoint(self, signal_dataset, small_model_dev_config, logger):
        fold_assignments = _make_synthetic_fold_assignments(signal_dataset)
        cv_results = run_lobo_cv(signal_dataset, fold_assignments, FEATURES, small_model_dev_config, logger)
        for model_name, fold_dict in cv_results.results.items():
            for fold_id, fold_result in fold_dict.items():
                assert set(fold_result.train_bearings).isdisjoint(set(fold_result.test_bearings))

    def test_both_bearings_are_test_exactly_once_across_folds(self, signal_dataset, small_model_dev_config, logger):
        fold_assignments = _make_synthetic_fold_assignments(signal_dataset)
        cv_results = run_lobo_cv(signal_dataset, fold_assignments, FEATURES, small_model_dev_config, logger)
        test_bearings_seen = []
        for fold_id, fold_result in cv_results.results["mean"].items():
            test_bearings_seen.extend(fold_result.test_bearings)
        assert sorted(test_bearings_seen) == ["set1::bearing_3", "set1::bearing_4"]

    def test_perturbing_test_fold_never_changes_fitted_model(self, signal_dataset, small_model_dev_config, logger):
        """Leakage test: corrupting the held-out bearing's feature values must not change
        the fitted linear model's coefficients (preprocessing + fitting only see the train fold)."""
        fold_assignments = _make_synthetic_fold_assignments(signal_dataset)

        clean_results = run_lobo_cv(signal_dataset, fold_assignments, FEATURES, small_model_dev_config, logger)
        clean_coef = clean_results.results["linear"]["fold_1"].fitted_model.coef_.copy()

        corrupted = signal_dataset.copy()
        mask = corrupted["bearing_id"] == "bearing_4"  # bearing_4 is fold_1's TEST bearing
        for feat in FEATURES:
            corrupted.loc[mask, feat] = 1e12  # extreme corruption of test-fold-only rows

        corrupted_results = run_lobo_cv(corrupted, fold_assignments, FEATURES, small_model_dev_config, logger)
        corrupted_coef = corrupted_results.results["linear"]["fold_1"].fitted_model.coef_.copy()

        assert np.allclose(clean_coef, corrupted_coef), (
            "Corrupting the held-out test bearing's features changed the fitted model — "
            "this means preprocessing or training leaked test-fold data into the fit."
        )

    def test_train_and_test_predictions_have_correct_lengths(self, signal_dataset, small_model_dev_config, logger):
        fold_assignments = _make_synthetic_fold_assignments(signal_dataset)
        cv_results = run_lobo_cv(signal_dataset, fold_assignments, FEATURES, small_model_dev_config, logger)
        fold_result = cv_results.results["random_forest"]["fold_1"]
        assert len(fold_result.y_test_pred) == len(fold_result.y_test_true)
        assert len(fold_result.y_train_pred) == len(fold_result.y_train_true)


# ──────────────────────────────────────────────────────────────────────────────
# Model trainer
# ──────────────────────────────────────────────────────────────────────────────

class TestModelTrainer:

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown candidate"):
            train_candidate("lightgbm", np.zeros((3, 2)), np.zeros(3), {}, seed=1)

    def test_random_forest_trains_and_predicts(self, small_model_dev_config):
        X = np.random.default_rng(0).normal(size=(20, 3))
        y = X[:, 0] * 2 + 1
        params = small_model_dev_config["model_dev"]["model_params"]["random_forest"]
        model = train_candidate("random_forest", X, y, params, seed=1)
        preds = model.predict(X)
        assert len(preds) == 20


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

class TestEvaluation:

    def test_evaluate_cv_results_computes_expected_mae(self, signal_dataset, small_model_dev_config, logger):
        fold_assignments = _make_synthetic_fold_assignments(signal_dataset)
        cv_results = run_lobo_cv(signal_dataset, fold_assignments, FEATURES, small_model_dev_config, logger)
        evaluation_result = evaluate_cv_results(cv_results, small_model_dev_config, logger)

        # Manually recompute MAE for the mean baseline on fold_1 and compare
        fold_result = cv_results.results["mean"]["fold_1"]
        expected_mae = compute_mae(fold_result.y_test_true, fold_result.y_test_pred)
        actual_mae = evaluation_result.model_evaluations["mean"].fold_metrics["fold_1"].metrics.mae
        assert actual_mae == pytest.approx(expected_mae)

    def test_stability_ratio_is_one_for_identical_fold_performance(self, signal_dataset, small_model_dev_config, logger):
        fold_assignments = _make_synthetic_fold_assignments(signal_dataset)
        cv_results = run_lobo_cv(signal_dataset, fold_assignments, FEATURES, small_model_dev_config, logger)
        evaluation_result = evaluate_cv_results(cv_results, small_model_dev_config, logger)
        # mean baseline's MAE is symmetric by construction of the synthetic dataset (identical
        # RUL trajectory shape for both bearings) -> stability ratio should be very close to 1
        assert evaluation_result.model_evaluations["mean"].stability_ratio == pytest.approx(1.0, abs=0.05)


# ──────────────────────────────────────────────────────────────────────────────
# Model selection — promotion criteria + best-model hierarchy
# ──────────────────────────────────────────────────────────────────────────────

def _make_fold_metrics(mae: float, near_failure_mae: float, overfitting_ratio: float = 1.0) -> ModelFoldMetrics:
    bundle = MetricBundle(
        mae=mae, rmse=mae * 1.2, r2=0.5, near_failure_mae=near_failure_mae,
        signed_error=0.0, late_prediction_rate=0.5, train_mae=mae * overfitting_ratio,
        overfitting_ratio=overfitting_ratio,
    )
    return ModelFoldMetrics(
        fold_id="fold_1", train_bearing="bearing_3", test_bearing="bearing_4",
        metrics=bundle, channel_specific_mae={}, largest_residuals=pd.DataFrame(),
    )


def _make_model_eval(model_name: str, fold_1_mae: float, fold_2_mae: float, near_failure_mae: float = None, overfitting_ratio: float = 1.0) -> ModelEvaluation:
    nf_mae = near_failure_mae if near_failure_mae is not None else fold_1_mae
    fm1 = _make_fold_metrics(fold_1_mae, nf_mae, overfitting_ratio)
    fm2 = _make_fold_metrics(fold_2_mae, nf_mae, overfitting_ratio)
    fold_metrics = {"fold_1": fm1, "fold_2": fm2}
    mean_mae = (fold_1_mae + fold_2_mae) / 2
    spread = abs(fold_1_mae - fold_2_mae)
    stability_ratio = max(fold_1_mae, fold_2_mae) / min(fold_1_mae, fold_2_mae)
    return ModelEvaluation(model_name=model_name, fold_metrics=fold_metrics, mean_mae=mean_mae, mae_spread=spread, stability_ratio=stability_ratio)


class TestModelSelection:

    def test_promotion_passes_for_clear_meaningful_improvement(self, small_model_dev_config, logger):
        evaluation_result = EvaluationResult(
            model_order=["mean", "median", "linear", "random_forest", "xgboost"],
            model_evaluations={
                "mean": _make_model_eval("mean", 1000, 1000),
                "median": _make_model_eval("median", 900, 900),
                "linear": _make_model_eval("linear", 500, 500),
                "random_forest": _make_model_eval("random_forest", 200, 200),
                "xgboost": _make_model_eval("xgboost", 100, 100),
            },
        )
        ledger = apply_promotion_criteria(evaluation_result, small_model_dev_config, logger)
        assert ledger.decisions["median"].promoted is True
        assert ledger.decisions["linear"].promoted is True
        assert ledger.decisions["random_forest"].promoted is True
        assert ledger.decisions["xgboost"].promoted is True

    def test_selection_picks_lowest_mae_when_all_promoted(self, small_model_dev_config, logger):
        evaluation_result = EvaluationResult(
            model_order=["mean", "median", "linear", "random_forest", "xgboost"],
            model_evaluations={
                "mean": _make_model_eval("mean", 1000, 1000),
                "median": _make_model_eval("median", 900, 900),
                "linear": _make_model_eval("linear", 500, 500),
                "random_forest": _make_model_eval("random_forest", 200, 200),
                "xgboost": _make_model_eval("xgboost", 100, 100),
            },
        )
        ledger = apply_promotion_criteria(evaluation_result, small_model_dev_config, logger)
        result = select_best_model(evaluation_result, ledger, small_model_dev_config, logger)
        assert result.selected_model == "xgboost"
        assert result.resolved_at_priority == 1

    def test_promotion_rejects_when_predecessor_not_beaten_on_both_folds(self, small_model_dev_config, logger):
        evaluation_result = EvaluationResult(
            model_order=["mean", "median"],
            model_evaluations={
                "mean": _make_model_eval("mean", 1000, 1000),
                "median": _make_model_eval("median", 900, 1100),  # worse on fold_2
            },
        )
        ledger = apply_promotion_criteria(evaluation_result, small_model_dev_config, logger)
        assert ledger.decisions["median"].promoted is False
        assert any("Rule 1 FAIL" in r for r in ledger.decisions["median"].reasons)

    def test_promotion_rejects_on_overfitting_evidence(self, small_model_dev_config, logger):
        evaluation_result = EvaluationResult(
            model_order=["mean", "median"],
            model_evaluations={
                "mean": _make_model_eval("mean", 1000, 1000),
                # median clearly beats mean by a wide margin, but is overfit-flagged
                "median": _make_model_eval("median", 100, 100, overfitting_ratio=0.1),
            },
        )
        ledger = apply_promotion_criteria(evaluation_result, small_model_dev_config, logger)
        assert ledger.decisions["median"].promoted is False
        assert any("Rule 4 FAIL" in r for r in ledger.decisions["median"].reasons)

    def test_demotion_fallback_when_nothing_beats_baselines(self, small_model_dev_config, logger):
        """No candidate beats the baselines -> must gracefully fall back, never raise."""
        evaluation_result = EvaluationResult(
            model_order=["mean", "median", "linear", "random_forest", "xgboost"],
            model_evaluations={
                "mean": _make_model_eval("mean", 1000, 1000),
                "median": _make_model_eval("median", 950, 950),
                "linear": _make_model_eval("linear", 1500, 1600),      # worse than median
                "random_forest": _make_model_eval("random_forest", 1400, 1700),  # worse than median
                "xgboost": _make_model_eval("xgboost", 1450, 1650),    # worse than median
            },
        )
        ledger = apply_promotion_criteria(evaluation_result, small_model_dev_config, logger)
        result = select_best_model(evaluation_result, ledger, small_model_dev_config, logger)  # must not raise
        assert result.selected_model == "median"
        assert result.resolved_at_priority == 0
        assert result.resolved_by == "demotion_fallback"

    def test_stability_tiebreak_prefers_stable_near_tied_model(self, small_model_dev_config, logger):
        # linear and random_forest are near-tied in mean MAE, but RF is unstable across folds
        evaluation_result = EvaluationResult(
            model_order=["mean", "median", "linear", "random_forest"],
            model_evaluations={
                "mean": _make_model_eval("mean", 1000, 1000),
                "median": _make_model_eval("median", 900, 900),
                "linear": _make_model_eval("linear", 300, 310),          # stable
                "random_forest": _make_model_eval("random_forest", 100, 500),  # near-tied mean but unstable
            },
        )
        # Force both into "promoted" state to isolate the tiebreak logic in select_best_model
        ledger = apply_promotion_criteria(evaluation_result, small_model_dev_config, logger)
        # Manually override promotion outcomes to guarantee both are hard-gate survivors,
        # isolating this test to the priority-2 stability tiebreak only.
        for name in ("linear", "random_forest"):
            ledger.decisions[name].promoted = True
        result = select_best_model(evaluation_result, ledger, small_model_dev_config, logger)
        # random_forest mean_mae=300 vs linear mean_mae=305 -> near-tied given margin=30
        # random_forest instability_ratio = 500/100 = 5.0 (unstable); linear ratio ~1.03 (stable)
        assert result.selected_model == "linear"
        assert result.resolved_at_priority == 2

    def test_integration_full_selection_never_raises_on_real_cv(self, noise_dataset, small_model_dev_config, logger):
        """End-to-end: pure-noise dataset must still produce a valid (baseline) selection, not a crash."""
        fold_assignments = _make_synthetic_fold_assignments(noise_dataset)
        cv_results = run_lobo_cv(noise_dataset, fold_assignments, FEATURES, small_model_dev_config, logger)
        evaluation_result = evaluate_cv_results(cv_results, small_model_dev_config, logger)
        ledger = apply_promotion_criteria(evaluation_result, small_model_dev_config, logger)
        result = select_best_model(evaluation_result, ledger, small_model_dev_config, logger)
        assert result.selected_model in evaluation_result.model_order


# ──────────────────────────────────────────────────────────────────────────────
# Model registry
# ──────────────────────────────────────────────────────────────────────────────

class TestModelRegistry:

    def test_fingerprint_deterministic_for_identical_object(self):
        obj = {"a": 1, "b": [1, 2, 3]}
        assert compute_model_fingerprint(obj) == compute_model_fingerprint(obj)

    def test_fingerprint_differs_for_different_objects(self):
        assert compute_model_fingerprint({"a": 1}) != compute_model_fingerprint({"a": 2})

    def test_register_model_round_trip(self, signal_dataset, small_model_dev_config, logger, tmp_path):
        fold_assignments = _make_synthetic_fold_assignments(signal_dataset)
        cv_results = run_lobo_cv(signal_dataset, fold_assignments, FEATURES, small_model_dev_config, logger)
        refit_model = cv_results.results["linear"]["fold_1"].fitted_model
        refit_preprocessor = cv_results.results["linear"]["fold_1"].preprocessor

        registry_info = register_model(
            "linear", cv_results, refit_model, refit_preprocessor, "test_exp_001",
            small_model_dev_config, tmp_path, logger,
        )
        model_path = tmp_path / registry_info["model_path"]
        refit_path = tmp_path / registry_info["refit_model_path"]
        assert model_path.exists()
        assert refit_path.exists()

        with open(refit_path, "rb") as f:
            loaded = pickle.load(f)
        assert loaded["selected_model_name"] == "linear"
        assert loaded["trained_on"] == "all_rtf"


# ──────────────────────────────────────────────────────────────────────────────
# Experiment tracker
# ──────────────────────────────────────────────────────────────────────────────

class TestExperimentTracker:

    def test_config_fingerprint_deterministic(self, base_config):
        assert compute_config_fingerprint(base_config) == compute_config_fingerprint(base_config)

    def test_config_fingerprint_changes_with_config(self, base_config):
        modified = dict(base_config)
        modified["model_dev"] = dict(base_config["model_dev"])
        modified["model_dev"]["random_seed"] = 999
        assert compute_config_fingerprint(base_config) != compute_config_fingerprint(modified)

    def test_experiment_id_deterministic(self):
        fp = "abc123def456"
        assert generate_experiment_id(fp, 42) == generate_experiment_id(fp, 42)
        assert generate_experiment_id(fp, 42) != generate_experiment_id(fp, 43)

    def test_experiment_log_appends(self, tmp_path, logger):
        log_path = tmp_path / "experiment_log.json"
        save_experiment_log({"experiment_id": "exp1", "random_seed": 1, "config_fingerprint": "fp1"}, log_path, logger)
        save_experiment_log({"experiment_id": "exp2", "random_seed": 2, "config_fingerprint": "fp2"}, log_path, logger)

        import json
        records = json.loads(log_path.read_text(encoding="utf-8"))
        assert len(records) == 2
        assert records[0]["experiment_id"] == "exp1"
        assert records[1]["experiment_id"] == "exp2"


# ──────────────────────────────────────────────────────────────────────────────
# Contract generator
# ──────────────────────────────────────────────────────────────────────────────

class TestContractGenerator:

    def test_contract_has_mandatory_n2_caveat(self, signal_dataset, small_model_dev_config, logger):
        fold_assignments = _make_synthetic_fold_assignments(signal_dataset)
        cv_results = run_lobo_cv(signal_dataset, fold_assignments, FEATURES, small_model_dev_config, logger)
        evaluation_result = evaluate_cv_results(cv_results, small_model_dev_config, logger)
        from src.model_dev.feature_importance import compute_feature_importance
        importance_result = compute_feature_importance(cv_results, signal_dataset, small_model_dev_config, logger)

        contract = build_phase7_contract(
            experiment_id="exp_test", config_fingerprint="fp", git_commit="abc123",
            selected_model="linear",
            registry_info={"refit_model_path": "models/x/refit_model.pkl", "model_path": "models/x/model.pkl", "refit_model_fingerprint": "fp1"},
            feature_columns=FEATURES, evaluation_result=evaluation_result, importance_result=importance_result,
            phase5_contract=_make_synthetic_contract(signal_dataset), config=small_model_dev_config, logger=logger,
        )
        assert contract["performance"]["n2_caveat"] is True
        assert contract["prediction_schema"]["input_features"] == FEATURES
        assert contract["consumer_expectations"]["must_not_use_shap_outputs_for_model_modification"] is True
        assert contract["schema_version"] == "1.0"


# ──────────────────────────────────────────────────────────────────────────────
# Prediction generator
# ──────────────────────────────────────────────────────────────────────────────

class TestPredictionGenerator:

    def test_predictions_cover_every_row_exactly_once(self, signal_dataset, small_model_dev_config, logger):
        fold_assignments = _make_synthetic_fold_assignments(signal_dataset)
        cv_results = run_lobo_cv(signal_dataset, fold_assignments, FEATURES, small_model_dev_config, logger)
        predictions = generate_predictions(cv_results, "mean", logger)
        assert len(predictions) == len(signal_dataset)
        assert set(predictions["bearing_id"].unique()) == {"bearing_3", "bearing_4"}


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

class TestReproducibility:

    def test_identical_seed_and_config_produce_identical_metrics(self, signal_dataset, small_model_dev_config, logger):
        fold_assignments = _make_synthetic_fold_assignments(signal_dataset)
        cv_a = run_lobo_cv(signal_dataset, fold_assignments, FEATURES, small_model_dev_config, logger)
        cv_b = run_lobo_cv(signal_dataset, fold_assignments, FEATURES, small_model_dev_config, logger)

        eval_a = evaluate_cv_results(cv_a, small_model_dev_config, logger)
        eval_b = evaluate_cv_results(cv_b, small_model_dev_config, logger)

        for model_name in cv_a.model_order:
            assert eval_a.model_evaluations[model_name].mean_mae == pytest.approx(
                eval_b.model_evaluations[model_name].mean_mae
            )


# ──────────────────────────────────────────────────────────────────────────────
# SHAP governance
# ──────────────────────────────────────────────────────────────────────────────

class TestShapGovernance:

    def test_shap_skipped_for_linear_model(self, signal_dataset, small_model_dev_config, logger):
        fold_assignments = _make_synthetic_fold_assignments(signal_dataset)
        cv_results = run_lobo_cv(signal_dataset, fold_assignments, FEATURES, small_model_dev_config, logger)
        result = compute_shap_for_model(cv_results, signal_dataset, "linear", small_model_dev_config, logger)
        assert result is None

    def test_shap_computed_for_random_forest(self, signal_dataset, small_model_dev_config, logger):
        fold_assignments = _make_synthetic_fold_assignments(signal_dataset)
        cv_results = run_lobo_cv(signal_dataset, fold_assignments, FEATURES, small_model_dev_config, logger)
        result = compute_shap_for_model(cv_results, signal_dataset, "random_forest", small_model_dev_config, logger)
        assert result is not None
        assert not result.global_ranking.empty

    def test_model_selection_module_never_imports_shap(self):
        """Structural governance check: model_selection.py must have no SHAP import,
        so SHAP outputs cannot influence model selection even by accident (architecture §11a)."""
        import ast

        import src.model_dev.model_selection as ms
        source = Path(ms.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)

        shap_imports = {m for m in imported_modules if "shap" in m.lower()}
        assert not shap_imports, (
            f"model_selection.py imports {shap_imports} — SHAP is interpretation-only "
            "and must not be able to influence model selection (architecture §11a)."
        )

    def test_select_best_model_signature_has_no_shap_parameter(self):
        """Belt-and-suspenders: even if shap were imported elsewhere, select_best_model's
        signature must have no parameter through which a SHAP result could be passed in."""
        import inspect

        from src.model_dev.model_selection import select_best_model
        params = set(inspect.signature(select_best_model).parameters.keys())
        assert not any("shap" in p.lower() for p in params)
