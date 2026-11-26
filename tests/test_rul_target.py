"""Tests for Phase 5 RUL Target Construction & Leakage Prevention module.

Tests cover:
- RUL label correctness and elapsed-time calculation
- Censored bearing handling (no fabricated labels)
- Temporal integrity validation (chronology, dtype, window)
- Failure index computation (t_fail, total_life)
- LOBO split correctness and zero overlap
- Leakage detection (L-01 through L-09)
- Dataset builder join correctness and monotonicity
- Label dictionary / manifest / contract schema
- Final validation rules
- Reproducibility (deterministic outputs for identical inputs)

All fixtures use a synthetic feature store scoped to `set1` only, with a
config whose `datasets.sets` registry is narrowed to just `set1` (copied
verbatim from the real config) — this keeps the unit_id-qualified code paths
exercised (unit_id = "set1::bearing_N") without needing synthetic data for
every registered dataset.
"""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.constants import DEFAULT_CONFIG_PATH
from src.logger import setup_logger

from src.rul_target.temporal_validator import validate_temporal_integrity
from src.rul_target.failure_indexer import build_failure_index, UnitFailureIndex
from src.rul_target.label_generator import generate_rul_labels
from src.rul_target.split_generator import generate_fold_assignments, generate_split_summary
from src.rul_target.leakage_validator import run_leakage_checks
from src.rul_target.dataset_builder import build_training_dataset
from src.rul_target.label_dictionary import build_label_dictionary
from src.rul_target.dataset_manifest import build_dataset_manifest
from src.rul_target.contract_generator import build_phase6_contract
from src.rul_target.validation import run_final_validation


def uid(bearing_id: str) -> str:
    """Qualified unit_id for the set1-only test fixtures."""
    return f"set1::{bearing_id}"


# ──────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def logger():
    return setup_logger(name="test_rul_target", config=load_config(DEFAULT_CONFIG_PATH))


@pytest.fixture(scope="module")
def base_config():
    config = load_config(DEFAULT_CONFIG_PATH)
    return config


def _make_synthetic_feature_store(
    bearing_ids=("bearing_1", "bearing_2", "bearing_3", "bearing_4"),
    n_files=10,
    start="2003-10-22 12:06:24",
    freq_minutes=10,
) -> pd.DataFrame:
    """Build a small synthetic feature store matching the Phase 4 grain.

    Grain: (unit_id, channel, timestamp). Two channels per bearing. All rows
    belong to the "set1" dataset, matching set1's real registered run_window
    (2003-10-22 to 2003-11-25) so temporal_validator's per-dataset window
    check passes without needing a custom override.
    """
    timestamps = pd.date_range(start=start, periods=n_files, freq=f"{freq_minutes}min")
    rows = []
    for bearing_id in bearing_ids:
        bearing_num = int(bearing_id.split("_")[1])
        channels = [(bearing_num - 1) * 2, (bearing_num - 1) * 2 + 1]
        for channel in channels:
            for i, ts in enumerate(timestamps):
                rows.append({
                    "dataset_id": "set1",
                    "bearing_id": bearing_id,
                    "unit_id": uid(bearing_id),
                    "channel": channel,
                    "timestamp": ts,
                    "td_rms": 0.1 + 0.01 * i,
                    "td_kurtosis": 3.0 + 0.05 * i,
                    "fd_peak_freq": 1000.0,
                    "trend_rms_slope": 0.001 * i,
                })
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@pytest.fixture
def synthetic_feature_store():
    return _make_synthetic_feature_store()


@pytest.fixture
def rul_config(base_config):
    """A config dict whose datasets registry is narrowed to set1 only.

    Reuses set1's real registry entry verbatim (run_window, bearing
    status/failure_mode) so get_rtf_units()/get_censored_units() return
    exactly ["set1::bearing_3", "set1::bearing_4"] / ["set1::bearing_1",
    "set1::bearing_2"], matching the synthetic feature store's bearings.
    """
    config = dict(base_config)
    config["datasets"] = dict(base_config["datasets"])
    config["datasets"]["sets"] = {"set1": dict(base_config["datasets"]["sets"]["set1"])}
    config["rul_target"] = dict(base_config["rul_target"])
    return config


# ──────────────────────────────────────────────────────────────────────────────
# Temporal validator tests
# ──────────────────────────────────────────────────────────────────────────────

class TestTemporalValidator:

    def test_passes_on_clean_synthetic_store(self, synthetic_feature_store, rul_config, logger):
        result = validate_temporal_integrity(synthetic_feature_store, rul_config, logger)
        assert result.passed is True
        assert len(result.fatals) == 0

    def test_fails_on_non_monotonic_timestamps(self, synthetic_feature_store, rul_config, logger):
        corrupted = synthetic_feature_store.copy()
        # Duplicate a timestamp within the same (unit_id, channel) group — this
        # produces a non-positive diff regardless of sort order (ties are fatal).
        mask = (corrupted["bearing_id"] == "bearing_3") & (corrupted["channel"] == 4)
        idx = corrupted[mask].index[:2]
        vals = corrupted.loc[idx, "timestamp"].values
        corrupted.loc[idx[1], "timestamp"] = vals[0]

        result = validate_temporal_integrity(corrupted, rul_config, logger)
        assert result.passed is False
        assert any("Non-strictly-increasing" in f for f in result.fatals)

    def test_fails_on_nat_timestamp(self, synthetic_feature_store, rul_config, logger):
        corrupted = synthetic_feature_store.copy()
        corrupted.loc[0, "timestamp"] = pd.NaT
        result = validate_temporal_integrity(corrupted, rul_config, logger)
        assert result.passed is False
        assert any("NaT" in f for f in result.fatals)

    def test_fails_on_missing_bearing(self, synthetic_feature_store, rul_config, logger):
        corrupted = synthetic_feature_store[synthetic_feature_store["bearing_id"] != "bearing_4"]
        result = validate_temporal_integrity(corrupted, rul_config, logger)
        assert result.passed is False
        assert any("Missing expected unit" in f for f in result.fatals)

    def test_fails_on_out_of_window_timestamp(self, synthetic_feature_store, rul_config, logger):
        corrupted = synthetic_feature_store.copy()
        corrupted.loc[0, "timestamp"] = pd.Timestamp("2010-01-01")
        result = validate_temporal_integrity(corrupted, rul_config, logger)
        assert result.passed is False
        assert any("outside frozen run window" in f for f in result.fatals)


# ──────────────────────────────────────────────────────────────────────────────
# Failure indexer tests
# ──────────────────────────────────────────────────────────────────────────────

class TestFailureIndexer:

    def test_rtf_units_get_t_fail_as_max_timestamp(self, synthetic_feature_store, rul_config, logger):
        index = build_failure_index(synthetic_feature_store, rul_config, logger)

        for bearing_id in ("bearing_3", "bearing_4"):
            idx = index[uid(bearing_id)]
            expected_t_fail = synthetic_feature_store[
                synthetic_feature_store["bearing_id"] == bearing_id
            ]["timestamp"].max()
            assert idx.is_censored is False
            assert idx.t_fail == expected_t_fail

    def test_censored_units_have_null_failure_data(self, synthetic_feature_store, rul_config, logger):
        index = build_failure_index(synthetic_feature_store, rul_config, logger)
        for bearing_id in ("bearing_1", "bearing_2"):
            idx = index[uid(bearing_id)]
            assert idx.is_censored is True
            assert idx.t_fail is None
            assert idx.total_life_minutes is None
            assert idx.total_cycles is None

    def test_total_life_matches_span(self, synthetic_feature_store, rul_config, logger):
        index = build_failure_index(synthetic_feature_store, rul_config, logger)
        idx = index[uid("bearing_3")]
        unit_rows = synthetic_feature_store[synthetic_feature_store["bearing_id"] == "bearing_3"]
        expected_span = (
            unit_rows["timestamp"].max() - unit_rows["timestamp"].min()
        ).total_seconds() / 60.0
        assert idx.total_life_minutes == pytest.approx(expected_span)


# ──────────────────────────────────────────────────────────────────────────────
# Label generator tests
# ──────────────────────────────────────────────────────────────────────────────

class TestLabelGenerator:

    def test_rul_minutes_elapsed_time_formula(self, synthetic_feature_store, rul_config, logger):
        index = build_failure_index(synthetic_feature_store, rul_config, logger)
        labels = generate_rul_labels(synthetic_feature_store, index, rul_config, logger)

        b3 = labels[labels["unit_id"] == uid("bearing_3")]
        t_fail = index[uid("bearing_3")].t_fail
        for _, row in b3.iterrows():
            expected = (t_fail - row["timestamp"]).total_seconds() / 60.0
            assert row["rul_minutes"] == pytest.approx(expected)

    def test_rul_at_failure_is_zero(self, synthetic_feature_store, rul_config, logger):
        index = build_failure_index(synthetic_feature_store, rul_config, logger)
        labels = generate_rul_labels(synthetic_feature_store, index, rul_config, logger)

        for bearing_id in ("bearing_3", "bearing_4"):
            t_fail = index[uid(bearing_id)].t_fail
            row_at_fail = labels[
                (labels["unit_id"] == uid(bearing_id)) & (labels["timestamp"] == t_fail)
            ]
            assert (row_at_fail["rul_minutes"] == 0.0).all()

    def test_censored_units_have_null_labels(self, synthetic_feature_store, rul_config, logger):
        index = build_failure_index(synthetic_feature_store, rul_config, logger)
        labels = generate_rul_labels(synthetic_feature_store, index, rul_config, logger)

        censored = labels[labels["unit_id"].isin([uid("bearing_1"), uid("bearing_2")])]
        assert censored["rul_minutes"].isna().all()
        assert censored["is_censored"].all()

    def test_rtf_labels_never_null(self, synthetic_feature_store, rul_config, logger):
        index = build_failure_index(synthetic_feature_store, rul_config, logger)
        labels = generate_rul_labels(synthetic_feature_store, index, rul_config, logger)

        rtf = labels[labels["unit_id"].isin([uid("bearing_3"), uid("bearing_4")])]
        assert rtf["rul_minutes"].notna().all()
        assert (~rtf["is_censored"]).all()

    def test_no_row_multiplication(self, synthetic_feature_store, rul_config, logger):
        index = build_failure_index(synthetic_feature_store, rul_config, logger)
        labels = generate_rul_labels(synthetic_feature_store, index, rul_config, logger)
        assert len(labels) == len(synthetic_feature_store)

    def test_rul_monotonically_non_increasing(self, synthetic_feature_store, rul_config, logger):
        index = build_failure_index(synthetic_feature_store, rul_config, logger)
        labels = generate_rul_labels(synthetic_feature_store, index, rul_config, logger)

        for (unit_id, ch), group in labels[~labels["is_censored"]].groupby(["unit_id", "channel"]):
            sorted_group = group.sort_values("timestamp")
            diffs = sorted_group["rul_minutes"].diff().dropna()
            assert (diffs <= 1e-9).all()

    def test_raises_on_timestamp_after_t_fail(self, synthetic_feature_store, rul_config, logger):
        """Fix 2: timestamps after t_fail produce negative rul_minutes → must raise ValueError."""
        index = build_failure_index(synthetic_feature_store, rul_config, logger)
        # Inject a row for bearing_3 with a timestamp 10 minutes after t_fail
        t_fail = index[uid("bearing_3")].t_fail
        extra_row = synthetic_feature_store[
            (synthetic_feature_store["bearing_id"] == "bearing_3") &
            (synthetic_feature_store["channel"] == synthetic_feature_store[
                synthetic_feature_store["bearing_id"] == "bearing_3"]["channel"].iloc[0])
        ].iloc[[0]].copy()
        extra_row["timestamp"] = t_fail + pd.Timedelta(minutes=10)
        corrupted_fs = pd.concat([synthetic_feature_store, extra_row], ignore_index=True)

        with pytest.raises(ValueError, match="rul_minutes < 0"):
            generate_rul_labels(corrupted_fs, index, rul_config, logger)

    def test_raises_on_missing_rtf_bearing_in_feature_store(self, synthetic_feature_store, rul_config, logger):
        """Fix 1: missing RTF unit in feature_store must raise ValueError, not silently proceed."""
        # Remove all rows for bearing_3 from the feature store
        truncated_fs = synthetic_feature_store[
            synthetic_feature_store["bearing_id"] != "bearing_3"
        ].copy()
        with pytest.raises(ValueError, match="RTF unit 'set1::bearing_3' has no rows"):
            build_failure_index(truncated_fs, rul_config, logger)


# ──────────────────────────────────────────────────────────────────────────────
# Split generator (LOBO) tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSplitGenerator:

    @pytest.fixture
    def rul_labels(self, synthetic_feature_store, rul_config, logger):
        index = build_failure_index(synthetic_feature_store, rul_config, logger)
        return generate_rul_labels(synthetic_feature_store, index, rul_config, logger)

    def test_fold_1_train_test_assignment(self, rul_labels, rul_config, logger):
        folds = generate_fold_assignments(rul_labels, rul_config, logger)
        f1_train = set(folds[folds["fold_1_role"] == "train"]["unit_id"].unique())
        f1_test = set(folds[folds["fold_1_role"] == "test"]["unit_id"].unique())
        # fold_1's test unit is the first RTF unit alphabetically: set1::bearing_3
        assert f1_train == {uid("bearing_4")}
        assert f1_test == {uid("bearing_3")}

    def test_fold_2_is_reverse_of_fold_1(self, rul_labels, rul_config, logger):
        folds = generate_fold_assignments(rul_labels, rul_config, logger)
        f2_train = set(folds[folds["fold_2_role"] == "train"]["unit_id"].unique())
        f2_test = set(folds[folds["fold_2_role"] == "test"]["unit_id"].unique())
        assert f2_train == {uid("bearing_3")}
        assert f2_test == {uid("bearing_4")}

    def test_censored_units_excluded_from_folds(self, rul_labels, rul_config, logger):
        folds = generate_fold_assignments(rul_labels, rul_config, logger)
        assert uid("bearing_1") not in folds["unit_id"].values
        assert uid("bearing_2") not in folds["unit_id"].values

    def test_zero_overlap_within_each_fold(self, rul_labels, rul_config, logger):
        folds = generate_fold_assignments(rul_labels, rul_config, logger)
        for fold_id in ("fold_1", "fold_2"):
            role_col = f"{fold_id}_role"
            train_units = set(folds[folds[role_col] == "train"]["unit_id"])
            test_units = set(folds[folds[role_col] == "test"]["unit_id"])
            assert train_units & test_units == set()

    def test_no_unassigned_rtf_rows(self, rul_labels, rul_config, logger):
        folds = generate_fold_assignments(rul_labels, rul_config, logger)
        assert folds["fold_1_role"].isna().sum() == 0
        assert folds["fold_2_role"].isna().sum() == 0

    def test_two_rtf_units_produce_exactly_two_folds(self, rul_labels, rul_config, logger):
        """With 2 RTF units registered, LOBO must generate exactly 2 folds (not a hardcoded pair)."""
        from src.rul_target.split_generator import get_fold_ids
        folds = generate_fold_assignments(rul_labels, rul_config, logger)
        assert get_fold_ids(folds) == ["fold_1", "fold_2"]

    def test_rejects_unsupported_protocol(self, rul_labels, rul_config, logger):
        bad_config = dict(rul_config)
        bad_config["rul_target"] = dict(rul_config["rul_target"])
        bad_config["rul_target"]["split"] = {"protocol": "random_split"}
        with pytest.raises(ValueError, match="Unsupported split protocol"):
            generate_fold_assignments(rul_labels, bad_config, logger)

    def test_split_summary_contains_honesty_caveat(self, rul_labels, rul_config, logger):
        folds = generate_fold_assignments(rul_labels, rul_config, logger)
        summary = generate_split_summary(folds, rul_config, logger)
        assert "n = 2" in summary or "n=2" in summary
        assert "Honesty Caveat" in summary


# ──────────────────────────────────────────────────────────────────────────────
# Leakage validator tests
# ──────────────────────────────────────────────────────────────────────────────

class TestLeakageValidator:

    @pytest.fixture
    def clean_pipeline_state(self, synthetic_feature_store, rul_config, logger):
        index = build_failure_index(synthetic_feature_store, rul_config, logger)
        labels = generate_rul_labels(synthetic_feature_store, index, rul_config, logger)
        folds = generate_fold_assignments(labels, rul_config, logger)
        return synthetic_feature_store, labels, folds

    def test_clean_pipeline_passes_all_checks(self, clean_pipeline_state, rul_config, logger):
        fs, labels, folds = clean_pipeline_state
        report = run_leakage_checks(fs, labels, folds, rul_config, logger)
        assert report.all_passed is True
        assert report.fatal_count == 0

    def test_l01_detects_unit_in_both_roles(self, clean_pipeline_state, rul_config, logger):
        fs, labels, folds = clean_pipeline_state
        corrupted = folds.copy()
        # fold_1 tests bearing_3 and trains bearing_4 (rtf_units sorted alphabetically,
        # first unit is tested in fold_1). Force bearing_4's unit to also appear as
        # test in fold_1 (duplicate row with conflicting role) to create real overlap.
        b4_row = corrupted[corrupted["unit_id"] == uid("bearing_4")].iloc[[0]].copy()
        b4_row["fold_1_role"] = "test"
        corrupted = pd.concat([corrupted, b4_row], ignore_index=True)

        report = run_leakage_checks(fs, labels, corrupted, rul_config, logger)
        l01 = next(c for c in report.checks if c.check_id == "L-01")
        assert l01.passed is False
        assert report.all_passed is False

    def test_l04_detects_duplicate_keys(self, clean_pipeline_state, rul_config, logger):
        fs, labels, folds = clean_pipeline_state
        corrupted_fs = pd.concat([fs, fs.iloc[[0]]], ignore_index=True)
        report = run_leakage_checks(corrupted_fs, labels, folds, rul_config, logger)
        l04 = next(c for c in report.checks if c.check_id == "L-04")
        assert l04.passed is False

    def test_l08_detects_censored_contamination(self, clean_pipeline_state, rul_config, logger):
        fs, labels, folds = clean_pipeline_state
        corrupted = folds.copy()
        contaminated_row = pd.DataFrame([{
            "unit_id": uid("bearing_1"), "dataset_id": "set1", "bearing_id": "bearing_1", "channel": 0,
            "timestamp": fs["timestamp"].iloc[0],
            "fold_1_role": "train", "fold_2_role": "train",
        }])
        corrupted = pd.concat([corrupted, contaminated_row], ignore_index=True)

        report = run_leakage_checks(fs, labels, corrupted, rul_config, logger)
        l08 = next(c for c in report.checks if c.check_id == "L-08")
        assert l08.passed is False

    def test_all_nine_checks_are_executed(self, clean_pipeline_state, rul_config, logger):
        fs, labels, folds = clean_pipeline_state
        report = run_leakage_checks(fs, labels, folds, rul_config, logger)
        check_ids = {c.check_id for c in report.checks}
        expected_ids = {f"L-{i:02d}" for i in range(1, 10)}
        assert check_ids == expected_ids


# ──────────────────────────────────────────────────────────────────────────────
# Dataset builder tests
# ──────────────────────────────────────────────────────────────────────────────

class TestDatasetBuilder:

    @pytest.fixture
    def pipeline_state(self, synthetic_feature_store, rul_config, logger):
        index = build_failure_index(synthetic_feature_store, rul_config, logger)
        labels = generate_rul_labels(synthetic_feature_store, index, rul_config, logger)
        folds = generate_fold_assignments(labels, rul_config, logger)
        return synthetic_feature_store, labels, folds

    def test_only_rtf_units_in_training_dataset(self, pipeline_state, rul_config, logger):
        fs, labels, folds = pipeline_state
        dataset = build_training_dataset(fs, labels, folds, rul_config, logger)
        assert set(dataset["unit_id"].unique()) == {uid("bearing_3"), uid("bearing_4")}

    def test_row_count_matches_fold_assignments(self, pipeline_state, rul_config, logger):
        fs, labels, folds = pipeline_state
        dataset = build_training_dataset(fs, labels, folds, rul_config, logger)
        assert len(dataset) == len(folds)

    def test_features_and_labels_present(self, pipeline_state, rul_config, logger):
        fs, labels, folds = pipeline_state
        dataset = build_training_dataset(fs, labels, folds, rul_config, logger)
        assert "td_rms" in dataset.columns
        assert "rul_minutes" in dataset.columns
        assert "fold_1_role" in dataset.columns
        assert "fold_2_role" in dataset.columns

    def test_no_null_ruls_in_training_dataset(self, pipeline_state, rul_config, logger):
        fs, labels, folds = pipeline_state
        dataset = build_training_dataset(fs, labels, folds, rul_config, logger)
        assert dataset["rul_minutes"].notna().all()

    def test_raises_on_monotonicity_violation(self, pipeline_state, rul_config, logger):
        fs, labels, folds = pipeline_state
        corrupted_labels = labels.copy()
        # Break monotonicity: set an early row's rul_minutes below a later row's
        b3_mask = corrupted_labels["unit_id"] == uid("bearing_3")
        idx = corrupted_labels[b3_mask].index
        corrupted_labels.loc[idx[0], "rul_minutes"] = -999.0
        with pytest.raises(ValueError, match="monotonicity|non-increasing|increases"):
            build_training_dataset(fs, corrupted_labels, folds, rul_config, logger)


# ──────────────────────────────────────────────────────────────────────────────
# Label dictionary / manifest / contract tests
# ──────────────────────────────────────────────────────────────────────────────

class TestArtifactGeneration:

    @pytest.fixture
    def full_pipeline_state(self, synthetic_feature_store, rul_config, logger):
        index = build_failure_index(synthetic_feature_store, rul_config, logger)
        labels = generate_rul_labels(synthetic_feature_store, index, rul_config, logger)
        folds = generate_fold_assignments(labels, rul_config, logger)
        dataset = build_training_dataset(synthetic_feature_store, labels, folds, rul_config, logger)
        return synthetic_feature_store, labels, folds, dataset

    def test_label_dictionary_schema(self, full_pipeline_state, rul_config, logger):
        fs, labels, folds, dataset = full_pipeline_state
        d = build_label_dictionary(labels, rul_config, logger)
        assert "rul_definition" in d
        assert "label_columns" in d
        assert any(c["name"] == "rul_minutes" for c in d["label_columns"])
        assert d["unit_classification"]["rtf_units"] == [uid("bearing_3"), uid("bearing_4")]
        assert d["unit_classification"]["censored_units"] == [uid("bearing_1"), uid("bearing_2")]

    def test_dataset_manifest_row_counts(self, full_pipeline_state, rul_config, logger, tmp_path):
        fs, labels, folds, dataset = full_pipeline_state
        output_paths = {"dummy": tmp_path / "dummy.parquet"}
        manifest = build_dataset_manifest(
            dataset, labels, folds, rul_config, output_paths, "abc123", logger
        )
        assert manifest["total_training_rows"] == len(dataset)
        assert manifest["config_fingerprint"] == "abc123"
        assert uid("bearing_3") in manifest["per_unit"]
        assert manifest["per_unit"][uid("bearing_1")]["type"] == "censored"
        assert manifest["per_unit"][uid("bearing_3")]["type"] == "rtf"

    def test_contract_schema_has_phase6_requirements(self, full_pipeline_state, rul_config, logger):
        fs, labels, folds, dataset = full_pipeline_state
        contract = build_phase6_contract(dataset, labels, folds, rul_config, "abc123", logger)
        assert "phase6_requirements" in contract
        assert "scaling" in contract["phase6_requirements"]
        assert contract["rul_specification"]["value_at_failure"] == 0.0
        assert contract["artifacts"]["training_dataset"]["total_rows"] == len(dataset)

    def test_manifest_deterministic_for_same_input(self, full_pipeline_state, rul_config, logger, tmp_path):
        fs, labels, folds, dataset = full_pipeline_state
        output_paths = {"dummy": tmp_path / "dummy.parquet"}
        m1 = build_dataset_manifest(dataset, labels, folds, rul_config, output_paths, "fp1", logger)
        m2 = build_dataset_manifest(dataset, labels, folds, rul_config, output_paths, "fp1", logger)
        assert m1["per_unit"] == m2["per_unit"]
        assert m1["fold_summary"] == m2["fold_summary"]


# ──────────────────────────────────────────────────────────────────────────────
# Final validation tests
# ──────────────────────────────────────────────────────────────────────────────

class TestFinalValidation:

    @pytest.fixture
    def full_pipeline_state(self, synthetic_feature_store, rul_config, logger):
        index = build_failure_index(synthetic_feature_store, rul_config, logger)
        labels = generate_rul_labels(synthetic_feature_store, index, rul_config, logger)
        folds = generate_fold_assignments(labels, rul_config, logger)
        dataset = build_training_dataset(synthetic_feature_store, labels, folds, rul_config, logger)
        return labels, folds, dataset

    def test_passes_on_clean_dataset(self, full_pipeline_state, rul_config, logger, tmp_path):
        labels, folds, dataset = full_pipeline_state
        fake_paths = {}
        for key in (
            "rul_labels", "training_dataset", "fold_assignments", "dataset_manifest",
            "label_dictionary", "dataset_contract", "leakage_report", "split_summary",
        ):
            p = tmp_path / f"{key}.txt"
            p.write_text("placeholder", encoding="utf-8")
            fake_paths[key] = p

        result = run_final_validation(dataset, labels, folds, rul_config, fake_paths, logger)
        assert result.all_passed is True
        assert result.fatal_count == 0

    def test_fails_on_missing_artifact(self, full_pipeline_state, rul_config, logger, tmp_path):
        labels, folds, dataset = full_pipeline_state
        fake_paths = {"rul_labels": tmp_path / "does_not_exist.parquet"}
        result = run_final_validation(dataset, labels, folds, rul_config, fake_paths, logger)
        assert result.all_passed is False

    def test_fails_on_negative_rul(self, full_pipeline_state, rul_config, logger, tmp_path):
        labels, folds, dataset = full_pipeline_state
        corrupted = dataset.copy()
        corrupted.loc[corrupted.index[0], "rul_minutes"] = -5.0
        fake_paths = {}
        for key in (
            "rul_labels", "training_dataset", "fold_assignments", "dataset_manifest",
            "label_dictionary", "dataset_contract", "leakage_report", "split_summary",
        ):
            p = tmp_path / f"{key}.txt"
            p.write_text("placeholder", encoding="utf-8")
            fake_paths[key] = p

        result = run_final_validation(corrupted, labels, folds, rul_config, fake_paths, logger)
        assert result.all_passed is False

    def test_fails_on_missing_fold_column(self, full_pipeline_state, rul_config, logger, tmp_path):
        labels, folds, dataset = full_pipeline_state
        corrupted = dataset.drop(columns=["fold_1_role"])
        fake_paths = {}
        for key in (
            "rul_labels", "training_dataset", "fold_assignments", "dataset_manifest",
            "label_dictionary", "dataset_contract", "leakage_report", "split_summary",
        ):
            p = tmp_path / f"{key}.txt"
            p.write_text("placeholder", encoding="utf-8")
            fake_paths[key] = p

        result = run_final_validation(corrupted, labels, folds, rul_config, fake_paths, logger)
        assert result.all_passed is False


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility tests
# ──────────────────────────────────────────────────────────────────────────────

class TestReproducibility:

    def test_label_generation_is_deterministic(self, synthetic_feature_store, rul_config, logger):
        index1 = build_failure_index(synthetic_feature_store, rul_config, logger)
        labels1 = generate_rul_labels(synthetic_feature_store, index1, rul_config, logger)

        index2 = build_failure_index(synthetic_feature_store, rul_config, logger)
        labels2 = generate_rul_labels(synthetic_feature_store, index2, rul_config, logger)

        pd.testing.assert_frame_equal(
            labels1.reset_index(drop=True), labels2.reset_index(drop=True)
        )

    def test_fold_assignment_is_deterministic(self, synthetic_feature_store, rul_config, logger):
        index = build_failure_index(synthetic_feature_store, rul_config, logger)
        labels = generate_rul_labels(synthetic_feature_store, index, rul_config, logger)

        folds1 = generate_fold_assignments(labels, rul_config, logger)
        folds2 = generate_fold_assignments(labels, rul_config, logger)

        pd.testing.assert_frame_equal(
            folds1.reset_index(drop=True), folds2.reset_index(drop=True)
        )
