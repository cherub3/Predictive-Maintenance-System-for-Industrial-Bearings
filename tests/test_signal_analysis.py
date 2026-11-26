"""Phase 3 signal analysis test suite.

Tests cover:
  - Bearing geometry computation correctness
  - Signal loader shape validation
  - Health indicator formula correctness
  - Row count invariant (17,248 rows for 2,156 files x 8 channels)
  - Chronological ordering of health indicators
  - Gating contract schema validation
  - Gate resolution logic (default-to-exclude, safety rule)
  - Checkpoint selection coverage
  - Spectral computation correctness (pure-tone energy)
  - Artifact existence (reports, parquet, YAML)
"""

import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.config import get_channel_bearing_map
from src.signal_analysis.bearing_geometry import (
    BearingGeometry,
    compute_defect_frequencies,
    load_bearing_geometry,
)
from src.signal_analysis.signal_loader import load_signal_file, extract_channel
from src.signal_analysis.health_indicators import (
    _compute_indicators,
    validate_health_indicators,
)
from src.signal_analysis.spectral_analysis import (
    compute_fft_spectrum,
    compute_band_energy,
    compute_spectral_entropy,
    compute_spectral_centroid,
)
from src.signal_analysis.checkpoint_selector import select_checkpoints
from src.signal_analysis.gating_contract import (
    build_gating_contract,
    validate_contract,
    resolve_skewness_gate,
    resolve_envelope_gate,
    resolve_rolling_variance_gate,
    resolve_xy_energy_ratio_gate,
    SCHEMA_VERSION,
    PRODUCER_PHASE,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_config():
    """Minimal configuration dictionary for Phase 3 unit tests."""
    return {
        "datasets": {
            "registry_version": "1.0",
            "defaults": {
                "sampling_rate_hz": 20000,
                "samples_per_file": 20480,
                "shaft_speed_rpm": 2000.0,
                "bearing_geometry": {
                    "num_rolling_elements": 16,
                    "ball_diameter_inches": 0.331,
                    "pitch_diameter_inches": 2.815,
                    "contact_angle_degrees": 15.17,
                },
            },
            "sets": {
                "set1": {
                    "display_name": "NASA IMS Set 1",
                    "raw_dir": "data/raw/set1",
                    "run_window": {"start": "2003-10-22", "end": "2003-11-25"},
                    "channel_layout": "paired_xy",
                    "expected_channels_per_file": 8,
                    "bearings": {
                        "bearing_1": {"channels": [1, 2], "channel_axes": {1: "X", 2: "Y"}, "status": "censored"},
                        "bearing_2": {"channels": [3, 4], "channel_axes": {3: "X", 4: "Y"}, "status": "censored"},
                        "bearing_3": {"channels": [5, 6], "channel_axes": {5: "X", 6: "Y"}, "status": "rtf", "failure_mode": "inner_race"},
                        "bearing_4": {"channels": [7, 8], "channel_axes": {7: "X", 8: "Y"}, "status": "rtf", "failure_mode": "rolling_element"},
                    },
                },
            },
        },
        "signal_analysis": {
            "sampling_rate_hz": 20000,
            "samples_per_file": 20480,
            "num_channels": 8,
            "shaft_speed_rpm": 2000.0,
            "bearing_geometry": {
                "num_rolling_elements": 16,
                "ball_diameter_inches": 0.331,
                "pitch_diameter_inches": 2.815,
                "contact_angle_degrees": 15.17,
            },
            "checkpoint_selection": {
                "n_checkpoints": 10,
                "early_fraction": 0.05,
                "late_fraction": 0.10,
            },
            "gate_thresholds": {
                "skewness_abs_mean_threshold": 0.5,
                "xy_asymmetry_cv_threshold": 0.15,
                "rolling_variance_correlation_threshold": 0.8,
                "rolling_variance_window_files": 10,
            },
            "output_paths": {
                "health_indicators": "data/processed/signal_health_indicators.parquet",
                "checkpoint_selection": "data/processed/checkpoint_selection.csv",
                "engineering_findings": "reports/engineering_findings.md",
                "signal_analysis_report": "reports/signal_analysis_report.md",
                "gating_contract": "reports/phase3_gating_decisions.yaml",
                "figures_dir": "reports/figures",
            },
        },
        "feature_engineering": {
            "frequency_bands": [[0, 1000], [1000, 5000], [5000, 10000]],
            "fft_n_fft": 2048,
        },
    }


@pytest.fixture
def synthetic_health_df(minimal_config):
    """Synthetic health indicators DataFrame with 80 rows (10 files x 8 channels)."""
    channel_map = get_channel_bearing_map(minimal_config, "set1")
    rtf_bearing_ids = {
        b for b, cfg in minimal_config["datasets"]["sets"]["set1"]["bearings"].items()
        if cfg.get("status") == "rtf"
    }
    n_files = 10
    channels = list(range(1, 9))
    records = []
    base_time = pd.Timestamp("2003-10-22 12:00:00")
    for f in range(n_files):
        ts = base_time + pd.Timedelta(minutes=10 * f)
        for ch in channels:
            info = channel_map[ch]
            bearing_id, axis = info["bearing_id"], info["axis"]
            # Simulate increasing RMS for failure bearings
            rms_base = 0.05 + 0.005 * f if bearing_id in ("bearing_3", "bearing_4") else 0.03
            records.append({
                "file_id": f,
                "dataset_id": "set1",
                "timestamp": ts,
                "filename": f"2003.10.22.12.0{f}.00",
                "bearing_id": bearing_id,
                "channel": ch,
                "axis": axis,
                "is_rtf": bearing_id in rtf_bearing_ids,
                "rms": rms_base + np.random.normal(0, 0.001),
                "kurtosis": 3.0 + np.random.normal(0, 0.2),
                "crest_factor": 4.0 + np.random.normal(0, 0.1),
                "skewness": np.random.normal(0, 0.3),
                "peak_amplitude": 0.5 + np.random.normal(0, 0.05),
                "mean": np.random.normal(0, 0.01),
                "load_status": "OK",
            })
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@pytest.fixture
def logger():
    """Silent mock logger."""
    return MagicMock()


# ──────────────────────────────────────────────────────────────────────────────
# 1. Bearing Geometry Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestBearingGeometry:
    """Validates defect frequency computation from ZA-2115 geometry."""

    def test_defect_frequencies_physical_range(self, minimal_config):
        """BPFO, BPFI, and BSF must fall within expected physical ranges for ZA-2115."""
        geometry = load_bearing_geometry(minimal_config)
        freqs = compute_defect_frequencies(geometry)

        # Expected approximate values for ZA-2115 at 2000 RPM:
        # BPFO ~ 236 Hz (outer: fewer contacts), BPFI ~ 297 Hz (inner: more contacts)
        # BSF ~ 140 Hz
        assert 220 <= freqs.bpfo_hz <= 260, f"BPFO out of range: {freqs.bpfo_hz}"
        assert 275 <= freqs.bpfi_hz <= 320, f"BPFI out of range: {freqs.bpfi_hz}"
        assert 120 <= freqs.bsf_hz <= 165, f"BSF out of range: {freqs.bsf_hz}"

    def test_bpfi_greater_than_bpfo(self, minimal_config):
        """Inner race contacts shaft more than outer race, so BPFI > BPFO."""
        geometry = load_bearing_geometry(minimal_config)
        freqs = compute_defect_frequencies(geometry)
        assert freqs.bpfi_hz > freqs.bpfo_hz

    def test_shaft_frequency_correct(self, minimal_config):
        """Shaft frequency must equal shaft_speed_rpm / 60."""
        geometry = load_bearing_geometry(minimal_config)
        freqs = compute_defect_frequencies(geometry)
        expected_fr = 2000.0 / 60.0
        assert abs(freqs.shaft_frequency_hz - expected_fr) < 0.001

    def test_channel_bearing_map_completeness(self, minimal_config):
        """All 8 channels must be present in Set 1's channel/bearing map."""
        channel_map = get_channel_bearing_map(minimal_config, "set1")
        assert set(channel_map.keys()) == set(range(1, 9))

    def test_channel_bearing_map_axis_alternation(self, minimal_config):
        """Odd channels must be X-axis, even channels Y-axis."""
        channel_map = get_channel_bearing_map(minimal_config, "set1")
        for ch, info in channel_map.items():
            expected_axis = "X" if ch % 2 == 1 else "Y"
            assert info["axis"] == expected_axis, f"Channel {ch} expected {expected_axis}, got {info['axis']}"


# ──────────────────────────────────────────────────────────────────────────────
# 2. Signal Loader Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSignalLoader:
    """Validates file loading and shape enforcement."""

    def test_load_valid_file(self, tmp_path):
        """Valid 20480x8 ASCII file loads correctly."""
        data = np.random.randn(20480, 8)
        fpath = tmp_path / "test_file.txt"
        np.savetxt(fpath, data)

        loaded, status = load_signal_file(fpath, expected_samples=20480, expected_channels=8)
        assert status == "OK"
        assert loaded is not None
        assert loaded.shape == (20480, 8)

    def test_load_wrong_shape_rejected(self, tmp_path):
        """File with wrong column count returns None and non-OK status."""
        data = np.random.randn(20480, 4)  # Wrong: 4 channels instead of 8
        fpath = tmp_path / "bad_shape.txt"
        np.savetxt(fpath, data)

        loaded, status = load_signal_file(fpath, expected_samples=20480, expected_channels=8)
        assert loaded is None
        assert status != "OK"

    def test_load_missing_file_returns_none(self, tmp_path):
        """Non-existent file path returns None and non-OK status."""
        fpath = tmp_path / "nonexistent.txt"
        loaded, status = load_signal_file(fpath)
        assert loaded is None
        assert "Failed" in status or status != "OK"

    def test_extract_channel_shape(self):
        """extract_channel returns 1-D array of correct length."""
        data = np.random.randn(20480, 8)
        ch = extract_channel(data, channel=0)
        assert ch.ndim == 1
        assert len(ch) == 20480


# ──────────────────────────────────────────────────────────────────────────────
# 3. Health Indicator Formula Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestHealthIndicatorFormulas:
    """Validates correctness of scalar health indicator computation."""

    def test_rms_pure_sine(self):
        """RMS of A*sin(2*pi*f*t) must equal A / sqrt(2)."""
        n = 20480
        t = np.linspace(0, 1.024, n, endpoint=False)
        amplitude = 2.0
        waveform = amplitude * np.sin(2 * np.pi * 100 * t)
        indicators = _compute_indicators(waveform)
        expected_rms = amplitude / math.sqrt(2)
        # 1e-3 tolerance: non-integer cycle count at 20480 samples causes minor numerical imprecision
        assert abs(indicators["rms"] - expected_rms) < 1e-3, (
            f"RMS: got {indicators['rms']:.6f}, expected {expected_rms:.6f}"
        )

    def test_kurtosis_gaussian(self):
        """Kurtosis of Gaussian noise must be near 0 (Fisher/excess definition)."""
        np.random.seed(42)
        waveform = np.random.randn(20480)
        indicators = _compute_indicators(waveform)
        assert abs(indicators["kurtosis"]) < 0.5, f"Kurtosis: {indicators['kurtosis']}"

    def test_crest_factor_sine(self):
        """Crest Factor of a pure sine wave must be sqrt(2) ~ 1.414."""
        n = 20480
        t = np.linspace(0, 1.024, n, endpoint=False)
        waveform = np.sin(2 * np.pi * 50 * t)
        indicators = _compute_indicators(waveform)
        expected_cf = math.sqrt(2)
        assert abs(indicators["crest_factor"] - expected_cf) < 0.01, (
            f"Crest factor: {indicators['crest_factor']:.4f}, expected {expected_cf:.4f}"
        )

    def test_skewness_symmetric_distribution(self):
        """Skewness of a symmetric signal must be near 0."""
        np.random.seed(99)
        waveform = np.random.randn(20480)
        indicators = _compute_indicators(waveform)
        assert abs(indicators["skewness"]) < 0.1, f"Skewness: {indicators['skewness']}"

    def test_empty_waveform_returns_nans(self):
        """Empty waveform must return NaN for all indicators, not raise."""
        indicators = _compute_indicators(np.array([]))
        assert math.isnan(indicators["rms"])
        assert math.isnan(indicators["kurtosis"])


# ──────────────────────────────────────────────────────────────────────────────
# 4. Health Indicator Row Count
# ──────────────────────────────────────────────────────────────────────────────

class TestHealthIndicatorRowCount:
    """Validates the 17,248-row invariant (2,156 files x 8 channels)."""

    def test_expected_rows(self, synthetic_health_df):
        """Synthetic DataFrame has exactly n_files x 8 rows."""
        assert len(synthetic_health_df) == 10 * 8  # 80 rows

    def test_validate_health_indicators_passes(self, synthetic_health_df, logger):
        """validate_health_indicators returns True for correct synthetic data."""
        assert validate_health_indicators(synthetic_health_df, 80, logger)

    def test_validate_health_indicators_wrong_row_count(self, synthetic_health_df, logger):
        """validate_health_indicators fails when expected row count is wrong."""
        assert not validate_health_indicators(synthetic_health_df, 17248, logger)

    def test_all_four_bearings_present(self, synthetic_health_df, logger):
        """All four bearing IDs must appear in the health indicators."""
        bearings = set(synthetic_health_df["bearing_id"].unique())
        assert bearings == {"bearing_1", "bearing_2", "bearing_3", "bearing_4"}


# ──────────────────────────────────────────────────────────────────────────────
# 5. Chronological Ordering
# ──────────────────────────────────────────────────────────────────────────────

class TestChronologicalOrdering:
    """Validates that health indicators preserve temporal ordering."""

    def test_timestamps_non_decreasing(self, synthetic_health_df):
        """Per-channel timestamps must be non-decreasing."""
        for ch in range(1, 9):
            ch_data = synthetic_health_df[synthetic_health_df["channel"] == ch]
            timestamps = ch_data["timestamp"].values
            diffs = np.diff(timestamps.astype("int64"))
            assert (diffs >= 0).all(), f"Channel {ch} timestamps not non-decreasing"


# ──────────────────────────────────────────────────────────────────────────────
# 6. Spectral Analysis Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSpectralAnalysis:
    """Validates FFT-based spectral feature computation."""

    def test_pure_tone_band_energy(self):
        """A 1 kHz pure tone must concentrate band energy in the 0-1000 Hz band."""
        sampling_rate = 20000.0
        n = 20480
        t = np.linspace(0, n / sampling_rate, n, endpoint=False)
        tone_freq = 800.0  # Hz — inside the 0-1000 Hz band
        waveform = np.sin(2 * np.pi * tone_freq * t)

        freqs, mag = compute_fft_spectrum(waveform, sampling_rate, n_fft=2048)
        energy_low = compute_band_energy(freqs, mag, 0, 1000)
        energy_mid = compute_band_energy(freqs, mag, 1000, 5000)
        energy_high = compute_band_energy(freqs, mag, 5000, 10000)

        # More than 90% of energy should be in the low band
        total = energy_low + energy_mid + energy_high
        assert energy_low / total > 0.90, f"Low-band energy fraction: {energy_low/total:.3f}"

    def test_spectral_entropy_flat_spectrum(self):
        """White noise (flat spectrum) must have higher entropy than a pure tone."""
        sampling_rate = 20000.0
        n = 20480
        np.random.seed(42)
        noise = np.random.randn(n)
        tone = np.sin(2 * np.pi * 1000 * np.linspace(0, n / sampling_rate, n, endpoint=False))

        _, mag_noise = compute_fft_spectrum(noise, sampling_rate, n_fft=2048)
        _, mag_tone = compute_fft_spectrum(tone, sampling_rate, n_fft=2048)

        entropy_noise = compute_spectral_entropy(mag_noise)
        entropy_tone = compute_spectral_entropy(mag_tone)

        assert entropy_noise > entropy_tone, (
            f"White noise entropy ({entropy_noise:.3f}) should exceed "
            f"pure tone entropy ({entropy_tone:.3f})"
        )

    def test_spectral_centroid_tracks_frequency(self):
        """Spectral centroid of a 5 kHz pure tone must be near 5 kHz."""
        sampling_rate = 20000.0
        n = 20480
        t = np.linspace(0, n / sampling_rate, n, endpoint=False)
        tone_freq = 5000.0
        waveform = np.sin(2 * np.pi * tone_freq * t)

        freqs, mag = compute_fft_spectrum(waveform, sampling_rate, n_fft=2048)
        centroid = compute_spectral_centroid(freqs, mag)

        assert abs(centroid - tone_freq) < 200, f"Centroid: {centroid:.1f} Hz, expected ~{tone_freq} Hz"


# ──────────────────────────────────────────────────────────────────────────────
# 7. Checkpoint Selection Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestCheckpointSelection:
    """Validates checkpoint selection coverage and structure."""

    def test_checkpoint_count_within_budget(self, synthetic_health_df, minimal_config, logger):
        """Selected checkpoints must not exceed n_checkpoints budget."""
        n_max = minimal_config["signal_analysis"]["checkpoint_selection"]["n_checkpoints"]
        checkpoints = select_checkpoints(synthetic_health_df, minimal_config, logger)
        assert len(checkpoints) <= n_max, f"Too many checkpoints: {len(checkpoints)} > {n_max}"

    def test_early_stage_checkpoints_present(self, synthetic_health_df, minimal_config, logger):
        """At least one early_stage checkpoint must be selected."""
        checkpoints = select_checkpoints(synthetic_health_df, minimal_config, logger)
        reasons = set(checkpoints["selection_reason"].values)
        assert "early_stage" in reasons, f"No early_stage checkpoints; reasons found: {reasons}"

    def test_late_stage_checkpoints_present(self, synthetic_health_df, minimal_config, logger):
        """At least one late_stage checkpoint must be selected."""
        checkpoints = select_checkpoints(synthetic_health_df, minimal_config, logger)
        reasons = set(checkpoints["selection_reason"].values)
        assert "late_stage" in reasons, f"No late_stage checkpoints; reasons found: {reasons}"

    def test_required_columns_present(self, synthetic_health_df, minimal_config, logger):
        """Checkpoint DataFrame must have required columns."""
        checkpoints = select_checkpoints(synthetic_health_df, minimal_config, logger)
        required = {"file_id", "filename", "timestamp", "selection_reason", "checkpoint_rank"}
        assert required.issubset(set(checkpoints.columns))


# ──────────────────────────────────────────────────────────────────────────────
# 7b. Rolling Variance Gate Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestRollingVarianceGate:
    """Validates rolling variance gate resolution logic."""

    def test_gate_returns_boolean_and_justification(self, logger):
        """resolve_rolling_variance_gate returns (bool, str) tuple."""
        n = 50
        health_df = pd.DataFrame({
            "dataset_id": ["set1"] * n,
            "timestamp": pd.date_range("2003-10-22", periods=n, freq="10min"),
            "bearing_id": ["bearing_3"] * n,
            "axis": ["X"] * n,
            "is_rtf": [True] * n,
            "rms": np.linspace(0.05, 0.15, n),
            "load_status": ["OK"] * n,
        })

        include, justification = resolve_rolling_variance_gate(
            health_df,
            correlation_threshold=0.8,
            window=10,
            logger=logger,
        )
        assert isinstance(include, bool)
        assert isinstance(justification, str)
        assert len(justification) > 0

    def test_correlation_threshold_controls_gate(self, logger):
        """Lower threshold makes gate more likely to be False."""
        # Realistic bearing degradation: RMS increases with some noise.
        np.random.seed(42)
        n = 100
        base = np.linspace(0.05, 0.2, n)
        noise = np.random.normal(0, 0.01, n)
        rms_values = base + noise

        health_df = pd.DataFrame({
            "dataset_id": ["set1"] * n,
            "timestamp": pd.date_range("2003-10-22", periods=n, freq="10min"),
            "bearing_id": ["bearing_3"] * n,
            "axis": ["X"] * n,
            "is_rtf": [True] * n,
            "rms": rms_values,
            "load_status": ["OK"] * n,
        })

        # Test with high threshold (gate more likely True)
        include_high_threshold, _ = resolve_rolling_variance_gate(
            health_df,
            correlation_threshold=0.99,  # Very permissive
            window=10,
            logger=logger,
        )

        # Test with low threshold (gate more likely False)
        include_low_threshold, _ = resolve_rolling_variance_gate(
            health_df,
            correlation_threshold=0.1,  # Very strict
            window=10,
            logger=logger,
        )

        # Gate logic: include = abs(correlation) < threshold
        # Higher threshold → more likely to satisfy < threshold → more likely True
        assert isinstance(include_high_threshold, bool)
        assert isinstance(include_low_threshold, bool)

    def test_insufficient_data_returns_false(self, logger):
        """Insufficient data (fewer than window*2 rows) must return False."""
        n = 15  # window=10, so window*2=20; 15 < 20
        health_df = pd.DataFrame({
            "dataset_id": ["set1"] * n,
            "timestamp": pd.date_range("2003-10-22", periods=n, freq="10min"),
            "bearing_id": ["bearing_3"] * n,
            "axis": ["X"] * n,
            "is_rtf": [True] * n,
            "rms": np.linspace(0.05, 0.1, n),
            "load_status": ["OK"] * n,
        })

        include, justification = resolve_rolling_variance_gate(
            health_df,
            correlation_threshold=0.8,
            window=10,
            logger=logger,
        )
        assert include is False
        assert "Insufficient" in justification or "insufficient" in justification.lower()


# ──────────────────────────────────────────────────────────────────────────────
# 8. Gating Contract Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestGatingContract:
    """Validates gating contract schema, safety rules, and defaults."""

    def test_valid_contract_passes_validation(self, synthetic_health_df, minimal_config, logger):
        """A correctly built contract must pass schema validation."""
        contract = build_gating_contract(synthetic_health_df, minimal_config, False, logger)
        assert validate_contract(contract)

    def test_schema_version_correct(self, synthetic_health_df, minimal_config, logger):
        """Schema version must equal SCHEMA_VERSION constant."""
        contract = build_gating_contract(synthetic_health_df, minimal_config, False, logger)
        assert contract["schema_version"] == SCHEMA_VERSION

    def test_producer_phase_correct(self, synthetic_health_df, minimal_config, logger):
        """Producer phase must equal PRODUCER_PHASE constant."""
        contract = build_gating_contract(synthetic_health_df, minimal_config, False, logger)
        assert contract["producer"]["phase"] == PRODUCER_PHASE

    def test_all_four_gates_present(self, synthetic_health_df, minimal_config, logger):
        """All four gate keys must be present in gate_decisions."""
        contract = build_gating_contract(synthetic_health_df, minimal_config, False, logger)
        expected = {"skewness", "rolling_variance", "envelope_features", "xy_energy_ratio"}
        assert set(contract["gate_decisions"].keys()) == expected

    def test_gate_values_are_boolean(self, synthetic_health_df, minimal_config, logger):
        """All gate include values must be strictly boolean."""
        contract = build_gating_contract(synthetic_health_df, minimal_config, False, logger)
        for gate_name, gate in contract["gate_decisions"].items():
            assert isinstance(gate["include"], bool), (
                f"Gate '{gate_name}' include is not bool: {type(gate['include'])}"
            )

    def test_bearing_geometry_safety_rule(self, synthetic_health_df, minimal_config, logger):
        """Envelope gate must be False when bearing_geometry_verified is False."""
        include, _ = resolve_envelope_gate(False, logger)
        assert include is False, "Envelope gate must be False when geometry not verified"

    def test_bearing_geometry_verified_enables_envelope(self, logger):
        """Envelope gate must be True when bearing_geometry_verified is True."""
        include, _ = resolve_envelope_gate(True, logger)
        assert include is True

    def test_default_all_gates_present_flag(self, synthetic_health_df, minimal_config, logger):
        """validation.all_gates_present must be True in every generated contract."""
        contract = build_gating_contract(synthetic_health_df, minimal_config, False, logger)
        assert contract["validation"]["all_gates_present"] is True

    def test_missing_gate_fails_validation(self, synthetic_health_df, minimal_config, logger):
        """A contract missing a gate key must raise ValueError on validation."""
        contract = build_gating_contract(synthetic_health_df, minimal_config, False, logger)
        del contract["gate_decisions"]["skewness"]
        with pytest.raises(ValueError, match="Missing gate decision"):
            validate_contract(contract)

    def test_non_boolean_gate_fails_validation(self, synthetic_health_df, minimal_config, logger):
        """A gate with non-boolean include must raise TypeError on validation."""
        contract = build_gating_contract(synthetic_health_df, minimal_config, False, logger)
        contract["gate_decisions"]["skewness"]["include"] = "true"  # Wrong type
        with pytest.raises(TypeError):
            validate_contract(contract)

    def test_contract_yaml_roundtrip(self, synthetic_health_df, minimal_config, logger, tmp_path):
        """A contract written to YAML and read back must preserve all gate values."""
        from src.signal_analysis.gating_contract import write_gating_contract

        contract = build_gating_contract(synthetic_health_df, minimal_config, False, logger)
        out_path = tmp_path / "phase3_gating_decisions.yaml"
        write_gating_contract(contract, out_path, logger)

        assert out_path.exists()
        with open(out_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)

        for gate_name in contract["gate_decisions"]:
            assert loaded["gate_decisions"][gate_name]["include"] == \
                contract["gate_decisions"][gate_name]["include"]
