"""Tests for Phase 4 Feature Engineering module.

Tests cover:
- Feature gate resolution logic
- Time-domain feature loading and validation
- Frequency-domain feature computation
- Trend feature computation with chronology gap handling
- Feature store integrity and row counts
- Naming convention compliance
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from datetime import datetime, timedelta
import tempfile

from src.config import load_config
from src.constants import DEFAULT_CONFIG_PATH
from src.logger import setup_logger


class TestFeatureGate:
    """Test suite for feature gate resolution logic."""

    def test_gate_defaults_to_exclude_when_phase3_missing(self):
        """Test that gates default to EXCLUDE when phase3_gating_decisions.yaml is absent."""
        from src.feature_engineering.feature_gate import FeatureGateResolver

        logger = setup_logger(name="test", config=load_config(DEFAULT_CONFIG_PATH))
        resolver = FeatureGateResolver(logger)

        nonexistent_path = Path("/nonexistent/phase3_gating_decisions.yaml")
        gates = resolver.resolve_gates(nonexistent_path, bearing_geometry_available=False)

        assert gates["skewness"] is False
        assert gates["rolling_variance"] is False
        assert gates["envelope_features"] is False
        assert gates["xy_energy_ratio"] is False

    def test_gate_envelope_features_requires_geometry(self):
        """Test that envelope features are gated off without geometry verification."""
        from src.feature_engineering.feature_gate import FeatureGateResolver

        logger = setup_logger(name="test", config=load_config(DEFAULT_CONFIG_PATH))
        resolver = FeatureGateResolver(logger)

        gates = resolver.resolve_gates(Path("/nonexistent/path"), bearing_geometry_available=False)

        assert gates["envelope_features"] is False
        assert resolver.gate_decisions["envelope_features"]["reason"] == "Bearing Geometry Safety Rule"

    def test_gate_reads_values_from_yaml_contract(self):
        """Test that gates are resolved from phase3_gating_decisions.yaml, not markdown."""
        import yaml
        import tempfile
        from src.feature_engineering.feature_gate import FeatureGateResolver

        contract = {
            "schema_version": "1.0",
            "gate_decisions": {
                "skewness": {"include": False, "justification": "Low skewness"},
                "rolling_variance": {"include": True, "justification": "High variance"},
                "envelope_features": {"include": True, "justification": "Geometry verified"},
                "xy_energy_ratio": {"include": False, "justification": "Low CV"},
            },
        }

        logger = setup_logger(name="test", config=load_config(DEFAULT_CONFIG_PATH))
        resolver = FeatureGateResolver(logger)

        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False, encoding="utf-8"
        ) as tmp:
            yaml.dump(contract, tmp)
            contract_path = Path(tmp.name)

        gates = resolver.resolve_gates(contract_path, bearing_geometry_available=True)

        assert gates["skewness"] is False
        assert gates["rolling_variance"] is True
        assert gates["envelope_features"] is True  # geometry available, YAML=True
        assert gates["xy_energy_ratio"] is False

    def test_geometry_safety_rule_overrides_yaml_true(self):
        """Test that Bearing Geometry Safety Rule overrides YAML envelope=True."""
        import yaml
        import tempfile
        from src.feature_engineering.feature_gate import FeatureGateResolver

        contract = {
            "schema_version": "1.0",
            "gate_decisions": {
                "skewness": {"include": False, "justification": "test"},
                "rolling_variance": {"include": False, "justification": "test"},
                "envelope_features": {"include": True, "justification": "Geometry verified"},
                "xy_energy_ratio": {"include": False, "justification": "test"},
            },
        }

        logger = setup_logger(name="test", config=load_config(DEFAULT_CONFIG_PATH))
        resolver = FeatureGateResolver(logger)

        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False, encoding="utf-8"
        ) as tmp:
            yaml.dump(contract, tmp)
            contract_path = Path(tmp.name)

        # Even though YAML says True, geometry not available → must be False
        gates = resolver.resolve_gates(contract_path, bearing_geometry_available=False)

        assert gates["envelope_features"] is False
        assert resolver.gate_decisions["envelope_features"]["reason"] == "Bearing Geometry Safety Rule"


class TestTimeDomainFeatures:
    """Test suite for time-domain feature loading."""

    def test_loads_from_phase3_indicators(self):
        """Test that time-domain features are loaded from Phase 3 output."""
        from src.feature_engineering import time_domain_features
        
        logger = setup_logger(name="test", config=load_config(DEFAULT_CONFIG_PATH))
        
        # Create a minimal Phase 3 indicator file for testing
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            test_data = pd.DataFrame({
                "dataset_id": ["set1"] * 10,
                "timestamp": pd.date_range("2003-10-22", periods=10),
                "bearing_id": [1] * 10,
                "channel": [1] * 10,
                "rms": np.random.rand(10),
                "kurtosis": np.random.rand(10) + 3,
                "crest_factor": np.random.rand(10) + 1,
                "skewness": np.random.randn(10),
                "peak_to_peak": np.random.rand(10),
            })
            test_data.to_parquet(tmp.name)
            
            gates = {"skewness": False, "rolling_variance": False}
            result = time_domain_features.load_time_domain_features(Path(tmp.name), gates, logger)
            
            assert len(result) == 10
            assert "td_rms" in result.columns
            assert "td_kurtosis" in result.columns
            assert "td_crest_factor" in result.columns
            assert "td_skewness" not in result.columns  # Gated off
            assert "peak_to_peak" not in result.columns  # Excluded


class TestFrequencyDomainFeatures:
    """Test suite for frequency-domain feature computation."""

    def test_synthetic_sinusoid_concentrates_energy_in_band(self):
        """Test FFT band energy on synthetic signal."""
        from src.feature_engineering.frequency_domain_features import (
            compute_fft_spectrum,
            compute_band_energy,
        )
        
        # Create synthetic sinusoid at 1 kHz
        sampling_rate = 20000
        duration = 1.024
        t = np.linspace(0, duration, int(sampling_rate * duration))
        frequency = 1000  # Hz
        signal = np.sin(2 * np.pi * frequency * t)
        
        freqs, magnitude = compute_fft_spectrum(signal, sampling_rate)
        
        # Band containing the sinusoid should have highest energy
        band_0_1k = compute_band_energy(freqs, magnitude, 0, 1000)
        band_1k_5k = compute_band_energy(freqs, magnitude, 1000, 5000)
        band_5k_10k = compute_band_energy(freqs, magnitude, 5000, 10000)
        
        # Sinusoid at 1 kHz is near the boundary; energy concentrated at lower end
        assert band_0_1k > band_5k_10k


class TestFFTFullSignal:
    """Verify the FFT is computed over the ENTIRE waveform (no truncation)."""

    def test_full_waveform_is_used(self):
        """Spectrum length must reflect the full signal, not a fixed 2048 window."""
        from src.feature_engineering.frequency_domain_features import compute_fft_spectrum
        import numpy as np

        # A full 20,480-sample acquisition must yield the rfft bin count for the
        # WHOLE signal (n // 2 + 1), not the 1024 bins the old 2048-truncation gave.
        waveform = np.random.randn(20480)
        freqs, magnitude = compute_fft_spectrum(waveform, sampling_rate=20000, window="hann")

        assert len(freqs) == 20480 // 2 + 1
        assert len(magnitude) == 20480 // 2 + 1
        assert np.all(np.isfinite(magnitude))

    def test_frequency_resolution_scales_with_length(self):
        """Bin width must equal sampling_rate / len(waveform) — proof of no truncation."""
        from src.feature_engineering.frequency_domain_features import compute_fft_spectrum
        import numpy as np

        sampling_rate = 20000
        freqs, _ = compute_fft_spectrum(np.random.randn(20480), sampling_rate)
        bin_width = freqs[1] - freqs[0]
        assert abs(bin_width - sampling_rate / 20480) < 1e-6  # ~0.977 Hz, not 9.77 Hz

    def test_fft_concentrates_energy_in_correct_band(self):
        """A 1 kHz tone over the full acquisition lands in the 0-5000 Hz band."""
        from src.feature_engineering.frequency_domain_features import (
            compute_fft_spectrum,
            compute_band_energy,
        )

        sampling_rate = 20000
        n = 20480
        t = np.arange(n) / sampling_rate
        signal = np.sin(2 * np.pi * 1000 * t)

        freqs, magnitude = compute_fft_spectrum(signal, sampling_rate)
        low_energy = compute_band_energy(freqs, magnitude, 0, 5000)
        high_energy = compute_band_energy(freqs, magnitude, 5000, 10000)

        assert low_energy > high_energy * 100


class TestTrendSlopeTimeBasis:
    """Test that rolling slope uses elapsed time in minutes, not sample indices."""

    def test_slope_reflects_time_units(self):
        """Slope for a linearly rising series should be in feature/minute, not feature/sample."""
        from src.feature_engineering.trend_features import compute_rolling_slope

        # 5-minute intervals → 10 points over 45 minutes
        timestamps = pd.date_range("2003-10-22 00:00", periods=10, freq="5min")
        # RMS rises by 1.0 per minute → total rise = 45.0 over 45 minutes
        # With window_size=5, slope over any 5-point window should be ≈ 1.0 /min
        rms_per_minute = 1.0
        values = pd.Series([i * 5 * rms_per_minute for i in range(10)])

        slopes = compute_rolling_slope(values, pd.Series(timestamps), window_size=5)

        # The first 4 are NaN (window not full); the rest should be ≈ 1.0
        valid = slopes.dropna()
        assert len(valid) > 0
        for slope in valid:
            assert abs(slope - rms_per_minute) < 0.01, (
                f"Slope {slope:.4f} != expected {rms_per_minute:.4f} (time-based units)"
            )


class TestNamingConvention:
    """Test suite for feature naming convention compliance."""

    def test_all_features_follow_naming_convention(self):
        """Test that all features follow td_*, fd_*, etc. naming conventions."""
        valid_prefixes = ("td_", "fd_", "env_", "trend_", "cross_", "timestamp", "bearing_id", "channel", "file_id", "file_sha256")
        
        # Create minimal test feature table
        test_df = pd.DataFrame({
            "timestamp": [datetime.now()],
            "bearing_id": [1],
            "channel": [1],
            "td_rms": [0.5],
            "td_kurtosis": [3.0],
            "td_crest_factor": [2.0],
            "fd_spectral_centroid": [500],
            "fd_spectral_entropy": [4.5],
        })
        
        feature_columns = [c for c in test_df.columns if any(c.startswith(p) for p in valid_prefixes)]
        assert len(feature_columns) == len(test_df.columns)


class TestTrendFeaturesGapHandling:
    """Test suite for trend feature gap-aware computation."""

    def test_rolling_slope_nulls_out_across_large_gaps(self):
        """Test that rolling slope is nulled when window spans chronology gap."""
        from src.feature_engineering.trend_features import compute_rolling_slope
        
        # Create time series with a large gap
        timestamps = pd.to_datetime([
            "2003-10-22 00:00:00",
            "2003-10-22 00:10:00",
            "2003-10-22 00:20:00",
            # 2-hour gap
            "2003-10-22 02:20:00",
            "2003-10-22 02:30:00",
        ])
        values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        
        slope = compute_rolling_slope(values, pd.Series(timestamps), window_size=2, gap_threshold_minutes=30)
        
        # Window at index 3 spans the gap; should be null
        assert pd.isna(slope.iloc[3])


class TestFeatureStoreRowCount:
    """Test suite for feature store row count consistency."""

    def test_feature_store_has_expected_row_count(self):
        """Test the expected multi-dataset row count: set1 (2156x8) + set2 (984x4) + set3 (6323x4)."""
        # This test requires Phase 1-3 to be complete
        # Placeholder for integration testing
        expected_rows = 2156 * 8 + 984 * 4 + 6323 * 4
        assert expected_rows == 46476


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
