"""Comprehensive test suite for Phase 7 — Maintenance Decision Engine.

Architecture §17. Tests verify:
  • Risk classification (§17a TestRiskClassifier)
  • Maintenance policy (§17a TestMaintenancePolicy)
  • Confidence estimation (§17a TestConfidenceEstimator)
  • Priority calculation (§17a TestPriorityCalculator)
  • Business rules BR-01–BR-05 (§17a TestBusinessRules)
  • Channel aggregation (§17a TestChannelAggregation)
  • Decision explainer (§17a TestDecisionExplainer)
  • Contract loader (§17a TestContractLoader)
  • Threshold validation (§17a TestThresholdValidation)
  • Integration / determinism (§17b)
  • Business rule precedence + conflict resolution (§17c)
  • Decision consistency (§17d)
  • Contract schema (§17e)
"""

import json
import math
from typing import Any, Dict
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.decision_engine.business_rules import (
    RULE_VERSION,
    BusinessRuleResult,
    apply_business_rules,
    apply_missing_prediction_fallback,
)
from src.decision_engine.confidence_estimator import (
    ConfidenceLevel,
    _apply_boundary_proximity,
    _apply_model_type_modifier,
    _apply_n2_penalty,
    estimate_confidence,
)
from src.decision_engine.contract_loader import validate_phase6_contract
from src.decision_engine.decision_engine import aggregate_channels, run_decision_pipeline
from src.decision_engine.decision_explainer import (
    _build_model_selection_note,
    generate_explanation,
)
from src.decision_engine.experiment_tracker import (
    compute_config_fingerprint,
    generate_experiment_id,
)
from src.decision_engine.maintenance_policy import (
    MaintenanceAction,
    get_maintenance_action,
    load_policy_map,
)
from src.decision_engine.priority_calculator import compute_priority, load_priority_matrix
from src.decision_engine.recommendation_engine import (
    Recommendation,
    assemble_recommendation,
    recommendations_to_dataframe,
)
from src.decision_engine.risk_classifier import (
    RiskCategory,
    classify_risk,
    validate_thresholds,
)


# ─── Test Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def thresholds():
    return {
        "critical_minutes": 2880,
        "high_minutes": 7200,
        "medium_minutes": 20160,
        "low_minutes": 43200,
    }


@pytest.fixture
def config(thresholds):
    return {
        "decision_engine": {
            "risk_thresholds": thresholds,
            "confidence": {
                "boundary_proximity_tolerance_minutes": 288,
                "model_type_modifiers": {
                    "mean": "LOW",
                    "median": "LOW",
                    "linear": "MEDIUM",
                    "random_forest": "HIGH",
                    "xgboost": "HIGH",
                },
            },
            "business_rules": {
                "channel_rul_spread_threshold_minutes": 5000,
                "missing_prediction_default_risk": "HIGH",
            },
            "explainability": {"max_top_features_in_explanation": 5},
            "priority_matrix": {
                "CRITICAL": {"HIGH": 1, "MEDIUM": 1, "LOW": 1},
                "HIGH": {"HIGH": 2, "MEDIUM": 2, "LOW": 3},
                "MEDIUM": {"HIGH": 3, "MEDIUM": 3, "LOW": 4},
                "LOW": {"HIGH": 4, "MEDIUM": 4, "LOW": 5},
                "HEALTHY": {"HIGH": 5, "MEDIUM": 5, "LOW": 5},
            },
            "policy_map": {
                "CRITICAL": "IMMEDIATE_SHUTDOWN",
                "HIGH": "URGENT_MAINTENANCE",
                "MEDIUM": "SCHEDULE_MAINTENANCE",
                "LOW": "ROUTINE_MONITORING",
                "HEALTHY": "NORMAL_OPERATION",
            },
            "rule_version": "1.0",
        }
    }


@pytest.fixture
def logger():
    import logging
    log = logging.getLogger("test_decision_engine")
    log.setLevel(logging.DEBUG)
    return log


@pytest.fixture
def minimal_phase6_contract():
    return {
        "schema_version": "1.0",
        "model": {"type": "median", "model_fingerprint": "abc123"},
        "performance": {
            "n2_caveat": True,
            "n2_caveat_text": "n=2 caveat text.",
            "mean_mae_minutes": 11530.0,
            "stability_ratio": 1.0,
        },
        "prediction_schema": {
            "grain": ["unit_id", "channel", "timestamp"],
            "input_features": ["feat_a", "feat_b"],
        },
        "experiment": {
            "experiment_id": "exp_abc123_seed42",
            "config_fingerprint": "abc123fingerprint",
        },
        "feature_importance": {"top_permutation_ranked_features": []},
    }


@pytest.fixture
def minimal_predictions_df():
    timestamps = pd.date_range("2003-10-22", periods=3, freq="10min")
    rows = []
    for ts in timestamps:
        for ch in range(1, 9):
            rows.append(
                {
                    "unit_id": "bearing_3",
                    "bearing_id": "bearing_3",
                    "channel": ch,
                    "timestamp": ts,
                    "predicted_rul_minutes": 14726.0,
                }
            )
    return pd.DataFrame(rows)


# ─── TestRiskClassifier ───────────────────────────────────────────────────────

class TestRiskClassifier:
    def test_classify_healthy(self, thresholds):
        assert classify_risk(50000, thresholds) == RiskCategory.HEALTHY

    def test_classify_low(self, thresholds):
        assert classify_risk(43000, thresholds) == RiskCategory.LOW

    def test_classify_medium(self, thresholds):
        assert classify_risk(14726, thresholds) == RiskCategory.MEDIUM

    def test_classify_high(self, thresholds):
        assert classify_risk(5000, thresholds) == RiskCategory.HIGH

    def test_classify_critical(self, thresholds):
        assert classify_risk(1000, thresholds) == RiskCategory.CRITICAL

    def test_exactly_at_critical_boundary(self, thresholds):
        assert classify_risk(2880, thresholds) == RiskCategory.CRITICAL

    def test_zero_rul_is_critical(self, thresholds):
        assert classify_risk(0.0, thresholds) == RiskCategory.CRITICAL

    def test_negative_rul_is_critical(self, thresholds):
        assert classify_risk(-1.0, thresholds) == RiskCategory.CRITICAL

    def test_just_above_critical_is_high(self, thresholds):
        assert classify_risk(2881, thresholds) == RiskCategory.HIGH


# ─── TestThresholdValidation ──────────────────────────────────────────────────

class TestThresholdValidation:
    def test_valid_thresholds_pass(self, thresholds, logger):
        validate_thresholds(thresholds, logger)  # must not raise

    def test_null_threshold_raises(self, logger):
        bad = {"critical_minutes": None, "high_minutes": 7200,
               "medium_minutes": 20160, "low_minutes": 43200}
        with pytest.raises(ValueError, match="null in configuration"):
            validate_thresholds(bad, logger)

    def test_ordering_violation_raises(self, logger):
        bad = {"critical_minutes": 10000, "high_minutes": 5000,
               "medium_minutes": 20160, "low_minutes": 43200}
        with pytest.raises(ValueError, match="ordering violated"):
            validate_thresholds(bad, logger)

    def test_zero_critical_raises(self, logger):
        bad = {"critical_minutes": 0, "high_minutes": 7200,
               "medium_minutes": 20160, "low_minutes": 43200}
        with pytest.raises(ValueError):
            validate_thresholds(bad, logger)


# ─── TestMaintenancePolicy ────────────────────────────────────────────────────

class TestMaintenancePolicy:
    def test_critical_maps_to_immediate_shutdown(self, config, logger):
        pm = load_policy_map(config, logger)
        assert pm[RiskCategory.CRITICAL] == MaintenanceAction.IMMEDIATE_SHUTDOWN

    def test_high_maps_to_urgent_maintenance(self, config, logger):
        pm = load_policy_map(config, logger)
        assert pm[RiskCategory.HIGH] == MaintenanceAction.URGENT_MAINTENANCE

    def test_medium_maps_to_schedule_maintenance(self, config, logger):
        pm = load_policy_map(config, logger)
        assert pm[RiskCategory.MEDIUM] == MaintenanceAction.SCHEDULE_MAINTENANCE

    def test_low_maps_to_routine_monitoring(self, config, logger):
        pm = load_policy_map(config, logger)
        assert pm[RiskCategory.LOW] == MaintenanceAction.ROUTINE_MONITORING

    def test_healthy_maps_to_normal_operation(self, config, logger):
        pm = load_policy_map(config, logger)
        assert pm[RiskCategory.HEALTHY] == MaintenanceAction.NORMAL_OPERATION


# ─── TestConfidenceEstimator ──────────────────────────────────────────────────

class TestConfidenceEstimator:
    def test_median_model_gives_low(self, thresholds, config, logger):
        level = estimate_confidence(14726, "median", True, thresholds, config, logger)
        assert level == ConfidenceLevel.LOW

    def test_rf_model_n2_penalty_gives_medium(self, thresholds, config, logger):
        # RF base=HIGH → n2 penalty → MEDIUM
        # No boundary proximity since 14726 is not near any threshold
        level = estimate_confidence(14726, "random_forest", True, thresholds, config, logger)
        assert level == ConfidenceLevel.MEDIUM

    def test_linear_model_n2_penalty_gives_low(self, thresholds, config, logger):
        # linear base=MEDIUM → n2 penalty → LOW
        level = estimate_confidence(14726, "linear", True, thresholds, config, logger)
        assert level == ConfidenceLevel.LOW

    def test_no_n2_penalty_rf_stays_high(self, thresholds, config, logger):
        level = estimate_confidence(14726, "random_forest", False, thresholds, config, logger)
        assert level == ConfidenceLevel.HIGH

    def test_boundary_proximity_downgrades(self, thresholds, config, logger):
        # RUL = critical_minutes ± tolerance → near boundary
        near_critical = float(thresholds["critical_minutes"]) + 100  # within tolerance
        # RF base=HIGH → n2=False → boundary → MEDIUM
        level = estimate_confidence(near_critical, "random_forest", False, thresholds, config, logger)
        assert level == ConfidenceLevel.MEDIUM

    def test_floor_is_low(self, thresholds, config, logger):
        # LOW cannot go below LOW regardless of penalties
        level = estimate_confidence(1000, "median", True, thresholds, config, logger)
        assert level == ConfidenceLevel.LOW

    def test_unknown_model_type_defaults_medium(self, thresholds, config, logger):
        level = estimate_confidence(14726, "unknown_model", False, thresholds, config, logger)
        assert level == ConfidenceLevel.MEDIUM


# ─── TestPriorityCalculator ───────────────────────────────────────────────────

class TestPriorityCalculator:
    def test_critical_high_confidence_is_1(self, config, logger):
        pm = load_priority_matrix(config, logger)
        assert compute_priority(RiskCategory.CRITICAL, ConfidenceLevel.HIGH, pm) == 1

    def test_critical_low_confidence_is_1(self, config, logger):
        pm = load_priority_matrix(config, logger)
        assert compute_priority(RiskCategory.CRITICAL, ConfidenceLevel.LOW, pm) == 1

    def test_high_high_is_2(self, config, logger):
        pm = load_priority_matrix(config, logger)
        assert compute_priority(RiskCategory.HIGH, ConfidenceLevel.HIGH, pm) == 2

    def test_high_low_is_3(self, config, logger):
        pm = load_priority_matrix(config, logger)
        assert compute_priority(RiskCategory.HIGH, ConfidenceLevel.LOW, pm) == 3

    def test_medium_low_is_4(self, config, logger):
        pm = load_priority_matrix(config, logger)
        assert compute_priority(RiskCategory.MEDIUM, ConfidenceLevel.LOW, pm) == 4

    def test_healthy_all_is_5(self, config, logger):
        pm = load_priority_matrix(config, logger)
        for conf in ConfidenceLevel:
            assert compute_priority(RiskCategory.HEALTHY, conf, pm) == 5

    def test_invalid_critical_priority_raises(self, config, logger):
        bad = dict(config)
        bad_de = dict(bad["decision_engine"])
        bad_pm = dict(bad_de["priority_matrix"])
        bad_pm["CRITICAL"] = {"HIGH": 2, "MEDIUM": 2, "LOW": 2}
        bad_de["priority_matrix"] = bad_pm
        bad["decision_engine"] = bad_de
        with pytest.raises(ValueError, match="must be 1"):
            load_priority_matrix(bad, logger)


# ─── TestBusinessRules ────────────────────────────────────────────────────────

class TestBusinessRules:
    def _policy_map(self, config, logger):
        return load_policy_map(config, logger)

    def test_br01_critical_locked_to_immediate_shutdown(self, config, logger):
        pm = self._policy_map(config, logger)
        result = apply_business_rules(
            rul_minutes=1000,
            risk_category=RiskCategory.CRITICAL,
            maintenance_action=MaintenanceAction.URGENT_MAINTENANCE,  # wrong action
            confidence_level=ConfidenceLevel.HIGH,
            rul_spread=0,
            model_type="random_forest",
            policy_map=pm,
            config=config,
            logger=logger,
        )
        assert result.maintenance_action == MaintenanceAction.IMMEDIATE_SHUTDOWN
        assert "BR-01" in result.rules_applied

    def test_br05_zero_rul_forces_critical(self, config, logger):
        pm = self._policy_map(config, logger)
        result = apply_business_rules(
            rul_minutes=0.0,
            risk_category=RiskCategory.HEALTHY,  # wrong classification
            maintenance_action=MaintenanceAction.NORMAL_OPERATION,
            confidence_level=ConfidenceLevel.HIGH,
            rul_spread=0,
            model_type="median",
            policy_map=pm,
            config=config,
            logger=logger,
        )
        assert result.risk_category == RiskCategory.CRITICAL
        assert result.maintenance_action == MaintenanceAction.IMMEDIATE_SHUTDOWN
        assert "BR-05" in result.rules_applied

    def test_br03_large_spread_downgrades_confidence(self, config, logger):
        pm = self._policy_map(config, logger)
        result = apply_business_rules(
            rul_minutes=14726,
            risk_category=RiskCategory.MEDIUM,
            maintenance_action=MaintenanceAction.SCHEDULE_MAINTENANCE,
            confidence_level=ConfidenceLevel.MEDIUM,
            rul_spread=9000,  # > threshold 5000
            model_type="median",
            policy_map=pm,
            config=config,
            logger=logger,
        )
        assert result.confidence_level == ConfidenceLevel.LOW
        assert "BR-03" in result.rules_applied

    def test_br04_constant_predictor_annotation(self, config, logger):
        pm = self._policy_map(config, logger)
        result = apply_business_rules(
            rul_minutes=14726,
            risk_category=RiskCategory.MEDIUM,
            maintenance_action=MaintenanceAction.SCHEDULE_MAINTENANCE,
            confidence_level=ConfidenceLevel.LOW,
            rul_spread=0,
            model_type="median",
            policy_map=pm,
            config=config,
            logger=logger,
        )
        assert "BR-04" in result.rules_applied
        assert result.br04_annotation is not None
        assert "constant" in result.br04_annotation.lower()

    def test_br04_not_fired_for_rf_model(self, config, logger):
        pm = self._policy_map(config, logger)
        result = apply_business_rules(
            rul_minutes=14726,
            risk_category=RiskCategory.MEDIUM,
            maintenance_action=MaintenanceAction.SCHEDULE_MAINTENANCE,
            confidence_level=ConfidenceLevel.MEDIUM,
            rul_spread=0,
            model_type="random_forest",
            policy_map=pm,
            config=config,
            logger=logger,
        )
        assert "BR-04" not in result.rules_applied

    def test_override_reason_populated_when_br01_fires(self, config, logger):
        pm = self._policy_map(config, logger)
        result = apply_business_rules(
            rul_minutes=1000,
            risk_category=RiskCategory.CRITICAL,
            maintenance_action=MaintenanceAction.ROUTINE_MONITORING,  # wrong
            confidence_level=ConfidenceLevel.HIGH,
            rul_spread=0,
            model_type="median",
            policy_map=pm,
            config=config,
            logger=logger,
        )
        assert result.override_reason is not None
        assert "BR-01" in result.override_reason

    def test_no_override_for_correct_base_decision(self, config, logger):
        pm = self._policy_map(config, logger)
        result = apply_business_rules(
            rul_minutes=14726,
            risk_category=RiskCategory.MEDIUM,
            maintenance_action=MaintenanceAction.SCHEDULE_MAINTENANCE,
            confidence_level=ConfidenceLevel.LOW,
            rul_spread=100,  # small spread, no BR-03
            model_type="random_forest",  # no BR-04
            policy_map=pm,
            config=config,
            logger=logger,
        )
        assert result.override_reason is None
        assert "BR-01" not in result.rules_applied
        assert "BR-05" not in result.rules_applied


# ─── TestBusinessRulePrecedence ───────────────────────────────────────────────

class TestBusinessRulePrecedence:
    """Business rule conflict-resolution tests. Architecture §17c."""

    def test_br01_cannot_be_overridden_by_br03(self, config, logger):
        """BR-01 (Tier 1) fixes action to IMMEDIATE_SHUTDOWN;
        BR-03 (Tier 3) may change confidence but not action."""
        pm = load_policy_map(config, logger)
        result = apply_business_rules(
            rul_minutes=1000,
            risk_category=RiskCategory.CRITICAL,
            maintenance_action=MaintenanceAction.URGENT_MAINTENANCE,
            confidence_level=ConfidenceLevel.MEDIUM,
            rul_spread=9000,  # BR-03 also fires
            model_type="random_forest",
            policy_map=pm,
            config=config,
            logger=logger,
        )
        # BR-01 must have locked the action
        assert result.maintenance_action == MaintenanceAction.IMMEDIATE_SHUTDOWN
        # BR-03 may also have changed confidence (acceptable — disjoint fields)
        assert "BR-01" in result.rules_applied

    def test_br05_and_br04_coexist(self, config, logger):
        """BR-05 (Tier 1) and BR-04 (Tier 3) can fire simultaneously.
        BR-05 changes risk/action; BR-04 adds annotation. Both applied."""
        pm = load_policy_map(config, logger)
        result = apply_business_rules(
            rul_minutes=0.0,
            risk_category=RiskCategory.LOW,  # should be overridden
            maintenance_action=MaintenanceAction.ROUTINE_MONITORING,
            confidence_level=ConfidenceLevel.HIGH,
            rul_spread=0,
            model_type="median",
            policy_map=pm,
            config=config,
            logger=logger,
        )
        assert result.risk_category == RiskCategory.CRITICAL
        assert "BR-05" in result.rules_applied
        assert "BR-04" in result.rules_applied

    def test_br03_and_br04_both_apply_to_disjoint_fields(self, config, logger):
        """BR-03 modifies confidence; BR-04 adds annotation. Neither overrides the other."""
        pm = load_policy_map(config, logger)
        result = apply_business_rules(
            rul_minutes=14726,
            risk_category=RiskCategory.MEDIUM,
            maintenance_action=MaintenanceAction.SCHEDULE_MAINTENANCE,
            confidence_level=ConfidenceLevel.MEDIUM,
            rul_spread=9000,  # triggers BR-03
            model_type="median",  # triggers BR-04
            policy_map=pm,
            config=config,
            logger=logger,
        )
        # BR-03 downgraded confidence
        assert result.confidence_level == ConfidenceLevel.LOW
        # BR-04 added annotation
        assert result.br04_annotation is not None
        # Risk/action unchanged by Tier 3 rules
        assert result.risk_category == RiskCategory.MEDIUM
        assert result.maintenance_action == MaintenanceAction.SCHEDULE_MAINTENANCE

    def test_br01_br05_agree_when_both_conditions_met(self, config, logger):
        """When RUL=0 and risk=CRITICAL: BR-01 and BR-05 produce the same outcome.
        Both should be structurally consistent — not a conflict."""
        pm = load_policy_map(config, logger)
        result = apply_business_rules(
            rul_minutes=0.0,
            risk_category=RiskCategory.CRITICAL,
            maintenance_action=MaintenanceAction.IMMEDIATE_SHUTDOWN,
            confidence_level=ConfidenceLevel.LOW,
            rul_spread=0,
            model_type="median",
            policy_map=pm,
            config=config,
            logger=logger,
        )
        # Final state remains correct regardless of which rule checked
        assert result.risk_category == RiskCategory.CRITICAL
        assert result.maintenance_action == MaintenanceAction.IMMEDIATE_SHUTDOWN


# ─── TestChannelAggregation ───────────────────────────────────────────────────

class TestChannelAggregation:
    def test_min_mean_count_correct(self, logger):
        rows = [
            {"unit_id": "bearing_3", "bearing_id": "bearing_3", "channel": 1, "timestamp": pd.Timestamp("2003-10-22"),
             "predicted_rul_minutes": 10000.0},
            {"unit_id": "bearing_3", "bearing_id": "bearing_3", "channel": 2, "timestamp": pd.Timestamp("2003-10-22"),
             "predicted_rul_minutes": 20000.0},
        ]
        df = pd.DataFrame(rows)
        agg = aggregate_channels(df, logger)
        assert len(agg) == 1
        row = agg.iloc[0]
        assert row["min_predicted_rul_minutes"] == 10000.0
        assert row["mean_predicted_rul_minutes"] == 15000.0
        assert row["channel_count"] == 2
        assert row["rul_spread"] == 10000.0

    def test_single_channel_spread_is_zero(self, logger):
        df = pd.DataFrame([
            {"unit_id": "b1", "bearing_id": "b1", "channel": 1, "timestamp": pd.Timestamp("2003-01-01"),
             "predicted_rul_minutes": 5000.0}
        ])
        agg = aggregate_channels(df, logger)
        assert agg.iloc[0]["rul_spread"] == 0.0

    def test_two_bearings_produce_separate_rows(self, logger):
        rows = []
        for bid in ["bearing_3", "bearing_4"]:
            rows.append({"unit_id": bid, "bearing_id": bid, "channel": 1,
                         "timestamp": pd.Timestamp("2003-10-22"),
                         "predicted_rul_minutes": 14726.0})
        agg = aggregate_channels(pd.DataFrame(rows), logger)
        assert len(agg) == 2
        assert set(agg["bearing_id"]) == {"bearing_3", "bearing_4"}

    def test_multiple_timestamps_produce_multiple_rows(self, logger):
        rows = []
        for ts in pd.date_range("2003-10-22", periods=5, freq="10min"):
            rows.append({"unit_id": "bearing_3", "bearing_id": "bearing_3", "channel": 1,
                         "timestamp": ts, "predicted_rul_minutes": 14726.0})
        agg = aggregate_channels(pd.DataFrame(rows), logger)
        assert len(agg) == 5


# ─── TestDecisionExplainer ────────────────────────────────────────────────────

class TestDecisionExplainer:
    def test_explanation_contains_bearing_id(self, config, logger):
        expl = generate_explanation(
            bearing_id="bearing_3",
            timestamp=pd.Timestamp("2003-10-22"),
            min_rul=14726.0,
            mean_rul=14726.0,
            channel_count=8,
            rul_spread=0,
            risk_category=RiskCategory.MEDIUM,
            maintenance_action=MaintenanceAction.SCHEDULE_MAINTENANCE,
            confidence_level=ConfidenceLevel.LOW,
            priority_score=4,
            model_type="median",
            n2_caveat=True,
            n2_caveat_text="n=2 caveat.",
            n_rtf_units=4,
            mean_mae_minutes=11530.0,
            experiment_id="exp_abc_seed42",
            override_reason=None,
            top_features=[],
            near_boundary=False,
            max_top_features=5,
            rule_version="1.0",
            config_fingerprint="abc123",
        )
        assert "bearing_3" in expl

    def test_constant_predictor_note_present_for_median(self, config, logger):
        note = _build_model_selection_note("median")
        assert "demotion fallback" in note
        assert "constant" in note.lower()

    def test_constant_predictor_note_absent_for_rf(self, config, logger):
        note = _build_model_selection_note("random_forest")
        assert "demotion fallback" not in note

    def test_n2_caveat_text_in_explanation(self, config, logger):
        expl = generate_explanation(
            bearing_id="bearing_3",
            timestamp=pd.Timestamp("2003-10-22"),
            min_rul=14726.0,
            mean_rul=14726.0,
            channel_count=8,
            rul_spread=0,
            risk_category=RiskCategory.MEDIUM,
            maintenance_action=MaintenanceAction.SCHEDULE_MAINTENANCE,
            confidence_level=ConfidenceLevel.LOW,
            priority_score=4,
            model_type="median",
            n2_caveat=True,
            n2_caveat_text="UNIQUE_CAVEAT_TEXT",
            n_rtf_units=4,
            mean_mae_minutes=11530.0,
            experiment_id="exp_abc_seed42",
            override_reason=None,
            top_features=[],
            near_boundary=False,
            max_top_features=5,
            rule_version="1.0",
            config_fingerprint="abc123",
        )
        assert "UNIQUE_CAVEAT_TEXT" in expl

    def test_override_reason_in_explanation_when_present(self, config, logger):
        expl = generate_explanation(
            bearing_id="bearing_4",
            timestamp=pd.Timestamp("2003-10-22"),
            min_rul=0.0,
            mean_rul=0.0,
            channel_count=8,
            rul_spread=0,
            risk_category=RiskCategory.CRITICAL,
            maintenance_action=MaintenanceAction.IMMEDIATE_SHUTDOWN,
            confidence_level=ConfidenceLevel.LOW,
            priority_score=1,
            model_type="median",
            n2_caveat=True,
            n2_caveat_text="n=2.",
            n_rtf_units=4,
            mean_mae_minutes=11530.0,
            experiment_id="exp_abc",
            override_reason="BR-05: zero RUL floor applied.",
            top_features=[],
            near_boundary=False,
            max_top_features=5,
            rule_version="1.0",
            config_fingerprint="abc",
        )
        assert "BR-05" in expl


# ─── TestContractLoader ───────────────────────────────────────────────────────

class TestContractLoader:
    def test_valid_contract_passes(self, minimal_phase6_contract, minimal_predictions_df, logger):
        result = validate_phase6_contract(minimal_phase6_contract, minimal_predictions_df, logger)
        assert result.passed

    def test_wrong_schema_version_fails(self, minimal_phase6_contract, minimal_predictions_df, logger):
        bad = dict(minimal_phase6_contract)
        bad["schema_version"] = "2.0"
        result = validate_phase6_contract(bad, minimal_predictions_df, logger)
        assert not result.passed
        assert any("schema_version" in f for f in result.fatals)

    def test_missing_n2_caveat_fails(self, minimal_phase6_contract, minimal_predictions_df, logger):
        bad = dict(minimal_phase6_contract)
        bad["performance"] = dict(bad["performance"])
        bad["performance"]["n2_caveat"] = False
        result = validate_phase6_contract(bad, minimal_predictions_df, logger)
        assert not result.passed

    def test_wrong_grain_fails(self, minimal_phase6_contract, minimal_predictions_df, logger):
        bad = dict(minimal_phase6_contract)
        bad["prediction_schema"] = dict(bad["prediction_schema"])
        bad["prediction_schema"]["grain"] = ["bearing_id"]  # incomplete
        result = validate_phase6_contract(bad, minimal_predictions_df, logger)
        assert not result.passed

    def test_model_type_extracted(self, minimal_phase6_contract, minimal_predictions_df, logger):
        result = validate_phase6_contract(minimal_phase6_contract, minimal_predictions_df, logger)
        assert result.model_type == "median"

    def test_n2_caveat_extracted(self, minimal_phase6_contract, minimal_predictions_df, logger):
        result = validate_phase6_contract(minimal_phase6_contract, minimal_predictions_df, logger)
        assert result.n2_caveat is True


# ─── TestDecisionConsistency ──────────────────────────────────────────────────

class TestDecisionConsistency:
    """Decision consistency tests. Architecture §17d."""

    def test_same_input_same_decision(self, thresholds, config, logger):
        pm = load_policy_map(config, logger)
        risk1 = classify_risk(14726, thresholds)
        risk2 = classify_risk(14726, thresholds)
        assert risk1 == risk2

    def test_rul_just_above_threshold_is_lower_risk(self, thresholds):
        critical = float(thresholds["critical_minutes"])
        assert classify_risk(critical + 1, thresholds) == RiskCategory.HIGH
        assert classify_risk(critical - 1, thresholds) == RiskCategory.CRITICAL

    def test_config_fingerprint_changes_when_threshold_changes(self, config):
        fp1 = compute_config_fingerprint(config)
        import copy
        cfg2 = copy.deepcopy(config)
        cfg2["decision_engine"]["risk_thresholds"]["critical_minutes"] = 3000
        fp2 = compute_config_fingerprint(cfg2)
        assert fp1 != fp2

    def test_experiment_id_stable_for_same_fingerprint(self):
        fp = "abc123def456789"
        id1 = generate_experiment_id(fp, "1.0")
        id2 = generate_experiment_id(fp, "1.0")
        assert id1 == id2

    def test_override_flag_set_when_br01_fires(self, config, logger):
        pm = load_policy_map(config, logger)
        result = apply_business_rules(
            rul_minutes=1000,
            risk_category=RiskCategory.CRITICAL,
            maintenance_action=MaintenanceAction.ROUTINE_MONITORING,
            confidence_level=ConfidenceLevel.HIGH,
            rul_spread=0,
            model_type="median",
            policy_map=pm,
            config=config,
            logger=logger,
        )
        assert result.override_reason is not None


# ─── TestContractSchema ───────────────────────────────────────────────────────

class TestContractSchema:
    """Phase 7→8 contract tests. Architecture §17e."""

    def test_rule_version_constant(self):
        assert RULE_VERSION == "1.0"

    def test_experiment_id_format(self):
        fp = "abcdef123456789012345678"
        eid = generate_experiment_id(fp, "1.0")
        assert eid.startswith("dec_")
        assert "rv10" in eid
        assert fp[:12] in eid

    def test_config_fingerprint_is_sha256(self, config):
        fp = compute_config_fingerprint(config)
        assert len(fp) == 64
        int(fp, 16)  # must be valid hex


# ─── TestIntegration ─────────────────────────────────────────────────────────

class TestIntegration:
    """Integration tests. Architecture §17b."""

    def test_full_pipeline_deterministic(
        self, minimal_phase6_contract, minimal_predictions_df, thresholds, config, logger
    ):
        """Same inputs → same recommendations. Architecture §25 Q4."""
        from src.decision_engine.contract_loader import validate_phase6_contract
        cv = validate_phase6_contract(
            minimal_phase6_contract, minimal_predictions_df, logger
        )
        bearing_df = aggregate_channels(minimal_predictions_df, logger)
        pm = load_policy_map(config, logger)
        pri_m = load_priority_matrix(config, logger)
        fp = compute_config_fingerprint(config)

        recs1 = run_decision_pipeline(bearing_df, cv, thresholds, pm, pri_m, config, fp, logger)
        recs2 = run_decision_pipeline(bearing_df, cv, thresholds, pm, pri_m, config, fp, logger)

        assert len(recs1) == len(recs2)
        for r1, r2 in zip(recs1, recs2):
            assert r1.risk_category == r2.risk_category
            assert r1.maintenance_action == r2.maintenance_action
            assert r1.priority_score == r2.priority_score

    def test_median_model_produces_br04_annotation(
        self, minimal_phase6_contract, minimal_predictions_df, config, logger
    ):
        """median model type must trigger BR-04 on every row."""
        from src.decision_engine.contract_loader import validate_phase6_contract
        cv = validate_phase6_contract(
            minimal_phase6_contract, minimal_predictions_df, logger
        )
        bearing_df = aggregate_channels(minimal_predictions_df, logger)
        pm = load_policy_map(config, logger)
        pri_m = load_priority_matrix(config, logger)
        fp = compute_config_fingerprint(config)

        recs = run_decision_pipeline(bearing_df, cv, config["decision_engine"]["risk_thresholds"],
                                     pm, pri_m, config, fp, logger)
        for r in recs:
            assert "BR-04" in (r.override_reason or "")

    def test_n2_caveat_true_in_every_recommendation(
        self, minimal_phase6_contract, minimal_predictions_df, config, logger
    ):
        from src.decision_engine.contract_loader import validate_phase6_contract
        cv = validate_phase6_contract(
            minimal_phase6_contract, minimal_predictions_df, logger
        )
        bearing_df = aggregate_channels(minimal_predictions_df, logger)
        pm = load_policy_map(config, logger)
        pri_m = load_priority_matrix(config, logger)
        fp = compute_config_fingerprint(config)

        recs = run_decision_pipeline(bearing_df, cv, config["decision_engine"]["risk_thresholds"],
                                     pm, pri_m, config, fp, logger)
        df = recommendations_to_dataframe(recs)
        assert df["n2_caveat"].all()

    def test_recommendations_have_required_schema_columns(
        self, minimal_phase6_contract, minimal_predictions_df, config, logger
    ):
        from src.decision_engine.contract_loader import validate_phase6_contract
        cv = validate_phase6_contract(
            minimal_phase6_contract, minimal_predictions_df, logger
        )
        bearing_df = aggregate_channels(minimal_predictions_df, logger)
        pm = load_policy_map(config, logger)
        pri_m = load_priority_matrix(config, logger)
        fp = compute_config_fingerprint(config)

        recs = run_decision_pipeline(bearing_df, cv, config["decision_engine"]["risk_thresholds"],
                                     pm, pri_m, config, fp, logger)
        df = recommendations_to_dataframe(recs)

        required = [
            "bearing_id", "timestamp", "min_predicted_rul_minutes",
            "mean_predicted_rul_minutes", "channel_count", "risk_category",
            "maintenance_action", "confidence_level", "priority_score",
            "is_business_rule_override", "override_reason", "explanation_text",
            "model_type", "n2_caveat", "rule_version", "config_fingerprint",
            "experiment_id",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"


# ─── TestBR02MissingPrediction ────────────────────────────────────────────────

class TestBR02MissingPrediction:
    """Tests for BR-02 missing-prediction fallback. Architecture §7.9a, §17c."""

    def test_missing_bearing_gets_high_risk_fallback(self, config, logger):
        """A bearing absent from bearing_df receives conservative HIGH risk."""
        result = apply_missing_prediction_fallback(
            bearing_id="bearing_missing",
            timestamp=pd.NaT,
            model_type="median",
            n2_caveat=True,
            n2_caveat_text="n=2 caveat.",
            mean_mae_minutes=11530.0,
            experiment_id="exp_abc_seed42",
            rule_version="1.0",
            config_fingerprint="abc123",
            config=config,
            logger=logger,
        )
        assert result["risk_category"] == RiskCategory.HIGH

    def test_missing_bearing_gets_urgent_maintenance(self, config, logger):
        """BR-02 fallback maps HIGH risk → URGENT_MAINTENANCE per policy map."""
        result = apply_missing_prediction_fallback(
            bearing_id="bearing_missing",
            timestamp=pd.NaT,
            model_type="median",
            n2_caveat=True,
            n2_caveat_text="n=2 caveat.",
            mean_mae_minutes=11530.0,
            experiment_id="exp_abc_seed42",
            rule_version="1.0",
            config_fingerprint="abc123",
            config=config,
            logger=logger,
        )
        assert result["maintenance_action"] == MaintenanceAction.URGENT_MAINTENANCE

    def test_missing_bearing_gets_low_confidence(self, config, logger):
        """BR-02 fallback always assigns LOW confidence (conservative)."""
        result = apply_missing_prediction_fallback(
            bearing_id="bearing_missing",
            timestamp=pd.NaT,
            model_type="median",
            n2_caveat=True,
            n2_caveat_text="n=2 caveat.",
            mean_mae_minutes=11530.0,
            experiment_id="exp_abc_seed42",
            rule_version="1.0",
            config_fingerprint="abc123",
            config=config,
            logger=logger,
        )
        assert result["confidence_level"] == ConfidenceLevel.LOW

    def test_missing_bearing_override_reason_contains_br02(self, config, logger):
        """BR-02 fallback sets override_reason containing 'BR-02'."""
        result = apply_missing_prediction_fallback(
            bearing_id="bearing_missing",
            timestamp=pd.NaT,
            model_type="median",
            n2_caveat=True,
            n2_caveat_text="n=2 caveat.",
            mean_mae_minutes=11530.0,
            experiment_id="exp_abc_seed42",
            rule_version="1.0",
            config_fingerprint="abc123",
            config=config,
            logger=logger,
        )
        assert "BR-02" in result["override_reason"]

    def test_missing_bearing_rul_is_nan(self, config, logger):
        """BR-02 fallback sets min/mean RUL to NaN (no prediction available)."""
        import math
        result = apply_missing_prediction_fallback(
            bearing_id="bearing_missing",
            timestamp=pd.NaT,
            model_type="random_forest",
            n2_caveat=True,
            n2_caveat_text="n=2 caveat.",
            mean_mae_minutes=500.0,
            experiment_id="exp_abc_seed42",
            rule_version="1.0",
            config_fingerprint="abc123",
            config=config,
            logger=logger,
        )
        assert math.isnan(result["min_predicted_rul_minutes"])
        assert math.isnan(result["mean_predicted_rul_minutes"])

    def test_pipeline_generates_br02_row_for_missing_bearing(
        self, minimal_phase6_contract, minimal_predictions_df, config, logger
    ):
        """run_decision_pipeline fires BR-02 for a bearing in expected_bearings
        that has no rows in bearing_df. Architecture §7.9a."""
        from src.decision_engine.contract_loader import validate_phase6_contract
        cv = validate_phase6_contract(
            minimal_phase6_contract, minimal_predictions_df, logger
        )
        bearing_df = aggregate_channels(minimal_predictions_df, logger)
        pm = load_policy_map(config, logger)
        pri_m = load_priority_matrix(config, logger)
        fp = compute_config_fingerprint(config)
        thresholds = config["decision_engine"]["risk_thresholds"]

        # bearing_df has only bearing_3; expected_bearings includes bearing_4 too
        bearing_df_3_only = bearing_df[bearing_df["bearing_id"] == "bearing_3"].copy()
        recs = run_decision_pipeline(
            bearing_df_3_only, cv, thresholds, pm, pri_m, config, fp, logger,
            expected_bearings=["bearing_3", "bearing_4"],
        )

        bearing_ids = [r.bearing_id for r in recs]
        assert "bearing_4" in bearing_ids

    def test_br02_fallback_row_is_marked_as_override(
        self, minimal_phase6_contract, minimal_predictions_df, config, logger
    ):
        """BR-02 fallback row has is_business_rule_override=True."""
        from src.decision_engine.contract_loader import validate_phase6_contract
        cv = validate_phase6_contract(
            minimal_phase6_contract, minimal_predictions_df, logger
        )
        bearing_df = aggregate_channels(minimal_predictions_df, logger)
        pm = load_policy_map(config, logger)
        pri_m = load_priority_matrix(config, logger)
        fp = compute_config_fingerprint(config)
        thresholds = config["decision_engine"]["risk_thresholds"]

        bearing_df_3_only = bearing_df[bearing_df["bearing_id"] == "bearing_3"].copy()
        recs = run_decision_pipeline(
            bearing_df_3_only, cv, thresholds, pm, pri_m, config, fp, logger,
            expected_bearings=["bearing_3", "bearing_4"],
        )
        fallback_recs = [r for r in recs if r.bearing_id == "bearing_4"]
        assert len(fallback_recs) == 1
        assert fallback_recs[0].is_business_rule_override is True

    def test_pipeline_no_missing_bearings_no_br02(
        self, minimal_phase6_contract, minimal_predictions_df, config, logger
    ):
        """When all expected bearings are present, BR-02 does not fire."""
        from src.decision_engine.contract_loader import validate_phase6_contract
        cv = validate_phase6_contract(
            minimal_phase6_contract, minimal_predictions_df, logger
        )
        bearing_df = aggregate_channels(minimal_predictions_df, logger)
        pm = load_policy_map(config, logger)
        pri_m = load_priority_matrix(config, logger)
        fp = compute_config_fingerprint(config)
        thresholds = config["decision_engine"]["risk_thresholds"]

        recs = run_decision_pipeline(
            bearing_df, cv, thresholds, pm, pri_m, config, fp, logger,
            expected_bearings=["bearing_3"],
        )
        br02_recs = [r for r in recs if r.override_reason and "BR-02" in r.override_reason]
        assert len(br02_recs) == 0

    def test_contract_expected_bearings_extracted(self, logger):
        """Phase 6 contract training_population.rtf_units is populated into
        ContractValidationResult.expected_bearings."""
        from src.decision_engine.contract_loader import validate_phase6_contract
        contract = {
            "schema_version": "1.0",
            "model": {
                "type": "median",
                "model_fingerprint": "abc",
                "training_population": {"rtf_units": ["bearing_3", "bearing_4"]},
            },
            "performance": {
                "n2_caveat": True,
                "n2_caveat_text": "n=2.",
                "mean_mae_minutes": 11530.0,
                "stability_ratio": 1.0,
            },
            "prediction_schema": {
                "grain": ["unit_id", "channel", "timestamp"],
                "input_features": [],
            },
            "experiment": {"experiment_id": "exp_abc", "config_fingerprint": "abc"},
            "feature_importance": {"top_permutation_ranked_features": []},
        }
        predictions = pd.DataFrame([
            {"unit_id": "bearing_3", "bearing_id": "bearing_3", "channel": 1,
             "timestamp": pd.Timestamp("2003-01-01"), "predicted_rul_minutes": 10000.0}
        ])
        result = validate_phase6_contract(contract, predictions, logger)
        assert "bearing_3" in result.expected_bearings
        assert "bearing_4" in result.expected_bearings


# ─── TestContradictoryRecommendation ─────────────────────────────────────────

class TestContradictoryRecommendation:
    """Tests for DV11 contradictory recommendation validation. Architecture §16c."""

    def _make_valid_df(self):
        return pd.DataFrame([{
            "bearing_id": "bearing_3",
            "timestamp": pd.Timestamp("2003-10-22"),
            "min_predicted_rul_minutes": 14726.0,
            "mean_predicted_rul_minutes": 14726.0,
            "channel_count": 2,
            "risk_category": "MEDIUM",
            "maintenance_action": "SCHEDULE_MAINTENANCE",
            "confidence_level": "LOW",
            "priority_score": 4,
            "is_business_rule_override": True,
            "override_reason": "BR-04 annotation",
            "explanation_text": "test",
            "model_type": "median",
            "n2_caveat": True,
            "rule_version": "1.0",
            "config_fingerprint": "abc",
            "experiment_id": "exp_abc",
        }])

    def _run_dv11(self, df, config, logger):
        from src.decision_engine.validation import run_decision_validation
        from src.decision_engine.contract_loader import ContractValidationResult
        cv = ContractValidationResult(passed=True, n2_caveat=True)
        manifest = {
            "experiment_id": "exp_abc",
            "config_fingerprint": "abc123",
            "rule_version": "1.0",
        }
        result = run_decision_validation(
            contract_validation=cv,
            recommendations_df=df,
            manifest=manifest,
            expected_bearings=sorted(df["bearing_id"].unique()),
            config=config,
            logger=logger,
        )
        return next(c for c in result.checks if c.name == "no_contradictory_recommendations")

    def test_valid_combination_passes(self, config, logger):
        """MEDIUM + SCHEDULE_MAINTENANCE is consistent → DV11 PASS."""
        df = self._make_valid_df()
        check = self._run_dv11(df, config, logger)
        assert check.passed

    def test_critical_normal_operation_fails(self, config, logger):
        """CRITICAL + NORMAL_OPERATION is contradictory → DV11 FATAL FAIL."""
        df = self._make_valid_df()
        df = df.copy()
        df["risk_category"] = "CRITICAL"
        df["maintenance_action"] = "NORMAL_OPERATION"
        df["priority_score"] = 1
        check = self._run_dv11(df, config, logger)
        assert not check.passed
        assert check.severity == "FATAL"

    def test_healthy_immediate_shutdown_fails(self, config, logger):
        """HEALTHY + IMMEDIATE_SHUTDOWN is contradictory → DV11 FATAL FAIL."""
        df = self._make_valid_df()
        df = df.copy()
        df["risk_category"] = "HEALTHY"
        df["maintenance_action"] = "IMMEDIATE_SHUTDOWN"
        df["priority_score"] = 5
        check = self._run_dv11(df, config, logger)
        assert not check.passed
        assert check.severity == "FATAL"

    def test_high_urgent_maintenance_passes(self, config, logger):
        """HIGH + URGENT_MAINTENANCE is consistent → DV11 PASS."""
        df = self._make_valid_df()
        df = df.copy()
        df["risk_category"] = "HIGH"
        df["maintenance_action"] = "URGENT_MAINTENANCE"
        df["priority_score"] = 2
        check = self._run_dv11(df, config, logger)
        assert check.passed

    def test_all_five_valid_policy_combinations_pass(self, config, logger):
        """All five base policy pairs are consistent → DV11 PASS."""
        from src.decision_engine.validation import _BASE_POLICY
        rows = []
        for risk, action in _BASE_POLICY.items():
            rows.append({
                "bearing_id": "bearing_3",
                "timestamp": pd.Timestamp("2003-10-22"),
                "min_predicted_rul_minutes": 14726.0,
                "mean_predicted_rul_minutes": 14726.0,
                "channel_count": 2,
                "risk_category": risk,
                "maintenance_action": action,
                "confidence_level": "LOW",
                "priority_score": 1 if risk == "CRITICAL" else 4,
                "is_business_rule_override": False,
                "override_reason": None,
                "explanation_text": "test",
                "model_type": "median",
                "n2_caveat": True,
                "rule_version": "1.0",
                "config_fingerprint": "abc",
                "experiment_id": "exp_abc",
            })
        df = pd.DataFrame(rows)
        check = self._run_dv11(df, config, logger)
        assert check.passed

    def test_dv11_severity_is_fatal(self, config, logger):
        """DV11 severity is always FATAL (architecture §16c)."""
        df = self._make_valid_df()
        check = self._run_dv11(df, config, logger)
        assert check.severity == "FATAL"


# ─── TestScoreUnlabelledBearing ───────────────────────────────────────────────

class TestScoreUnlabelledBearing:
    """The in-service inference path for bearings that have not failed."""

    # Fields the batch recommendations.parquet exposes that a live score must match.
    _RECOMMENDATION_FIELDS = {
        "bearing_id", "min_predicted_rul_minutes", "mean_predicted_rul_minutes",
        "risk_category", "maintenance_action", "confidence_level",
        "priority_score", "model_type", "n2_caveat",
    }

    def test_healthy_bearing_routes_to_normal_operation(self, config, logger):
        from src.decision_engine.scoring import score_unlabelled_bearing
        from src.model_dev.baseline_models import ConstantPredictor

        # A healthy in-service bearing: predicted RUL above the LOW threshold (43200).
        model = ConstantPredictor(value=50000.0)
        result = score_unlabelled_bearing(
            bearing_id="bearing_1",
            features={"td_rms": 0.15, "fd_band_energy_0_1000": 1.2},
            model_artifact=model,
            config=config,
            logger=logger,
            model_type="median",
        )

        assert result["bearing_id"] == "bearing_1"
        assert result["risk_category"] == "HEALTHY"
        assert result["maintenance_action"] == "NORMAL_OPERATION"
        assert result["confidence_level"] == "LOW"          # median + n=2 penalty
        assert result["constant_predictor_warning"] is True
        assert result["min_predicted_rul_minutes"] == 50000.0

    def test_output_schema_matches_recommendations(self, config, logger):
        from src.decision_engine.scoring import score_unlabelled_bearing
        from src.model_dev.baseline_models import ConstantPredictor

        result = score_unlabelled_bearing(
            bearing_id="bearing_2",
            features=[0.15, 1.2, 0.3],
            model_artifact=ConstantPredictor(value=1000.0),  # near-failure → CRITICAL
            config=config,
            logger=logger,
            model_type="median",
        )

        assert self._RECOMMENDATION_FIELDS.issubset(result.keys())
        # Safety floor: a near-zero RUL must be CRITICAL and top priority.
        assert result["risk_category"] == "CRITICAL"
        assert result["priority_score"] == 1
