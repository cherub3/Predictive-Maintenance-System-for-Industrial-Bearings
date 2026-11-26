"""Decision engine — orchestrates channel aggregation and the decision pipeline.

Architecture §7.2 and §8. Drives the transformation from (bearing_id, channel,
timestamp) grain predictions to (bearing_id, timestamp) grain recommendations
by calling all downstream modules in the fixed order defined in §8 and §8a.
"""

import logging
from typing import Any, Dict, List

import pandas as pd

from src.decision_engine.business_rules import (
    RULE_VERSION,
    BusinessRuleResult,
    apply_business_rules,
    apply_missing_prediction_fallback,
)
from src.decision_engine.confidence_estimator import estimate_confidence
from src.decision_engine.contract_loader import ContractValidationResult
from src.decision_engine.decision_explainer import generate_explanation
from src.decision_engine.maintenance_policy import MaintenanceAction, get_maintenance_action
from src.decision_engine.priority_calculator import compute_priority
from src.decision_engine.recommendation_engine import (
    Recommendation,
    assemble_recommendation,
)
from src.decision_engine.risk_classifier import RiskCategory, classify_risk


def aggregate_channels(predictions_df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Group by (unit_id, timestamp); aggregate channels.

    Architecture §7.2. Conservative aggregation: min_predicted_rul_minutes
    is the primary decision input. Mean is informational. Channel count and
    inter-channel spread are computed for BR-03 and explanation generation.

    Grouped by `unit_id` (qualified dataset_id::bearing_id), NOT bare
    bearing_id — the same bearing_id string (e.g. "bearing_3") can refer to
    physically unrelated bearings from different datasets, both present in
    model_predictions.parquet once Set 2/3 are included. The output column is
    still named `bearing_id` for compatibility with every downstream Phase 7
    consumer (business rules, explanations, reports, dashboard), but it now
    holds the qualified unit_id value — every consumer already treats this as
    an opaque identifier string, so this is a value fix, not a schema change.

    Args:
        predictions_df: Raw Phase 6 predictions at (unit_id, channel, timestamp) grain.
        logger: Pipeline logger.

    Returns:
        DataFrame at (unit_id, timestamp) grain with columns:
          bearing_id (holding the qualified unit_id), min_predicted_rul_minutes,
          mean_predicted_rul_minutes, channel_count, rul_spread.
    """
    grouped = predictions_df.groupby(["unit_id", "timestamp"])["predicted_rul_minutes"]
    bearing_df = pd.DataFrame(
        {
            "min_predicted_rul_minutes": grouped.min(),
            "mean_predicted_rul_minutes": grouped.mean(),
            "channel_count": grouped.count(),
            "rul_spread": grouped.max() - grouped.min(),
        }
    ).reset_index()
    bearing_df = bearing_df.rename(columns={"unit_id": "bearing_id"})

    logger.info(
        "Channel aggregation complete: %d bearing-timestamp rows "
        "(bearings: %s, timestamps per bearing: %s)",
        len(bearing_df),
        bearing_df["bearing_id"].unique().tolist(),
        bearing_df.groupby("bearing_id").size().to_dict(),
    )
    return bearing_df


def run_decision_pipeline(
    bearing_df: pd.DataFrame,
    contract_validation: ContractValidationResult,
    thresholds: Dict[str, Any],
    policy_map: Dict[RiskCategory, MaintenanceAction],
    priority_matrix: Dict[str, Dict[str, int]],
    config: Dict[str, Any],
    config_fingerprint: str,
    logger: logging.Logger,
    expected_bearings: List[str] = None,
) -> List[Recommendation]:
    """Apply the complete decision pipeline to every bearing-timestamp row.

    Architecture §8 pipeline order (per row):
      1. risk_classifier.classify_risk
      2. maintenance_policy.get_maintenance_action
      3. confidence_estimator.estimate_confidence
      4. priority_calculator.compute_priority
      5. business_rules.apply_business_rules (BR-01, BR-05, BR-03, BR-04)
      6. decision_explainer.generate_explanation
      7. recommendation_engine.assemble_recommendation

    After the main loop, any bearing in expected_bearings absent from bearing_df
    receives a BR-02 missing-prediction fallback recommendation (architecture §7.9a).

    Args:
        bearing_df: Channel-aggregated DataFrame from aggregate_channels.
        contract_validation: Validated Phase 6 contract metadata.
        thresholds: Validated risk threshold dict.
        policy_map: Loaded maintenance policy mapping.
        priority_matrix: Loaded priority matrix.
        config: Full project configuration dict.
        config_fingerprint: SHA-256 of decision_engine config block.
        logger: Pipeline logger.
        expected_bearings: Authoritative list of bearings that must appear in
            output (from Phase 6 contract training_population.rtf_units).
            Bearings absent from bearing_df trigger BR-02. Defaults to None
            (no BR-02 detection).

    Returns:
        List of Recommendation instances, one per bearing-timestamp row,
        plus one BR-02 fallback row per missing bearing.
    """
    de_cfg = config.get("decision_engine", {})
    expl_cfg = de_cfg.get("explainability", {})
    max_top_features = int(expl_cfg.get("max_top_features_in_explanation", 5))
    conf_cfg = de_cfg.get("confidence", {})
    tolerance = float(conf_cfg.get("boundary_proximity_tolerance_minutes", 0.0))

    model_type = contract_validation.model_type
    n2_caveat = contract_validation.n2_caveat
    n2_caveat_text = contract_validation.n2_caveat_text
    n_rtf_units = contract_validation.n_rtf_units
    mean_mae = contract_validation.mean_mae_minutes
    experiment_id = contract_validation.experiment_id
    top_features = contract_validation.top_permutation_ranked_features

    # Pre-compute threshold boundary values for near-boundary detection
    boundary_values = [
        float(thresholds["critical_minutes"]),
        float(thresholds["high_minutes"]),
        float(thresholds["medium_minutes"]),
        float(thresholds["low_minutes"]),
    ]

    recommendations: List[Recommendation] = []

    for _, row in bearing_df.iterrows():
        bearing_id = row["bearing_id"]
        timestamp = row["timestamp"]
        min_rul = float(row["min_predicted_rul_minutes"])
        mean_rul = float(row["mean_predicted_rul_minutes"])
        channel_count = int(row["channel_count"])
        rul_spread = float(row["rul_spread"])

        # Step 1: Risk classification
        risk = classify_risk(min_rul, thresholds)

        # Step 2: Base maintenance action
        action = get_maintenance_action(risk, policy_map)

        # Step 3: Confidence estimation
        near_boundary = any(abs(min_rul - bv) <= tolerance for bv in boundary_values)
        confidence = estimate_confidence(
            min_rul, model_type, n2_caveat, thresholds, config, logger
        )

        # Step 4: Priority is computed AFTER business rules (see final_priority
        # below), since a business rule may change the risk/confidence inputs.

        # Step 5: Business rules (BR-01, BR-05, BR-03, BR-04)
        br_result: BusinessRuleResult = apply_business_rules(
            rul_minutes=min_rul,
            risk_category=risk,
            maintenance_action=action,
            confidence_level=confidence,
            rul_spread=rul_spread,
            model_type=model_type,
            policy_map=policy_map,
            config=config,
            logger=logger,
        )

        # Priority must be re-computed if risk or confidence changed after BR
        final_priority = compute_priority(
            br_result.risk_category, br_result.confidence_level, priority_matrix
        )

        # Combine override reason and annotation for the override_reason field
        combined_override = br_result.override_reason
        if br_result.br04_annotation:
            if combined_override:
                combined_override = combined_override + " | " + br_result.br04_annotation
            else:
                combined_override = br_result.br04_annotation

        # Step 6: Generate explanation (§13d seven-question checklist)
        explanation = generate_explanation(
            bearing_id=bearing_id,
            timestamp=timestamp,
            min_rul=min_rul,
            mean_rul=mean_rul,
            channel_count=channel_count,
            rul_spread=rul_spread,
            risk_category=br_result.risk_category,
            maintenance_action=br_result.maintenance_action,
            confidence_level=br_result.confidence_level,
            priority_score=final_priority,
            model_type=model_type,
            n2_caveat=n2_caveat,
            n2_caveat_text=n2_caveat_text,
            n_rtf_units=n_rtf_units,
            mean_mae_minutes=mean_mae,
            experiment_id=experiment_id,
            override_reason=combined_override,
            top_features=top_features,
            near_boundary=near_boundary,
            max_top_features=max_top_features,
            rule_version=RULE_VERSION,
            config_fingerprint=config_fingerprint,
        )

        # Step 7: Assemble recommendation
        rec = assemble_recommendation(
            bearing_id=bearing_id,
            timestamp=timestamp,
            min_rul=min_rul,
            mean_rul=mean_rul,
            channel_count=channel_count,
            risk_category=br_result.risk_category,
            maintenance_action=br_result.maintenance_action,
            confidence_level=br_result.confidence_level,
            priority_score=final_priority,
            explanation_text=explanation,
            model_type=model_type,
            n2_caveat=n2_caveat,
            rule_version=RULE_VERSION,
            config_fingerprint=config_fingerprint,
            experiment_id=experiment_id,
            override_reason=combined_override,
        )
        recommendations.append(rec)

    # BR-02: Missing-prediction fallback (architecture §7.9a Tier 2)
    # Detect bearings that are expected (per Phase 6 contract) but absent from
    # bearing_df (i.e., they had no prediction rows in model_predictions.parquet).
    if expected_bearings:
        produced_bearings = {r.bearing_id for r in recommendations}
        missing_bearings = [b for b in expected_bearings if b not in produced_bearings]
        for missing_bid in missing_bearings:
            fallback_fields = apply_missing_prediction_fallback(
                bearing_id=missing_bid,
                timestamp=pd.NaT,
                model_type=model_type,
                n2_caveat=n2_caveat,
                n2_caveat_text=n2_caveat_text,
                mean_mae_minutes=mean_mae,
                experiment_id=experiment_id,
                rule_version=RULE_VERSION,
                config_fingerprint=config_fingerprint,
                config=config,
                logger=logger,
            )
            fb_risk = fallback_fields["risk_category"]
            fb_action = fallback_fields["maintenance_action"]
            fb_confidence = fallback_fields["confidence_level"]
            fb_priority = compute_priority(fb_risk, fb_confidence, priority_matrix)
            fb_override = fallback_fields["override_reason"]

            fb_explanation = generate_explanation(
                bearing_id=missing_bid,
                timestamp=pd.NaT,
                min_rul=fallback_fields["min_predicted_rul_minutes"],
                mean_rul=fallback_fields["mean_predicted_rul_minutes"],
                channel_count=fallback_fields["channel_count"],
                rul_spread=0.0,
                risk_category=fb_risk,
                maintenance_action=fb_action,
                confidence_level=fb_confidence,
                priority_score=fb_priority,
                model_type=model_type,
                n2_caveat=n2_caveat,
                n2_caveat_text=n2_caveat_text,
                n_rtf_units=n_rtf_units,
                mean_mae_minutes=mean_mae,
                experiment_id=experiment_id,
                override_reason=fb_override,
                top_features=top_features,
                near_boundary=False,
                max_top_features=max_top_features,
                rule_version=RULE_VERSION,
                config_fingerprint=config_fingerprint,
            )

            fb_rec = assemble_recommendation(
                bearing_id=missing_bid,
                timestamp=pd.NaT,
                min_rul=fallback_fields["min_predicted_rul_minutes"],
                mean_rul=fallback_fields["mean_predicted_rul_minutes"],
                channel_count=fallback_fields["channel_count"],
                risk_category=fb_risk,
                maintenance_action=fb_action,
                confidence_level=fb_confidence,
                priority_score=fb_priority,
                explanation_text=fb_explanation,
                model_type=model_type,
                n2_caveat=n2_caveat,
                rule_version=RULE_VERSION,
                config_fingerprint=config_fingerprint,
                experiment_id=experiment_id,
                override_reason=fb_override,
            )
            recommendations.append(fb_rec)

    logger.info(
        "Decision pipeline complete: %d recommendations generated", len(recommendations)
    )
    return recommendations
