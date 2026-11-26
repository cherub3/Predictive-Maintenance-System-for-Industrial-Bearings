"""Feature dictionary generator for documentation and governance.

The feature dictionary provides complete metadata for every feature in the
feature store, enabling traceability, onboarding, and governance.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any


FEATURE_DICTIONARY_TEMPLATE = {
    "td_rms": {
        "feature_name": "td_rms",
        "short_description": "Root Mean Square vibration energy",
        "engineering_meaning": "Proportional to mechanical energy dissipated by the vibrating structure",
        "units": "Signal amplitude units (typically acceleration in g or velocity in in/s)",
        "physical_formula": "RMS = sqrt(mean(signal^2))",
        "source_module": "time_domain_features.py (reuses Phase 3 computation)",
        "input_signal": "Raw vibration waveform, 20,480 samples at 20 kHz",
        "dependencies": ["Phase 3 signal_health_indicators.parquet"],
        "expected_behaviour": "Flat during health; gradual rise during degradation; accelerating rise near failure",
        "feature_family": "Time-Domain Statistical",
        "family_prefix": "td_",
        "phase_introduced": 3,
        "version": "1.0",
        "status": "included",
        "gate_condition": "None (always included)",
    },
    "td_kurtosis": {
        "feature_name": "td_kurtosis",
        "short_description": "Kurtosis (impulsiveness of amplitude distribution)",
        "engineering_meaning": "Measures presence of high-amplitude, short-duration impact events",
        "units": "Dimensionless",
        "physical_formula": "Kurtosis = E[(X - μ)^4] / σ^4 (Gaussian baseline ≈ 3)",
        "source_module": "time_domain_features.py (reuses Phase 3 computation)",
        "input_signal": "Raw vibration waveform, 20,480 samples at 20 kHz",
        "dependencies": ["Phase 3 signal_health_indicators.parquet"],
        "expected_behaviour": "≈3 in health; rises during degradation; may decline near failure (documented non-monotonic behavior)",
        "feature_family": "Time-Domain Statistical",
        "family_prefix": "td_",
        "phase_introduced": 3,
        "version": "1.0",
        "status": "included",
        "gate_condition": "None (always included)",
        "notes": "Non-monotonic near failure is expected physical behavior, not a computation error",
    },
    "td_crest_factor": {
        "feature_name": "td_crest_factor",
        "short_description": "Crest Factor (peak-to-RMS ratio)",
        "engineering_meaning": "How 'spiky' the waveform is relative to average energy",
        "units": "Dimensionless ratio",
        "physical_formula": "Crest Factor = max(|signal|) / RMS",
        "source_module": "time_domain_features.py (reuses Phase 3 computation)",
        "input_signal": "Raw vibration waveform, 20,480 samples at 20 kHz",
        "dependencies": ["Phase 3 signal_health_indicators.parquet"],
        "expected_behaviour": "Low in health; rises ahead of RMS during degradation; may fall as RMS denominator grows faster",
        "feature_family": "Time-Domain Statistical",
        "family_prefix": "td_",
        "phase_introduced": 3,
        "version": "1.0",
        "status": "included",
        "gate_condition": "None (always included)",
    },
    "td_skewness": {
        "feature_name": "td_skewness",
        "short_description": "Skewness (asymmetry of amplitude distribution)",
        "engineering_meaning": "Whether positive and negative excursions are balanced",
        "units": "Dimensionless",
        "physical_formula": "Skewness = E[(X - μ)^3] / σ^3",
        "source_module": "time_domain_features.py (reuses Phase 3 computation)",
        "input_signal": "Raw vibration waveform, 20,480 samples at 20 kHz",
        "dependencies": ["Phase 3 signal_health_indicators.parquet"],
        "expected_behaviour": "Uncertain; engineering hypothesis to be validated against Phase 3 findings",
        "feature_family": "Time-Domain Statistical",
        "family_prefix": "td_",
        "phase_introduced": 3,
        "version": "1.0",
        "status": "included_conditional",
        "gate_condition": "Included only if Phase 3 engineering_findings.md documents observable skewness trend",
    },
}


def generate_feature_dictionary(
    included_features: list,
    gates_decisions: Dict[str, Dict],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Generate feature dictionary with all feature metadata.

    Args:
        included_features: List of feature names included in final store
        gates_decisions: Feature gate decisions
        logger: Logger instance

    Returns:
        dict with complete feature dictionary
    """
    logger.info("Generating feature dictionary")

    feature_dict = {
        "version": "1.0",
        "phase": 4,
        "generated": datetime.utcnow().isoformat() + "Z",
        "features": {},
    }

    # Add known features from template
    for feat_name, feat_meta in FEATURE_DICTIONARY_TEMPLATE.items():
        if feat_name in included_features:
            feature_dict["features"][feat_name] = feat_meta

    logger.info(f"Feature dictionary generated with {len(feature_dict['features'])} features")
    return feature_dict


def save_feature_dictionary(
    dictionary: Dict[str, Any],
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """Save feature dictionary to JSON file.

    Args:
        dictionary: Feature dictionary
        output_path: Path to write feature_dictionary.json
        logger: Logger instance
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dictionary, f, indent=2, default=str)

    logger.info(f"Feature dictionary saved to {output_path}")


from datetime import datetime
