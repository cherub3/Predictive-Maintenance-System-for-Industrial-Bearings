"""Feature manifest generator for versioning and reproducibility.

The feature manifest records complete metadata about the feature store:
- Version information and build timestamp
- Dependencies on prior phases (Phase 1, Phase 3)
- Configuration used for feature computation
- Gate decisions and rationale
- List of included/excluded features
- Quality checks and counts
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


def generate_feature_manifest(
    feature_store_path: Path,
    phase_version: str,
    phase_3_dependency_version: str,
    feature_catalog_version: str,
    configuration: Dict[str, Any],
    gates_decisions: Dict[str, Dict],
    included_features: list,
    excluded_features: list,
    feature_count: int,
    record_count: int,
    processing_duration_seconds: float,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Generate a comprehensive feature manifest.

    Args:
        feature_store_path: Path where feature_store.parquet will be saved
        phase_version: Phase 4 version string
        phase_3_dependency_version: Phase 3 output version
        feature_catalog_version: Feature catalog documentation version
        configuration: Feature engineering configuration dict
        gates_decisions: Feature gate decisions with rationale
        included_features: List of included feature names
        excluded_features: List of excluded features with reasons
        feature_count: Number of features in store
        record_count: Number of rows in feature store
        processing_duration_seconds: Time taken to compute features
        logger: Logger instance

    Returns:
        dict with complete manifest structure
    """
    logger.info("Generating feature manifest")

    manifest = {
        "feature_store_metadata": {
            "version": "1.0",
            "phase": 4,
            "phase_version": phase_version,
            "phase_3_dependency_version": phase_3_dependency_version,
            "feature_catalog_version": feature_catalog_version,
            "build_timestamp": datetime.utcnow().isoformat() + "Z",
            "built_by": "Feature Engineering Pipeline (Phase 4)",
            "configuration_version": "1.0",
            "dataset_version": "NASA IMS Set 1+2+3 (multi-dataset)",
            "dataset_source": "NASA IMS Bearing Dataset, Sets 1/2/3 combined",
            "number_of_features": feature_count,
            "number_of_records": record_count,
            "processing_duration_seconds": round(processing_duration_seconds, 2),
            "feature_store_location": str(feature_store_path),
        },
        "feature_gate_decisions": gates_decisions,
        "dependencies": [
            {
                "dependency": "Phase 1 Metadata",
                "version": "1.0",
                "file": "data/processed/metadata.parquet",
                "usage": "Bearing ID, channel ID, timestamp, file structure",
            },
            {
                "dependency": "Phase 3 Health Indicators",
                "version": "1.0",
                "file": "data/processed/signal_health_indicators.parquet",
                "usage": "RMS, Kurtosis, Crest Factor, Skewness (reused, not recomputed)",
            },
            {
                "dependency": "Phase 3 Bearing Geometry",
                "version": "1.0" if gates_decisions.get("envelope_features", {}).get("included") else "N/A",
                "file": "src/signal_analysis/bearing_geometry.py (if verified)",
                "usage": "Defect frequency constants for envelope features",
            },
        ],
        "features_included": included_features,
        "features_excluded": excluded_features,
    }

    logger.info(f"Manifest generated with {feature_count} features, {record_count} records")
    return manifest


def save_feature_manifest(
    manifest: Dict[str, Any],
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """Save feature manifest to JSON file.

    Args:
        manifest: Manifest dictionary
        output_path: Path to write manifest.json
        logger: Logger instance
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    logger.info(f"Feature manifest saved to {output_path}")
