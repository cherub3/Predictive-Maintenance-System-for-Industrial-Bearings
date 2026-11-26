"""Phase 4: Feature Engineering Pipeline Main Entry Point."""

import sys
from pathlib import Path
import importlib.util
import time

import pandas as pd

from src.config import load_config
from src.constants import (
    DEFAULT_CONFIG_PATH,
    EXIT_SUCCESS,
    EXIT_ERROR_CONFIG,
    EXIT_ERROR_FEATURE_ENGINEERING,
    PROJECT_ROOT,
)
from src.logger import setup_logger
from src.feature_engineering.feature_gate import FeatureGateResolver
from src.feature_engineering.feature_store_builder import (
    build_feature_store,
    save_feature_store,
    validate_feature_store,
)
from src.feature_engineering.feature_manifest import (
    generate_feature_manifest,
    save_feature_manifest,
)
from src.feature_engineering.feature_dictionary import (
    generate_feature_dictionary,
    save_feature_dictionary,
)


def run_feature_engineering_pipeline() -> int:
    """Execute the complete Phase 4 feature engineering pipeline."""
    start_time = time.time()

    try:
        config = load_config(DEFAULT_CONFIG_PATH)
    except Exception as e:
        print(f"Configuration load failed: {e}")
        return EXIT_ERROR_CONFIG

    logger = setup_logger(name="feature_engineering", config=config)
    logger.info("=" * 80)
    logger.info("PHASE 4 FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 80)

    try:
        # Step 1: Resolve feature gates
        logger.info("Step 1: Resolving feature gates based on Phase 3 findings")
        
        phase3_contract_path = Path(config["paths"]["report_dir"]) / "phase3_gating_decisions.yaml"
        bearing_geometry_module = None
        bearing_geometry_available = False

        try:
            geometry_path = PROJECT_ROOT / "src" / "signal_analysis" / "bearing_geometry.py"
            if geometry_path.exists():
                spec = importlib.util.spec_from_file_location("bearing_geometry", geometry_path)
                bearing_geometry_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(bearing_geometry_module)
                bearing_geometry_available = True
                logger.info("Phase 3 bearing geometry module loaded")
            else:
                logger.warning("Phase 3 bearing_geometry.py not found")
        except Exception as e:
            logger.warning(f"Failed to load Phase 3 bearing geometry: {e}")

        gate_resolver = FeatureGateResolver(logger)
        gates = gate_resolver.resolve_gates(phase3_contract_path, bearing_geometry_available)
        gate_decisions = gate_resolver.get_gate_decisions()

        # Step 2: Load metadata from Phase 1
        logger.info("Step 2: Loading Phase 1 metadata")
        
        metadata_path = Path(config["paths"]["processed_data_dir"]) / "metadata.parquet"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Phase 1 metadata not found at {metadata_path}")

        metadata_df = pd.read_parquet(metadata_path)
        metadata_df["timestamp"] = pd.to_datetime(metadata_df["timestamp"])
        logger.info(f"Loaded {len(metadata_df)} metadata records")

        # Step 3: Check Phase 3 health indicators
        logger.info("Step 3: Checking Phase 3 health indicators")
        
        phase3_indicators_path = Path(config["paths"]["processed_data_dir"]) / "signal_health_indicators.parquet"
        if not phase3_indicators_path.exists():
            raise FileNotFoundError(
                f"Phase 3 indicators not found at {phase3_indicators_path}"
            )

        logger.info("Phase 3 indicators found")

        # Step 4: Build feature store
        logger.info("Step 4: Building feature store")

        feature_store = build_feature_store(
            metadata_df,
            phase3_indicators_path,
            bearing_geometry_module,
            gates,
            config,
            logger,
        )

        # Step 5: Validate feature store
        logger.info("Step 5: Validating feature store")
        
        if not validate_feature_store(feature_store, logger):
            raise ValueError("Feature store validation failed")

        # Step 6: Generate manifest and dictionary
        logger.info("Step 6: Generating feature artifacts")
        
        included_features = [c for c in feature_store.columns if any(c.startswith(p) for p in ("td_", "fd_", "env_", "trend_", "cross_"))]
        excluded_features = [
            {"feature": "Peak-to-Peak Amplitude", "reason": "Redundant with RMS + Crest Factor"},
            {"feature": "Raw Shannon Entropy", "reason": "Redundant with Kurtosis/Skewness"},
            {"feature": "Variance", "reason": "Redundant with RMS"},
        ]

        manifest = generate_feature_manifest(
            Path(config["feature_engineering"]["feature_store_output"]),
            phase_version="4.0",
            phase_3_dependency_version="1.0",
            feature_catalog_version="1.0",
            configuration=config.get("feature_engineering", {}),
            gates_decisions=gate_decisions,
            included_features=included_features,
            excluded_features=excluded_features,
            feature_count=len(included_features),
            record_count=len(feature_store),
            processing_duration_seconds=time.time() - start_time,
            logger=logger,
        )

        manifest_path = Path(config["feature_engineering"]["feature_manifest_output"])
        save_feature_manifest(manifest, manifest_path, logger)

        dictionary = generate_feature_dictionary(included_features, gate_decisions, logger)
        dictionary_path = Path(config["feature_engineering"]["feature_dictionary_output"])
        save_feature_dictionary(dictionary, dictionary_path, logger)

        # Step 7: Save feature store
        logger.info("Step 7: Saving feature store")
        
        feature_store_path = Path(config["feature_engineering"]["feature_store_output"])
        save_feature_store(feature_store, feature_store_path, logger)

        elapsed = time.time() - start_time
        logger.info("=" * 80)
        logger.info("PHASE 4 FEATURE ENGINEERING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Feature store: {feature_store_path}")
        logger.info(f"Rows: {len(feature_store)}, Columns: {len(feature_store.columns)}")
        logger.info(f"Total elapsed time: {elapsed:.1f} seconds")

        return EXIT_SUCCESS

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return EXIT_ERROR_FEATURE_ENGINEERING
    except Exception as e:
        logger.error(f"Feature engineering pipeline failed: {e}", exc_info=True)
        return EXIT_ERROR_FEATURE_ENGINEERING


if __name__ == "__main__":
    exit_code = run_feature_engineering_pipeline()
    sys.exit(exit_code)
