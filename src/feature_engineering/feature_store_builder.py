"""Feature store builder: orchestrates all feature families into a single table."""

import time

from src.config import get_all_dataset_ids


def build_feature_store(
    metadata_df, phase3_indicators_path, bearing_geometry_module, gates, config, logger
):
    """Build complete feature store by orchestrating all feature families across all datasets."""
    logger.info("Beginning feature store construction")
    start_time = time.time()

    from . import time_domain_features
    from . import frequency_domain_features
    from . import envelope_features
    from . import trend_features
    from . import cross_channel_features

    logger.info("Step 1: Loading time-domain features from Phase 3")
    try:
        td_df = time_domain_features.load_time_domain_features(phase3_indicators_path, gates, logger)
        valid, msg = time_domain_features.validate_time_domain_features(td_df, logger)
        if not valid:
            raise ValueError(msg)
    except Exception as e:
        logger.error(f"Failed to load time-domain features: {e}")
        raise

    logger.info("Step 2: Building frequency-domain features")
    try:
        fe_config = config.get("feature_engineering", {})
        frequency_bands = fe_config.get("frequency_bands", [[0, 1000], [1000, 5000], [5000, 10000]])
        sampling_rate = config.get("datasets", {}).get("defaults", {}).get("sampling_rate_hz", 20000)
        fd_df = frequency_domain_features.build_frequency_domain_features(
            metadata_df, config, frequency_bands, sampling_rate, fe_config, logger
        )
        valid, msg = frequency_domain_features.validate_frequency_domain_features(fd_df, logger)
        if not valid:
            raise ValueError(msg)
    except Exception as e:
        logger.error(f"Failed to build frequency-domain features: {e}")
        raise

    env_df = None
    if gates.get("envelope_features", False):
        logger.info("Step 3: Building envelope/defect-frequency features (gated ON)")
        try:
            sampling_rate = config.get("datasets", {}).get("defaults", {}).get("sampling_rate_hz", 20000)
            env_df = envelope_features.build_envelope_features(
                metadata_df, bearing_geometry_module, sampling_rate,
                config, logger
            )
            valid, msg = envelope_features.validate_envelope_features(env_df, logger)
            if not valid:
                raise ValueError(msg)
        except Exception as e:
            logger.warning(f"Envelope features failed; excluding: {e}")
            env_df = None
    else:
        logger.info("Step 3: Envelope/defect-frequency features (gated OFF)")

    logger.info("Step 4: Building trend/derived features")
    try:
        fe_config = config.get("feature_engineering", {})
        trend_window_size = fe_config.get("trend_window_size", 5)
        gap_threshold = fe_config.get("chronology_gap_threshold_minutes", 30)
        trend_df = trend_features.build_trend_features(
            td_df, gates, trend_window_size, gap_threshold, logger
        )
        valid, msg = trend_features.validate_trend_features(trend_df, logger)
        if not valid:
            raise ValueError(msg)
    except Exception as e:
        logger.error(f"Failed to build trend features: {e}")
        raise

    cross_df = None
    if gates.get("xy_energy_ratio", False):
        logger.info("Step 5: Building cross-channel features (gated ON)")
        try:
            cross_df = cross_channel_features.build_cross_channel_features(td_df, gates, config, logger)
            if not cross_df.empty:
                valid, msg = cross_channel_features.validate_cross_channel_features(cross_df, logger)
                if not valid:
                    raise ValueError(msg)
        except Exception as e:
            logger.warning(f"Cross-channel features failed: {e}")
            cross_df = None
    else:
        logger.info("Step 5: Cross-channel features (gated OFF)")

    logger.info("Step 6: Merging all feature families")
    feature_store = td_df.copy()

    # Every merge is keyed on (dataset_id, timestamp, bearing_id, channel) — NOT
    # just the latter three — because bearing_id and channel numbers repeat across
    # datasets (e.g. channel 1 exists in set1, set2, AND set3, referring to
    # physically unrelated bearings). Merging without dataset_id risks either a
    # silent row explosion or (with non-overlapping timestamps, as today)
    # accidentally-correct behavior that would break the moment two datasets ever
    # shared a timestamp.
    merge_keys = ["dataset_id", "timestamp", "bearing_id", "channel"]

    fd_merge = fd_df[[c for c in fd_df.columns if c in merge_keys or c.startswith("fd_")]].drop_duplicates()
    feature_store = feature_store.merge(fd_merge, on=merge_keys, how="left")

    trend_merge = trend_df[[c for c in trend_df.columns if c in merge_keys or c.startswith("trend_")]].drop_duplicates()
    feature_store = feature_store.merge(trend_merge, on=merge_keys, how="left")

    if env_df is not None and not env_df.empty:
        env_merge = env_df[[c for c in env_df.columns if c in merge_keys or c.startswith("env_")]].drop_duplicates()
        feature_store = feature_store.merge(env_merge, on=merge_keys, how="left")

    if cross_df is not None and not cross_df.empty:
        # cross_channel_features is per (dataset_id, bearing_id, timestamp) — it
        # has no `channel` column (it aggregates across a bearing's X/Y channels),
        # so its merge key omits `channel`.
        cross_merge = cross_df.drop_duplicates()
        feature_store = feature_store.merge(
            cross_merge, on=["dataset_id", "timestamp", "bearing_id"], how="left"
        )

    # Expected row count is derived from Phase 1 metadata's own per-file channel
    # counts (not a hardcoded Set-1-only constant), so this stays correct as
    # datasets are added or removed from the registry.
    dataset_ids = get_all_dataset_ids(config)
    expected_rows = len(td_df)
    actual_rows = len(feature_store)
    logger.info(f"Feature store row count: {actual_rows} (expected {expected_rows}, across {len(dataset_ids)} datasets)")
    if actual_rows != expected_rows:
        raise ValueError(
            f"Feature store row count mismatch: expected {expected_rows}, got {actual_rows}. "
            "Data loss or duplication occurred during merge."
        )

    feature_store = feature_store.sort_values(by=["dataset_id", "bearing_id", "channel", "timestamp"]).reset_index(drop=True)
    elapsed = time.time() - start_time
    logger.info(f"Feature store built in {elapsed:.1f} seconds; shape: {feature_store.shape}")

    return feature_store


def save_feature_store(feature_store, output_path, logger):
    """Save feature store to parquet format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_store.to_parquet(output_path, index=False, compression="snappy")
    logger.info(f"Feature store saved to {output_path}")


def validate_feature_store(feature_store, logger):
    """Validate feature store integrity."""
    logger.info("Validating feature store integrity")

    traceability_cols = ["dataset_id", "timestamp", "bearing_id", "channel"]
    missing = [c for c in traceability_cols if c not in feature_store.columns]
    if missing:
        logger.error(f"Missing traceability columns: {missing}")
        return False

    if feature_store["bearing_id"].isna().any() or feature_store["channel"].isna().any():
        logger.error("Found null values in bearing_id or channel")
        return False

    valid_bearing_ids = {"bearing_1", "bearing_2", "bearing_3", "bearing_4"}
    if not (feature_store["bearing_id"].isin(valid_bearing_ids).all()):
        logger.error("Invalid bearing_id values")
        return False

    # Channel numbers are per-dataset local (1-8 for paired_xy, 1-4 for
    # single_axis); range 1-8 covers every registered dataset's valid channels.
    if not (feature_store["channel"].isin(range(1, 9)).all()):
        logger.error("Invalid channel values")
        return False

    logger.info(f"Feature store validation passed; {len(feature_store)} rows, {len(feature_store.columns)} columns")
    return True
