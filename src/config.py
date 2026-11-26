"""Configuration manager for loading and retrieving project configuration options.

Decouples hardcoded parameters from the Python codebase by resolving environment
overrides and loading configurations from YAML format.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from src.constants import DEFAULT_CONFIG_PATH


class ConfigurationError(Exception):
    """Raised when there is an issue loading or parsing the configuration file."""
    pass


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads configuration dictionary from a YAML file.

    Args:
        config_path: Path to the yaml config file. Defaults to DEFAULT_CONFIG_PATH.

    Returns:
        Dict[str, Any]: Parsed configuration parameters.

    Raises:
        ConfigurationError: If file not found or invalid YAML syntax is encountered.
    """
    path = config_path or DEFAULT_CONFIG_PATH

    if not path.exists():
        raise ConfigurationError(
            f"Configuration file not found at: {path.absolute()}"
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if not isinstance(config, dict):
                raise ConfigurationError(
                    f"Configuration format is invalid. Expected a dictionary, got {type(config)}"
                )
            return config
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Failed to parse YAML configuration: {exc}") from exc
    except Exception as exc:
        raise ConfigurationError(f"Unexpected error loading configuration: {exc}") from exc


# ---------------------------------------------------------------------------
# Datasets registry helpers
#
# These are the ONLY functions in the codebase that should read the nested
# `datasets.sets.*` shape directly. Every consumer of RTF/censored unit
# lists, per-dataset channel maps, or per-unit metadata should call one of
# these instead of walking the config dict itself, so the registry's shape
# can evolve in one place. A "unit_id" is the qualified, globally-unique
# identifier `f"{dataset_id}::{bearing_id}"` (e.g. "set2::bearing_1") — this
# is what disambiguates "bearing_3" in Set 1 (inner-race RTF) from
# "bearing_3" in Set 3 (outer-race RTF), which are physically unrelated
# bearings that happen to share a name.
# ---------------------------------------------------------------------------

def _get_datasets_registry(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return the `datasets` registry block, failing closed if absent/malformed."""
    registry = config.get("datasets")
    if not isinstance(registry, dict) or "sets" not in registry:
        raise ConfigurationError(
            "config.yaml is missing a valid 'datasets' registry block "
            "(expected keys: registry_version, defaults, sets)."
        )
    if not isinstance(registry["sets"], dict) or not registry["sets"]:
        raise ConfigurationError("'datasets.sets' must be a non-empty mapping of dataset_id -> dataset config.")
    return registry


def make_unit_id(dataset_id: str, bearing_id: str) -> str:
    """Build the qualified, globally-unique unit identifier for a bearing.

    Grain key used throughout the pipeline is `(unit_id, channel, timestamp)`,
    equivalent to `(dataset_id, bearing_id, channel, timestamp)` but requiring
    no change to existing `groupby("bearing_id")` / `isin(rtf_units)` call
    sites once `bearing_id` columns are populated with qualified unit_ids.
    """
    return f"{dataset_id}::{bearing_id}"


def parse_unit_id(unit_id: str) -> "tuple[str, str]":
    """Split a qualified unit_id back into (dataset_id, bearing_id)."""
    if "::" not in unit_id:
        raise ConfigurationError(f"'{unit_id}' is not a qualified unit_id (expected 'dataset_id::bearing_id').")
    dataset_id, bearing_id = unit_id.split("::", 1)
    return dataset_id, bearing_id


def get_rtf_units(config: Dict[str, Any]) -> List[str]:
    """Return the sorted list of qualified unit_ids with status == 'rtf' across all datasets."""
    registry = _get_datasets_registry(config)
    units = []
    for dataset_id, dataset_cfg in registry["sets"].items():
        for bearing_id, bearing_cfg in dataset_cfg.get("bearings", {}).items():
            if bearing_cfg.get("status") == "rtf":
                units.append(make_unit_id(dataset_id, bearing_id))
    return sorted(units)


def get_censored_units(config: Dict[str, Any]) -> List[str]:
    """Return the sorted list of qualified unit_ids with status == 'censored' across all datasets."""
    registry = _get_datasets_registry(config)
    units = []
    for dataset_id, dataset_cfg in registry["sets"].items():
        for bearing_id, bearing_cfg in dataset_cfg.get("bearings", {}).items():
            if bearing_cfg.get("status") == "censored":
                units.append(make_unit_id(dataset_id, bearing_id))
    return sorted(units)


def get_dataset_for_unit(config: Dict[str, Any], unit_id: str) -> Dict[str, Any]:
    """Return combined dataset + bearing metadata for a qualified unit_id.

    Result includes dataset-level fields (raw_dir, run_window, channel_layout,
    expected_channels_per_file, display_name) merged with the bearing's own
    entry (channels, channel_axes, status, failure_mode if present), plus
    `dataset_id`/`bearing_id`/`unit_id` for traceability.
    """
    registry = _get_datasets_registry(config)
    dataset_id, bearing_id = parse_unit_id(unit_id)
    dataset_cfg = registry["sets"].get(dataset_id)
    if dataset_cfg is None:
        raise ConfigurationError(f"Unknown dataset_id '{dataset_id}' (from unit_id '{unit_id}').")
    bearing_cfg = dataset_cfg.get("bearings", {}).get(bearing_id)
    if bearing_cfg is None:
        raise ConfigurationError(f"Unknown bearing_id '{bearing_id}' in dataset '{dataset_id}' (from unit_id '{unit_id}').")

    result = {
        "dataset_id": dataset_id,
        "bearing_id": bearing_id,
        "unit_id": unit_id,
        "display_name": dataset_cfg.get("display_name"),
        "raw_dir": dataset_cfg.get("raw_dir"),
        "run_window": dataset_cfg.get("run_window"),
        "channel_layout": dataset_cfg.get("channel_layout"),
        "expected_channels_per_file": dataset_cfg.get("expected_channels_per_file"),
    }
    result.update(bearing_cfg)
    return result


def get_dataset_config(config: Dict[str, Any], dataset_id: str) -> Dict[str, Any]:
    """Return the raw per-dataset registry block for one dataset_id.

    Includes `raw_dir`, `run_window`, `channel_layout`, `expected_channels_per_file`,
    `display_name`, `bearings`, and any dataset-level extras (e.g.
    `last_valid_file_timestamp`, `excluded_files`), plus `dataset_id` itself.
    Use this for whole-dataset operations (e.g. Phase 1 ingestion, which processes
    one dataset's files at a time); use `get_dataset_for_unit` when you have a
    specific bearing's qualified unit_id.
    """
    registry = _get_datasets_registry(config)
    dataset_cfg = registry["sets"].get(dataset_id)
    if dataset_cfg is None:
        raise ConfigurationError(f"Unknown dataset_id '{dataset_id}'.")
    result = {"dataset_id": dataset_id}
    result.update(dataset_cfg)
    return result


def get_channel_bearing_map(config: Dict[str, Any], dataset_id: str) -> Dict[int, Dict[str, str]]:
    """Build a {channel_number: {"bearing_id": ..., "axis": ...}} map for one dataset.

    Replaces the old Set-1-only hardcoded CHANNEL_BEARING_MAP module constant —
    this is derived from the registry so it is correct for any dataset's
    channel_layout (paired_xy or single_axis) without a code change.
    """
    registry = _get_datasets_registry(config)
    dataset_cfg = registry["sets"].get(dataset_id)
    if dataset_cfg is None:
        raise ConfigurationError(f"Unknown dataset_id '{dataset_id}'.")

    channel_map: Dict[int, Dict[str, str]] = {}
    for bearing_id, bearing_cfg in dataset_cfg.get("bearings", {}).items():
        channel_axes = bearing_cfg.get("channel_axes", {})
        for channel, axis in channel_axes.items():
            channel_map[int(channel)] = {"bearing_id": bearing_id, "axis": axis}
    return channel_map


def get_all_dataset_ids(config: Dict[str, Any]) -> List[str]:
    """Return the sorted list of all registered dataset_ids (e.g. ['set1', 'set2', 'set3'])."""
    registry = _get_datasets_registry(config)
    return sorted(registry["sets"].keys())
