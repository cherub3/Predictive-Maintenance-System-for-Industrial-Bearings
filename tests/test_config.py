"""Tests for the datasets registry helpers in src/config.py.

Covers both the real project config (integration-style: the registry as
actually declared in config/config.yaml) and synthetic fixtures (unit-style:
fail-closed behaviour on malformed/missing registries).
"""

import pytest

from src.config import (
    ConfigurationError,
    get_all_dataset_ids,
    get_censored_units,
    get_channel_bearing_map,
    get_dataset_config,
    get_dataset_for_unit,
    get_rtf_units,
    load_config,
    make_unit_id,
    parse_unit_id,
)
from src.constants import DEFAULT_CONFIG_PATH


@pytest.fixture(scope="module")
def real_config():
    return load_config(DEFAULT_CONFIG_PATH)


def test_make_unit_id_and_parse_unit_id_roundtrip():
    unit_id = make_unit_id("set2", "bearing_1")
    assert unit_id == "set2::bearing_1"
    dataset_id, bearing_id = parse_unit_id(unit_id)
    assert dataset_id == "set2"
    assert bearing_id == "bearing_1"


def test_parse_unit_id_without_separator_raises():
    with pytest.raises(ConfigurationError):
        parse_unit_id("bearing_1")


def test_get_all_dataset_ids(real_config):
    assert get_all_dataset_ids(real_config) == ["set1", "set2", "set3"]


def test_get_rtf_units_across_all_three_sets(real_config):
    rtf_units = get_rtf_units(real_config)
    assert rtf_units == [
        "set1::bearing_3",
        "set1::bearing_4",
        "set2::bearing_1",
        "set3::bearing_3",
    ]


def test_get_censored_units_across_all_three_sets(real_config):
    censored_units = get_censored_units(real_config)
    assert censored_units == sorted([
        "set1::bearing_1", "set1::bearing_2",
        "set2::bearing_2", "set2::bearing_3", "set2::bearing_4",
        "set3::bearing_1", "set3::bearing_2", "set3::bearing_4",
    ])


def test_rtf_and_censored_units_are_disjoint_and_complete(real_config):
    """Every bearing in every dataset must be classified exactly once."""
    rtf = set(get_rtf_units(real_config))
    censored = set(get_censored_units(real_config))
    assert rtf.isdisjoint(censored)
    assert len(rtf) + len(censored) == 12  # 3 datasets x 4 bearings


def test_get_dataset_for_unit_set1_rtf_bearing(real_config):
    info = get_dataset_for_unit(real_config, "set1::bearing_3")
    assert info["dataset_id"] == "set1"
    assert info["bearing_id"] == "bearing_3"
    assert info["status"] == "rtf"
    assert info["failure_mode"] == "inner_race"
    assert info["channel_layout"] == "paired_xy"
    assert info["expected_channels_per_file"] == 8
    assert info["channels"] == [5, 6]


def test_get_dataset_for_unit_set2_rtf_bearing_single_axis(real_config):
    info = get_dataset_for_unit(real_config, "set2::bearing_1")
    assert info["status"] == "rtf"
    assert info["failure_mode"] == "outer_race"
    assert info["channel_layout"] == "single_axis"
    assert info["expected_channels_per_file"] == 4
    assert info["channels"] == [1]


def test_get_dataset_for_unit_unknown_dataset_raises(real_config):
    with pytest.raises(ConfigurationError):
        get_dataset_for_unit(real_config, "set99::bearing_1")


def test_get_dataset_for_unit_unknown_bearing_raises(real_config):
    with pytest.raises(ConfigurationError):
        get_dataset_for_unit(real_config, "set1::bearing_99")


def test_get_channel_bearing_map_set1_paired_xy(real_config):
    channel_map = get_channel_bearing_map(real_config, "set1")
    assert len(channel_map) == 8
    assert channel_map[1] == {"bearing_id": "bearing_1", "axis": "X"}
    assert channel_map[2] == {"bearing_id": "bearing_1", "axis": "Y"}
    assert channel_map[5] == {"bearing_id": "bearing_3", "axis": "X"}
    assert channel_map[8] == {"bearing_id": "bearing_4", "axis": "Y"}


def test_get_channel_bearing_map_set2_single_axis(real_config):
    channel_map = get_channel_bearing_map(real_config, "set2")
    assert len(channel_map) == 4
    assert channel_map[1] == {"bearing_id": "bearing_1", "axis": "X"}
    assert channel_map[4] == {"bearing_id": "bearing_4", "axis": "X"}


def test_get_channel_bearing_map_unknown_dataset_raises(real_config):
    with pytest.raises(ConfigurationError):
        get_channel_bearing_map(real_config, "not_a_real_set")


def test_get_dataset_config_set1(real_config):
    cfg = get_dataset_config(real_config, "set1")
    assert cfg["dataset_id"] == "set1"
    assert cfg["raw_dir"] == "data/raw/set1"
    assert cfg["channel_layout"] == "paired_xy"
    assert cfg["expected_channels_per_file"] == 8
    assert set(cfg["bearings"].keys()) == {"bearing_1", "bearing_2", "bearing_3", "bearing_4"}


def test_get_dataset_config_set3_has_corrected_run_window_and_exclusion(real_config):
    cfg = get_dataset_config(real_config, "set3")
    assert cfg["run_window"]["end"] == "2004-04-18"
    assert cfg["excluded_files"] == ["2004.04.18.02.42.55"]


def test_get_dataset_config_unknown_dataset_raises(real_config):
    with pytest.raises(ConfigurationError):
        get_dataset_config(real_config, "set99")


def test_missing_datasets_registry_raises_fail_closed():
    broken_config = {"project": {"name": "test"}}
    with pytest.raises(ConfigurationError):
        get_rtf_units(broken_config)
    with pytest.raises(ConfigurationError):
        get_censored_units(broken_config)


def test_empty_sets_registry_raises_fail_closed():
    broken_config = {"datasets": {"registry_version": "1.0", "sets": {}}}
    with pytest.raises(ConfigurationError):
        get_rtf_units(broken_config)
