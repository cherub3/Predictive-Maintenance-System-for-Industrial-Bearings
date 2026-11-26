"""Health indicator trend plots across the run-to-failure timeline.

Generates time-series plots of RMS, Kurtosis, and Crest Factor for each
bearing, independently per registered dataset. All plots use the
non-interactive Agg backend and are saved to disk.

Plots are generated **per dataset_id**, not just per bearing_id: the same
bearing_id string (e.g. "bearing_3") refers to physically different bearings
in different datasets (Set 1's bearing_3 has an inner-race defect; Set 3's
bearing_3 has an outer-race defect), so grouping only by bearing_id across
datasets would splice two unrelated experiments' trends onto one line.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend; must precede pyplot import
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from src.config import get_all_dataset_ids, get_dataset_config


_INDICATORS = [
    ("rms", "RMS (g)", "RMS Vibration"),
    ("kurtosis", "Kurtosis (dimensionless)", "Kurtosis"),
    ("crest_factor", "Crest Factor (dimensionless)", "Crest Factor"),
]

_FAILURE_COLORS = ["#2196F3", "#4CAF50", "#F44336", "#FF9800"]


def _bearing_label(bearing_id: str, bearing_cfg: Dict[str, Any]) -> str:
    """Builds a human-readable label from the registry's status/failure_mode."""
    if bearing_cfg.get("status") == "rtf":
        failure_mode = bearing_cfg.get("failure_mode", "unknown").replace("_", " ")
        return f"{bearing_id.replace('_', ' ').title()} ({failure_mode.title()} Defect)"
    return f"{bearing_id.replace('_', ' ').title()} (Healthy)"


def plot_all_bearings_indicator(
    health_df: pd.DataFrame,
    dataset_id: str,
    config: Dict[str, Any],
    indicator: str,
    ylabel: str,
    title_suffix: str,
    figures_dir: Path,
    logger: logging.Logger,
) -> None:
    """Plots a single health indicator for all of one dataset's bearings on one figure.

    Args:
        health_df: Full health indicators DataFrame (all datasets).
        dataset_id: Which dataset to plot.
        config: Full project configuration dictionary.
        indicator: Column name to plot (e.g. 'rms').
        ylabel: Y-axis label string.
        title_suffix: Title component string.
        figures_dir: Root figures output directory.
        logger: Pipeline logger.
    """
    dataset_cfg = get_dataset_config(config, dataset_id)
    bearings = dataset_cfg.get("bearings", {})
    bearing_ids = sorted(bearings.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes_flat = axes.flatten()

    # Use X-axis channel only for trend clarity (one line per bearing)
    x_channels = health_df[
        (health_df["dataset_id"] == dataset_id) & (health_df["axis"] == "X")
    ].copy()
    x_channels["timestamp"] = pd.to_datetime(x_channels["timestamp"])

    for i, bearing_id in enumerate(bearing_ids):
        ax = axes_flat[i]
        bearing_data = (
            x_channels[x_channels["bearing_id"] == bearing_id]
            .sort_values("timestamp")
        )

        if bearing_data.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        color = _FAILURE_COLORS[i % len(_FAILURE_COLORS)]
        ax.plot(
            bearing_data["timestamp"],
            bearing_data[indicator],
            color=color,
            linewidth=0.8,
            alpha=0.85,
        )
        ax.set_title(_bearing_label(bearing_id, bearings[bearing_id]), fontsize=10)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
        ax.grid(True, alpha=0.3, linewidth=0.5)

    fig.suptitle(
        f"{title_suffix} — All Bearings (X-axis channel)\n"
        f"{dataset_cfg.get('display_name', dataset_id)} Run-to-Failure",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    out_dir = figures_dir / "health_trends"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset_id}_{indicator}_all_bearings.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path.name}")


def plot_xy_comparison(
    health_df: pd.DataFrame,
    dataset_id: str,
    bearing_id: str,
    bearing_label: str,
    indicator: str,
    ylabel: str,
    figures_dir: Path,
    logger: logging.Logger,
) -> None:
    """Plots X vs Y axis channels for a single bearing to reveal directional asymmetry.

    Only meaningful for paired_xy datasets — callers should only invoke this
    for datasets whose channel_layout has a Y axis.

    Args:
        health_df: Full health indicators DataFrame (all datasets).
        dataset_id: Which dataset the bearing belongs to.
        bearing_id: Bearing identifier string (e.g. 'bearing_3').
        bearing_label: Human-readable label for the plot title.
        indicator: Health indicator column name.
        ylabel: Y-axis label string.
        figures_dir: Root figures output directory.
        logger: Pipeline logger.
    """
    bearing_data = health_df[
        (health_df["dataset_id"] == dataset_id) & (health_df["bearing_id"] == bearing_id)
    ].copy()
    bearing_data["timestamp"] = pd.to_datetime(bearing_data["timestamp"])

    x_data = bearing_data[bearing_data["axis"] == "X"].sort_values("timestamp")
    y_data = bearing_data[bearing_data["axis"] == "Y"].sort_values("timestamp")

    if y_data.empty:
        logger.warning(f"[{dataset_id}] No Y-axis data for {bearing_id}; skipping XY comparison plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x_data["timestamp"], x_data[indicator], label="X-axis", color="#1565C0", linewidth=0.9, alpha=0.9)
    ax.plot(y_data["timestamp"], y_data[indicator], label="Y-axis", color="#C62828", linewidth=0.9, alpha=0.9)
    ax.set_title(
        f"{ylabel} — {bearing_label} (X vs Y)\n{dataset_id}",
        fontsize=11,
    )
    ax.set_xlabel("Timestamp")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    plt.tight_layout()

    out_dir = figures_dir / "channel_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset_id}_{bearing_id}_{indicator}_xy.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path.name}")


def generate_all_health_trend_plots(
    health_df: pd.DataFrame,
    config: Dict[str, Any],
    figures_dir: Path,
    logger: logging.Logger,
) -> None:
    """Generates the complete set of health indicator trend figures, per dataset.

    Args:
        health_df: Full health indicators DataFrame (all datasets).
        config: Full project configuration dictionary.
        figures_dir: Root figures output directory.
        logger: Pipeline logger.
    """
    logger.info("Generating health indicator trend plots...")

    for dataset_id in get_all_dataset_ids(config):
        dataset_cfg = get_dataset_config(config, dataset_id)
        bearings = dataset_cfg.get("bearings", {})

        for col, ylabel, title in _INDICATORS:
            plot_all_bearings_indicator(
                health_df, dataset_id, config, col, ylabel, title, figures_dir, logger
            )

        # X vs Y comparison for this dataset's run-to-failure bearing(s), only
        # for paired_xy datasets (single_axis datasets have no Y channel).
        if dataset_cfg.get("channel_layout") == "paired_xy":
            rtf_bearing_ids = [b for b, cfg in bearings.items() if cfg.get("status") == "rtf"]
            for bearing_id in rtf_bearing_ids:
                label = _bearing_label(bearing_id, bearings[bearing_id])
                for col, ylabel, _ in _INDICATORS:
                    plot_xy_comparison(
                        health_df, dataset_id, bearing_id, label, col, ylabel, figures_dir, logger
                    )

    logger.info("Health trend plots complete")
