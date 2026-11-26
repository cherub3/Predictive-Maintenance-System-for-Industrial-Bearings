"""Spectral checkpoint comparison plots for Tier 2 analysis.

Plots FFT magnitude spectra for selected checkpoint files, enabling visual
comparison of spectral evolution across early, mid, and late degradation stages.
Generated independently per registered dataset, for that dataset's own
run-to-failure (RTF) bearing(s) — each dataset has its own channel layout and
failure mode, so channel numbers and defect-frequency annotations cannot be
hardcoded to Set 1's layout.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import get_all_dataset_ids, get_channel_bearing_map, get_dataset_config
from src.signal_analysis.spectral_analysis import compute_fft_spectrum, analyse_checkpoint_spectrum
from src.signal_analysis.signal_loader import load_signal_file
from src.signal_analysis.bearing_geometry import DefectFrequencies

# Maps a bearing's documented failure_mode to the corresponding characteristic
# defect frequency attribute on DefectFrequencies.
_FAILURE_MODE_TO_DEFECT_FREQ_ATTR = {
    "inner_race": "bpfi_hz",
    "outer_race": "bpfo_hz",
    "rolling_element": "bsf_hz",
}
_FAILURE_MODE_TO_LABEL = {
    "inner_race": "BPFI",
    "outer_race": "BPFO",
    "rolling_element": "BSF",
}


def _rtf_bearing_x_channels(config: Dict[str, Any], dataset_id: str) -> Dict[str, int]:
    """Returns {bearing_id: X-axis channel number} for a dataset's RTF bearings."""
    dataset_cfg = get_dataset_config(config, dataset_id)
    channel_map = get_channel_bearing_map(config, dataset_id)
    result = {}
    for bearing_id, bearing_cfg in dataset_cfg.get("bearings", {}).items():
        if bearing_cfg.get("status") != "rtf":
            continue
        for channel, info in channel_map.items():
            if info["bearing_id"] == bearing_id and info["axis"] == "X":
                result[bearing_id] = channel
                break
    return result


def plot_spectral_evolution(
    checkpoint_df: pd.DataFrame,
    config: Dict[str, Any],
    defect_freqs: Optional[DefectFrequencies],
    figures_dir: Path,
    logger: logging.Logger,
) -> None:
    """Plots FFT spectra for checkpoint files overlaid by degradation stage.

    Generates one figure per (dataset, run-to-failure bearing), showing
    spectral evolution from early to late stage. Defect frequencies are
    annotated using the bearing's own documented failure mode, if bearing
    geometry is available.

    Args:
        checkpoint_df: Checkpoint selection DataFrame, enriched with an
            `absolute_path` column (dataset_id + filename joined against
            Phase 1 metadata).
        config: Full project configuration dictionary.
        defect_freqs: Computed defect frequencies (or None if geometry unavailable).
        figures_dir: Root figures output directory.
        logger: Pipeline logger.
    """
    sa_cfg = config["signal_analysis"]
    sampling_rate = float(sa_cfg["sampling_rate_hz"])
    freq_bands = config["feature_engineering"]["frequency_bands"]

    stage_colors = {
        "early_stage": "#2196F3",
        "mid_stage": "#4CAF50",
        "kurtosis_peak": "#FF9800",
        "late_stage": "#F44336",
    }

    for dataset_id in get_all_dataset_ids(config):
        dataset_cfg = get_dataset_config(config, dataset_id)
        dataset_checkpoints = checkpoint_df[checkpoint_df["dataset_id"] == dataset_id]
        if dataset_checkpoints.empty:
            continue

        target_channels = _rtf_bearing_x_channels(config, dataset_id)

        for bearing_id, channel_1indexed in target_channels.items():
            logger.info(f"[{dataset_id}] Generating spectral evolution plot for {bearing_id}...")
            checkpoints_sorted = dataset_checkpoints.sort_values("timestamp")

            fig, ax = plt.subplots(figsize=(14, 6))

            expected_channels = dataset_cfg["expected_channels_per_file"]
            plotted = 0
            for _, cp_row in checkpoints_sorted.iterrows():
                if pd.isna(cp_row.get("absolute_path")):
                    continue
                file_path = Path(cp_row["absolute_path"])
                data, status = load_signal_file(
                    file_path, expected_channels=expected_channels, logger=logger
                )
                if data is None:
                    continue

                waveform = data[:, channel_1indexed - 1]
                freqs, mag = compute_fft_spectrum(waveform, sampling_rate)

                reason = cp_row["selection_reason"]
                color = stage_colors.get(reason, "#607D8B")
                label = f"{reason} ({pd.Timestamp(cp_row['timestamp']).strftime('%b %d')})"

                ax.semilogy(freqs, mag, color=color, alpha=0.4, linewidth=0.7, label=label)
                plotted += 1

            # Annotate the defect frequency matching this bearing's documented failure mode
            failure_mode = dataset_cfg["bearings"][bearing_id].get("failure_mode")
            if defect_freqs is not None and failure_mode in _FAILURE_MODE_TO_DEFECT_FREQ_ATTR:
                attr = _FAILURE_MODE_TO_DEFECT_FREQ_ATTR[failure_mode]
                freq_val = getattr(defect_freqs, attr)
                label_text = f"{_FAILURE_MODE_TO_LABEL[failure_mode]} ({freq_val:.1f} Hz)"
                ax.axvline(freq_val, color="red", linestyle="--", alpha=0.8, linewidth=1.5, label=label_text)

            # Annotate band boundaries
            for i, band in enumerate(freq_bands):
                ax.axvspan(band[0], band[1], alpha=0.05, color=["blue", "green", "purple"][i])

            ax.set_xlim(0, sampling_rate / 2)
            ax.set_xlabel("Frequency (Hz)", fontsize=10)
            ax.set_ylabel("Magnitude (log scale)", fontsize=10)
            ax.set_title(
                f"Spectral Evolution — {bearing_id.replace('_', ' ').title()} (X-axis)\n"
                f"{dataset_cfg.get('display_name', dataset_id)} — {plotted} checkpoints overlaid",
                fontsize=11,
            )

            # Deduplicate legend by reason
            handles, labels = ax.get_legend_handles_labels()
            seen = set()
            unique_handles, unique_labels = [], []
            for h, l in zip(handles, labels):
                key = l.split(" (")[0]
                if key not in seen:
                    seen.add(key)
                    unique_handles.append(h)
                    unique_labels.append(l)
            ax.legend(unique_handles, unique_labels, fontsize=7, loc="upper right", ncol=2)
            ax.grid(True, alpha=0.3, which="both")

            plt.tight_layout()
            out_dir = figures_dir / "spectral_checkpoints"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{dataset_id}_{bearing_id}_spectral_evolution.png"
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved: {out_path.name}")


def plot_band_energy_evolution(
    health_df: pd.DataFrame,
    checkpoint_df: pd.DataFrame,
    config: Dict[str, Any],
    figures_dir: Path,
    logger: logging.Logger,
) -> None:
    """Plots spectral band energy evolution across checkpoints, per dataset.

    Args:
        health_df: Full health indicators DataFrame (for timestamp reference).
        checkpoint_df: Checkpoint selection DataFrame, enriched with an
            `absolute_path` column.
        config: Full project configuration dictionary.
        figures_dir: Root figures output directory.
        logger: Pipeline logger.
    """
    sa_cfg = config["signal_analysis"]
    sampling_rate = float(sa_cfg["sampling_rate_hz"])
    freq_bands = config["feature_engineering"]["frequency_bands"]
    band_labels = [f"{b[0]}-{b[1]} Hz" for b in freq_bands]
    band_colors = ["#1565C0", "#2E7D32", "#6A1B9A"]

    for dataset_id in get_all_dataset_ids(config):
        dataset_cfg = get_dataset_config(config, dataset_id)
        dataset_checkpoints = checkpoint_df[checkpoint_df["dataset_id"] == dataset_id]
        if dataset_checkpoints.empty:
            continue

        target_channels = _rtf_bearing_x_channels(config, dataset_id)
        if not target_channels:
            continue
        # Plot the first RTF bearing's X-axis channel (matches the original
        # single-bearing behaviour, generalized per dataset).
        bearing_id, target_channel = sorted(target_channels.items())[0]

        logger.info(f"[{dataset_id}] Generating band energy evolution plot ({bearing_id})...")

        sorted_checkpoints = dataset_checkpoints.sort_values("timestamp")
        timestamps = []
        band_energies_over_time = [[] for _ in freq_bands]

        expected_channels = dataset_cfg["expected_channels_per_file"]
        for _, cp_row in sorted_checkpoints.iterrows():
            if pd.isna(cp_row.get("absolute_path")):
                continue
            file_path = Path(cp_row["absolute_path"])
            data, status = load_signal_file(
                file_path, expected_channels=expected_channels, logger=logger
            )
            if data is None:
                continue

            waveform = data[:, target_channel - 1]
            result = analyse_checkpoint_spectrum(
                waveform,
                sampling_rate,
                freq_bands,
            )
            timestamps.append(pd.Timestamp(cp_row["timestamp"]))
            for i, energy in enumerate(result["band_energies"]):
                band_energies_over_time[i].append(energy)

        if not timestamps:
            logger.warning(f"[{dataset_id}] No checkpoint data for band energy plot")
            continue

        fig, ax = plt.subplots(figsize=(12, 5))
        for i, (energies, label, color) in enumerate(
            zip(band_energies_over_time, band_labels, band_colors)
        ):
            ax.semilogy(timestamps, energies, marker="o", markersize=4,
                        label=label, color=color, linewidth=1.2)

        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Band Energy (log scale)")
        ax.set_title(
            f"Spectral Band Energy Evolution — {bearing_id.replace('_', ' ').title()} X-axis\n"
            f"{dataset_cfg.get('display_name', dataset_id)} — Checkpoint Files Only",
            fontsize=11,
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        plt.tight_layout()

        out_dir = figures_dir / "spectral_checkpoints"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{dataset_id}_band_energy_evolution_{bearing_id}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {out_path.name}")
