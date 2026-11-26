"""Phase 3 signal analysis report generator.

Generates reports/signal_analysis_report.md — a comprehensive analysis report
documenting methodology, results, and artifact inventory for Phase 3.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.config import get_all_dataset_ids, get_dataset_config


def generate_signal_analysis_report(
    health_df: pd.DataFrame,
    checkpoint_df: pd.DataFrame,
    gate_contract: Dict[str, Any],
    bearing_geometry_verified: bool,
    config: Dict[str, Any],
    figures_generated: List[str],
    elapsed_seconds: float,
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """Generates the comprehensive signal analysis report.

    Args:
        health_df: Health indicators DataFrame.
        checkpoint_df: Checkpoint selection DataFrame.
        gate_contract: Gating contract dictionary.
        bearing_geometry_verified: Whether geometry was verified.
        config: Full project configuration dictionary.
        figures_generated: List of generated figure filenames.
        elapsed_seconds: Total Phase 3 pipeline execution time.
        output_path: Destination path for the report.
        logger: Pipeline logger.
    """
    ok_rows = health_df[health_df["load_status"] == "OK"]
    total_files = ok_rows["filename"].nunique()
    total_rows = len(ok_rows)
    failed_rows = len(health_df) - total_rows

    sa_cfg = config["signal_analysis"]
    defaults = config["datasets"]["defaults"]
    sampling_rate = int(defaults["sampling_rate_hz"])
    samples_per_file = int(defaults["samples_per_file"])
    shaft_rpm = float(defaults["shaft_speed_rpm"])
    dataset_ids = get_all_dataset_ids(config)

    gate_decisions = gate_contract["gate_decisions"]
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Per-(dataset, bearing) stats — grouping by bearing_id alone would splice
    # together physically different bearings that happen to share a name across
    # datasets (e.g. Set 1's bearing_3 vs. Set 3's bearing_3).
    per_dataset_bearing_sections = []
    for dataset_id in dataset_ids:
        dataset_cfg = get_dataset_config(config, dataset_id)
        bearing_blocks = []
        for bearing_id in sorted(dataset_cfg.get("bearings", {}).keys()):
            bdf = ok_rows[(ok_rows["dataset_id"] == dataset_id) & (ok_rows["bearing_id"] == bearing_id) & (ok_rows["axis"] == "X")]
            if bdf.empty:
                bearing_blocks.append(f"**{bearing_id.replace('_', ' ').title()}:** No data")
                continue
            bearing_blocks.append(f"""**{bearing_id.replace('_', ' ').title()}:**
- Records: {len(bdf)}
- Mean RMS: {bdf['rms'].mean():.6f} g
- Max RMS: {bdf['rms'].max():.6f} g
- Mean Kurtosis: {bdf['kurtosis'].mean():.2f}
- Max Kurtosis: {bdf['kurtosis'].max():.2f}
- Max Crest Factor: {bdf['crest_factor'].max():.2f}""")
        per_dataset_bearing_sections.append(
            f"#### {dataset_cfg.get('display_name', dataset_id)} (`{dataset_id}`)\n\n" + "\n\n".join(bearing_blocks)
        )

    figures_section = "\n".join(f"- `{f}`" for f in figures_generated) if figures_generated else "- No figures generated"

    content = f"""# Phase 3: Signal Understanding & Exploratory Signal Analysis
## Signal Analysis Report

**Generated:** {generated_at}
**Dataset:** NASA IMS Bearing Dataset ({', '.join(dataset_ids)} combined)
**Pipeline elapsed time:** {elapsed_seconds:.1f} seconds

---

## 1. Executive Summary

Phase 3 performed Tier 1 time-domain health indicator computation across all {total_files}
acquisition files ({total_rows:,} channel-level records), and Tier 2 spectral checkpoint
analysis on {len(checkpoint_df)} selected files. Gate decisions for {len(gate_decisions)}
conditional Phase 4 features were resolved through quantitative statistical analysis.

---

## 2. Analysis Methodology

### 2.1 Tier 1 — Time-Domain Health Indicators

All {total_files} acquisition files were processed to compute four scalar health indicators
per channel:

| Indicator | Formula | Physical Meaning |
|---|---|---|
| RMS | sqrt(mean(x^2)) | Signal energy; broad bearing wear proxy |
| Kurtosis | E[(x-mu)^4] / sigma^4 | Impulsiveness; early-stage defect sensitivity |
| Crest Factor | peak / RMS | Localised impact detection |
| Skewness | E[(x-mu)^3] / sigma^3 | Distributional asymmetry |

- Sampling rate: {sampling_rate:,} Hz
- Samples per file: {samples_per_file:,}
- Window duration: {samples_per_file / sampling_rate:.3f} seconds
- Kurtosis definition: Fisher (excess), unbiased estimator

### 2.2 Tier 2 — Spectral Checkpoint Analysis

{len(checkpoint_df)} files were selected for FFT spectral analysis:

| Selection Reason | Count |
|---|---|
{chr(10).join(f'| {r} | {c} |' for r, c in checkpoint_df['selection_reason'].value_counts().items())}

FFT parameters:
- Window function: Hann
- FFT size: {config['feature_engineering'].get('fft_n_fft', 2048)}
- Frequency resolution: {sampling_rate / config['feature_engineering'].get('fft_n_fft', 2048):.2f} Hz

---

## 3. Health Indicator Results

### 3.1 Data Quality

- Total rows processed: {len(health_df):,}
- Successfully loaded: {total_rows:,} ({100*total_rows/len(health_df):.1f}%)
- Failed to load: {failed_rows:,}

### 3.2 Per-Dataset, Per-Bearing Summary Statistics (X-axis channels)

{chr(10).join(per_dataset_bearing_sections)}

---

## 4. Bearing Geometry Assessment

- **Bearing type:** Rexnord ZA-2115 Double-Row
- **Geometry verified:** {bearing_geometry_verified}
- **Shaft speed:** {shaft_rpm:.0f} RPM

{"Defect frequencies were computed from verified geometry parameters and are available for Phase 4 envelope analysis." if bearing_geometry_verified else "Geometry could not be verified. Per Bearing Geometry Safety Rule, envelope features are excluded from Phase 4."}

---

## 5. Feature Gate Decisions

These decisions are stored in `reports/phase3_gating_decisions.yaml` for Phase 4 consumption.

| Gate | Include | Justification Summary |
|---|---|---|
| skewness | {gate_decisions['skewness']['include']} | {gate_decisions['skewness']['justification'][:80]}... |
| rolling_variance | {gate_decisions['rolling_variance']['include']} | {gate_decisions['rolling_variance']['justification'][:80]}... |
| envelope_features | {gate_decisions['envelope_features']['include']} | {gate_decisions['envelope_features']['justification'][:80]}... |
| xy_energy_ratio | {gate_decisions['xy_energy_ratio']['include']} | {gate_decisions['xy_energy_ratio']['justification'][:80]}... |

---

## 6. Generated Artifacts

### 6.1 Data Artifacts
- `data/processed/signal_health_indicators.parquet` — {total_rows:,} rows x 13 columns
- `data/processed/checkpoint_selection.csv` — {len(checkpoint_df)} checkpoint records

### 6.2 Reports
- `reports/engineering_findings.md` — Engineering observations and interpretations
- `reports/signal_analysis_report.md` — This document
- `reports/phase3_gating_decisions.yaml` — Machine-readable Phase 4 gate contract

### 6.3 Figures
{figures_section}

---

## 7. Known Limitations

1. Bearing 1 sensor DC offset inflates absolute RMS relative to other bearings.
2. Kurtosis non-monotonicity near end-of-life is a physical phenomenon, not an artefact.
3. Checkpoint spectral analysis covers {len(checkpoint_df)} of {total_files} files (~{100*len(checkpoint_df)/total_files:.0f}%).
4. No pre-failure defect onset timestamp is available from the dataset.
5. Spectral entropy as computed here uses the FFT magnitude spectrum; phase information is discarded.

---

*Report generated by Phase 3 Signal Analysis Pipeline*
*Engineering findings: reports/engineering_findings.md*
*Gate contract: reports/phase3_gating_decisions.yaml*
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"Signal analysis report written: {output_path}")
