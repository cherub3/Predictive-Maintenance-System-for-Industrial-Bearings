# Phase 3: Signal Understanding & Exploratory Signal Analysis
## Signal Analysis Report

**Generated:** 2026-07-31 18:11:47 UTC
**Dataset:** NASA IMS Bearing Dataset (set1, set2, set3 combined)
**Pipeline elapsed time:** 312.4 seconds

---

## 1. Executive Summary

Phase 3 performed Tier 1 time-domain health indicator computation across all 9463
acquisition files (46,476 channel-level records), and Tier 2 spectral checkpoint
analysis on 60 selected files. Gate decisions for 4
conditional Phase 4 features were resolved through quantitative statistical analysis.

---

## 2. Analysis Methodology

### 2.1 Tier 1 — Time-Domain Health Indicators

All 9463 acquisition files were processed to compute four scalar health indicators
per channel:

| Indicator | Formula | Physical Meaning |
|---|---|---|
| RMS | sqrt(mean(x^2)) | Signal energy; broad bearing wear proxy |
| Kurtosis | E[(x-mu)^4] / sigma^4 | Impulsiveness; early-stage defect sensitivity |
| Crest Factor | peak / RMS | Localised impact detection |
| Skewness | E[(x-mu)^3] / sigma^3 | Distributional asymmetry |

- Sampling rate: 20,000 Hz
- Samples per file: 20,480
- Window duration: 1.024 seconds
- Kurtosis definition: Fisher (excess), unbiased estimator

### 2.2 Tier 2 — Spectral Checkpoint Analysis

60 files were selected for FFT spectral analysis:

| Selection Reason | Count |
|---|---|
| mid_stage | 18 |
| early_stage | 15 |
| late_stage | 15 |
| kurtosis_peak | 12 |

FFT parameters:
- Window function: Hann
- FFT size: 2048
- Frequency resolution: 9.77 Hz

---

## 3. Health Indicator Results

### 3.1 Data Quality

- Total rows processed: 46,476
- Successfully loaded: 46,476 (100.0%)
- Failed to load: 0

### 3.2 Per-Dataset, Per-Bearing Summary Statistics (X-axis channels)

#### NASA IMS Set 1 (`set1`)

**Bearing 1:**
- Records: 2156
- Mean RMS: 0.152221 g
- Max RMS: 0.174975 g
- Mean Kurtosis: 1.00
- Max Kurtosis: 1.70
- Max Crest Factor: 7.12

**Bearing 2:**
- Records: 2156
- Mean RMS: 0.149490 g
- Max RMS: 0.211860 g
- Mean Kurtosis: 0.26
- Max Kurtosis: 0.72
- Max Crest Factor: 6.41

**Bearing 3:**
- Records: 2156
- Mean RMS: 0.161843 g
- Max RMS: 0.593610 g
- Mean Kurtosis: 1.98
- Max Kurtosis: 71.58
- Max Crest Factor: 25.95

**Bearing 4:**
- Records: 2156
- Mean RMS: 0.146809 g
- Max RMS: 0.237261 g
- Mean Kurtosis: 2.10
- Max Kurtosis: 71.57
- Max Crest Factor: 19.45
#### NASA IMS Set 2 (`set2`)

**Bearing 1:**
- Records: 984
- Mean RMS: 0.106111 g
- Max RMS: 0.725001 g
- Mean Kurtosis: 0.83
- Max Kurtosis: 14.11
- Max Crest Factor: 9.33

**Bearing 2:**
- Records: 984
- Mean RMS: 0.098753 g
- Max RMS: 0.218294 g
- Mean Kurtosis: 0.15
- Max Kurtosis: 2.45
- Max Crest Factor: 7.93

**Bearing 3:**
- Records: 984
- Mean RMS: 0.104657 g
- Max RMS: 0.187405 g
- Mean Kurtosis: 1.33
- Max Kurtosis: 3.70
- Max Crest Factor: 12.75

**Bearing 4:**
- Records: 984
- Mean RMS: 0.060269 g
- Max RMS: 0.148351 g
- Mean Kurtosis: 0.14
- Max Kurtosis: 1.65
- Max Crest Factor: 9.24
#### NASA IMS Set 3 (`set3`)

**Bearing 1:**
- Records: 6323
- Mean RMS: 0.074179 g
- Max RMS: 0.155143 g
- Mean Kurtosis: 0.43
- Max Kurtosis: 2.06
- Max Crest Factor: 10.26

**Bearing 2:**
- Records: 6323
- Mean RMS: 0.081054 g
- Max RMS: 0.272873 g
- Mean Kurtosis: 0.17
- Max Kurtosis: 0.94
- Max Crest Factor: 9.30

**Bearing 3:**
- Records: 6323
- Mean RMS: 0.072400 g
- Max RMS: 0.758765 g
- Mean Kurtosis: 0.15
- Max Kurtosis: 16.75
- Max Crest Factor: 10.61

**Bearing 4:**
- Records: 6323
- Mean RMS: 0.055733 g
- Max RMS: 0.311968 g
- Mean Kurtosis: 0.19
- Max Kurtosis: 0.88
- Max Crest Factor: 8.69

---

## 4. Bearing Geometry Assessment

- **Bearing type:** Rexnord ZA-2115 Double-Row
- **Geometry verified:** True
- **Shaft speed:** 2000 RPM

Defect frequencies were computed from verified geometry parameters and are available for Phase 4 envelope analysis.

---

## 5. Feature Gate Decisions

These decisions are stored in `reports/phase3_gating_decisions.yaml` for Phase 4 consumption.

| Gate | Include | Justification Summary |
|---|---|---|
| skewness | False | Mean absolute skewness across all channels = 0.0525; threshold = 0.5. Gate set t... |
| rolling_variance | True | Pearson r(rolling_slope, rolling_variance) = 0.3564; threshold = 0.8. Gate set t... |
| envelope_features | True | Bearing geometry verified. Envelope features enabled pending signal characterist... |
| xy_energy_ratio | False | X/Y RMS ratio CV = 0.0454 across all bearings and timestamps; threshold = 0.15. ... |

---

## 6. Generated Artifacts

### 6.1 Data Artifacts
- `data/processed/signal_health_indicators.parquet` — 46,476 rows x 13 columns
- `data/processed/checkpoint_selection.csv` — 60 checkpoint records

### 6.2 Reports
- `reports/engineering_findings.md` — Engineering observations and interpretations
- `reports/signal_analysis_report.md` — This document
- `reports/phase3_gating_decisions.yaml` — Machine-readable Phase 4 gate contract

### 6.3 Figures
- `reports\figures\channel_comparison\bearing_3_crest_factor_xy.png`
- `reports\figures\channel_comparison\bearing_3_kurtosis_xy.png`
- `reports\figures\channel_comparison\bearing_3_rms_xy.png`
- `reports\figures\channel_comparison\bearing_4_crest_factor_xy.png`
- `reports\figures\channel_comparison\bearing_4_kurtosis_xy.png`
- `reports\figures\channel_comparison\bearing_4_rms_xy.png`
- `reports\figures\channel_comparison\set1_bearing_3_crest_factor_xy.png`
- `reports\figures\channel_comparison\set1_bearing_3_kurtosis_xy.png`
- `reports\figures\channel_comparison\set1_bearing_3_rms_xy.png`
- `reports\figures\channel_comparison\set1_bearing_4_crest_factor_xy.png`
- `reports\figures\channel_comparison\set1_bearing_4_kurtosis_xy.png`
- `reports\figures\channel_comparison\set1_bearing_4_rms_xy.png`
- `reports\figures\health_trends\crest_factor_all_bearings.png`
- `reports\figures\health_trends\kurtosis_all_bearings.png`
- `reports\figures\health_trends\rms_all_bearings.png`
- `reports\figures\health_trends\set1_crest_factor_all_bearings.png`
- `reports\figures\health_trends\set1_kurtosis_all_bearings.png`
- `reports\figures\health_trends\set1_rms_all_bearings.png`
- `reports\figures\health_trends\set2_crest_factor_all_bearings.png`
- `reports\figures\health_trends\set2_kurtosis_all_bearings.png`
- `reports\figures\health_trends\set2_rms_all_bearings.png`
- `reports\figures\health_trends\set3_crest_factor_all_bearings.png`
- `reports\figures\health_trends\set3_kurtosis_all_bearings.png`
- `reports\figures\health_trends\set3_rms_all_bearings.png`
- `reports\figures\spectral_checkpoints\band_energy_evolution_bearing3.png`
- `reports\figures\spectral_checkpoints\bearing_3_spectral_evolution.png`
- `reports\figures\spectral_checkpoints\bearing_4_spectral_evolution.png`
- `reports\figures\spectral_checkpoints\set1_band_energy_evolution_bearing_3.png`
- `reports\figures\spectral_checkpoints\set1_bearing_3_spectral_evolution.png`
- `reports\figures\spectral_checkpoints\set1_bearing_4_spectral_evolution.png`
- `reports\figures\spectral_checkpoints\set2_bearing_1_spectral_evolution.png`
- `reports\figures\spectral_checkpoints\set3_bearing_3_spectral_evolution.png`
- `reports\figures\spectral_checkpoints\set2_band_energy_evolution_bearing_1.png`
- `reports\figures\spectral_checkpoints\set3_band_energy_evolution_bearing_3.png`

---

## 7. Known Limitations

1. Bearing 1 sensor DC offset inflates absolute RMS relative to other bearings.
2. Kurtosis non-monotonicity near end-of-life is a physical phenomenon, not an artefact.
3. Checkpoint spectral analysis covers 60 of 9463 files (~1%).
4. No pre-failure defect onset timestamp is available from the dataset.
5. Spectral entropy as computed here uses the FFT magnitude spectrum; phase information is discarded.

---

*Report generated by Phase 3 Signal Analysis Pipeline*
*Engineering findings: reports/engineering_findings.md*
*Gate contract: reports/phase3_gating_decisions.yaml*
