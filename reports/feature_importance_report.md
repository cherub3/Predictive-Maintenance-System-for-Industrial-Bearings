# Phase 6 — Feature Importance Report

> **SHAP governance (architecture §11a):** SHAP is used ONLY for model interpretation. It is computed after model selection and is never used to choose between candidate models. Observations below are for human review only.

---

## random_forest

### Permutation Importance (primary, out-of-sample)

| Feature | Importance (mean MAE increase) |
|---------|-------------------------------:|
| fd_band_energy_1000_5000 | 3125.1531 |
| td_rms | 101.5044 |
| fd_band_energy_0_1000 | 37.2462 |
| env_bsf_energy | 27.1035 |
| env_bpfi_energy | 1.5439 |
| fd_spectral_entropy | 0.4551 |
| td_crest_factor | -2.8605 |
| fd_spectral_centroid | -6.6051 |
| trend_rms_slope | -6.6195 |
| fd_band_energy_5000_10000 | -15.2259 |

### Native Model Importance (cross-check)

| Feature | Native Importance |
|---------|-------------------:|
| fd_band_energy_1000_5000 | 0.7736 |
| td_kurtosis | 0.0956 |
| fd_band_energy_0_1000 | 0.0493 |
| fd_spectral_entropy | 0.0204 |
| fd_band_energy_5000_10000 | 0.0161 |
| td_rms | 0.0143 |
| fd_spectral_centroid | 0.0125 |
| trend_kurtosis_slope | 0.0052 |
| trend_rms_variance | 0.0045 |
| env_bsf_energy | 0.0025 |

**Caution:** top-5 permutation vs. native importance overlap is only 2/5 features — native gain may be biased toward high-cardinality features (architecture §11).

### SHAP Global Ranking (mean |SHAP|)

| Feature | Mean |SHAP| |
|---------|------------:|
| fd_band_energy_1000_5000 | 17547.8495 |
| fd_band_energy_0_1000 | 3378.7155 |
| td_kurtosis | 2881.3242 |
| fd_spectral_entropy | 1792.0856 |
| fd_band_energy_5000_10000 | 1223.6952 |
| td_rms | 1180.8064 |
| fd_spectral_centroid | 563.8168 |
| trend_kurtosis_slope | 184.9404 |
| trend_rms_variance | 130.9131 |
| env_bsf_energy | 113.0853 |

### SHAP Local Examples (near-failure region)

- **bearing_3** (fold fold_1, channel 5, actual RUL=1432.4 min): fd_band_energy_1000_5000=-23556.72, td_kurtosis=+4518.61, td_rms=-1130.90, td_crest_factor=-546.25, fd_band_energy_5000_10000=+494.78
- **bearing_3** (fold fold_1, channel 5, actual RUL=1422.4 min): fd_band_energy_1000_5000=-26450.67, td_kurtosis=+3837.43, fd_band_energy_5000_10000=+708.02, fd_spectral_centroid=-627.45, td_rms=-390.90
- **bearing_3** (fold fold_1, channel 5, actual RUL=772.4 min): fd_band_energy_1000_5000=-23637.52, td_kurtosis=+4529.53, td_rms=-1111.68, td_crest_factor=-656.36, fd_band_energy_5000_10000=+514.15
- **bearing_3** (fold fold_1, channel 5, actual RUL=762.4 min): fd_band_energy_1000_5000=-23395.79, td_kurtosis=+5076.90, td_rms=-1200.47, td_crest_factor=-656.27, fd_spectral_centroid=-582.98
- **bearing_3** (fold fold_1, channel 5, actual RUL=752.4 min): fd_band_energy_1000_5000=-23560.75, td_kurtosis=+4744.58, td_rms=-1118.90, td_crest_factor=-656.35, fd_spectral_centroid=-521.80

## linear

### Permutation Importance (primary, out-of-sample)

| Feature | Importance (mean MAE increase) |
|---------|-------------------------------:|
| td_rms | 7177.5338 |
| fd_band_energy_1000_5000 | 2288.6113 |
| fd_spectral_entropy | 1405.7958 |
| fd_band_energy_5000_10000 | 1210.1252 |
| fd_spectral_centroid | 291.7439 |
| td_crest_factor | 213.2180 |
| td_kurtosis | 58.6550 |
| trend_rms_slope | 36.1033 |
| fd_band_energy_0_1000 | 35.5662 |
| trend_kurtosis_slope | 5.2971 |

### Native Model Importance (cross-check)

| Feature | Native Importance |
|---------|-------------------:|
| td_rms | 29785.9566 |
| fd_spectral_entropy | 20820.8644 |
| fd_band_energy_1000_5000 | 10306.7634 |
| fd_band_energy_5000_10000 | 9612.5791 |
| fd_spectral_centroid | 5446.3603 |
| env_bpfo_energy | 3995.5424 |
| fd_band_energy_0_1000 | 3923.9591 |
| env_bpfi_energy | 2165.6613 |
| td_crest_factor | 2152.8495 |
| td_kurtosis | 1678.9294 |

## xgboost

### Permutation Importance (primary, out-of-sample)

| Feature | Importance (mean MAE increase) |
|---------|-------------------------------:|
| fd_band_energy_1000_5000 | 2172.4690 |
| trend_rms_variance | 111.9509 |
| fd_spectral_entropy | 98.2464 |
| td_kurtosis | 80.4751 |
| fd_spectral_centroid | 57.9720 |
| td_rms | 49.5963 |
| trend_rms_slope | 33.8311 |
| env_bsf_energy | 24.7186 |
| fd_band_energy_0_1000 | 4.2214 |
| env_bpfi_energy | -9.2921 |

### Native Model Importance (cross-check)

| Feature | Native Importance |
|---------|-------------------:|
| fd_band_energy_1000_5000 | 0.4310 |
| td_rms | 0.1608 |
| fd_band_energy_0_1000 | 0.1004 |
| fd_spectral_entropy | 0.0757 |
| td_kurtosis | 0.0676 |
| fd_spectral_centroid | 0.0343 |
| fd_band_energy_5000_10000 | 0.0324 |
| trend_rms_variance | 0.0239 |
| env_bsf_energy | 0.0168 |
| td_crest_factor | 0.0153 |

### SHAP Global Ranking (mean |SHAP|)

| Feature | Mean |SHAP| |
|---------|------------:|
| fd_band_energy_1000_5000 | 15532.5039 |
| fd_spectral_entropy | 2877.4331 |
| td_kurtosis | 2639.4534 |
| fd_band_energy_0_1000 | 2555.9585 |
| td_rms | 2482.8325 |
| fd_band_energy_5000_10000 | 1421.8739 |
| fd_spectral_centroid | 859.4927 |
| env_bsf_energy | 537.5914 |
| trend_rms_slope | 452.6990 |
| trend_rms_variance | 411.8110 |

### SHAP Local Examples (near-failure region)

- **bearing_3** (fold fold_1, channel 5, actual RUL=1432.4 min): fd_band_energy_1000_5000=-21964.43, td_kurtosis=+7861.59, td_rms=-6381.14, fd_spectral_entropy=-3020.37, fd_band_energy_5000_10000=+2059.38
- **bearing_3** (fold fold_1, channel 5, actual RUL=1422.4 min): fd_band_energy_1000_5000=-20687.09, td_rms=-6297.32, td_kurtosis=+4950.55, fd_spectral_entropy=-2205.78, fd_band_energy_5000_10000=+1767.52
- **bearing_3** (fold fold_1, channel 5, actual RUL=772.4 min): fd_band_energy_1000_5000=-21961.04, td_kurtosis=+8091.09, td_rms=-6466.89, fd_spectral_entropy=-3126.95, fd_band_energy_5000_10000=+1963.64
- **bearing_3** (fold fold_1, channel 5, actual RUL=762.4 min): fd_band_energy_1000_5000=-21766.56, td_kurtosis=+8316.72, td_rms=-6334.57, fd_spectral_entropy=-3092.20, fd_band_energy_5000_10000=+2001.93
- **bearing_3** (fold fold_1, channel 5, actual RUL=752.4 min): fd_band_energy_1000_5000=-21959.75, td_kurtosis=+7966.77, td_rms=-6466.77, fd_spectral_entropy=-3103.76, fd_band_energy_5000_10000=+2001.93

---

**n=2 caveat:** importance rankings are derived from two folds and are indicative of mechanism, not statistically stable rankings (architecture §11).

_Generated by Phase 6 pipeline. Do not edit manually._