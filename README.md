# Bearing RUL Prediction (NASA IMS)

Predicts the Remaining Useful Life of rolling-element bearings from raw vibration signals and converts those predictions into auditable, risk-tiered maintenance recommendations.

---

## Results

Measured on the **NASA IMS Bearing Dataset, Sets 1–3 combined** (9,463 raw files, 46,476 channel signals, 951,828,480 samples inspected; 0 missing, 0 corrupted).

**Validation method:** Leave-One-Bearing-Out (LOBO) cross-validation, 4 folds. Those 4 run-to-failure units come from only **3 independent rig experiments** (Set 1 contributes two bearings from the same run), so no classical confidence interval is placed on any number below — these are methodology-validation results, not fleet-calibrated accuracy.

| Area | Result |
|---|---|
| Raw files processed | 9,463 (set1: 2,156 · set2: 984 · set3: 6,323), 0 missing, 0 corrupted |
| Signals / samples inspected | 46,476 signals / 951,828,480 samples |
| Features admitted | 14 of 17 candidates (3 excluded as redundant) |
| Cross-validation | Leave-One-Bearing-Out, 4 folds / 3 independent experiments |
| Leakage checks | 9/9 passed |
| Models evaluated | mean, median, linear, Random Forest, XGBoost |
| Mean MAE | XGBoost 12,589 min vs. mean-baseline 16,539 min (~24% lower) |
| Near-failure MAE (RUL ≤ 1,440 min), averaged over the 4 folds | XGBoost ≈1.4 days vs. mean-baseline ≈15.6 days |
| Model frozen for production | `mean` baseline — no flexible model currently clears the promotion gate (see Key Findings) |
| Maintenance recommendations generated | 11,619, from the frozen baseline model |
| Automated tests | 273 passing |
| Dashboard | 5 persona views (Executive, Maintenance Manager, Reliability Engineer, ML Engineer, Auditor) |

---

## Pipeline

1. **Data Ingestion** (`src/data_ingestion`) — scans the raw NASA IMS files and extracts metadata into `data/processed/metadata.parquet`.
2. **Data Validation** (`src/data_validation`) — audits file integrity, duplicates, NaNs, and chronology; writes `reports/quality_report.md`.
3. **Signal Analysis** (`src/signal_analysis`) — computes health indicators (RMS, kurtosis, FFT/envelope spectra) and feature gates.
4. **Feature Engineering** (`src/feature_engineering`) — builds the 14-feature store (`data/processed/feature_store.parquet`) from the admitted candidates.
5. **RUL Target & Leakage** (`src/rul_target`) — labels RUL, builds the LOBO fold assignments, runs the 9 leakage checks.
6. **Model Development** (`src/model_dev`) — trains and evaluates the baselines, Random Forest and XGBoost under LOBO CV, and applies the promotion gate.
7. **Decision Engine** (`src/decision_engine`) — converts the frozen model's predictions into risk-tiered maintenance recommendations.
8. **Dashboard** (`src/dashboard`) — Streamlit app exposing 5 persona views over the frozen artifacts.

---

## Key findings

**FFT truncation bug, found and fixed.** The original `compute_fft_spectrum` truncated every 20,480-sample acquisition to its first 2,048 samples before the FFT — using only 10% of each signal and inflating the frequency-bin width tenfold (9.77 Hz instead of 0.977 Hz). Fixed to compute the FFT over the full waveform. (`ENGINEERING_DECISIONS.md`, ED-002; `config/config.yaml:451`)

**NASA's own archive documentation understates Set 3 by ~2 weeks.** The bundled NASA IMS README states Set 3 runs 2004-03-04 to 2004-04-04, but the downloaded files run continuously to 2004-04-18. Bearing_3's actual degradation-to-failure signature only appears in that undocumented two-week window — at the README's stated cutoff the signal is still indistinguishable from baseline. (`ENGINEERING_DECISIONS.md`, ED-005)

**XGBoost measurably outperforms the baseline, but isn't promoted — and isn't the model making recommendations.** Under 4-fold LOBO CV, XGBoost's mean MAE (12,589 min) is ~24% lower than the `mean` baseline (16,539 min), and its near-failure MAE (RUL ≤ 1,440 min) averages ≈1.4 days versus the baseline's ≈15.6 days across the 4 folds. However, it does not beat both naive baselines on *every* fold, and with a single training bearing per fold the overfitting policy flags flexible models by construction — so the architecture's promotion gate keeps the conservative `mean` baseline frozen for production. **All 11,619 maintenance recommendations are generated from that frozen baseline model, not from XGBoost.** (`reports/evaluation_report.md`; `reports/model_selection_report.md`; `ENGINEERING_DECISIONS.md`, ED-006; `reports/decision_report.md`)

---

## Limitations

- Only 3 independent rig experiments back the 4 LOBO folds — results are methodology demonstrations, not statistically validated fleet accuracy.
- No flexible model (Random Forest, XGBoost) currently clears promotion, regardless of average MAE advantage, under the "beat every fold" + single-training-bearing overfitting rules.
- Risk-tier thresholds in `config/config.yaml` are engineering placeholders pending validation by a qualified reliability engineer before any operational use.
- A known, non-blocking Phase 5 contract gap: the `is_censored` column is declared in the Phase 5 contract but dropped from the physical training dataset (documented in `reports/PHASE_6_IMPLEMENTATION_REPORT.md`).

---

## Reproduce

Requires **Python ≥ 3.10**.

```bash
pip install -r requirements.txt
```

Place the raw NASA IMS files per [data/README.md](data/README.md), then run the test suite:

```bash
pytest tests/        # 273 passing
```

Run the pipeline phases in order (each consumes the previous phase's frozen artifacts):

```bash
python -m src.data_ingestion.metadata_extractor
python -m src.data_validation.validator
python -m src.signal_analysis.main
python -m src.feature_engineering.main
python -m src.rul_target.main
python -m src.model_dev.main
python -m src.decision_engine.main
```

Launch the dashboard:

```bash
streamlit run src/dashboard/app.py
```

## Repository structure

```text
.
├── config/
│   └── config.yaml                # Single source of truth for all parameters
├── src/
│   ├── data_ingestion/            # Phase 1: metadata extraction
│   ├── data_validation/           # Phase 2: integrity + quality audit
│   ├── signal_analysis/           # Phase 3: health indicators, spectral gating
│   ├── feature_engineering/       # Phase 4: time/frequency/envelope/trend features
│   ├── rul_target/                # Phase 5: RUL labels, LOBO splits, leakage checks
│   ├── model_dev/                 # Phase 6: baselines, candidates, selection
│   ├── decision_engine/           # Phase 7: risk -> action -> priority
│   └── dashboard/                 # Phase 8: Streamlit multi-persona UI
├── data/
│   ├── raw/                       # Not committed — see data/README.md
│   ├── processed/                 # Pipeline output (parquet/csv/duckdb caches)
│   └── external/
├── models/                        # Serialized model artifacts per experiment (not committed)
├── reports/                       # Phase contracts, manifests, implementation reports
├── tests/                         # 273 automated tests
└── docs/                          # Setup, coding standards, contribution guide
```

## Full report

[reports/RUL_project_report.pdf](reports/RUL_project_report.pdf) — a 5-page summary of the problem, architecture, key findings and limitations above.

## Author

Priyanshi Bhadauria
