# DASHBOARD INFORMATION ARCHITECTURE
(Pre-Phase 8 Dashboard Design)

## 1. Executive Summary

This document defines the Information Architecture (IA) for the Bearing Remaining-Useful-Life Prediction & Maintenance Decision System dashboard. The IA dictates what information is presented, to whom, in what hierarchy, and for what decisions. This architecture is designed purely to structure information derived from Phases 1 through 7, ensuring it aligns with the strict governance, statistical limitations (n=2 caveat), and categorical confidence rules established by the engineering pipeline.

No visual design, charts, layouts, or implementation technologies are defined in this document.

---

## 2. User Personas

### 2.1. Executive
- **Primary goals:** Understand fleet risk profile, ensure business continuity, optimize maintenance budgets.
- **Questions they need answered:** Are any critical assets at risk of imminent failure? What is the upcoming maintenance workload? What is the business impact?
- **Information required:** Overall fleet health summary, operational risk distribution, high-level maintenance recommendations.
- **Information NOT required:** Sensor anomalies, spectral bands, machine learning metrics (MAE, R²), file lineage.
- **Drill-down needs:** Minimal; only needs to see which facility/asset is driving risk.
- **Decision frequency:** Weekly / Monthly / Ad-hoc during crises.

### 2.2. Maintenance Manager
- **Primary goals:** Effectively allocate resources, schedule downtime, prevent unplanned outages.
- **Questions they need answered:** Which bearings need maintenance this week? What is the priority? How confident is the system?
- **Information required:** Maintenance priorities (1-5), risk categories (CRITICAL, HIGH, MEDIUM, LOW), recommended actions, scheduling horizons.
- **Information NOT required:** Signal analysis reports, feature importance, model hyper-parameters.
- **Drill-down needs:** Moderate; needs to drill down from a fleet list to a specific bearing's recommended action and risk category.
- **Decision frequency:** Daily / Weekly.

### 2.3. Reliability Engineer
- **Primary goals:** Understand the physical degradation of assets, diagnose failure modes, validate system warnings.
- **Questions they need answered:** Is this a bearing defect or a sensor artefact? What stage of degradation is this? Is it impulsive or broadband?
- **Information required:** RMS evolution, Kurtosis behaviour (non-monotonicity), Crest factor, spectral band energies (0-1k, 1k-5k, 5k-10k), Bearing 1 sensor drift artefact.
- **Information NOT required:** High-level budget summaries, model evaluation fold splits.
- **Drill-down needs:** Deep; needs to navigate from a bearing alert down into the time-domain and frequency-domain physical markers over time.
- **Decision frequency:** Daily / On-alert.

### 2.4. Machine Learning Engineer
- **Primary goals:** Monitor model performance, ensure feature integrity, track data drift (if applicable).
- **Questions they need answered:** Is the fallback baseline (median) still active? What features are driving predictions? Is there data leakage?
- **Information required:** Model type (median), MAE, Overfitting ratios, SHAP findings, permutation importance, LOBO cross-validation metrics.
- **Information NOT required:** Daily technician scheduling, high-level operational KPIs.
- **Drill-down needs:** Deep; needs to see per-fold performance, raw feature values, and local SHAP explanations for specific predictions.
- **Decision frequency:** Monthly / Post-deployment review.

### 2.5. Auditor / Governance
- **Primary goals:** Ensure compliance, trace decision provenance, verify safety protocols and methodology.
- **Questions they need answered:** Are the statistical limitations (n=2) being displayed? Are decisions traceable to frozen contracts? Is there leakage?
- **Information required:** Manifest hashes, dataset contracts, leakage report status, Phase 6 model fallback justification, n=2 caveat visibility.
- **Information NOT required:** Maintenance technician schedules, real-time vibration streaming.
- **Drill-down needs:** High; needs to trace a final decision all the way back to the originating pipeline artifact and contract fingerprint.
- **Decision frequency:** Quarterly / Annual audits.

---

## 3. Information Hierarchy

The insights discovered in Phase 1-7 are strictly organized into the following mutually exclusive tiers:

### Tier 1: Executive KPIs
- Fleet Risk Profile (Count of Bearings by Risk Category)
- Pending Maintenance Workload (Count of Recommended Actions)

### Tier 2: Operational Monitoring
- System Confidence Limits (Categorical: LOW)
- Current Risk Categories by Asset

### Tier 3: Maintenance Decisions
- Recommended Actions (e.g., SCHEDULE_MAINTENANCE)
- Priority Scores (1 to 5)
- Operational Risk Thresholds (Placeholder values)

### Tier 4: Engineering Diagnostics
- Primary Degradation Marker (RMS)
- Early-Stage Sensitivity Marker (Kurtosis)
- Complementary Early Indicator (Crest Factor)
- Spectral Band Behaviour (1,000-5,000 Hz concentration)
- Known Artefacts (Bearing 1 DC offset)

### Tier 5: Machine Learning Diagnostics
- Baseline Model Bias (Median fallback, early-warning bias)
- LOBO CV Performance Metrics (MAE)
- Feature Importance Alignment (td_rms as top feature)

### Tier 6: Governance & Audit
- Small Sample Size Caveat (n=2 limitation)
- Decision Traceability (Experiment ID, Config Fingerprint)
- Leakage Prevention Validation

---

## 4. KPI Inventory

| KPI Name | Description | Source Phase | Source Artifact | Business/Engineering Value | Dashboard Priority | Intended Audience |
|---|---|---|---|---|---|---|
| **Fleet Risk Status** | Aggregate count of bearings in CRITICAL or HIGH risk. | Phase 7 | `decision_report.md` | Business: Prevents catastrophic failure. | Critical | Executive, Maintenance Manager |
| **Pending Maintenance Actions** | Total number of generated SCHEDULE_MAINTENANCE actions. | Phase 7 | `decision_report.md` | Business: Drives daily resource allocation. | Critical | Maintenance Manager |
| **System Confidence Level** | Categorical confidence (HIGH/MEDIUM/LOW). Currently restricted to LOW. | Phase 7 | `decision_report.md` | Eng/Gov: Prevents false precision reliance. | Critical | All Personas |
| **Statistical Limitation (n=2)** | Permanent caveat indicating results are methodology validation. | Phase 5 | `split_summary.md` | Governance: Protects liability. | Critical | Executive, Auditor |
| **Primary Health Indicator (RMS)** | Absolute RMS magnitude tracking late-stage wear. | Phase 3 | `engineering_findings.md` | Eng: Ground-truth physical wear metric. | Important | Reliability Engineer |
| **Early Warning Indicator (Kurtosis)** | Impulsive impact metric tracking early defect onset. | Phase 3 | `engineering_findings.md` | Eng: Maximizes lead time for repairs. | Important | Reliability Engineer |
| **Model Prediction Error (MAE)** | Mean Absolute Error from LOBO CV folds. | Phase 6 | `evaluation_report.md` | Eng/ML: Validates system accuracy limits. | Optional | ML Engineer |

---

## 5. Information Requirements by Persona

- **Executive:** Tier 1 and Tier 6 (specifically the n=2 limitation to govern business risk).
- **Maintenance Manager:** Tier 1, Tier 2, and Tier 3.
- **Reliability Engineer:** Tier 2, Tier 3, and Tier 4.
- **Machine Learning Engineer:** Tier 4 and Tier 5.
- **Auditor / Governance:** Tier 6, supported by tracing down to Tier 3.

---

## 6. User Journey

1. **What should a user learn first? (Landing Page)**
   The user must immediately learn the current Fleet Risk Status, the Pending Maintenance Actions, and crucially, the System Confidence Level and the Statistical Limitation (n=2) caveat.
2. **What should they investigate next? (Level 1 Drill-down)**
   They should investigate the specific bearings that triggered the risk or maintenance recommendations, viewing Priority Scores and Recommended Actions (Tier 2 and Tier 3).
3. **What information should only appear after drill-down? (Level 2+ Drill-down)**
   Engineering Diagnostics (RMS trends, Kurtosis spikes, Frequency band energies) should only appear when investigating a specific bearing. Machine Learning diagnostics (Feature Importance, MAE) and deep Governance artifacts (fingerprints, manifests) should be heavily nested or on completely separate specialized views.
4. **What information should never appear on the landing page?**
   SHAP values, permutation importance, contract hashes, sensor drift artefacts (Bearing 1 DC offset), and model baseline fallbacks.

---

## 7. Drill-down Hierarchy

**Landing Page** *(Fleet overview, Executive KPIs, Mandatory Governance Caveats)*
↓
**Operations Summary** *(Asset-level risk distributions, Priority listings)*
↓
**Maintenance Detail** *(Specific asset recommendations, Threshold values, Action schedules)*
↓
**Bearing Detail** *(Historical risk trajectory of a single bearing)*
↓
**Signal Analysis** *(Engineering diagnostics: RMS, Kurtosis, Crest Factor, Spectra)*
↓
**Model Diagnostics** *(ML diagnostics: feature importance, baseline bias, MAE)*
↓
**Governance** *(Artifact hashes, LOBO leakage validations, Contract verifications)*

**Explanation of Hierarchy:**
The hierarchy obeys the "progressive disclosure" principle. It moves from high-impact business decisions (top) to deep technical and physical causality (bottom). It ensures non-technical stakeholders are not overwhelmed by signal processing artifacts, while providing complete traceability for engineers and auditors who require deep physical and methodological verification.

---

## 8. Traceability Matrix

| Information Element | Originating Phase | Originating Report | Originating Artifact | Confidence | Update Freq | Downstream Dependencies |
|---|---|---|---|---|---|---|
| Maintenance Recommendation | Phase 7 | `decision_report.md` | `recommendations.parquet` | Categorical | On Inference | Operations Scheduling |
| Risk Category | Phase 7 | `decision_report.md` | `recommendations.parquet` | Categorical | On Inference | Maintenance Action Rule |
| n=2 Caveat / Methodology Flag | Phase 5 | `split_summary.md` | `decision_report.md` | High (Factual) | Static (until new data) | Dashboard Display Rule |
| RMS Magnitude | Phase 3 | `engineering_findings.md` | `feature_store.parquet` | High | On Inference | ML Model |
| Kurtosis Magnitude | Phase 3 | `engineering_findings.md` | `feature_store.parquet` | High | On Inference | Engineering Display |
| Feature Importance (SHAP) | Phase 6 | `feature_importance.md` | Model artifacts | Medium (n=2) | On Train | ML Diagnostics |
| MAE Metric | Phase 6 | `evaluation_report.md` | Model artifacts | Medium (n=2) | On Train | Model Selection |
| Experiment Fingerprints | Phase 7 | `decision_validation_report.md`| `decision_manifest.json` | High | On Train/Infer | Auditor Tracing |

---

## 9. Dashboard Content Priority

1. **Mandatory/Critical:** Fleet Risk, Recommended Actions, n=2 Caveat, Categorical Confidence (LOW).
2. **Important:** Priority Scores, RMS Trends, Kurtosis Spikes, Decision Traceability.
3. **Optional (Deep Drill-down):** Spectral Bands, SHAP values, MAE, Contract Hashes.

---

## 10. Out of Scope

The following items are strictly OUT OF SCOPE for this document and must be defined in Phase 8:
- Dashboard layouts, grids, or spatial arrangements.
- Visual design elements, UI aesthetics, themes, typography, or colors.
- Specific charts, graphs, or visual representations of data.
- Interactive filters, date pickers, or navigation controls.
- Implementation technologies (e.g., React, Streamlit, Power BI, Plotly).

---

## 11. Assumptions

- The dashboard will primarily read from the frozen artifacts (e.g., `recommendations.parquet`, `feature_store.parquet`, `decision_report.md`).
- The statistical limitations (n=2) and model fallback (median) will remain true until a new dataset is acquired, necessitating their permanent visibility on the dashboard.
- The Risk Thresholds provided by Phase 7 are placeholders and will be treated as configurable parameters rather than absolute truths.

---

## 12. Recommendations for Phase 8

- **Design for Conservatism:** Ensure the UI reflects the conservative nature of the Phase 6 median model (early-warning bias).
- **Mandatory Caveat Placement:** Phase 8 must determine a persistent UI location for the "n=2 / Methodology Validation Only" caveat where it cannot be missed or scrolled past.
- **Progressive Disclosure:** Phase 8 layouts must respect the Drill-down Hierarchy to avoid cognitive overload for Executive and Maintenance Manager personas.

---

## Final Review Statement

**APPROVE INFORMATION ARCHITECTURE**

**Justification:** This Information Architecture successfully categorizes all project insights, strictly defines the information boundaries required for five distinct user personas, establishes a rigorous 6-tier information hierarchy, maps the user journey through progressive disclosure, and ensures complete traceability back to the frozen Phase 1-7 artifacts. No UI design, charts, or visual technologies have been proposed, remaining strictly within the boundaries of an information design exercise. It enforces the critical governance and statistical limitations required by the engineering pipeline.
