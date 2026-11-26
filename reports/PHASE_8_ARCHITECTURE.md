# Phase 8 — Architecture Design: Dashboard & Decision Support System

## 1. Executive Summary

This document defines the Phase 8 architecture for the Bearing Remaining-Useful-Life (RUL) Prediction & Maintenance Decision System dashboard. It governs how the frozen insights and artifacts from Phases 3–7 are consumed, organized, and presented to support diverse stakeholders (Executives, Maintenance Managers, Reliability Engineers, ML Engineers, and Auditors). The architecture strictly enforces progressive disclosure, engineering traceability, and scientific integrity without proposing visual styling, layout frameworks, or implementation technologies.

## 2. Objectives

- **Decision Support:** Enable rapid, traceable maintenance decisions based on objective engineering evidence.
- **Traceability:** Maintain unbroken provenance from business recommendations down to raw Phase 3 signal characteristics and Phase 7 contract hashes.
- **Scientific Integrity:** Faithfully present frozen evaluation metrics and decision boundaries without reinterpretation, interpolation, or smoothing.
- **Governance:** Permanently enforce the visibility of statistical limitations (e.g., the n=2 methodology caveat) to prevent liability and false precision.

## 3. Users

The dashboard architecture serves five primary personas, inheriting requirements from the Information Architecture:
1. **Executive:** Requires high-level fleet risk and business impact.
2. **Maintenance Manager:** Requires actionable priorities and scheduling details.
3. **Reliability Engineer:** Requires deep physical degradation markers (RMS, Kurtosis, Spectra).
4. **Machine Learning Engineer:** Requires model biases, SHAP values, and CV metrics.
5. **Auditor / Governance:** Requires contract hashes, lineage traces, and validation logs.

## 4. Dashboard Modes

Dashboard operating modes are architectural abstractions that control information visibility and navigation limits based on the active persona, without adding new pages.

- **Executive Mode:** Purpose is strategic oversight. Visible information: Landing, Governance. Hidden information: Diagnostics, Signal Analysis, individual sensor readings. Permitted navigation: High-level summaries only. Target persona: Executive.
- **Operations Mode:** Purpose is task execution. Visible information: Landing, Maintenance Center, Bearing Detail. Hidden information: ML importance scores, Signal Analysis spectra. Target persona: Maintenance Manager.
- **Reliability Engineering Mode:** Purpose is physical diagnosis. Visible information: Bearing Detail, Signal Analysis. Hidden information: High-level budget KPIs, Model Diagnostics. Target persona: Reliability Engineer.
- **Machine Learning Diagnostics Mode:** Purpose is AI transparency. Visible information: Model Diagnostics, Bearing Detail. Hidden information: Technician routing, P&L. Target persona: ML Engineer.
- **Governance Mode:** Purpose is compliance verification. Visible information: Governance & Audit, Landing. Hidden information: Live sensor trends. Target persona: Auditor.

## 5. Navigation Architecture

Navigation follows a strict, progressive "funnel" architecture. Information flows from high-level, low-complexity business summaries down to high-complexity engineering and governance data.

**Hierarchy:**
`Landing Dashboard` → `Maintenance Center` → `Bearing Detail` → `Signal Analysis` → `Model Diagnostics` → `Governance & Audit`

Users traverse deeper into the architecture based on their persona's need for evidence. Lateral navigation is permitted between sibling tabs of a specific asset (e.g., Bearing Detail ↔ Signal Analysis).

## 6. Page Architecture

### 6.1. Landing Dashboard
- **Purpose:** Executive fleet overview and critical business risk summary.
- **Target Persona:** Executive, Maintenance Manager.
- **Questions Answered:** Are any assets at imminent risk? What is the system's confidence?
- **Source Artifacts:** `decision_report.md`, `split_summary.md`
- **Allowed Drill-downs:** To Maintenance Center.
- **Prohibited Information:** ML metrics, specific sensor readings, model baselines.

### 6.2. Maintenance Center
- **Purpose:** Centralized queue for actionable maintenance tasks.
- **Target Persona:** Maintenance Manager.
- **Questions Answered:** What actions must be scheduled today? What are the priority scores?
- **Source Artifacts:** `recommendations.parquet`, `decision_report.md`
- **Allowed Drill-downs:** To Bearing Detail.
- **Prohibited Information:** Complex ML importance scores, frequency spectra.

### 6.3. Bearing Detail
- **Purpose:** Consolidated risk and recommendation view for a single physical asset.
- **Target Persona:** Reliability Engineer, Maintenance Manager.
- **Questions Answered:** Why is this specific bearing flagged? What is its historical risk trajectory?
- **Source Artifacts:** `recommendations.parquet`, `feature_store.parquet` (high-level summaries).
- **Allowed Drill-downs:** To Signal Analysis, Model Diagnostics.
- **Prohibited Information:** Fleet-wide aggregates.

### 6.4. Signal Analysis
- **Purpose:** Deep-dive physical diagnostic evidence.
- **Target Persona:** Reliability Engineer.
- **Questions Answered:** Is this an impulsive or broadband defect? Is the sensor drifting?
- **Source Artifacts:** `feature_store.parquet`, `engineering_findings.md`
- **Allowed Drill-downs:** None (Terminal node).
- **Prohibited Information:** Business cost, maintenance schedules.

### 6.5. Model Diagnostics
- **Purpose:** Transparent display of AI reasoning and baseline performance.
- **Target Persona:** ML Engineer.
- **Questions Answered:** What is the MAE? Which features drove the prediction? Why was the median model selected?
- **Source Artifacts:** `evaluation_report.md`, `feature_importance_report.md`, `model_selection_report.md`
- **Allowed Drill-downs:** To Governance.
- **Prohibited Information:** Business P&L, technician routing.

### 6.6. Governance & Audit
- **Purpose:** Traceability and compliance verification.
- **Target Persona:** Auditor / Governance.
- **Questions Answered:** Are the decisions based on the frozen Phase 7 contract? Is the n=2 limitation acknowledged?
- **Source Artifacts:** `decision_manifest.json`, `phase7_decision_contract.yaml`, `split_summary.md`, `leakage_report.md`
- **Allowed Drill-downs:** None (Terminal node).
- **Prohibited Information:** Live sensor data streams.

## 7. Component Architecture

- **Summary Cards:** Top-level metric displays (e.g., Total Bearings at Risk). Responsible for immediate status communication.
- **Data Tables:** Paginated, sortable grids (e.g., Maintenance Queue). Responsible for displaying detailed lists of records.
- **Trend Panels:** Time-series views of degradation (e.g., RMS over time). Responsible for displaying temporal context.
- **Diagnostic Panels:** Complex matrices or multi-variable views (e.g., Feature Importance/SHAP). Responsible for deep technical explainability.
- **Recommendation Panels:** Explicitly formatted action blocks (e.g., "SCHEDULE MAINTENANCE, PRIORITY 4"). Responsible for conveying the Phase 7 business rule output.
- **Governance Panels:** Immutable text blocks (e.g., Manifest Hashes, n=2 caveat). Responsible for enforcing liability protections.
- **Notification Areas:** Alert banners. Responsible for highlighting system validation failures or critical anomalies.
- **Detail Panes:** Collapsible or modal views for secondary information on a selected row.

## 8. Dashboard Empty States

Empty states define the expected behaviour when data sub-sets yield no results, ensuring user confidence rather than confusion.
- **No recommendations:** Display explicit "System Nominal" or "No Maintenance Actions Required". The page must still render healthy asset counts.
- **No HIGH/CRITICAL risk:** Display green/healthy indicators explicitly stating "0 items" rather than hiding the section entirely.
- **No maintenance required:** An empty Maintenance Center table explicitly states "No tasks" rather than disappearing.
- **No SHAP information:** Show "Diagnostic Unavailable" for that specific bearing, maintaining the structural integrity of the Diagnostic Panel.
- **No predictions:** For censored bearings (if viewed), explicitly indicate "Censored Asset - No RUL Prediction" instead of rendering blank charts.
- **Empty filters / Empty search results:** Show "0 Results Found" and offer a 1-click action to clear filters.
- **Missing optional diagnostics:** Show explanatory placeholder text (e.g., "Phase 3 evidence excluded this feature").
- **Never Hidden:** Fleet health summary blocks and the n=2 statistical caveat must never be hidden, even during complete empty states.

## 9. Interaction Architecture

- **Navigation:** Top or side persistent navigation menu mapping to the Page Architecture.
- **Drill-down:** Clicking a summary card or table row injects context (e.g., `bearing_id`) into the child page, restricting the view to that asset.
- **Back Navigation:** Breadcrumb trails must persist across all drill-downs to allow context-aware backtracking.
- **Deep Linking Assumptions:** The architecture must support URL routing that explicitly encodes filter states and selected assets (e.g., `/bearing-detail?id=bearing_3`) to allow sharing insights.

## 10. Dashboard State Architecture

The dashboard state model dictates how context is managed across the application.
- **Global filter state:** Includes selected experiment, selected time window (if multiple exist), and current persona. This state is global and affects all pages.
- **Current navigation context / Current drill-down level:** Page-local state (pagination, local sorting) that exists only while viewing a specific view.
- **Selected bearing state:** Dynamically persists as page-local state when navigating deep asset hierarchies, but can cross boundaries as defined in Cross-Page Synchronization.
- **State persistence:** Global state persists continuously across navigation actions via URL routing or session storage.
- **State restoration:** Upon reload, the dashboard restores state purely from the URL deep link, falling back to the Landing Dashboard if absent.

## 11. Cross-Page Synchronization

Cross-page synchronization dictates how context flows between logically connected views.
- **Example Flow:** Bearing 3 selected (Maintenance Center) → Signal Analysis → Model Diagnostics → Governance.
- **Persisting Context:** The `bearing_id` context persists when navigating laterally between asset-specific pages (Bearing Detail, Signal Analysis, Model Diagnostics) so the user does not have to re-select the asset.
- **Resetting Context:** Drilling "up" to the Landing Dashboard or Governance pages intentionally drops the `bearing_id` context to return to fleet-level views.
- **Filter Propagation:** Global time filters propagate everywhere. Local priority filters remain constrained to the Maintenance Center and reset upon exit.
- **Drill-down Continuity:** Clicking an asset directly passes the identifier to initialize the state of the target detail page seamlessly.

## 12. Session Philosophy

The dashboard behaves as **one continuous analytical session** rather than independent, stateless pages.
- **Justification:** Decision-makers are investigating complex, interconnected artifacts to make high-stakes maintenance decisions. Loss of context breaks their mental model of risk tracing.
- **Navigation Continuity:** The outer shell and global filters remain static, providing continuous orientation while the inner content routes change.
- **State Continuity:** Filter selections apply instantly and travel with the user throughout the session.
- **Traceability Continuity:** The active Phase 7 Experiment Fingerprint remains permanently affixed to the session header, guaranteeing all viewed data belongs to the same pipeline run.

## 13. Persona Switching Behaviour

Switching between operating modes (e.g., Executive → Reliability Engineer) is treated as a "View As" UX toggle for UI simplification, assuming standard overarching authorization to the frozen reports.
- **What state persists:** The currently selected asset (`bearing_id`) and global time filters persist to allow a user to view the exact same engineering problem through a different professional lens.
- **What state resets:** Persona-specific local filters (e.g., sorting preferences in the Maintenance Center) are reset to defaults.
- **Information visibility:** Switching persona dynamically reveals or collapses navigation tabs in accordance with the Dashboard Modes definitions.
- **Drill-down behaviour:** If an Executive switches to ML Engineer while on the Landing Page, they remain on the Landing Page but their navigation menu expands to allow deeper diagnostics drill-downs.

## 14. Filter Architecture

- **Global Filters:** Time horizon (if multiple experiments exist). Allowed on all pages.
- **Local Filters:**
  - **Bearing Filter:** Restricts view to specific assets. (Allowed on Maintenance Center, Bearing Detail, Diagnostics).
  - **Risk Filter:** Restricts by CRITICAL, HIGH, MEDIUM, LOW. (Allowed on Landing, Maintenance Center).
  - **Priority Filter:** Restricts by Priority 1-5. (Allowed on Maintenance Center).
- **Persona-specific Filters:** "My View" defaults based on user role login (if implemented).

## 15. Data Flow

The dashboard operates purely as a read-only presentation layer. It does not compute new features, run ML inference, or evaluate business rules.

**Flow:**
Local Storage / Cloud Storage → Data Loading Layer → Validation Layer (Schema/Hash Check) → State Management (Filtering/Routing) → UI Component Rendering.

## 16. Artifact Loading Strategy

To balance performance with context availability, artifact ingestion follows a tiered strategy:
- **Startup Loading:** Essential lightweight artifacts (`decision_report.md`, `split_summary.md`, `decision_manifest.json`) load immediately to render the Landing page and establish Governance context.
- **Lazy Loading:** Heavy artifacts (`feature_store.parquet`, `recommendations.parquet`) are loaded lazily or partitioned only when the user navigates to the Maintenance Center or Signal Analysis pages.
- **On-demand Loading:** Highly specific diagnostic data (e.g., SHAP values) load only when a user drills deeply into Model Diagnostics.
- **Cache Philosophy:** Parquet files are aggressively cached in-memory during the continuous session to prevent slow re-reads from disk upon navigating back and forth.
- **Reload Behaviour:** A hard user refresh clears the cache and re-triggers validation of all artifacts.
- **Never Preloaded:** High-cardinality raw frequency spectra matrices are never preloaded to protect the client memory footprint.
- **Validation Order:** Manifest validation MUST occur prior to Schema checks, which MUST occur prior to full Data loading.

## 17. Artifact Mapping

| Target Page | Consumed Artifacts |
|---|---|
| Landing Dashboard | `decision_report.md`, `split_summary.md` |
| Maintenance Center | `recommendations.parquet`, `decision_report.md` |
| Bearing Detail | `recommendations.parquet` |
| Signal Analysis | `feature_store.parquet`, `engineering_findings.md` |
| Model Diagnostics | `evaluation_report.md`, `feature_importance_report.md`, `model_selection_report.md` |
| Governance & Audit | `decision_manifest.json`, `phase7_decision_contract.yaml`, `decision_validation_report.md`, `leakage_report.md` |

## 18. Dashboard Lifecycle

The lifecycle dictates system responsibilities from boot to shutdown:
1. **Dashboard Startup:** System initializes and reads environment configurations.
2. **Artifact Discovery:** System strictly scans the `data/` and `reports/` directories for expected frozen Phase 3-7 artifacts.
3. **Validation:** System verifies schemas and fingerprint hashes against the manifest. Failures halt the boot process.
4. **Artifact Loading:** System loads Tier 1 (Startup) artifacts into memory.
5. **Navigation Enabled:** UI unlocks and the Landing Dashboard renders.
6. **Normal Operation:** User interacts, triggering lazy/on-demand loading of Tier 2/Tier 3 artifacts.
7. **Refresh:** Triggered manually; flushes the cache and restarts from Step 2.
8. **Graceful Shutdown:** System cleanly releases memory and file handles.

## 19. Validation Strategy

The dashboard must validate its inputs on load:
- **Missing Artifacts:** If a required parquet or JSON file is missing, the dashboard enters "Degraded Mode," disabling affected pages and displaying an error banner.
- **Schema Mismatch:** If columns in `recommendations.parquet` do not match the expected Phase 7 schema, the dashboard refuses to load the file to prevent silent misinterpretation.
- **Fingerprint Mismatch:** The dashboard checks if the `decision_manifest.json` hash matches the expected deployed version. If it fails, an Auditor Alert is triggered.
- **Failure Behaviour:** The system fails closed. It is safer to show no data than corrupted or misaligned engineering data.

## 20. Governance

- **Statistical Limitations:** The Phase 5 "n=2 Caveat" (methodology validation only, not fleet guarantees) must be permanently affixed to the Landing Dashboard and Governance pages. It cannot be minimized or closed.
- **Dashboard Versioning:** The dashboard itself must carry a version number that corresponds to the Phase 7 contract version it is designed to consume.
- **Decision Traceability:** Every recommendation must expose its originating Phase 6 Model ID and Phase 7 Experiment Fingerprint in the UI.

## 21. Testing Strategy

- **Unit Testing:** Validate that data loading functions correctly parse the frozen artifacts.
- **Integration Testing:** Verify that filter states correctly propagate across drill-down routes.
- **Persona Testing:** Ensure that the Executive view strictly hides ML diagnostics, and that Auditor views successfully load manifest hashes.
- **Fuzz Testing:** Inject malformed parquet files to ensure the Validation Layer safely fails closed.

## 22. Risk Register (Phase 8 specific)

| Risk | Impact | Mitigation |
|---|---|---|
| Dashboard misinterprets parquet timestamps | High (False alarms) | Enforce strict timezone awareness and schema validation on data load. |
| Stakeholders ignore n=2 limitation | High (Liability) | Affix the caveat permanently to the landing page; prohibit removal via UI. |
| Dashboard latency rendering 17k rows | Medium (UX) | Implement server-side/lazy pagination for data tables. |

## 23. Acceptance Criteria

1. Dashboard consumes ONLY the approved frozen artifacts from Phases 3-7.
2. All five personas can navigate their respective user journeys without encountering prohibited information on their primary screens.
3. Drill-down functionality preserves context (e.g., `bearing_id`) across pages.
4. The Validation Layer correctly intercepts and halts the loading of malformed or missing artifacts.
5. The n=2 caveat and categorical confidence limitations are permanently visible.
6. Every displayed recommendation provides a link or trace to its generating fingerprint.

## 24. Implementation Notes

- The system should leverage a reactive state-management paradigm to handle global filtering and routing.
- Data loading should be asynchronous to prevent UI blocking during large parquet reads.

## 25. Out of Scope

- Visual aesthetics (Colors, Themes, Typography, Icons).
- Selection of specific charting libraries or component frameworks (e.g., React, Streamlit, Plotly).
- CSS or responsive grid layouts.
- Backend database implementations (system relies on frozen flat files/parquets).
- Any re-architecture or re-design of existing pages/functionalities.

## 26. Architecture Assumptions

- The frozen artifacts reside in a consistent, accessible file path relative to the dashboard execution environment.
- The dashboard operates in an environment where file I/O permissions are granted for the `reports/` and `data/` directories.

## 27. Phase 8 → Final Delivery Contract

The Phase 8 architecture defined herein constitutes the final blueprint for the presentation layer. Upon authorization, the implementation phase will execute this architecture using the chosen technology stack, strictly adhering to the information boundaries, validation requirements, and traceability mandates specified above.

---

## Final Self Review

1. **Would you approve this architecture before implementation?** Yes. It clearly delineates boundaries, ensures traceability, heavily mitigates UI risk with state management and loading strategies, and prevents scope creep into visual design.
2. **Is it consistent with all frozen phases?** Yes. It maps precisely to the outputs of Phases 3-7.
3. **Does it preserve engineering traceability?** Yes. The Governance page and mandatory fingerprint linking ensure full provenance.
4. **Does it preserve scientific integrity?** Yes. By failing closed on validation errors and displaying the n=2 caveat permanently, it prevents false precision.
5. **Does it avoid reinterpreting engineering results?** Yes. It operates strictly as a read-only presentation layer.
6. **Would a Principal BI Architect approve this architecture?** Yes. The component, filter, state, loading, and drill-down architectures reflect advanced industry best practices for enterprise BI deployments.

**APPROVE ARCHITECTURE**

**Justification:** The proposed architecture successfully translates the strict engineering and statistical realities of the Phase 1-7 pipeline into a scalable, secure, and governed presentation framework. The added documentation refinements (State Architecture, Loading Strategy, Empty States, Cross-Page Synchronization, Session Philosophy, Persona Switching, Lifecycle, and Modes) rigorously harden the blueprint against implementation ambiguities without expanding scope or altering navigation. This document fulfills the exact parameters of a finalized architecture design phase.
