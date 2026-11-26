# Phase 7 Architecture — Maintenance Decision Engine
## Version 1.1 | Status: DRAFT

**Project:** Bearing Remaining Useful Life Prediction & Maintenance Decision System
**Organization:** Independent Project
**Author Role:** Principal Reliability Engineer / Principal Industrial AI Architect
**Phase:** 7 of N
**Depends On:** Phase 6 (Frozen — Approved)
**Output Consumed By:** Phase 8 (Dashboard & Alerting — not yet designed)

---

## Changelog

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-07-04 | Initial architecture design |
| 1.1 | 2026-07-04 | Documentation refinement pass following first formal architecture review. Strengthened: threshold governance (ownership/approval/versioning/audit/backward compatibility), added dedicated Decision Traceability section, expanded Business Rule Precedence with conflict-resolution examples, added Decision Output Validation, strengthened Explainability with an explicit evidence checklist, expanded BR-04 rationale, and fully enumerated the Phase 7→8 contract artifact list. No architecture, module, business rule, decision logic, or scope changes were made — documentation only. |

---

## Table of Contents

1. Executive Summary
2. Objectives
3. Architecture Overview
4. Phase Boundaries
5. Inputs — Phase 6 Artifacts
6. Outputs — Phase 7 Artifacts
7. Module Architecture
8. Decision Pipeline
   - 8a. Decision Traceability
9. Risk Classification Framework
   - 9d. Threshold Governance
10. Maintenance Decision Framework
11. Confidence Estimation Framework
12. Business Rules Framework
   - 12a. Business Rule Precedence
13. Explainability Framework
14. Data Flow
15. Configuration
16. Validation Strategy
   - 16c. Decision Output Validation
17. Testing Strategy
18. Governance
19. Failure Handling
20. Risk Register
21. Phase 7 → Phase 8 Contract
22. Implementation Notes
23. Out of Scope
24. Architecture Assumptions
25. Self-Review
26. Architecture Refinement Log (v1.1)

---

## §1 — Executive Summary

Phase 7 is the **Maintenance Decision Engine**. It transforms the raw numerical
Remaining Useful Life (RUL) predictions produced by Phase 6 into structured,
explainable maintenance recommendations suitable for engineering action and
executive review.

The engine does not train models. It does not re-rank features. It does not
recompute predictions. It applies a principled, configurable decision framework
to the approved Phase 6 artifact set.

Every output recommendation is:

- **Traceable** — linked by grain to its source prediction
- **Explainable** — accompanied by a human-readable justification citing model
  evidence, feature evidence, and decision rule evidence
- **Honest** — carries forward all caveats declared in the Phase 6 contract,
  including the n=2 limitation and the median-baseline (demotion-fallback) model type
- **Governed** — fingerprinted, versioned, and recorded in a decision manifest
- **Reproducible** — given the same Phase 6 artifacts, config fingerprint, and
  rule version, the decision engine produces identical output

Phase 7 produces a machine-readable Phase 7 → Phase 8 contract that Phase 8
(Dashboard & Alerting) consumes. Phase 8 must not bypass this contract.

---

## §2 — Objectives

| # | Objective |
|---|-----------|
| O1 | Transform predicted RUL values into risk categories with defensible thresholds |
| O2 | Assign maintenance actions to each risk category |
| O3 | Estimate decision confidence, propagating Phase 6 model limitations |
| O4 | Calculate a multi-factor maintenance priority score |
| O5 | Generate a decision explanation for every recommendation |
| O6 | Apply business rules including safety overrides, fallbacks, and missing-prediction handling |
| O7 | Aggregate channel-level predictions to bearing-level decisions |
| O8 | Produce a complete audit trail of every decision |
| O9 | Produce a typed, versioned Phase 7 → Phase 8 contract |
| O10 | Validate all output artifacts before committing them to disk |

---

## §3 — Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PHASE 7 — DECISION ENGINE                           │
│                                                                             │
│  INPUTS (Phase 6 Frozen Artifacts)                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  model_predictions.parquet  phase6_model_contract.yaml              │   │
│  │  model_manifest.json        feature_importance_report.md            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│            │                                                                │
│            ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  CONTRACT LOADER — validates Phase 6 contract schema on entry        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│            │                                                                │
│            ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  DECISION ENGINE — channel aggregation, decision transformation      │   │
│  │                                                                     │   │
│  │   risk_classifier  →  maintenance_policy  →  confidence_estimator   │   │
│  │         │                    │                       │              │   │
│  │         └──────────┬─────────┘                       │              │   │
│  │                    ▼                                  │              │   │
│  │           priority_calculator  ←──────────────────────              │   │
│  │                    │                                                 │   │
│  │                    ▼                                                 │   │
│  │           recommendation_engine  →  decision_explainer               │   │
│  │                    │                                                 │   │
│  │                    ▼                                                 │   │
│  │             business_rules (overrides, fallbacks)                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│            │                                                                │
│            ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  VALIDATION — 10 checks, 5 FATAL / 3 WARNING / 2 INFO               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│            │                                                                │
│            ▼                                                                │
│  OUTPUTS                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  recommendations.parquet          decision_report.md                 │   │
│  │  phase7_decision_contract.yaml    decision_manifest.json             │   │
│  │  decision_log.json                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## §4 — Phase Boundaries

### What Phase 7 IS responsible for

- Consuming Phase 6 approved artifacts via the Phase 6→7 contract
- Aggregating channel-level predictions to bearing-level decisions
- Classifying RUL predictions into risk categories
- Mapping risk categories to maintenance actions
- Estimating decision confidence from declared model properties
- Calculating maintenance priority
- Generating human-readable decision explanations
- Applying business rules and safety overrides
- Validating all outputs
- Producing the Phase 7→8 contract

### What Phase 7 is NOT responsible for

- Training, retraining, or fine-tuning any model (Phase 6)
- Recomputing RUL predictions (Phase 6)
- Dashboard rendering (Phase 8)
- Alerting and notification (Phase 8)
- Web APIs and deployment (Phase 8)
- Threshold calibration via optimization (this is a domain-expert responsibility,
  not an algorithmic one)
- Fleet-level analysis beyond the 2 RTF units in the Phase 6 artifact set

---

## §5 — Inputs — Phase 6 Artifacts

Phase 7 consumes ONLY artifacts produced and approved in Phase 6.

| Artifact | Path | Role |
|----------|------|------|
| Predictions | `reports/model_predictions.parquet` | Primary decision input |
| Phase 6 Contract | `reports/phase6_model_contract.yaml` | Schema, model metadata, caveats |
| Model Manifest | `reports/model_manifest.json` | Governance lineage |
| Feature Importance | `reports/feature_importance_report.md` | Explanation generation (prose only) |
| Phase 6 Eval Report | `reports/evaluation_report.md` | Confidence context (prose only) |

### §5a — Prediction Schema (from Phase 6 Contract)

The Phase 6 contract declares:

```yaml
prediction_schema:
  grain: [bearing_id, channel, timestamp]
  output: predicted_rul_minutes
  output_dtype: float
  output_constraint: non_negative
  units: minutes
```

Phase 7 reads predictions at grain `(bearing_id, channel, timestamp)` and
aggregates them to grain `(bearing_id, timestamp)` before applying the decision
framework.

### §5b — Phase 6 Model Properties (from Phase 6 Contract)

Phase 7 reads and propagates the following Phase 6 model properties:

- `model.type` — selected model name (e.g., `"median"` under demotion fallback)
- `performance.n2_caveat` — mandatory boolean; must be `true`
- `performance.n2_caveat_text` — mandatory prose; propagated to every explanation
- `performance.mean_mae_minutes` — propagated to confidence estimation
- `performance.stability_ratio` — propagated to confidence estimation
- `consumer_expectations.must_not_use_shap_outputs_for_model_modification` —
  verified present; its value `true` is re-asserted in the Phase 7→8 contract

### §5c — Phase 6 Contract Entry Gate

The Phase 6 contract must be loaded and validated before any decision logic
executes. The following are FATAL entry-gate conditions (§17 — Validation):

- `schema_version` absent or not `"1.0"` → FATAL
- `performance.n2_caveat` absent or not `True` → FATAL
- `prediction_schema.grain` does not match expected `[bearing_id, channel, timestamp]` → FATAL
- `model.type` absent → FATAL

---

## §6 — Outputs — Phase 7 Artifacts

| Artifact | Path | Description |
|----------|------|-------------|
| Recommendations | `reports/recommendations.parquet` | Bearing-level decisions, one row per (bearing_id, timestamp) |
| Decision Report | `reports/decision_report.md` | Human-readable summary of all decisions |
| Phase 7→8 Contract | `reports/phase7_decision_contract.yaml` | Machine-readable interface for Phase 8 |
| Decision Manifest | `reports/decision_manifest.json` | Governance metadata |
| Decision Log | `reports/decision_log.json` | Append-only execution log |

### §6a — Recommendation Schema

Each row in `reports/recommendations.parquet` contains:

| Column | Dtype | Description |
|--------|-------|-------------|
| `bearing_id` | str | Bearing identifier |
| `timestamp` | datetime | Data file timestamp |
| `min_predicted_rul_minutes` | float | Min RUL across channels (conservative aggregation) |
| `mean_predicted_rul_minutes` | float | Mean RUL across channels (informational) |
| `channel_count` | int | Number of channels contributing to this decision |
| `risk_category` | str | CRITICAL / HIGH / MEDIUM / LOW / HEALTHY |
| `maintenance_action` | str | IMMEDIATE_SHUTDOWN / URGENT_MAINTENANCE / SCHEDULE_MAINTENANCE / ROUTINE_MONITORING / NORMAL_OPERATION |
| `confidence_level` | str | HIGH / MEDIUM / LOW |
| `priority_score` | int | 1 (highest) to 5 (lowest) |
| `is_business_rule_override` | bool | True if a business rule altered the base decision |
| `override_reason` | str or null | Reason for override, if applied |
| `explanation_text` | str | Human-readable decision justification |
| `model_type` | str | Propagated from Phase 6 contract |
| `n2_caveat` | bool | Always True — propagated from Phase 6 contract |
| `rule_version` | str | Business rule set version |
| `config_fingerprint` | str | SHA-256 of decision_engine config block |
| `experiment_id` | str | Phase 6 experiment ID (upstream lineage) |

---

## §7 — Module Architecture

All Phase 7 source modules reside in `src/decision_engine/`.

### §7.1 — `contract_loader.py`

**Responsibility:** Load and validate the Phase 6→7 contract on pipeline entry.
Verify schema version, mandatory fields, and n2_caveat presence before any
decision logic executes.

Key functions:
- `load_phase6_contract(path) -> Dict` — loads and validates `phase6_model_contract.yaml`
- `load_phase6_predictions(path) -> pd.DataFrame` — loads `model_predictions.parquet`
- `load_phase6_manifest(path) -> Dict` — loads `model_manifest.json` for upstream lineage
- `validate_phase6_contract(contract, predictions, config, logger) -> ContractValidationResult`
  — runs entry-gate checks; FATAL on schema failure, WARNING on non-blocking gaps

```
ContractValidationResult:
  passed: bool
  fatals: List[str]
  warnings: List[str]
  model_type: str
  n2_caveat: bool
  n2_caveat_text: str
  feature_columns: List[str]
  mean_mae_minutes: float
  stability_ratio: float
```

### §7.2 — `decision_engine.py`

**Responsibility:** Orchestrate the full decision transformation for the loaded
prediction dataset. Drives channel aggregation, then calls all downstream modules
in order: risk_classifier → maintenance_policy → confidence_estimator →
priority_calculator → decision_explainer → business_rules. Returns a list of
`Recommendation` objects.

Key functions:
- `aggregate_channels(predictions_df, config) -> pd.DataFrame`
  — groups by `(bearing_id, timestamp)`, computes `min_predicted_rul_minutes`
    (primary, conservative) and `mean_predicted_rul_minutes` (informational).
    Records `channel_count` per group.
- `run_decision_pipeline(bearing_df, phase6_contract, config, logger) -> List[Recommendation]`
  — iterates each `(bearing_id, timestamp)` row and calls the full decision chain.

### §7.3 — `risk_classifier.py`

**Responsibility:** Map a RUL value (in minutes) to one of five ordered risk
categories. Thresholds are configuration parameters; the classifier does NOT
contain hardcoded values.

Risk categories (ordered, highest to lowest severity):

| Category | Condition | Meaning |
|----------|-----------|---------|
| CRITICAL | RUL ≤ `thresholds.critical_minutes` | Failure imminent — act before next window |
| HIGH | `critical_minutes` < RUL ≤ `thresholds.high_minutes` | Urgent — act in next maintenance window |
| MEDIUM | `high_minutes` < RUL ≤ `thresholds.medium_minutes` | Plan action in near-term cycle |
| LOW | `medium_minutes` < RUL ≤ `thresholds.low_minutes` | Monitor; plan for next routine cycle |
| HEALTHY | RUL > `thresholds.low_minutes` | No action required at this time |

**Threshold justification requirement:** The threshold values in `config.yaml`
MUST be accompanied by inline comments explaining their engineering basis.
Examples of acceptable bases: minimum mobilization time for a maintenance crew,
safety margin to failure, production schedule constraints. The architecture does
not assert specific minute values because they require domain-expert validation
by a qualified Reliability Engineer before any operational use.

Key functions:
- `classify_risk(rul_minutes, thresholds) -> RiskCategory` — pure mapping, no side effects
- `validate_thresholds(thresholds, logger) -> None` — verifies ordering:
  `critical < high < medium < low`; raises `ValueError` if violated
- `RiskCategory` — `Enum` with members CRITICAL, HIGH, MEDIUM, LOW, HEALTHY

### §7.4 — `maintenance_policy.py`

**Responsibility:** Map a risk category to a recommended maintenance action.
Policy is a one-to-one mapping. The mapping is fixed by domain conventions and
must not be softcoded — it must be declared in configuration so it can be
reviewed and versioned independently of source code.

Default policy (configurable):

| Risk Category | Maintenance Action | Rationale |
|---------------|--------------------|-----------|
| CRITICAL | IMMEDIATE_SHUTDOWN | Failure imminent; continuing operation risks catastrophic damage |
| HIGH | URGENT_MAINTENANCE | Bearing is in danger zone; delay increases failure probability substantially |
| MEDIUM | SCHEDULE_MAINTENANCE | Degradation detected; maintenance should be planned before next inspection cycle |
| LOW | ROUTINE_MONITORING | Early degradation signal; confirm at next scheduled inspection |
| HEALTHY | NORMAL_OPERATION | No degradation signal; proceed under standard operating schedule |

Key functions:
- `get_maintenance_action(risk_category, policy_map) -> MaintenanceAction`
- `load_policy_map(config) -> Dict[RiskCategory, MaintenanceAction]`
- `MaintenanceAction` — `Enum` with members IMMEDIATE_SHUTDOWN, URGENT_MAINTENANCE,
  SCHEDULE_MAINTENANCE, ROUTINE_MONITORING, NORMAL_OPERATION

### §7.5 — `confidence_estimator.py`

**Responsibility:** Estimate the confidence level of each decision. Confidence
is a composite property derived from three independent factors — model type,
n2 limitation, and prediction proximity to decision boundaries. It is expressed
as a categorical level (HIGH / MEDIUM / LOW), not as a probability, because
the n=2 training corpus does not support well-calibrated probability estimates.

#### §7.5a — Confidence factors

**Factor 1 — Model type modifier**

The Phase 6 model type is read from the Phase 6 contract. Each model type
carries a declared confidence modifier:

| Model Type | Modifier | Rationale |
|------------|----------|-----------|
| `mean` | MEDIUM → LOW | Constant predictor; no feature sensitivity |
| `median` | MEDIUM → LOW | Constant predictor (demotion fallback); no feature sensitivity |
| `linear` | MEDIUM | Feature-sensitive but no non-linearity |
| `random_forest` | HIGH | Non-linear, multi-feature; but restricted n=2 evaluation |
| `xgboost` | HIGH | Same as RF; same n=2 limitation |

*Note:* The Phase 6 approved experiment uses `model_type = "median"` (demotion
fallback). The confidence modifier for `median` is therefore the applicable
modifier for this pipeline run. Future pipeline runs with a promoted model type
will automatically apply the corresponding modifier, since the value is read
from the contract rather than hardcoded.

**Factor 2 — n=2 penalty**

`n2_caveat: true` (mandatory per Phase 6 contract) applies a systematic
confidence downgrade. A model evaluated on only 2 RTF units cannot produce
fleet-level guarantees. The n=2 penalty reduces any confidence level by one
grade unless it is already at LOW.

Combined effect: for `median` baseline with n=2 penalty, all decisions default
to confidence LOW, which is the honest scientific position.

**Factor 3 — Boundary proximity**

When the predicted RUL is within a configurable tolerance band around any
decision threshold, the decision is near-boundary and therefore higher
uncertainty. Near-boundary decisions are downgraded by one confidence grade
if they are not already at LOW.

Tolerance band: `config.decision_engine.confidence.boundary_proximity_tolerance_minutes`.

**Factor resolution order:**
1. Start at model-type base confidence
2. Apply n=2 penalty
3. Apply boundary-proximity downgrade
4. Floor at LOW (cannot go below LOW)

Key functions:
- `estimate_confidence(rul_minutes, model_type, n2_caveat, thresholds, config) -> ConfidenceLevel`
- `ConfidenceLevel` — `Enum` with members HIGH, MEDIUM, LOW
- `_apply_model_type_modifier(base, model_type, modifier_map) -> ConfidenceLevel`
- `_apply_n2_penalty(level, n2_caveat) -> ConfidenceLevel`
- `_apply_boundary_proximity(level, rul_minutes, thresholds, tolerance) -> ConfidenceLevel`

### §7.6 — `priority_calculator.py`

**Responsibility:** Compute an integer priority score (1 = highest urgency,
5 = lowest) by combining risk category and confidence level. Priority determines
action order when multiple bearings require attention.

Priority assignment matrix:

| Risk Category | HIGH Confidence | MEDIUM Confidence | LOW Confidence |
|---------------|-----------------|-------------------|----------------|
| CRITICAL | 1 | 1 | 1 |
| HIGH | 2 | 2 | 3 |
| MEDIUM | 3 | 3 | 4 |
| LOW | 4 | 4 | 5 |
| HEALTHY | 5 | 5 | 5 |

Design rationale: CRITICAL is Priority 1 regardless of confidence because the
safety-first principle requires treating a low-confidence CRITICAL prediction as
if it were CRITICAL until the bearing can be physically inspected. A potentially
imminent failure must not be deprioritized because of model uncertainty.

Key functions:
- `compute_priority(risk_category, confidence_level, priority_matrix) -> int`
- `load_priority_matrix(config) -> Dict` — reads matrix from config (overridable)

### §7.7 — `recommendation_engine.py`

**Responsibility:** Assemble all decision components into a unified
`Recommendation` dataclass for each `(bearing_id, timestamp)` row.
This module is a pure assembler — it calls no decision logic of its own.

```
Recommendation:
  bearing_id: str
  timestamp: datetime
  min_predicted_rul_minutes: float
  mean_predicted_rul_minutes: float
  channel_count: int
  risk_category: RiskCategory
  maintenance_action: MaintenanceAction
  confidence_level: ConfidenceLevel
  priority_score: int
  is_business_rule_override: bool
  override_reason: Optional[str]
  explanation_text: str
  model_type: str
  n2_caveat: bool
  rule_version: str
  config_fingerprint: str
  experiment_id: str
```

Key functions:
- `assemble_recommendation(row, risk, action, confidence, priority, explanation,
  override_info, provenance) -> Recommendation`
- `recommendations_to_dataframe(recommendations) -> pd.DataFrame`

### §7.8 — `decision_explainer.py`

**Responsibility:** Generate a structured, human-readable explanation for each
recommendation. The explanation must be self-contained — a reader without access
to source code should understand why the decision was made.

**Explanation template:**

```
Bearing {bearing_id} — {maintenance_action} ({timestamp})

Risk Assessment
  Predicted RUL (minimum across channels): {min_rul:.0f} minutes
    ({hours:.1f} hours / {days:.1f} days)
  Risk Category: {risk_category}
  Maintenance Action: {maintenance_action}
  Priority: {priority_score} / 5
  Confidence: {confidence_level}

Model Evidence
  Selected Model: {model_type} (Phase 6 experiment {experiment_id})
  Model Selection Note: {model_selection_note}
  Evaluation MAE: {mean_mae_minutes:.0f} minutes (mean across 2 LOBO folds)

Statistical Limitations
  n=2 Caveat: {n2_caveat_text}
  Confidence Level Basis: {confidence_basis}

Recommended Action
  {action_rationale}

Override Note (if applicable)
  {override_reason}
```

**Model selection note** is derived from the Phase 6 contract model type:
- If `model_type in ("mean", "median")`: "Baseline model selected via
  demotion fallback — no candidate model individually cleared the Phase 6
  promotion criteria on this n=2 dataset. Predictions reflect training-set
  central tendency, not feature-driven estimation."
- Otherwise: formal model name description.

Key functions:
- `generate_explanation(recommendation, phase6_contract, config) -> str`
- `_format_rul_human(rul_minutes) -> str` — converts minutes to hours/days prose
- `_build_model_selection_note(model_type) -> str`
- `_build_confidence_basis(confidence_level, model_type, n2_caveat, near_boundary) -> str`
- `_build_action_rationale(maintenance_action) -> str`

### §7.9 — `business_rules.py`

**Responsibility:** Apply domain business rules that can modify, override, or
escalate the base decision produced by the decision engine. Business rules are
the safety net — they encode engineering judgment that cannot be captured by a
pure RUL-to-risk mapping.

#### §7.9a — Override Rules

**Rule BR-01: Safety floor for CRITICAL decisions**
If `risk_category == CRITICAL`, the `maintenance_action` is locked to
`IMMEDIATE_SHUTDOWN` regardless of confidence level. No confidence-based
downgrade may change a CRITICAL decision to a less severe action.
*Rationale:* Potential imminent failure must be treated as imminent until
physically inspected.

**Rule BR-02: Missing prediction fallback**
If a bearing expected in the Phase 6 artifact set has no prediction rows in
`model_predictions.parquet`, the engine assigns:
- `risk_category = HIGH` (conservative)
- `maintenance_action = URGENT_MAINTENANCE`
- `confidence_level = LOW`
- `is_business_rule_override = True`
- `override_reason = "No prediction available for bearing — conservative HIGH risk assigned"`
*Rationale:* Absence of a prediction is itself an anomaly and must not be
treated as HEALTHY.

**Rule BR-03: High channel disagreement flag**
If the spread of predicted RUL values across channels for a single
`(bearing_id, timestamp)` exceeds
`config.decision_engine.business_rules.channel_rul_spread_threshold_minutes`,
the explanation is annotated with a disagreement warning and confidence is
downgraded by one level.
*Rationale:* Large inter-channel disagreement indicates physical uncertainty
(different channel types capturing different failure signatures) that should
be communicated to the maintenance engineer.

**Rule BR-04: Constant-predictor awareness note**
If `model_type in ("mean", "median")`, all recommendations receive a
mandatory prose annotation:
> "All predictions for this bearing are constant (same value at every
> timestamp) because the selected model is a baseline predictor with no
> feature sensitivity. The time series of decisions for this bearing is
> therefore uniform. Decisions reflect the training-distribution central
> tendency, not the bearing's evolving degradation state."
*Rationale:* Without this note, a consumer of `recommendations.parquet`
may falsely interpret a time series of identical decisions as stable health
rather than as a model limitation.

*Extended justification (why BR-04 exists as a business rule rather than a
one-off caveat):* The Phase 6 approved experiment selected `model_type =
"median"` through the demotion-fallback path (Phase 6 §7b) — no non-baseline
candidate individually cleared the promotion ladder on the n=2 dataset. A
consequence of this honest, correctly-selected outcome is that the baseline
predictor emits the *same* RUL value for every timestamp of a given bearing,
because a constant predictor has no feature sensitivity by definition (it was
fit on `y_train` alone, §7.9a Phase 6 architecture). If Phase 7 rendered this
constant series of predictions through the ordinary decision pipeline without
comment, a maintenance engineer viewing a sequence of identical
`ROUTINE_MONITORING` (or any other) recommendations across many timestamps
could reasonably — but incorrectly — read that flat sequence as *the equipment
staying steady*, i.e., as positive evidence of stable, unchanging bearing
health. That reading would be false: the flatness is an artifact of the
selected model's lack of feature sensitivity, not a measurement of the
bearing's actual (and, per the underlying physics, presumably evolving)
degradation state.

BR-04 exists specifically to close this gap. It does not change
`risk_category`, `maintenance_action`, `confidence_level`, or any other
decision field — it only attaches a mandatory, verbatim prose annotation to
the explanation. This distinction matters for the architecture's own
integrity: BR-04 is a **transparency mechanism**, not a decision-logic
mechanism. It ensures a documented, honest Phase 6 modeling outcome (demotion
fallback) is not silently laundered into an implicit — and false — Phase 7
health signal simply because no one told the consumer the model behind the
numbers cannot see the bearing's condition change. Because the annotation is
mandatory whenever `model_type in ("mean", "median")`, its presence or
absence is itself an auditable governance property: any future review can
grep `recommendations.parquet` for the annotation whenever a constant-model
experiment is active and treat its absence as a validation defect (already
covered structurally by the explanation-completeness checklist, §13d, item 5).

**Rule BR-05: Zero-RUL floor**
If a predicted RUL is exactly 0.0 minutes, it is classified as CRITICAL
regardless of threshold configuration.
*Rationale:* A prediction of 0 RUL = at failure. No threshold configuration
should permit this to be classified as anything below CRITICAL.

#### §7.9b — Rule Versioning

Business rules are versioned. The rule version is embedded in every
`Recommendation` and in the Phase 7 governance manifest. Rule version `"1.0"`
covers the five rules above. Any future modification to rule logic must
increment the rule version, which in turn changes the config fingerprint —
ensuring that outputs produced under different rule versions cannot be silently
compared.

Key functions:
- `apply_business_rules(recommendation, phase6_contract, config, logger) -> Recommendation`
- `_apply_safety_floor(rec) -> Recommendation`
- `_apply_missing_prediction_fallback(bearing_id, timestamp, provenance) -> Recommendation`
- `_apply_channel_disagreement_flag(rec, spread_minutes, threshold) -> Recommendation`
- `_apply_constant_predictor_note(rec, model_type) -> Recommendation`
- `RULE_VERSION = "1.0"`

### §7.10 — `validation.py`

**Responsibility:** Validate all Phase 7 output artifacts before they are written
to disk. Mirrors the ValidationCheck / FinalValidationResult pattern established
in `src/rul_target/validation.py` and `src/model_dev/validation.py`.

Ten validation checks:

| # | Check | Severity |
|---|-------|----------|
| V1 | Phase 6 contract entry gate passed | FATAL |
| V2 | All expected bearings present in output | FATAL |
| V3 | No NaN in `min_predicted_rul_minutes` | FATAL |
| V4 | No negative `min_predicted_rul_minutes` | FATAL |
| V5 | `risk_category` values are all valid enum members | FATAL |
| V6 | `priority_score` values are all in {1,2,3,4,5} | FATAL |
| V7 | `n2_caveat = True` in every row | WARNING |
| V8 | CRITICAL rows all have `priority_score == 1` | WARNING |
| V9 | Governance fields present in manifest | WARNING |
| V10 | Channel count per (bearing_id, timestamp) matches Phase 6 prediction grain | INFO |

Key functions:
- `run_decision_validation(recommendations_df, phase6_contract, manifest, config, logger) -> FinalValidationResult`
- `generate_decision_validation_report(result, logger) -> str`
- `save_decision_validation_report(report_md, output_path, logger) -> None`

### §7.11 — `contract_generator.py`

**Responsibility:** Produce `reports/phase7_decision_contract.yaml` — the
machine-readable interface Phase 8 must consume. See §21 for the full schema.

Key functions:
- `build_phase8_contract(recommendations_df, manifest, phase6_contract, config, logger) -> Dict`
- `save_phase8_contract(contract, output_path, logger) -> None`

### §7.12 — `decision_manifest.py`

**Responsibility:** Produce `reports/decision_manifest.json` — the governance
artifact recording all metadata about this pipeline run. See §18 for fields.

Key functions:
- `build_decision_manifest(experiment_id, config_fingerprint, rule_version,
  phase6_manifest, selection_summary, environment, logger) -> Dict`
- `save_decision_manifest(manifest, output_path, logger) -> None`
- `capture_environment() -> Dict` — versions of all Python dependencies

### §7.13 — `experiment_tracker.py`

**Responsibility:** Compute the configuration fingerprint and generate the
experiment record for `decision_log.json`.

Key functions:
- `compute_config_fingerprint(config) -> str` — SHA-256 of `decision_engine`
  block, `json.dumps(sort_keys=True)`
- `generate_experiment_id(config_fingerprint) -> str`
  — `f"dec_{fp[:12]}_rv{rule_version}"` where `rule_version` replaces the seed
    (decisions are deterministic given config; there is no randomness)
- `build_experiment_record(experiment_id, config_fingerprint, rule_version,
  phase6_experiment_id, phase6_config_fingerprint) -> Dict`
- `save_experiment_log(record, output_path, logger) -> None` — atomic append

### §7.14 — `main.py`

**Responsibility:** 15-step pipeline orchestrator. Each step is numbered,
logged, and exits with `EXIT_ERROR_DECISION_ENGINE = 80` on any unhandled
exception.

```
Step  1  — Load project configuration
Step  2  — Compute config fingerprint; generate experiment ID
Step  3  — Load and validate Phase 6 contract (entry gate)
Step  4  — Load Phase 6 predictions
Step  5  — Load Phase 6 manifest (upstream lineage)
Step  6  — Validate risk threshold ordering
Step  7  — Aggregate channels to bearing-level (min + mean RUL)
Step  8  — Run decision pipeline (risk / policy / confidence / priority)
Step  9  — Apply business rules
Step 10  — Generate decision explanations
Step 11  — Assemble recommendation dataframe
Step 12  — Run final validation (10 checks)
Step 13  — Generate and save decision report
Step 14  — Generate and save Phase 7→8 contract
Step 15  — Generate and save decision manifest + decision log
```

EXIT code convention (inherited from prior phases):
- `EXIT_ERROR_DECISION_ENGINE = 80` added to `src/constants.py` between
  `EXIT_ERROR_MODEL_DEV = 70` and `EXIT_ERROR_UNKNOWN = 99`

---

## §8 — Decision Pipeline

The decision pipeline transforms one aggregated bearing row into a
`Recommendation`. The pipeline is strictly sequential — no step may influence
any earlier step.

```
Input: (bearing_id, timestamp, min_rul, mean_rul, channel_count)
                │
                ▼
   ┌────────────────────────────┐
   │ 1. risk_classifier         │
   │    RUL → RiskCategory      │
   └──────────────┬─────────────┘
                  │
                  ▼
   ┌────────────────────────────┐
   │ 2. maintenance_policy      │
   │    RiskCategory → Action   │
   └──────────────┬─────────────┘
                  │
                  ▼
   ┌────────────────────────────┐
   │ 3. confidence_estimator    │
   │    (model_type, n2,        │
   │     boundary_proximity)    │
   │    → ConfidenceLevel       │
   └──────────────┬─────────────┘
                  │
                  ▼
   ┌────────────────────────────┐
   │ 4. priority_calculator     │
   │    (RiskCategory,          │
   │     ConfidenceLevel)       │
   │    → int (1–5)             │
   └──────────────┬─────────────┘
                  │
                  ▼
   ┌────────────────────────────┐
   │ 5. business_rules          │
   │    overrides + fallbacks   │
   └──────────────┬─────────────┘
                  │
                  ▼
   ┌────────────────────────────┐
   │ 6. decision_explainer      │
   │    → explanation_text      │
   └──────────────┬─────────────┘
                  │
                  ▼
   ┌────────────────────────────┐
   │ 7. recommendation_engine   │
   │    assembles Recommendation│
   └────────────────────────────┘
```

### §8a — Decision Traceability

Every recommendation produced by Phase 7 must be traceable end-to-end: from
the raw Phase 6 prediction that fed the decision, through every module that
touched it, to the exact governance record that identifies the pipeline run
that produced it. This subsection documents that lineage explicitly. It adds
no new field and no new module — every element of the chain below already
exists in §5–§7, §12, §13, and §18; this section only makes the chain
explicit as a single, auditable lineage rather than something a reader must
reconstruct from separate module descriptions.

```
Phase 6 Prediction
  (bearing_id, channel, timestamp, predicted_rul_minutes)
  — source: reports/model_predictions.parquet, grain (bearing_id, channel, timestamp)
        │
        ▼
Channel Aggregation
  (bearing_id, timestamp, min_rul, mean_rul, channel_count)
  — §7.2 decision_engine.aggregate_channels
        │
        ▼
Risk Classification
  min_rul → RiskCategory
  — §7.3 risk_classifier.classify_risk, thresholds per §9d
        │
        ▼
Maintenance Policy
  RiskCategory → MaintenanceAction (base action, pre-override)
  — §7.4 maintenance_policy.get_maintenance_action
        │
        ▼
Confidence Assignment
  (model_type, n2_caveat, boundary_proximity) → ConfidenceLevel
  — §7.5 confidence_estimator.estimate_confidence
        │
        ▼
Priority Assignment
  (RiskCategory, ConfidenceLevel) → priority_score (1–5)
  — §7.6 priority_calculator.compute_priority
        │
        ▼
Business Rules Applied
  BR-01..BR-05 evaluated in precedence order (§12a)
  → final MaintenanceAction, final ConfidenceLevel, override flags/reasons
  — §7.9 business_rules.apply_business_rules
        │
        ▼
Final Recommendation
  fully assembled Recommendation row
  — §7.7 recommendation_engine.assemble_recommendation
        │
        ▼
Decision Explanation
  human-readable justification citing every step above
  — §7.8 decision_explainer.generate_explanation
        │
        ▼
Decision Manifest / Decision Log
  experiment_id, config_fingerprint, rule_version, phase6 upstream lineage
  — §18a decision_manifest.json, §18b decision_log.json
        │
        ▼
Phase 8
  consumes recommendations.parquet + phase7_decision_contract.yaml only
  — §21 Phase 7 → Phase 8 Contract
```

**Traceability guarantee:** given any single row of `recommendations.parquet`,
a reviewer can walk this chain backward — from `explanation_text` and
`override_reason`, to the business rule that fired, to the confidence and
priority inputs, to the risk category and threshold that triggered it, to the
`experiment_id`/`config_fingerprint` that identifies the exact Phase 6
prediction and Phase 7 configuration that produced it — without needing access
to source code. This is the same traceability discipline already required by
§13 (Explainability) and §18 (Governance); §8a exists only to present it as one
continuous, ordered chain.

---

## §9 — Risk Classification Framework

### §9a — Rationale for five-tier classification

Five risk tiers are used rather than three (Low/Medium/High) because:
1. CRITICAL and HIGH require fundamentally different response times (immediate
   vs. next window) — collapsing them would mask urgency
2. LOW and HEALTHY are operationally different — LOW warrants planned attention;
   HEALTHY warrants only continued monitoring
3. Five tiers are standard in reliability engineering practice (e.g., FMEA risk
   matrices) and are interview-defensible

### §9b — Threshold configuration philosophy

Threshold values are domain knowledge, not machine learning outputs. They must
be set by a qualified Reliability Engineer based on:

- **Critical threshold:** Time required to perform an emergency shutdown and
  prevent catastrophic bearing seizure (hours to days, depending on bearing
  criticality and equipment type)
- **High threshold:** Time required to mobilize a maintenance crew and complete
  an unplanned maintenance intervention
- **Medium threshold:** Time required to acquire replacement parts and schedule
  a planned maintenance window
- **Low threshold:** Time representing the outer boundary of routine monitoring
  vigilance — beyond this, the bearing is considered healthy for the current
  inspection cycle

The architecture provides a configuration schema with placeholder values. The
placeholder values are structural stubs only and must be replaced with
domain-validated values before any operational deployment.

### §9c — Threshold ordering constraint

The validation framework enforces:
```
0 < critical < high < medium < low
```
Any configuration violating this ordering causes a FATAL validation error
before any decision logic executes.

### §9d — Threshold Governance

Thresholds are the single most consequential configuration surface in Phase 7:
they convert a numerical RUL prediction into an operational maintenance
decision. Because they are domain knowledge rather than a computed artifact,
they require a governance process distinct from ordinary configuration
changes. This subsection documents that process. It does not assign values —
it defines how values are owned, approved, changed, and audited once they
exist.

#### §9d.1 — Threshold ownership

| Role | Responsibility |
|------|-----------------|
| **Threshold Owner** (Reliability Engineering) | Sole authority to propose or change `risk_thresholds` values in `config/config.yaml`. Owns the engineering justification for each value (§9b). |
| **Architecture Owner** (Principal Industrial AI Architect) | Verifies that any proposed threshold change conforms to the ordering constraint (§9c) and does not require a change to decision logic, risk category count, or module architecture. |
| **Pipeline Operator** | Executes the Phase 7 pipeline with the currently approved threshold set. Has no authority to modify thresholds. |

A threshold change is a **configuration change**, never a **code change**. If a
proposed change cannot be expressed purely as a value in `risk_thresholds`, it
is out of scope for a threshold update and must be routed back through a formal
architecture revision.

#### §9d.2 — Approval workflow

1. Threshold Owner drafts the proposed values with written engineering
   justification (mobilization time, safety margin, or scheduling horizon
   evidence per §9b).
2. Architecture Owner reviews the ordering constraint and confirms no
   downstream module requires modification to accept the new values.
3. Proposed values are recorded in a **Threshold Change Record** (a dated
   entry appended to the threshold changelog described in §9d.3) prior to
   being written into `config/config.yaml`.
4. Only after the Threshold Change Record exists may the values be written
   into the live configuration and the pipeline re-run.

This workflow requires no new tooling — it is a documentation and process
discipline layered on top of the existing configuration file and the existing
config-fingerprint mechanism (§18c).

#### §9d.3 — Threshold versioning

Threshold sets are versioned implicitly through the existing config
fingerprint mechanism (§18c): any change to `risk_thresholds` changes
`config_fingerprint`, which changes `experiment_id`. No separate threshold
version number is introduced, because introducing one would duplicate a
mechanism the architecture already has.

A **Threshold Change Record** is maintained as a simple changelog (a table in
`reports/decision_report.md` or a sibling file) with columns: date, prior
values, new values, justification, approver. This is documentation, not a new
deliverable — it uses the reporting artifact already defined in §6.

#### §9d.4 — Configuration governance

Threshold values live only in `config/config.yaml::decision_engine.risk_thresholds`.
They are never duplicated, hardcoded, or overridden elsewhere in source code —
this is already guaranteed by §7.3 (`risk_classifier.py` contains no hardcoded
values) and is reaffirmed here as a governance requirement, not merely an
implementation detail: any future code review that finds a threshold value
outside this configuration path is a governance violation, independent of
whether the value happens to be correct.

#### §9d.5 — Audit requirements

Every recommendation row already carries `config_fingerprint` (§18d). This is
sufficient to audit which threshold set produced any historical decision:
given a `config_fingerprint`, the corresponding `config.yaml` snapshot (retained
under standard version control practice for the project) reconstructs the
exact threshold values in force at decision time. No additional audit field is
required beyond what §18 already defines.

#### §9d.6 — Reproducibility after threshold updates

Reproducibility is defined **per configuration**, not globally: given a fixed
`config_fingerprint`, the decision engine is fully deterministic and
reproducible (§25, answer to Question 4). A threshold update produces a *new*
`config_fingerprint` and therefore a *new* experiment lineage — it does not
retroactively alter, invalidate, or require recomputation of any
already-produced `recommendations.parquet` under the prior fingerprint.
Reproducibility is preserved within each threshold version; it was never
claimed to hold *across* threshold versions, because a threshold change is by
definition a change in decision policy.

#### §9d.7 — Backward compatibility of historical decisions

Historical decisions are never mutated when thresholds change. Each Phase 7
pipeline run is a distinct, independently governed artifact identified by its
own `experiment_id` and `config_fingerprint` (§18a). A threshold update
requires a new pipeline run producing a new `recommendations.parquet`; the
prior run's output file, decision manifest, and decision log remain on disk
unchanged and remain internally consistent (every field in the prior file
still refers to the threshold set that was in force when it was produced).
Phase 8 or any downstream consumer must treat recommendations from different
`config_fingerprint` values as belonging to different decision policies and
must not silently merge or compare them (this expectation is already stated in
the Phase 7→8 contract, §21, `must_not_compare_experiments_with_different_config_fingerprints`).

---

## §10 — Maintenance Decision Framework

### §10a — Mapping completeness

Every possible risk category maps to exactly one maintenance action. There are
no unmapped categories and no categories that map to multiple actions. The
mapping is total and functional.

### §10b — Override safety

Business rule BR-01 ensures that CRITICAL risk cannot be downgraded by any
subsequent logic. The action IMMEDIATE_SHUTDOWN is protected from override
by any confidence-based or context-based reasoning. This is the safety floor.

### §10c — Action semantics

| Action | Operational Meaning |
|--------|---------------------|
| IMMEDIATE_SHUTDOWN | Stop the equipment now. Do not await the next maintenance window. |
| URGENT_MAINTENANCE | Complete bearing replacement at the next available maintenance window, bringing it forward if necessary. |
| SCHEDULE_MAINTENANCE | Add bearing replacement to the near-term maintenance schedule. Do not skip the next planned inspection. |
| ROUTINE_MONITORING | Confirm bearing health at the next scheduled inspection. No immediate action required. |
| NORMAL_OPERATION | Continue standard operating schedule. No specific maintenance action needed. |

---

## §11 — Confidence Estimation Framework

### §11a — Why categorical confidence, not probability

The n=2 RTF training corpus does not support calibrated probability estimation.
A model trained on 2 bearings and evaluated on 2 bearings cannot produce a
probability of failure that generalizes to an unseen fleet. Reporting a
numerical confidence probability (e.g., "87% confident") would be scientifically
misleading.

HIGH / MEDIUM / LOW is the honest confidence vocabulary for this dataset size.

### §11b — Confidence is read from the Phase 6 contract, not re-estimated

Phase 7 does not re-estimate model quality. It reads the declared properties
from the Phase 6 contract (`model_type`, `n2_caveat`, `mean_mae_minutes`,
`stability_ratio`) and translates them into a confidence level per the rules in
§7.5. This is the correct separation of concerns.

### §11c — Honesty requirement

All consumer-facing outputs must communicate:
1. The model type and its limitations
2. The n=2 caveat
3. The confidence level and its basis
4. The recommendation that a physical inspection be prioritized for CRITICAL and
   HIGH decisions

---

## §12 — Business Rules Framework

### §12a — Business Rule Precedence

The five business rules (BR-01 through BR-05, §7.9a) are not evaluated as an
unordered set — they are evaluated in a fixed, deterministic sequence, and
each rule's authority over the decision is explicitly ranked. This subsection
documents that precedence completely. No new rule is introduced here; this is
an explicit statement of an ordering that the architecture already implies
through the rule descriptions in §7.9a.

#### §12a.1 — Execution order

Rules execute in this fixed order on every recommendation, regardless of which
rules ultimately fire:

| Order | Rule | Category | Can be overridden by a later rule? |
|-------|------|----------|--------------------------------------|
| 1 | **BR-01** Safety floor (CRITICAL → IMMEDIATE_SHUTDOWN) | Action override | No |
| 2 | **BR-05** Zero-RUL floor (RUL=0 → CRITICAL) | Risk override | No |
| 3 | **BR-02** Missing prediction fallback (→ HIGH risk) | Risk override (absent-data path) | No — mutually exclusive with 1/2/4/5 (fires only when no prediction exists at all) |
| 4 | **BR-03** Channel disagreement (confidence downgrade) | Confidence modifier | Cannot be overridden by BR-04; cannot override BR-01/02/05 |
| 5 | **BR-04** Constant-predictor annotation | Explanation annotation | Never overrides anything — annotation only |

#### §12a.2 — Override hierarchy

The five rules fall into three tiers of authority, from highest to lowest:

**Tier 1 — Decision-altering, non-overridable (BR-01, BR-05).**
These rules change the final `risk_category` and/or `maintenance_action` and
cannot be reversed by any rule that runs after them. They exist to enforce a
safety floor: no confidence level, no channel disagreement, and no annotation
logic may ever downgrade a CRITICAL classification or a zero-RUL reading.

**Tier 2 — Decision-substituting under missing evidence (BR-02).**
This rule only fires in the absence of a prediction — a structurally distinct
condition from Tier 1 (which fires on the *presence* of an extreme prediction).
Because BR-02 only applies when no prediction row exists for a bearing at all,
it cannot conflict with BR-01 or BR-05, which require a prediction value to
evaluate against. BR-02 is therefore precedence-independent from Tier 1: the
two conditions are mutually exclusive by construction, not by rule ordering.

**Tier 3 — Modifying and annotating, non-decision-altering (BR-03, BR-04).**
These rules never change `risk_category` or `maintenance_action`. BR-03
modifies `confidence_level` only. BR-04 modifies `explanation_text` only. Both
can coexist with any Tier 1 or Tier 2 outcome without conflict, because they
operate on different output fields.

#### §12a.3 — Conflict resolution examples

**Example 1 — BR-01 vs. BR-03 (safety floor vs. confidence modifier).**
A bearing is classified CRITICAL (min_rul below the critical threshold) *and*
exhibits high inter-channel disagreement (BR-03 condition also true).
Resolution: BR-01 fixes `maintenance_action = IMMEDIATE_SHUTDOWN` and this is
never touched by BR-03. BR-03 is still permitted to fire and downgrade
`confidence_level` by one grade, and the explanation records both the CRITICAL
safety floor and the channel-disagreement caveat. There is no true conflict:
BR-01 owns the action field, BR-03 owns the confidence field. Both apply
simultaneously without contradiction.

**Example 2 — BR-01 vs. BR-05 (both decision-altering, both true).**
A prediction is exactly 0 minutes RUL. This value is already below any
sensible `critical_minutes` threshold, so BR-01's classification and BR-05's
floor agree on CRITICAL by construction. Because thresholds are constrained to
be positive (§9c: `0 < critical < ...`), any RUL of 0 is always ≤ critical, so
this is not an actual conflict — BR-05 is a defensive floor for a case BR-01
already covers, guaranteeing CRITICAL even if a future threshold configuration
error somehow permitted `critical_minutes` to be zero or negative and bypass
threshold ordering validation (§9c's ordering check as the primary defense;
BR-05 as the redundant safety net).

**Example 3 — BR-02 vs. BR-03/BR-04 (missing data vs. modifiers).**
A bearing has no prediction row (BR-02 fires). Because there is no
`min_predicted_rul_minutes` value and no per-channel spread to evaluate, BR-03
cannot evaluate its spread condition and does not fire. BR-04 evaluates
`model_type` independent of the prediction value, so a constant-predictor
annotation can still be attached even to a BR-02 fallback row (the annotation
is about the *model*, not this specific prediction). There is no conflict:
BR-03 is structurally inapplicable when BR-02 fires; BR-04 is independent of
both.

**Example 4 — BR-03 vs. BR-04 (two Tier-3 rules, both true).**
A bearing produced by the `median` model (BR-04 condition true) also shows
high channel disagreement (BR-03 condition true). Both fire: confidence is
downgraded once by BR-03, and the explanation carries both the
constant-predictor note and the channel-disagreement note. Tier-3 rules are
additive by design — they modify disjoint output fields (`confidence_level`
and `explanation_text` annotations respectively) and never need arbitration
between each other.

#### §12a.4 — Deterministic behaviour

Because (a) the execution order is fixed, (b) Tier 1 rules are structurally
non-overridable, (c) Tier 2 is mutually exclusive with Tier 1 by construction,
and (d) Tier 3 rules operate on disjoint fields, the outcome of business rule
application is a pure function of the input row and the configuration
(`config_fingerprint`). The same `(bearing_id, timestamp, predictions,
config)` input always produces the same business-rule outcome — there is no
ordering ambiguity, no rule execution order that depends on runtime state, and
no randomness. This determinism is the same property already claimed for the
decision pipeline as a whole in §25, Question 4; §12a documents specifically
why the business rule layer preserves it.

### §12b — Override audit trail

Every override is recorded in:
- `is_business_rule_override = True` in the recommendation row
- `override_reason` field with a prose description
- The decision manifest lists all override counts by rule ID

### §12c — Fallback correctness under model failure

If the Phase 6 model produces NaN predictions for a bearing (which should be
caught by Phase 6 validation but is also guarded here), business rule BR-02
treats the bearing as HIGH risk. NaN predictions are never forwarded to the
decision engine or appear in `recommendations.parquet`.

---

## §13 — Explainability Framework

### §13a — Principles

Every recommendation must be explainable to:

1. **A maintenance engineer** — "What should I do and why?"
2. **A reliability manager** — "What is the evidence and confidence?"
3. **A Principal Reliability Engineer** — "Is the recommendation scientifically defensible?"
4. **An auditor** — "What model, what data, what rules produced this recommendation?"

### §13b — Evidence sources

The explanation draws from:

| Evidence Source | Field |
|-----------------|-------|
| Predicted RUL (min across channels) | `min_predicted_rul_minutes` |
| Risk threshold that was triggered | `risk_category` config threshold |
| Model type and selection note | Phase 6 contract `model.type` |
| Evaluation performance | Phase 6 contract `performance.mean_mae_minutes` |
| n=2 caveat text | Phase 6 contract `performance.n2_caveat_text` |
| Business rule override reason | `override_reason` |
| Channel count and spread | `channel_count`, spread computed in aggregation |

### §13c — SHAP in explanations

Phase 6 generated SHAP explanations (`reports/feature_importance_report.md`).
Phase 7 may reference the top Phase 6 SHAP-ranked features by name in the
explanation prose. However:

- Phase 7 does NOT re-run SHAP
- Phase 7 does NOT import `shap`
- Phase 7 references SHAP feature names as static strings read from the Phase 6
  contract `feature_importance.top_permutation_ranked_features`
- This is interpretation-only; it does not influence the decision

When `model_type in ("mean", "median")`, SHAP feature references are suppressed
because a constant-predictor model's "feature importance" is not meaningful for
decision explanation. The explanation notes this explicitly.

### §13d — Mandatory explanation checklist

Every `explanation_text` generated by `decision_explainer.py` (§7.8) must
answer all seven of the following questions. This is a strengthening of the
existing template (§7.8) into an explicit, checkable contract — no new content
is added to the explanation that was not already specified by the template; this
subsection exists so the completeness of the template can be verified
independently of reading its prose form.

| # | Question | Answered by (existing field / source) |
|---|----------|------------------------------------------|
| 1 | What prediction produced this decision? | `min_predicted_rul_minutes` (and `mean_predicted_rul_minutes`, `channel_count` for context) — §7.8 "Risk Assessment" block |
| 2 | Which business rules fired? | `is_business_rule_override`, `override_reason` — §7.8 "Override Note" block, per the precedence chain in §12a |
| 3 | Which rule determined the final recommendation? | The highest-precedence rule that altered the action or risk category (Tier 1/2 per §12a.2), named explicitly in `override_reason` when an override occurred; when no override rule fires, the explanation states the recommendation followed the base risk→action mapping (§10a) unmodified |
| 4 | Which evidence was used? | The full §13b evidence-source table — RUL value, threshold triggered, channel spread |
| 5 | Which assumptions were applied? | The n=2 caveat text (`n2_caveat_text`), the constant-predictor note when applicable (BR-04, §7.9a), and the confidence basis (`_build_confidence_basis`, §7.8) |
| 6 | Which model produced the prediction? | `model_type`, model selection note (`_build_model_selection_note`, §7.8) |
| 7 | Which experiment generated the model? | `experiment_id` (Phase 6, propagated via `phase6_upstream` lineage, §18a) |

This checklist strengthens traceability only — it does not add a field to the
`Recommendation` schema (§6a) or a new explanation section beyond what §7.8
already defines. Every answer already has a home in the existing template; this
table is the acceptance criterion a reviewer uses to confirm the template was
implemented completely.

---

## §14 — Data Flow

```
reports/model_predictions.parquet
  (bearing_id, channel, timestamp, predicted_rul_minutes)
  8,624 rows × 4 columns
         │
         ▼ groupby(bearing_id, timestamp).agg(min, mean, count)
         │
bearing-level dataframe
  (bearing_id, timestamp, min_rul, mean_rul, channel_count)
  ~1,078 rows (2 bearings × ~539 timestamps each)
  [exact count depends on Phase 5 file set; architecture is robust to actual count]
         │
         ▼ decision pipeline (per row)
         │
List[Recommendation]  (one per bearing-timestamp row)
         │
         ├──► recommendations.parquet   (all rows, all columns)
         ├──► decision_report.md        (human-readable summary)
         ├──► phase7_decision_contract.yaml
         ├──► decision_manifest.json
         └──► decision_log.json
```

---

## §15 — Configuration

The Phase 7 configuration block is added to `config/config.yaml` as the
`decision_engine` key. All thresholds, weights, and rule parameters are
declared here. No decision logic is hardcoded.

```yaml
decision_engine:

  # Risk classification thresholds (minutes).
  # VALUES ARE DOMAIN-EXPERT PLACEHOLDERS — must be validated by a
  # qualified Reliability Engineer before operational use.
  risk_thresholds:
    critical_minutes: null    # e.g., minimum mobilization time for emergency shutdown
    high_minutes: null        # e.g., minimum unplanned maintenance window lead time
    medium_minutes: null      # e.g., planned maintenance scheduling horizon
    low_minutes: null         # e.g., outer monitoring vigilance boundary

  # Confidence estimation parameters
  confidence:
    boundary_proximity_tolerance_minutes: null  # e.g., 10% of critical threshold
    model_type_modifiers:                       # base confidence by model type
      mean: "LOW"
      median: "LOW"
      linear: "MEDIUM"
      random_forest: "HIGH"
      xgboost: "HIGH"

  # Business rules
  business_rules:
    channel_rul_spread_threshold_minutes: null  # spread above which BR-03 fires
    missing_prediction_default_risk: "HIGH"

  # Explanation formatting
  explainability:
    max_top_features_in_explanation: 5

  # Priority matrix (rows = risk category, cols = confidence level)
  # Values must be integers 1–5
  priority_matrix:
    CRITICAL:
      HIGH: 1
      MEDIUM: 1
      LOW: 1
    HIGH:
      HIGH: 2
      MEDIUM: 2
      LOW: 3
    MEDIUM:
      HIGH: 3
      MEDIUM: 3
      LOW: 4
    LOW:
      HIGH: 4
      MEDIUM: 4
      LOW: 5
    HEALTHY:
      HIGH: 5
      MEDIUM: 5
      LOW: 5

  # Maintenance action mapping
  policy_map:
    CRITICAL: "IMMEDIATE_SHUTDOWN"
    HIGH: "URGENT_MAINTENANCE"
    MEDIUM: "SCHEDULE_MAINTENANCE"
    LOW: "ROUTINE_MONITORING"
    HEALTHY: "NORMAL_OPERATION"

  # Business rule version (must match RULE_VERSION in business_rules.py)
  rule_version: "1.0"
```

**Implementation note:** The `null` placeholder values for thresholds and the
`channel_rul_spread_threshold_minutes` must be replaced with domain-validated
values. The pipeline will raise a FATAL error if any threshold required for
classification remains `null` at runtime.

---

## §16 — Validation Strategy

Validation follows the `ValidationCheck / FinalValidationResult` pattern
established in Phase 4, Phase 5, and Phase 6.

### §16a — Ten validation checks

| ID | Name | Severity | Description |
|----|------|----------|-------------|
| DV1 | `phase6_contract_entry_gate` | FATAL | Phase 6 contract schema version and n2_caveat present |
| DV2 | `all_bearings_present` | FATAL | Both bearing_3 and bearing_4 appear in output |
| DV3 | `no_nan_rul` | FATAL | No NaN in `min_predicted_rul_minutes` |
| DV4 | `no_negative_rul` | FATAL | No negative `min_predicted_rul_minutes` |
| DV5 | `valid_risk_categories` | FATAL | All values in {CRITICAL,HIGH,MEDIUM,LOW,HEALTHY} |
| DV6 | `valid_priority_scores` | FATAL | All values in {1,2,3,4,5} |
| DV7 | `n2_caveat_propagated` | WARNING | `n2_caveat == True` in every row |
| DV8 | `critical_rows_priority_1` | WARNING | All CRITICAL rows have `priority_score == 1` |
| DV9 | `governance_fields_in_manifest` | WARNING | `experiment_id`, `config_fingerprint`, `rule_version` present |
| DV10 | `channel_count_matches_grain` | INFO | Channel counts consistent with Phase 6 prediction grain |

### §16b — Exit policy

- Any FATAL failure → pipeline exits with `EXIT_ERROR_DECISION_ENGINE = 80`
- WARNING failures → pipeline continues, warnings recorded in manifest
- INFO → recorded, no action

### §16c — Decision Output Validation

§16a validates artifact integrity (schema, presence, non-null, non-negative).
This subsection strengthens that validation by explicitly stating the PASS /
WARNING / FATAL criteria for internal *consistency* of the generated
decisions themselves — i.e., whether the combination of fields on a single
`Recommendation` row is a combination the decision pipeline (§8) could
actually have produced. Every scenario below is already prevented or caught by
an existing check (DV2, DV5, DV6, DV8) or an existing structural guarantee
(§9a mapping totality, §12a precedence); this subsection exists to state the
PASS/WARNING/FATAL criteria explicitly rather than leave them implicit in the
check descriptions.

| Scenario | Criterion | Severity | Covered by |
|----------|-----------|----------|------------|
| **Impossible risk/priority combination** — e.g. a `CRITICAL` row with `priority_score != 1`, or a `HEALTHY` row with `priority_score == 1` | FATAL if any `CRITICAL` row has `priority_score != 1` (violates the non-overridable safety floor, §12a.2 Tier 1); WARNING if any other risk/priority pairing falls outside the declared priority matrix cells (§7.6) | CRITICAL case: FATAL (structural safety violation). Other mismatches: WARNING (matrix drift, not a safety violation) | DV8 (existing FATAL-adjacent check, formalized here as FATAL for the CRITICAL case specifically) |
| **Missing recommendation** — a bearing/timestamp pair present in Phase 6 predictions has no corresponding row in `recommendations.parquet` | FATAL — every bearing present in the Phase 6 artifact set must produce exactly one recommendation row per timestamp; a silent drop is a pipeline defect, not a business-rule outcome (BR-02 produces a *substitute* row, never a *missing* one) | FATAL | DV2 (extended: DV2 already checks bearing presence; this scenario is the timestamp-level generalization of the same check) |
| **Confidence without prediction** — a row has a non-null `confidence_level` but `min_predicted_rul_minutes` is NaN, or is a BR-02 fallback row with a confidence level other than LOW | FATAL for NaN-with-confidence (a confidence estimate must never be attached to a value that failed DV3); WARNING if a BR-02 fallback row reports confidence other than LOW (§7.9a BR-02 always assigns LOW; any other value indicates the fallback path was not applied correctly) | FATAL / WARNING as described | DV3 (NaN case) plus the BR-02 rule definition (§7.9a) |
| **Contradictory recommendation** — e.g. `maintenance_action == NORMAL_OPERATION` while `risk_category == CRITICAL`, or `is_business_rule_override == False` while `override_reason` is non-null (or the reverse) | FATAL — the risk→action mapping (§9a) is total and functional; any row where the action does not match the risk category's declared mapping (after any legitimate BR-01 override) indicates a broken pipeline invariant, not a valid decision | FATAL | DV5 + DV6 jointly (valid category and valid action are necessary but not sufficient; this scenario is the joint consistency check between the two) |
| **Invalid business-rule combination** — e.g. both `BR-01` and `BR-02` recorded as having fired on the same row (structurally impossible per §12a.2 Tier 1/Tier 2 mutual exclusivity), or `override_reason` naming a rule ID that does not exist in the five defined rules (§7.9a) | FATAL — this can only occur from a coding defect in `business_rules.py`, since the rule engine is designed to make such combinations unreachable (§12a.4 determinism); its occurrence indicates the precedence guarantee itself has been violated | FATAL | New explicit statement — not previously named as a check, but already structurally impossible per §12a; stated here as an acceptance criterion for the eventual implementation's test suite (§17c) rather than as a new runtime check |

This subsection adds no new module, no new artifact, and no new pipeline
stage. It restates existing check outcomes (DV2, DV3, DV5, DV6, DV8) and
existing structural guarantees (§9a, §12a) as explicit acceptance criteria for
decision-level consistency, and directs the future test suite (§17c,
`TestBusinessRules`) to assert that the "invalid business-rule combination"
scenario is unreachable rather than merely untested.

---

## §17 — Testing Strategy

All Phase 7 tests reside in `tests/test_decision_engine.py`.

### §17a — Unit tests

| Test Class | Tests | What it verifies |
|------------|-------|------------------|
| `TestRiskClassifier` | 8 | One test per boundary + each category + null threshold FATAL |
| `TestMaintenancePolicy` | 5 | Each risk category maps to correct action |
| `TestConfidenceEstimator` | 9 | Model-type modifier, n2 penalty, boundary proximity, combined |
| `TestPriorityCalculator` | 15 | All 15 cells of the priority matrix |
| `TestBusinessRules` | 8 | BR-01 through BR-05 + override audit trail |
| `TestChannelAggregation` | 4 | min/mean/count correct, single-channel edge case |
| `TestDecisionExplainer` | 4 | Template completeness, constant-predictor note, n2 text |
| `TestContractLoader` | 6 | Entry gate FATALs, valid contract passes |
| `TestThresholdValidation` | 4 | Ordering violation raises ValueError |

### §17b — Integration tests

| Test | What it verifies |
|------|------------------|
| `test_full_pipeline_on_phase6_artifacts` | End-to-end pipeline with actual Phase 6 artifacts → 0 FATAL checks |
| `test_decision_pipeline_deterministic` | Run twice with same config → identical recommendations |
| `test_config_fingerprint_changes_on_threshold_change` | Changing any threshold changes fingerprint |

### §17c — Business rule tests

| Test | What it verifies |
|------|------------------|
| `test_critical_is_priority_1_regardless_of_confidence` | Every CRITICAL row → priority 1 |
| `test_missing_bearing_gets_high_risk_fallback` | Drop bearing from predictions → BR-02 fires |
| `test_zero_rul_always_critical` | RUL=0 → CRITICAL regardless of threshold |
| `test_channel_spread_downgrades_confidence` | Large channel spread → confidence -1 grade |
| `test_constant_predictor_note_in_explanation` | `model_type=median` → BR-04 annotation present |

### §17d — Decision consistency tests

| Test | What it verifies |
|------|------------------|
| `test_rul_just_above_threshold_is_lower_risk` | RUL = threshold + 1 → one tier lower than RUL = threshold − 1 |
| `test_same_rul_same_model_same_decision` | Identical rows → identical decisions |
| `test_override_recorded_in_output` | BR-01 override → `is_business_rule_override=True` in row |

### §17e — Contract tests

| Test | What it verifies |
|------|------------------|
| `test_phase8_contract_schema_version` | `schema_version == "1.0"` |
| `test_phase8_contract_has_n2_caveat` | `n2_caveat` present and True |
| `test_phase8_contract_has_consumer_expectations` | All required consumer_expectations keys present |
| `test_phase8_contract_grain_matches_recommendations` | Contract grain field matches output grain |

**Minimum coverage target:** 80% line coverage across all `src/decision_engine/`
modules. Coverage report generated during CI run.

---

## §18 — Governance

### §18a — Decision Manifest

`reports/decision_manifest.json` contains:

```json
{
  "phase": "7",
  "module": "decision_engine",
  "experiment_id": "dec_{config_fp[:12]}_rv{rule_version}",
  "config_fingerprint": "<sha256 of decision_engine config block>",
  "rule_version": "1.0",
  "git_commit": "<sha or 'unknown'>",
  "environment": { "<library>": "<version>" },
  "phase6_upstream": {
    "experiment_id": "<phase6 experiment_id>",
    "config_fingerprint": "<phase6 config_fingerprint>",
    "model_type": "<selected model>",
    "model_fingerprint": "<phase6 refit_model_fingerprint>"
  },
  "decision_summary": {
    "total_recommendations": "<int>",
    "bearing_count": 2,
    "risk_category_counts": { "CRITICAL": 0, "HIGH": 0, ... },
    "business_rule_override_count": "<int>",
    "override_counts_by_rule": { "BR-01": 0, "BR-02": 0, ... }
  },
  "statistical_limitations": {
    "n_rtf_units": 2,
    "n2_caveat": true,
    "model_selection_outcome": "demotion_fallback",
    "confidence_basis": "categorical_only_n2_insufficient_for_probabilities"
  },
  "produced_at": "<iso8601 utc timestamp>"
}
```

### §18b — Decision Log

`reports/decision_log.json` is an append-only JSON list of execution records.
Each record contains the experiment ID, config fingerprint, rule version,
timestamp, and a summary of decision counts. Atomic append via `.tmp` +
`Path.replace()`.

### §18c — Config fingerprint

`compute_config_fingerprint(config)` computes SHA-256 of the `decision_engine`
block serialized as `json.dumps(sort_keys=True)`. This ensures that any change
to thresholds, rule parameters, or priority matrix produces a distinct
experiment ID — making it impossible to silently compare decisions produced
under different configurations.

### §18d — Recommendation provenance

Every row in `recommendations.parquet` carries:
- `experiment_id` — links to the decision manifest
- `config_fingerprint` — links to the exact configuration used
- `rule_version` — links to the business rule set
- `model_type` — from Phase 6 contract (not re-derived)
- `n2_caveat` — always True; required presence verified by DV7

---

## §19 — Failure Handling

### §19a — Fatal failures (pipeline exits `EXIT_ERROR_DECISION_ENGINE = 80`)

| Condition | Handler |
|-----------|---------|
| Phase 6 contract missing or unreadable | FATAL: log path, exit 80 |
| Phase 6 contract fails entry-gate validation | FATAL: log violation, exit 80 |
| `model_predictions.parquet` missing | FATAL: log path, exit 80 |
| Phase 6 manifest missing | FATAL: log path, exit 80 |
| Risk threshold ordering violated | FATAL: log offending values, exit 80 |
| Null threshold at classification time | FATAL: log field name, exit 80 |
| Any FATAL validation check fails | FATAL: log check name, exit 80 |

### §19b — Warning conditions (pipeline continues)

| Condition | Handler |
|-----------|---------|
| Phase 6 feature importance report missing | WARNING: explanations omit feature evidence |
| Channel count < 8 for some bearing-timestamp | WARNING: log affected rows |
| All decisions are same risk category (expected with median model) | WARNING: logged with explanation |
| Business rule override count > 50% of rows | WARNING: logged for review |

### §19c — Atomic writes

All output files use `.tmp` + `Path.replace()` pattern (inherited from Phases
3–5). A partial write cannot corrupt a previously-committed artifact.

---

## §20 — Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| R1: Threshold values are not domain-validated | High (placeholder values) | High (decisions are meaningless) | Architecture requires null → validated replacement before operational use; pipeline raises FATAL on null thresholds |
| R2: Median model produces uniform decisions | Certain (known demotion fallback) | Medium (less actionable time series) | BR-04 mandatory annotation; architecture explicitly documents this; Phase 8 must display the annotation |
| R3: n=2 caveat ignored by Phase 8 consumer | Medium | High (false confidence in decisions) | `must_honour_n2_caveat` in Phase 7→8 contract consumer_expectations; DV7 validation check |
| R4: Config fingerprint drift between runs | Low (deterministic) | Medium (silent comparison of incompatible outputs) | Config fingerprint embedded in every recommendation row |
| R5: Rule version mismatch between runs | Low | Medium (comparison across incompatible rule sets) | Rule version embedded in every recommendation row and manifest |
| R6: SHAP feature names referenced in explanations no longer match current features | Low (Phase 6 frozen) | Low (prose only, no decision impact) | Features read from Phase 6 contract which is frozen; documented as explanation-only |

---

## §21 — Phase 7 → Phase 8 Contract

`reports/phase7_decision_contract.yaml` is the sole machine-readable interface
Phase 8 may consume from Phase 7.

### §21a — Schema

```yaml
schema_version: "1.0"
producer:
  phase: "7"
  module: "decision_engine"
  experiment_id: "<dec_...>"
  config_fingerprint: "<sha256>"
  rule_version: "1.0"
  git_commit: "<sha or unknown>"

upstream_lineage:
  phase6_experiment_id: "<exp_...>"
  phase6_config_fingerprint: "<sha256>"
  phase6_model_type: "median"
  phase6_model_fingerprint: "<sha256>"
  phase6_n2_caveat: true

decision_schema:
  grain: [bearing_id, timestamp]
  risk_categories: [CRITICAL, HIGH, MEDIUM, LOW, HEALTHY]
  maintenance_actions:
    - IMMEDIATE_SHUTDOWN
    - URGENT_MAINTENANCE
    - SCHEDULE_MAINTENANCE
    - ROUTINE_MONITORING
    - NORMAL_OPERATION
  confidence_levels: [HIGH, MEDIUM, LOW]
  priority_range: [1, 5]

artifact_paths:
  recommendations: "reports/recommendations.parquet"
  decision_report: "reports/decision_report.md"
  decision_manifest: "reports/decision_manifest.json"

performance_context:
  phase6_mean_mae_minutes: <float>
  phase6_stability_ratio: <float>
  n2_caveat: true
  n2_caveat_text: "<propagated from Phase 6 contract>"
  model_type: "median"
  model_selection_outcome: "demotion_fallback"

statistical_limitations:
  n_rtf_units: 2
  confidence_type: "categorical_only"
  probability_estimates: false
  extrapolation_permitted: false

consumer_expectations:
  must_honour_n2_caveat: true
  must_display_confidence_level_with_every_recommendation: true
  must_display_model_type_and_selection_outcome: true
  must_not_aggregate_across_rule_versions: true
  must_not_compare_experiments_with_different_config_fingerprints: true
  must_honour_business_rule_override_flags: true
  must_not_use_shap_outputs_for_model_modification: true
  apply_domain_expert_validated_thresholds_before_operational_use: true
```

### §21b — Contract versioning

`schema_version: "1.0"` is the initial version. Any change to the schema
(new fields, removed fields, changed semantics) must increment the schema
version and be documented in the contract changelog.

### §21c — Explicit artifact enumeration

The contract in §21a is not the only surface Phase 8 consumes — the contract
*references* `recommendations.parquet` and `decision_manifest.json` rather
than duplicating their full content inline. This subsection enumerates, in
one place, every artifact and every field Phase 8 receives from Phase 7,
tracing each back to where it is already defined elsewhere in this document.
No artifact listed below is new; this is a completeness index over §6a, §18,
and §21a.

| Transferred Item | Source Field(s) | Defined In |
|-------------------|------------------|------------|
| **Maintenance recommendation** | `maintenance_action` | `recommendations.parquet` row, §6a |
| **Confidence category** | `confidence_level` | `recommendations.parquet` row, §6a; basis defined in §11 |
| **Risk category** | `risk_category` | `recommendations.parquet` row, §6a; classification defined in §9 |
| **Priority** | `priority_score` | `recommendations.parquet` row, §6a; matrix defined in §7.6 |
| **Explanation** | `explanation_text` | `recommendations.parquet` row, §6a; template defined in §7.8, completeness checklist in §13d |
| **Rule trace** | `is_business_rule_override`, `override_reason` | `recommendations.parquet` row, §6a; precedence and firing logic defined in §12a, audit trail in §12b |
| **Decision manifest** | Full `decision_manifest.json` payload | §18a — includes governance summary, decision counts, override counts by rule ID, statistical limitations |
| **Configuration fingerprint** | `config_fingerprint` | Present on every recommendation row (§6a, §18d) *and* in the contract producer block (§21a); governs reproducibility per §9d.6/§9d.7 |
| **Decision version** | `rule_version` (business rule set version) + `schema_version` (contract schema version) | `rule_version`: §7.9b, present on every row and in the manifest; `schema_version`: §21a/§21b, contract-level only |
| **Producer metadata** | `producer.phase`, `producer.module`, `producer.experiment_id`, `producer.git_commit` | §21a `producer` block; mirrors the governance fields defined for Phase 6 in the upstream contract |
| **Consumer expectations** | `consumer_expectations.*` (8 declared expectations) | §21a `consumer_expectations` block, enumerated in full below |
| **Validation requirements** | Implicit: Phase 8 must not consume a `recommendations.parquet` that failed any FATAL check in §16a/§16c | §16a (10 checks), §16c (decision-consistency criteria); the contract's `consumer_expectations` block is the mechanism by which these requirements are communicated to Phase 8 |

**Full enumeration of `consumer_expectations` (already declared in §21a, restated
here for completeness against the list above):**

1. `must_honour_n2_caveat`
2. `must_display_confidence_level_with_every_recommendation`
3. `must_display_model_type_and_selection_outcome`
4. `must_not_aggregate_across_rule_versions`
5. `must_not_compare_experiments_with_different_config_fingerprints`
6. `must_honour_business_rule_override_flags`
7. `must_not_use_shap_outputs_for_model_modification`
8. `apply_domain_expert_validated_thresholds_before_operational_use`

No consumer expectation is added or removed by this subsection — items 1–8
are exactly the set already present in §21a's `consumer_expectations` block.

---

## §22 — Implementation Notes

| Note | Detail |
|------|--------|
| IN-01 | `experiment_id` format for Phase 7 is `dec_{config_fp[:12]}_rv{rule_version}` rather than the Phase 6 `exp_{fp[:12]}_seed{seed}` convention. Decisions are deterministic (no random seed); rule version replaces seed as the discriminating suffix. |
| IN-02 | The `decision_engine.py` module is the orchestrator within `src/decision_engine/`; `main.py` is the top-level pipeline entry point. This mirrors the Phase 5/6 pattern where a `main.py` drives module-level orchestrators. |
| IN-03 | `recommendations.parquet` grain is `(bearing_id, timestamp)` — one row per bearing per timestamp. This is different from Phase 6 predictions grain `(bearing_id, channel, timestamp)`. The grain change is achieved by channel aggregation in Step 7 of `main.py`. |
| IN-04 | `model_type = "median"` (demotion fallback) means all predictions for a given bearing are identical (the same constant). The time series of decisions for each bearing is therefore uniform. BR-04 annotates every recommendation with this fact. This is honest. |
| IN-05 | Phase 7 does NOT import `shap`. Feature names referenced in explanations come from `phase6_model_contract.yaml::feature_importance.top_permutation_ranked_features` as static strings. |
| IN-06 | Null thresholds in config raise a FATAL validation error. The Phase 7 architecture deliberately defers threshold specification to domain experts. |
| IN-07 | `EXIT_ERROR_DECISION_ENGINE = 80` is added to `src/constants.py` between `EXIT_ERROR_MODEL_DEV = 70` and `EXIT_ERROR_UNKNOWN = 99`, continuing the 10-increment exit code convention. |
| IN-08 | Atomic file writes use `.tmp` + `Path.replace()` in all output writers, consistent with Phases 3–6. |
| IN-09 | The `decision_report.md` includes a mandatory header section titled "Statistical Limitations" that appears before any decision table. This section must not be removed or reduced in Phase 8 display. |

---

## §23 — Out of Scope

The following are explicitly excluded from Phase 7:

| Excluded Item | Phase |
|---------------|-------|
| Dashboard and data visualisation | Phase 8 |
| Real-time alerting | Phase 8 |
| Web API / REST interface | Phase 8 |
| Email / SMS notifications | Phase 8 |
| Fleet-level aggregation beyond 2 RTF units | Out of project scope |
| Re-training or fine-tuning any model | Phase 6 (frozen) |
| Threshold optimisation via ML | Not in scope — domain expertise required |
| Calibrated probability estimates | Not scientifically supported (n=2) |
| Survival analysis or hazard modelling | Not in scope |

---

## §24 — Architecture Assumptions

| ID | Assumption |
|----|------------|
| A1 | Phase 6 artifacts are frozen and will not be modified after Phase 7 implementation begins |
| A2 | `reports/model_predictions.parquet` contains predictions for both bearing_3 and bearing_4 with grain (bearing_id, channel, timestamp) |
| A3 | All 8 channels are present for each (bearing_id, timestamp); channel count variations are handled by BR-03 if spread is large |
| A4 | The decision engine runs after Phase 6 and before Phase 8; it does not run in real-time or streaming mode |
| A5 | Threshold values will be populated by a domain expert before any operational use; the implementation report will flag which thresholds remain as null placeholders |
| A6 | The Phase 6 experiment that produced `model_predictions.parquet` will not be re-run during Phase 7 implementation |
| A7 | `git_commit` may be "unknown" (no `.git` repository in this project) — this is acceptable, consistent with Phase 6 treatment |
| A8 | `model_type = "median"` (demotion fallback) is the current Phase 6 outcome; the Phase 7 architecture is written to be model-type-agnostic so that a future Phase 6 re-run with a promoted model produces valid Phase 7 decisions without code changes |

---

## §25 — Self-Review

### 1. Would you approve this architecture before implementation?

Yes. The architecture is complete, internally consistent, and follows all
conventions established in prior frozen phases. Every module has defined
responsibilities, no two modules share responsibility for the same logic, and
the decision pipeline is strictly sequential with no circular dependencies.

### 2. Is it scientifically sound?

Yes. The architecture is honest about the underlying science:

- Categorical confidence (HIGH/MEDIUM/LOW) rather than calibrated probabilities
  is the correct choice for n=2
- The median-baseline demotion-fallback outcome is propagated transparently via
  BR-04 and the confidence modifier map
- SHAP feature names are referenced only as prose evidence, not as decision inputs
- The n=2 caveat is structurally enforced (DV7 validation, every recommendation
  row, Phase 7→8 contract consumer_expectations)
- Threshold values are explicitly deferred to domain experts rather than invented

### 3. Is it maintainable?

Yes. Module boundaries are clean. Business rules are versioned and configurable.
Thresholds are in config, not hardcoded. The priority matrix is in config and
overridable. The decision pipeline is linear and traceable. All output artifacts
are atomically written.

### 4. Is it reproducible?

Yes. The config fingerprint mechanism (SHA-256 of the `decision_engine` block)
ensures that any change to thresholds, rule parameters, or priority matrix
produces a distinct experiment ID. Given the same Phase 6 artifacts and config,
the engine produces byte-identical output. There is no randomness in Phase 7.

### 5. Is it interview-defensible?

Yes. Every design choice is justified:

- Five risk tiers: standard reliability engineering (FMEA matrices)
- Categorical confidence: honest n=2 limitation
- Minimum RUL across channels: safety-first conservative aggregation
- Constant-predictor annotation (BR-04): scientific integrity requirement
- Deferral of threshold values to domain experts: avoids invented numbers

A Principal Reliability Engineer asking "why this threshold?" will receive the
answer: "The framework is designed; the specific value requires your domain
validation." That is the correct engineering answer.

### 6. Would a Principal Reliability Engineer approve this design?

Yes — with the condition that threshold values are populated with domain-validated
inputs before operational use. The architecture explicitly requires this and
gates the runtime on it (FATAL on null thresholds).

---

## FINAL VERDICT

**APPROVE ARCHITECTURE**

The Phase 7 Architecture is complete, scientifically sound, maintainable,
reproducible, and consistent with the frozen Phase 5 and Phase 6 architectures.
All 15 required sections are present. All design choices are justified. The
architecture enforces honesty about the n=2 limitation and the median-baseline
demotion-fallback outcome at every layer — from confidence estimation through
business rules to the Phase 7→8 contract consumer_expectations.

The single outstanding action before implementation begins: populate null
threshold values in `config/config.yaml` with domain-expert-validated
engineering inputs. The architecture provides the framework; the numbers require
a qualified Reliability Engineer.

---

## §26 — Architecture Refinement Log (v1.1)

This section documents the documentation-refinement pass applied after the
first formal architecture review of Phase 7 Architecture v1.0. It exists so
that a reviewer can verify, without re-reading the entire document, exactly
what changed and confirm that no architecture, module, business rule,
decision logic, or scope was altered.

### §26a — Refinement Summary

Eight review concerns were raised and addressed, all as documentation
strengthening:

| # | Concern | Section(s) Strengthened |
|---|---------|--------------------------|
| 1 | Threshold Governance | §9d (new) — ownership, approval workflow, versioning, configuration governance, audit requirements, reproducibility after updates, backward compatibility of historical decisions |
| 2 | Decision Traceability | §8a (new) — full prediction-to-Phase-8 lineage chain with traceability guarantee |
| 3 | Business Rule Precedence | §12a (expanded) — execution order table, override hierarchy tiers, four worked conflict-resolution examples, determinism argument |
| 4 | Decision Validation | §16c (new) — PASS/WARNING/FATAL criteria for five decision-consistency scenarios, each mapped to an existing check or structural guarantee |
| 5 | Explainability | §13d (new) — seven-question mandatory explanation checklist mapped to existing template fields |
| 6 | BR-04 Justification | §7.9a BR-04 (expanded) — extended justification distinguishing the transparency mechanism from decision logic |
| 7 | Phase 7 → Phase 8 Contract | §21c (new) — complete enumeration of every transferred artifact and field, traced to its defining section |
| 8 | Final Review | §26 (this section) — refinement summary, change classification, re-verdict |

### §26b — Documentation Changes

- Added 4 new subsections: §8a, §9d, §13d, §16c, §21c, §26 (6 new subsections
  total, all additive documentation)
- Expanded 2 existing subsections in place: §12a (Business Rules Framework),
  §7.9a BR-04 rationale
- Updated Table of Contents to reference all new subsections
- Updated version header (1.0 → 1.1) and Changelog table

### §26c — Architecture Changes

**NONE.** No module was added, removed, or renamed. No module responsibility
(§7) was altered. No item in the Module Architecture, Decision Pipeline (§8),
Risk Classification Framework (§9a–§9c), Maintenance Decision Framework (§10),
or Confidence Estimation Framework (§11) was changed.

### §26d — Business Rule Changes

**NONE.** BR-01 through BR-05 (§7.9a) remain exactly as originally specified.
§12a documents their existing precedence relationship explicitly; it does not
alter which conditions trigger which rule, nor which fields each rule may
modify.

### §26e — Decision Logic Changes

**NONE.** Risk classification (§9), the risk→action mapping (§10a), confidence
estimation (§11), and priority calculation (§7.6) are unchanged. No threshold
value was introduced — §9d discusses only the governance *process* around
thresholds, which remain configuration placeholders (`null`) pending
domain-expert validation, exactly as in v1.0.

### §26f — Scope Changes

**NONE.** No new deliverable, artifact, module, or pipeline stage was added.
§21c enumerates artifacts and fields that were already declared in §6a, §18,
and §21a of v1.0 — it does not introduce anything Phase 8 will receive that it
was not already going to receive.

### §26g — Implementation Changes

**NONE.** No source code, configuration values, or file was created or
modified as part of this refinement pass. This remains an architecture-only
document; implementation has not begun.

### §26h — Re-Verdict

**APPROVE ARCHITECTURE**

The refinement pass strengthens governance clarity, decision traceability,
business rule precedence, decision-output validation, explainability
completeness, BR-04's scientific justification, and the Phase 7→8 contract's
completeness — without altering the underlying architecture in any respect
verified in §26c–§26g. The architecture remains scientifically sound,
maintainable, reproducible, and interview-defensible per the original §25
self-review, now with stronger documentary support for each of those claims.

---

*Phase 7 Architecture v1.1 — DRAFT*
*Do not implement until formally reviewed and approved. Wait for Phase 7
implementation authorization.*
