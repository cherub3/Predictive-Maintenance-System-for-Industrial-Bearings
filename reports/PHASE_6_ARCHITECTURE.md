# Phase 6 Architecture — Model Development & Evaluation

**Status:** ARCHITECTURE ONLY — NO IMPLEMENTATION
**Phase:** 6 — Model Development & Evaluation
**Architecture Version:** 1.1
**Date:** 2026-07-04
**Authoring Roles:** Principal ML Scientist · Principal Data Scientist · Principal Reliability Engineer · Principal AI Engineer
**Upstream Contract:** `reports/phase5_dataset_contract.yaml`, `data/processed/training_dataset.parquet` (8,624 rows × 21 cols, RTF-only, frozen), `data/processed/splits/fold_assignments.parquet`
**Downstream Consumer:** Phase 7 — Maintenance Decision & Business Optimization

---

## 1. Executive Summary

Phase 6 consumes the frozen, leakage-free Phase 5 training dataset and produces an **evaluated RUL prediction model** together with the evidence needed to trust it: baseline comparisons, honest cross-validated metrics, interpretability outputs, and a governed, reproducible model artifact. It is the first phase in which a model is trained.

The architecture is governed by the same physical constraint that shaped Phase 5: the supervised problem has an effective sample size of **n = 2 run-to-failure (RTF) units** (`bearing_3`, inner-race defect; `bearing_4`, rolling-element defect). This single fact dictates every design decision in this phase — the choice of a **fixed-configuration baseline-first strategy over aggressive hyperparameter search**, the refusal to do nested cross-validation (which is statistically impossible with two units), the selection of **Leave-One-Bearing-Out (LOBO)** as the *only* honest evaluation protocol (inherited from the Phase 5 contract), and the framing of all results as **methodology validation, not production accuracy**.

Phase 6 deliberately **does not** chase a headline accuracy number. With two failure trajectories, a leaderboard of tuned gradient-boosting variants would be scientific theatre. Instead the phase is designed to answer one defensible question end-to-end: *given a leakage-free dataset and an honest protocol, does a disciplined model progression (naive → linear → ensemble) behave sensibly, and can we explain why?* The deliverable is a **trustworthy, reproducible, interpretable modelling pipeline**, not a bearing-fleet SLA.

The design targets **industrial engineering quality without enterprise over-engineering**: config-driven, experiment-tracked, fingerprinted, contract-governed, atomically written, buildable by one engineer, and interview-defensible at every decision.

---

## 2. Objectives

1. Establish naive baselines first, so every advanced model is measured against a meaningful floor.
2. Train a disciplined progression of candidate models under the **fixed** LOBO protocol from Phase 5, with no leakage reintroduced.
3. Fit all input scaling and any imputation **inside each training fold only** (honouring the Phase 5 contract's `consumer_expectations`).
4. Evaluate with metrics chosen for RUL regression, with primary/secondary/rejected metrics justified.
5. Produce global and local interpretability (permutation importance + SHAP) to make the model defensible, not a black box.
6. Govern every experiment: experiment IDs, config fingerprints, seeds, training manifests, evaluation reports, lineage back to Phase 5.
7. Publish a machine-readable **Phase 6 → Phase 7 contract** describing the selected model artifact, its prediction schema, and its consumer obligations.
8. State the n=2 statistical limitation everywhere a number appears, so no downstream reader over-claims.

---

## 3. Responsibilities

**In scope (Phase 6 owns):**
- Baseline model definition and evaluation (naive predictors, linear).
- Candidate model training (tree ensembles) under LOBO.
- In-fold feature scaling and in-fold imputation of documented NaN provenance.
- Cross-validation orchestration (consuming Phase 5 folds, never recomputing them).
- Constrained, optional, leakage-safe hyperparameter search (default: off).
- Metric computation and evaluation reporting.
- Feature importance (permutation + SHAP), global and local.
- Model selection, model registry, and the trained model artifact.
- Prediction generation on held-out folds.
- Experiment tracking, governance metadata, and the Phase 6 → Phase 7 contract.

**Explicitly NOT Phase 6's responsibility:**

| Activity | Phase 6 | Later phase |
|----------|:-------:|:-----------:|
| Model training / selection / evaluation | ✅ | ❌ |
| In-fold scaling & imputation | ✅ | ❌ |
| Feature importance / SHAP | ✅ | ❌ |
| Model registry & artifact governance | ✅ | ❌ |
| **Recomputing RUL labels / splits / leakage checks** | ❌ (consumes Phase 5 as-is) | Phase 5 (frozen) |
| **Inventing new features** | ❌ (uses the 14 contract features only) | Phase 4 (frozen) |
| Maintenance thresholds / alarm policy | ❌ | Phase 7 |
| Cost-optimal decision / scheduling | ❌ | Phase 7 |
| Dashboard / visualization product | ❌ | Phase 7+ |
| Deployment / serving infrastructure | ❌ | Phase 8+ |
| Multi-dataset generalization (IMS Set 2/3) | ❌ | Out of scope |

**Responsibility boundary rule:** Phase 6 never touches data outside its current **train-fold** view when fitting anything. It never modifies labels, folds, feature definitions, or leakage rules — those are frozen upstream. If a task would require recomputing an upstream artifact, it does not belong to Phase 6.

---

## 4. Inputs

| Input | Source | Purpose | Validation on entry |
|-------|--------|---------|---------------------|
| `data/processed/training_dataset.parquet` | Phase 5 | Features + labels + fold roles (8,624 × 21) | Row/col count, schema, grain uniqueness, dtypes vs contract |
| `data/processed/splits/fold_assignments.parquet` | Phase 5 | Row → fold/role (LOBO) | Presence, unit-disjointness re-asserted (defensive) |
| `reports/phase5_dataset_contract.yaml` | Phase 5 | **Authoritative machine-readable interface** | `schema_version` gate; column/unit/label presence; fail-closed on mismatch |
| `reports/dataset_manifest.json` | Phase 5 | Upstream lineage + config fingerprint | Version + fingerprint recorded into Phase 6 lineage |
| `config/config.yaml` (`model_dev` section) | Project | All Phase 6 parameters | Required keys present, types valid |

Phase 6 **never** parses `.md` narrative reports for control flow — only the `.yaml`/`.json` machine contracts (same governance rule as Phases 3–5). On entry Phase 6 **re-validates** the Phase 5 contract and **fails closed** if `schema_version` or declared columns/units do not match the actual dataset (Phase 5 contract §16, R-06).

---

## 5. Outputs (Artifacts)

| Artifact | Format | Consumer | Purpose |
|----------|--------|----------|---------|
| `models/<experiment_id>/model.pkl` | Pickle/joblib | **Phase 7 (primary)** | Serialized selected model + in-fold preprocessing pipeline |
| `models/<experiment_id>/refit_model.pkl` | Pickle/joblib | Phase 7 | Model refit on **all** RTF data for deployment (see §8.4) |
| `reports/model_predictions.parquet` | Parquet | Phase 7 / audit | Held-out LOBO predictions per row (grain-aligned) |
| `reports/phase6_model_contract.yaml` | YAML | **Phase 7 (machine-readable contract)** | Model spec, prediction schema, metrics, consumer obligations |
| `reports/evaluation_report.md` | Markdown | Human review | All metrics per model per fold + baselines + honesty caveat |
| `reports/model_selection_report.md` | Markdown | Human review | Candidate comparison, selection rationale |
| `reports/feature_importance_report.md` | Markdown | Human review | Permutation + SHAP global/local narrative |
| `reports/experiment_log.json` | JSON | Governance | Every experiment: ID, config fingerprint, seed, metrics, lineage |
| `reports/model_manifest.json` | JSON | Governance | Selected-model provenance, fingerprints, environment |
| `reports/model_validation_report.md` | Markdown | Human review | Every validation rule + pass/warn/fail |

Contract vs. narrative separation is preserved: `.yaml`/`.json` are machine inputs for the next phase; `.md` are human-only and never parsed downstream.

---

## 6. Dataset Shape & The Channel Grain (Design Item — data understanding)

The Phase 5 training dataset grain is **(bearing_id, channel, timestamp)** — each RTF acquisition timestamp produces **two rows** (channel 0 and channel 1 of the same bearing). Both rows at a timestamp carry the **same** `rul_minutes` (RUL is a property of the bearing at a time, not of a channel).

This has three architectural consequences that Phase 6 must handle explicitly:

1. **Fold grouping is by `bearing_id`, never by channel or row.** Inherited directly from Phase 5 fold assignments — no recomputation.
2. **Row-level training is permitted; row-level *evaluation of independence is not*.** Because the two channels of a bearing share a label and a trajectory, they are not independent samples. Metrics are computed at row grain for convenience but **interpreted at unit grain** — the honest number of independent test observations per fold is **one bearing**, not thousands of rows. This is stated in every report (§13, §6a-equivalent honesty guardrail).
3. **Channel handling is a config-declared modelling choice**, defaulting to **treat channel as an additional categorical/duplicate observation** (simplest, no feature invention). An optional config flag may **aggregate channels per timestamp** (e.g., mean/max of features) to yield a per-timestamp grain; this is off by default because it would alter the dataset the Phase 5 contract froze. Any aggregation happens **inside the fold**, never across the train/test boundary.

No new features are engineered here (that is Phase 4, frozen). Phase 6 uses exactly the **14 feature columns** enumerated in the Phase 5 contract.

---

## 7. Baseline Strategy (Design Item — baselines)

Baselines are **mandatory and evaluated first**. Every advanced model is reported *relative to* these floors; a model that cannot beat them is not selected regardless of sophistication.

| Order | Baseline | What it predicts | Why it exists |
|-------|----------|------------------|---------------|
| 0 | **Mean-RUL predictor** | Constant = mean `rul_minutes` of the **training fold** | Absolute floor. Any model must beat "predict the average." Fit-on-train-only (the mean is a statistic → must not see the test bearing, echoing L-07). |
| 1 | **Median-RUL predictor** | Constant = median `rul_minutes` of the training fold | Robust-to-skew floor; RUL distributions are right-skewed (long healthy life, short failure region). If median beats mean, the skew matters — an interpretable diagnostic. |
| 2 | **Linear Regression** | Linear map from the 14 features → RUL | First *real* model. Establishes whether the features carry any linear degradation signal at all. Transparent, fast, no tuning, fully interview-defensible. Its coefficients are a sanity check against SHAP later. |
| 3 | **Random Forest** | Bagged trees | First **non-linear, interaction-capturing** model. Degradation features (kurtosis slope, band energy) are expected to be non-linear near failure. RF is low-variance, needs little tuning, and gives a native importance cross-check. |
| 4 | **Gradient Boosting (XGBoost)** | Boosted trees | Strongest tabular candidate. Introduced **last and only after** simpler models are understood, so its gains (if any) are attributable and not mistaken for the whole story. |

**Justification of the progression:** complexity is added **one controlled step at a time**, each step answering a specific question (is there signal? is it linear? are interactions needed? does boosting add anything?). This is the scientifically correct order because it makes every performance delta *attributable*. Jumping straight to XGBoost would produce a number with no explanation of *why* it is what it is — indefensible in review and useless for reliability engineering, where the mechanism matters as much as the metric.

### 7a. Baseline Engineering Purpose Table

Every model in the progression exists because it answers a specific, irreducible engineering question. If a model cannot answer its question, its successor has no interpretive foundation.

| Model | Engineering question it answers | What a "pass" means | What a "fail" means |
|-------|--------------------------------|---------------------|---------------------|
| **Mean-RUL** | Is there any signal at all beyond the dataset average? | Features carry at least some degradation information. | Features are uninformative; the pipeline cannot support supervised RUL prediction on this data. |
| **Median-RUL** | Does the RUL distribution's skew affect the baseline floor? | Skew is meaningful; the right-skewed healthy-life region biases the mean. If median beats mean, downstream models should be evaluated against median as the tighter floor. | Skew is negligible; mean is sufficient as the floor. |
| **Linear Regression** | Do the 14 features carry a *linear* degradation signal? | A linear relationship exists; coefficients are interpretable, and serve as a sanity check for SHAP later. | Non-linear interactions are required to capture degradation — motivates tree models. |
| **Random Forest** | Do non-linear feature interactions improve prediction? | Interactions matter; bagged trees capture them with low variance, and native importance provides a cross-check for permutation + SHAP. | The degradation signal is already well-captured linearly; additional complexity is not justified. |
| **XGBoost** | Does sequential boosting extract residual signal that bagging misses? | Boosting captures subtle degradation patterns that RF cannot; the gain is attributable because RF was understood first. | RF already captures the signal; boosting adds complexity without benefit, and the simpler model is preferred (§8.3). |

### 7b. Model Promotion Criteria

Models advance through the progression **only if** they satisfy explicit, objective promotion rules. No model is adopted on the basis of appearing to perform well — it must demonstrate that its added complexity is justified.

**Promotion rule set (all must hold for a model to be promoted over its predecessor):**

1. **Baseline exceeded on both folds.** The candidate must beat the immediately preceding model's **primary metric (MAE)** on **both** LOBO folds, not just on average. A model that wins on one failure mode but collapses on the other is not promoted — it has learned a mode-specific pattern, not a generalizable degradation signal.

2. **Improvement is meaningful, not noise.** Because n=2 does not support classical statistical significance, "meaningful" is defined operationally: the MAE improvement must be **large enough to change a maintenance decision** (i.e., not a sub-minute difference on a target spanning thousands of minutes). The near-failure MAE (§10.2) is examined specifically — an overall MAE improvement that comes entirely from better predictions in the low-risk early-life region, while near-failure accuracy is unchanged, is not meaningful for the downstream Phase 7 decision.

3. **Complexity justified.** The candidate must demonstrate that its additional complexity captures a phenomenon the simpler model cannot. For tree models over linear: evidence of non-linear feature interactions (e.g., near-failure predictions where the linear model systematically under-predicts). For XGBoost over RF: evidence of residual structure (e.g., consistent improvement in the near-failure slice, not just a marginal overall MAE drop). This is documented in `model_selection_report.md`.

4. **No evidence of overfitting.** The candidate must satisfy the overfitting policy (§10.5). A model that achieves near-zero training error while its held-out error is comparable to or worse than a simpler model is rejected regardless of its test MAE.

5. **Reproducible.** The candidate must produce identical metrics across two runs with the same config + seed (§16). A non-reproducible result is not a valid promotion basis.

**Demotion rule:** if a more complex model fails any of these criteria, the simpler predecessor is selected. The progression is **not** a tournament where the most complex survivor wins by default — it is a ladder where each rung must earn its height.

**Decision documentation:** every promotion or rejection decision is recorded in `model_selection_report.md` with the specific criterion that determined the outcome.

---

## 8. Model Candidates & Selection (Design Item — candidates)

### 8.1 In scope

| Model | Rationale for inclusion |
|-------|-------------------------|
| Mean / Median baseline | Mandatory floors (§7). |
| Linear Regression (+ optional Ridge) | Transparent signal check; Ridge optional for the p=14 vs. low-n regime. |
| Random Forest | Non-linear, low-variance, robust default; native + permutation importance. |
| XGBoost | Best-in-class tabular gradient boosting; the primary "advanced" candidate. |

### 8.2 Intentionally excluded (with reasons)

| Excluded | Why excluded (this phase, this dataset) |
|----------|------------------------------------------|
| **LightGBM** | Adds a second boosting library with no scientific difference from XGBoost at this scale (8,624 rows). Extra dependency, extra governance surface, zero expected insight. Excluded to avoid complexity for its own sake. Reconsider only if XGBoost is compute-bound (it is not here). |
| **CatBoost** | Its headline advantage is native categorical handling; Phase 5 features are all continuous. No categorical-heavy input → no justified benefit. Excluded. |
| **Deep learning (LSTM/CNN/Transformer)** | Requires far more trajectories than **n=2** to avoid gross overfitting; would be scientific malpractice on this dataset. Sequence models are noted as future work if more RTF units become available. Excluded. |
| **Stacking / blending ensembles** | Meta-learning on two folds cannot be validated without a third level of held-out units, which do not exist. Excluded as un-evaluatable. |

Every exclusion is a **deliberate, defensible scope decision**, not an oversight — each names the specific reason it adds no validatable value on this dataset.

### 8.3 Selection rule — "Best Model" definition

"Best model" is **not** the model with the lowest single number. It is the model that best satisfies a **hierarchical selection rule** designed for an n=2, safety-relevant, methodology-validation context. The hierarchy is applied in strict order — a criterion is only consulted if the preceding criteria do not resolve the decision.

**Selection hierarchy (applied in order):**

| Priority | Criterion | Rationale |
|----------|-----------|-----------|
| 1 | **Lowest primary metric (MAE) averaged across both LOBO folds** | MAE is the maintenance-relevant error in minutes; this is the first filter, not the last. |
| 2 | **Stable performance across both folds** | A model whose two-fold MAE values differ by more than a config-declared **instability threshold** (e.g., >2× ratio) is flagged: it may have learned a mode-specific pattern. If two candidates have similar average MAE but one is stable and the other is not, the stable model is preferred — consistency across failure modes is more trustworthy than a lower mean that hides a collapse on one bearing. |
| 3 | **No evidence of overfitting** | Evaluated per the overfitting policy (§10.5). A model that satisfies criteria 1–2 but shows clear train/test divergence or suspiciously low error is rejected. |
| 4 | **Simpler model preferred when performance is effectively tied** | "Effectively tied" means the MAE difference is too small to change a maintenance decision (operationally: within a config-declared **equivalence margin** in minutes). When tied, prefer the model with fewer parameters, fewer hyperparameters, and faster training — i.e., Linear Regression over RF, RF over XGBoost. This is Occam's razor applied operationally, not aesthetically. |
| 5 | **More interpretable model preferred as a final tiebreaker** | If criteria 1–4 do not resolve, prefer the model whose predictions are easier to explain to a reliability engineer: Linear > RF ≈ XGBoost (with SHAP, tree models are similarly interpretable, but linear coefficients are directly inspectable). |

**Hard gates (must pass before the hierarchy is consulted):**
- The candidate must beat **both** naive baselines (mean and median) on **both** folds.
- The candidate must satisfy the promotion criteria (§7b).
- The candidate must be reproducible (identical metrics on re-run with same config + seed).

A model that fails any hard gate is **excluded from consideration** regardless of its MAE.

**Documentation:** the full selection hierarchy evaluation — including which criterion resolved the decision and the metric values that determined it — is written to `model_selection_report.md`.

### 8.4 Refit-for-handoff

The LOBO evaluation gives the *honest performance estimate*. For the artifact handed to Phase 7, the selected model **type + configuration** is refit on **all RTF data** (`refit_model.pkl`) so Phase 7 receives a model that used every available failure trajectory. Both the fold models (for provenance) and the refit model (for use) are registered. The refit model's expected performance is quoted as the LOBO estimate, **explicitly labelled as an out-of-sample estimate from n=2**, never as an independent validation of the refit artifact itself.

---

## 9. Hyperparameter Strategy (Design Item — hyperparameters)

This is the most easily-abused part of the phase and is designed conservatively **because of n=2**.

- **Default: no search.** Every model runs with **fixed, documented, sensible defaults** declared in config. This is the primary, recommended path and the one reported by default.
- **Why not nested CV / grid / Bayesian search:** honest hyperparameter selection requires an inner validation loop that is **disjoint at the unit level** from the outer test unit. With only two RTF units, the outer LOBO loop already consumes both units; there is **no third unit** to form an inner unit-disjoint validation set. Aggressive tuning would therefore either (a) leak the test unit into selection, or (b) tune on the single training unit's own trajectory and overfit it. Neither is defensible. **The correct engineering answer on n=2 is to not tune.**
- **Optional, leakage-safe, constrained search (config-gated, default off):** if enabled, hyperparameters are selected using a **within-training-unit temporal inner split** — carved from the *early life* of the **training** bearing only (past-only), exactly the mechanism Phase 5 reserved in its `split_spec.val_from_train_early_fraction`. This never touches the held-out test bearing. Search is restricted to a **small, hand-declared grid** (not unbounded), with **early stopping** on the inner validation slice for boosting models. Any run of this path records the full grid, the selected point, and the seed in the experiment log, and the report states that tuning was performed on a within-unit slice and is therefore weaker evidence than the untuned result.
- **Reproducibility:** a single global `random_seed` (config) is threaded into every stochastic estimator (RF bootstrap, XGBoost subsampling) and every split operation. Seeds are recorded per experiment. Two runs with the same config + seed produce identical models and metrics.

**Interview-defensible summary:** *"With two failure units we do not tune, because there is no leakage-free way to select hyperparameters; we report fixed, sensible defaults and expose an optional within-train-unit search only as a clearly-labelled, weaker-evidence branch."*

---

## 10. Evaluation Strategy (Design Item — metrics)

### 10.1 Protocol (inherited, not redesigned)

Evaluation uses the **LOBO CV folds emitted by Phase 5** — two folds, each testing on one held-out bearing. Phase 6 **consumes** `fold_assignments.parquet` and **never recomputes** splits. Metrics are computed on the held-out fold's rows, then reported **per fold** and **as a two-fold summary (mean + explicit spread)** — never as a single blended number that hides which failure mode failed.

### 10.2 Metrics

| Metric | Role | Justification |
|--------|------|---------------|
| **MAE (minutes)** | **Primary** | Directly interpretable ("on average, RUL is off by X minutes"), robust to the right-skewed RUL distribution, and the natural currency for the Phase 7 maintenance lead-time decision. Chosen as the selection metric. |
| **RMSE (minutes)** | Secondary | Penalizes large errors more heavily; a large RMSE/MAE gap signals a few catastrophic misses (often in the critical near-failure region) that MAE alone hides. Reported alongside MAE, not used for selection. |
| **R²** | Secondary (diagnostic) | Communicates variance explained vs. the mean baseline. Interpreted cautiously: with n=2 units it can be misleading, so it is reported as a *diagnostic relative to baseline*, never as a headline. |
| **Near-failure MAE** (subset where RUL ≤ threshold) | Secondary (reliability-critical) | The maintenance decision only cares about accuracy *near failure*. Overall MAE can look good while the model is useless in the critical region. This slice makes that visible. Threshold is config-declared. |
| **Prediction-error asymmetry** (mean signed error / late-prediction rate) | Secondary (safety) | In predictive maintenance, **over-predicting RUL (late warning) is more dangerous than under-predicting**. We report the signed error and the fraction of dangerously-late predictions so Phase 7 understands the model's risk direction. Diagnostic, not a selection gate in this phase. |

### 10.3 Rejected metrics

| Rejected | Why |
|----------|-----|
| **MAPE / sMAPE** | RUL → 0 at failure; percentage error **divides by (near-)zero** exactly in the most safety-critical region, producing infinite/exploding values. Fundamentally unsuitable for a target that reaches zero. Rejected. |
| **Classification metrics (accuracy/F1/AUC)** | The task is regression (RUL in minutes). Thresholding into classes throws away the signal and pre-empts the Phase 7 decision policy. Rejected. |
| **Single pooled metric across both folds without spread** | Hides which failure mode the model cannot handle. Rejected in favour of per-fold + spread reporting. |

### 10.4 Honesty guardrail

`evaluation_report.md` states, adjacent to every number, that metrics are the average/spread of **exactly two evaluation runs (n=2 units)**, carry **no classical confidence interval**, and are **methodology validation, not fleet accuracy** (mirrors Phase 5 §6a/§6b). This is a load-bearing constraint the whole phase is designed around, not a disclaimer.

### 10.5 Overfitting Policy

With n=2 units and thousands of rows per trajectory, overfitting to a single training bearing's degradation pattern is the **dominant modelling risk** in Phase 6. The architecture defines explicit identification, monitoring, and rejection criteria.

**Identification — how overfitting is detected:**

| Signal | How measured | What it indicates |
|--------|-------------|-------------------|
| **Train-vs-test MAE discrepancy** | Compare MAE on the training fold's own rows vs. the held-out fold's rows, per LOBO fold | A large gap (train MAE ≪ test MAE) means the model has memorized the training trajectory rather than learning generalizable degradation features. |
| **Inconsistent LOBO behaviour** | Compare MAE across the two LOBO folds for the same model | A model that performs well when trained on bearing_3 but poorly when trained on bearing_4 (or vice versa) has likely overfit to the specific characteristics of one failure mode. |
| **Suspiciously low error** | Absolute MAE on training data is near zero, or test MAE is lower than expected given dataset noise | With real sensor data, some irreducible measurement noise exists. Near-zero training error on 4,000+ rows with 14 features suggests the model has learned the noise, not the signal. |
| **Complexity-for-no-gain** | A more complex model's test MAE is comparable to or worse than a simpler predecessor's | The additional complexity is fitting noise, not capturing new structure. |

**Monitoring strategy:**
- Training MAE is computed for every model on every fold (cheap, always available).
- The train/test MAE ratio is reported per model per fold in `evaluation_report.md`.
- A config-declared **overfitting threshold** (e.g., train/test MAE ratio < 0.3) triggers a WARNING.

**Rejection criteria:**
- A model whose train/test MAE ratio is below the config-declared threshold on **either** fold receives a WARNING and is **flagged in `model_selection_report.md`**.
- A flagged model is **not automatically rejected** — with n=2, some train/test gap is expected because the two failure modes are genuinely different (inner-race vs. rolling-element). The flag requires the `model_selection_report.md` to explicitly state whether the gap is attributable to failure-mode difference or to memorization, with evidence (e.g., does the near-failure accuracy improve or degrade?).
- A model that is both flagged for overfitting **and** fails to beat the simpler predecessor on the test fold is **rejected** — no ambiguity.

**Acceptable behaviour:** a moderate train/test gap where (a) the test MAE still beats baselines on both folds, (b) near-failure accuracy is preserved, and (c) the gap is consistent across both folds (not mode-specific) is acceptable and documented.

### 10.6 Error Analysis

Aggregate metrics (MAE, RMSE) summarize model quality but do not explain *where* or *why* the model fails. Error analysis is a **mandatory** post-evaluation step that decomposes prediction errors into interpretable dimensions.

**Error analysis dimensions (all computed on held-out folds only):**

| Dimension | What it reveals | Why it matters |
|-----------|-----------------|----------------|
| **Prediction error vs. time-to-failure** | Plot residual (predicted − actual RUL) as a function of actual RUL | Reveals whether the model is accurate near failure (safety-critical) or only in the easy early-life region. A model with low overall MAE but large errors in the last 10% of life is **dangerous for maintenance decisions.** |
| **Near-failure error** (RUL ≤ threshold) | MAE/RMSE restricted to the config-declared near-failure region | Directly answers "how accurate is the model when it matters most?" — the near-failure MAE is reported alongside overall MAE in evaluation_report.md. |
| **Bearing-specific error** | MAE per held-out bearing (i.e., per LOBO fold) | Reveals whether the model generalizes across failure modes. A model that predicts inner-race degradation well but not rolling-element degradation (or vice versa) has limited utility. Already captured by per-fold reporting but **explicitly broken down here with the bearing name, not just fold ID.** |
| **Channel-specific error** | MAE per channel within the held-out bearing | Reveals whether the model treats both sensor channels equally or systematically favours one. With two channels per bearing sharing the same label, a large channel disparity suggests the model is relying on channel-specific noise rather than the shared degradation signal. |
| **Largest residuals** | Identify the N rows (config-declared, e.g., top 10) with the largest absolute residual on each held-out fold | Inspection target: are the worst predictions at a specific degradation stage? A specific channel? A specific time region? This is the reliability engineer's diagnostic — the individual predictions where the model would most mislead a maintenance decision. |
| **Systematic bias** | Mean signed error (predicted − actual) on each fold | Positive bias = model over-predicts RUL (predicts more remaining life than actual) = **late warning, dangerous**. Negative bias = under-predicts = **early warning, conservative**. The direction matters more than the magnitude for maintenance safety. Reported in `evaluation_report.md` and the Phase 7 contract. |

**Output:** error analysis results are documented in the relevant sections of `evaluation_report.md`. No new artifacts are created — this is a decomposition of metrics already computed, not a separate pipeline stage.

---

## 11. Feature Importance & Interpretability (Design Item — importance)

Interpretability is a **first-class deliverable**, not an appendix — a reliability model that cannot explain *why* it predicts imminent failure is not trustworthy for maintenance decisions.

| Technique | Scope | Purpose |
|-----------|-------|---------|
| **Permutation importance** | Global, model-agnostic | Measures true out-of-sample reliance by shuffling each feature on the **held-out fold**. Model-agnostic, so it compares linear/RF/XGB on equal footing. Primary global method. |
| **SHAP** (TreeExplainer for tree models) | Global + local | Global: mean \|SHAP\| ranking. Local: per-prediction attribution, especially in the near-failure region, so an engineer can see *which degradation feature* drove a specific RUL alarm. |
| **Native model importance** (RF/XGB gain) | Global (cross-check) | Cheap consistency check against permutation + SHAP. Divergence between native gain and permutation importance is itself reported as a caution flag (gain can be biased toward high-cardinality features). |
| **Linear coefficients** | Global (cross-check) | The Linear Regression baseline's standardized coefficients are compared against SHAP; agreement on the top degradation drivers increases trust. |

**Comparison strategy:** the three global views (permutation, SHAP, native) are **overlaid** in `feature_importance_report.md`; a feature is called a "trusted driver" only if it ranks highly across methods. Local SHAP examples are drawn from the near-failure region of the held-out bearing. All importance is computed **on held-out folds** (out-of-sample), never on training rows, to avoid over-crediting memorized features.

**n=2 caveat:** importance rankings are themselves derived from two folds and are reported as **indicative of mechanism**, not as statistically stable rankings.

### 11a. SHAP Governance Policy

SHAP is a powerful interpretability tool and correspondingly easy to misuse. This policy defines its **permitted and prohibited uses** explicitly.

**SHAP is used ONLY for model interpretation — never for model selection.**

| Permitted use | Prohibited use |
|---------------|----------------|
| Explaining *why* the selected model made a specific prediction (local attribution) | Using SHAP values to choose between candidate models |
| Identifying which degradation features drive near-failure RUL drops (engineering insight) | Filtering, re-ranking, or eliminating features based on SHAP values |
| Cross-checking against permutation importance and linear coefficients (consistency validation) | Using SHAP consistency/inconsistency as a model-selection criterion |
| Communicating model behaviour to reliability engineers and interviewers (trust-building) | Discarding a model because its SHAP attribution "doesn't look right" |

**Why this boundary exists:** SHAP explains *how* a fitted model uses its features; it does not tell you whether the model is *correct*. A model with clean, interpretable SHAP plots can still overfit. A model with surprising SHAP patterns may have discovered a legitimate non-obvious signal. Using SHAP to influence model selection conflates interpretability with accuracy and introduces a subjective, non-reproducible decision into what must be an objective, metric-driven pipeline.

**Enforcement:** model selection (§8.3) references only MAE, fold stability, overfitting evidence, and complexity — never SHAP. SHAP analysis runs **after** model selection, not before. The `feature_importance_report.md` narrative may note surprising SHAP patterns as **observations for human review**, but these observations do not feed back into the selection decision. If SHAP reveals that the selected model relies on a feature in a physically implausible way, this is **documented as a finding** and surfaced for the reviewer's judgement — it does not automatically trigger re-selection.

---

## 12. Module Architecture (Design Item — modules)

All modules are **design specifications only** in this phase. No cycles; dependencies flow downward. Every module reads config, never hardcodes.

| Module | Responsibility | Inputs | Outputs | Depends on |
|--------|----------------|--------|---------|-----------|
| `config` (loader, existing) | Provide `model_dev` params | `config.yaml` | typed config dict | — |
| `contract_loader` | Load + validate Phase 5 contract; fail-closed on mismatch | `phase5_dataset_contract.yaml`, dataset | validated schema handle | config |
| `data_loader` | Load training dataset + fold assignments; re-assert unit-disjointness | dataset, folds | in-memory frames | contract_loader |
| `preprocessing` | **In-fold** scaling + imputation pipeline (fit on train fold only) | train-fold frame | fitted transform | data_loader |
| `baseline_models` | Mean/median/linear floors | folds | baseline predictions/metrics | preprocessing |
| `model_trainer` | Fit a candidate model on a train fold with a fitted preprocessor | train fold, config | fitted estimator+pipeline | preprocessing |
| `cross_validation` | Orchestrate LOBO over Phase 5 folds; collect per-fold results | folds, trainer | per-fold predictions/metrics | model_trainer |
| `hyperparameter_search` | Optional, config-gated, within-train-unit temporal search | train fold | selected params | cross_validation |
| `metrics` | Compute MAE/RMSE/R²/near-failure/asymmetry | predictions, labels | metric records | — |
| `evaluation` | Aggregate metrics per fold + summary + baselines | cross_validation, metrics | `evaluation_report.md`, metric tables | metrics |
| `feature_importance` | Permutation + native importance | fitted models, held-out folds | importance tables | cross_validation |
| `shap_analysis` | Global + local SHAP | fitted tree models, folds | SHAP outputs | feature_importance |
| `model_selection` | Apply §8.3 rule; pick model; refit-for-handoff (§8.4) | evaluation results | selected + refit model | evaluation |
| `model_registry` | Version, fingerprint, persist model artifacts | selected/refit models | `models/<exp_id>/*.pkl` | model_selection |
| `prediction_generator` | Emit held-out predictions at dataset grain | fold models, folds | `model_predictions.parquet` | cross_validation |
| `experiment_tracker` | Assign experiment IDs; log config fingerprint, seed, metrics, lineage | all runs | `experiment_log.json` | all |
| `model_manifest` | Selected-model provenance + environment capture | registry, tracker | `model_manifest.json` | model_registry |
| `phase7_contract` | Emit machine-readable model contract | manifest, selected model | `phase6_model_contract.yaml` | model_manifest |
| `validation` | Aggregate all rules → report, gate exit code | all | `model_validation_report.md` + exit code | all |
| `main` | Orchestrate ordered pipeline, logging, exit codes | config | pipeline run | all |
| `tests/` | §15 test suite | — | — | all |

**Ordering invariant enforced by `main`:** contract_loader → data_loader → (per fold: preprocessing → baseline_models → model_trainer → cross_validation [+ optional hyperparameter_search]) → evaluation → feature_importance → shap_analysis → model_selection → model_registry → prediction_generator → manifest/contract → validation. **Preprocessing is always fit inside the fold**; it is structurally impossible for a scaler to see the held-out bearing.

---

## 13. Processing Pipeline & Data Flow

```
phase5_dataset_contract.yaml ─┐
training_dataset.parquet ─────┼─> contract_loader (schema gate, fail-closed)
fold_assignments.parquet ─────┘            │
                                           v
                                    data_loader (re-assert unit-disjoint)
                                           │
              ┌────────── for each LOBO fold (train unit / test unit) ──────────┐
              │  preprocessing.fit(train fold only)  ── scaler + imputer         │
              │            │                                                     │
              │            v                                                     │
              │  baseline_models ─┐                                              │
              │  model_trainer  ──┼─> fitted model + pipeline                    │
              │  [hyperparameter_search: within-train-unit early-life slice]     │
              │            │                                                     │
              │            v                                                     │
              │  predict(held-out test unit) ─> per-fold predictions + metrics   │
              └────────────────────────────────┬────────────────────────────────┘
                                                v
                     evaluation ─> evaluation_report.md (per fold + spread + n=2 caveat)
                                                │
                                                v
                 feature_importance + shap_analysis ─> feature_importance_report.md
                                                │
                                                v
                 model_selection ─> pick model (§8.3) + refit on all RTF (§8.4)
                                                │
                                                v
   model_registry ─> models/<exp_id>/model.pkl, refit_model.pkl
   prediction_generator ─> model_predictions.parquet
   experiment_tracker ─> experiment_log.json
                                                │
                                                v
        model_manifest.json + phase6_model_contract.yaml
                                                │
                                                v
                        validation ─> model_validation_report.md ─> exit code
```

---

## 14. Experiment Lifecycle (Design Item — lifecycle)

Each **experiment** is one full evaluation of one model family under LOBO with one config + seed, and is the unit of governance.

1. **Declare:** config block fully specifies the run (models enabled, seed, metric thresholds, tuning on/off). No parameter lives in code.
2. **Fingerprint:** SHA-256 of the serialized `model_dev` config → **config fingerprint** (reusing the Phase 3–5 `_compute_config_version` pattern). Combined with a timestamp/counter into a stable **experiment ID**.
3. **Execute:** LOBO loop; every model, every fold, every metric captured.
4. **Track:** `experiment_tracker` appends a record (experiment ID, config fingerprint, git commit, seed, per-fold metrics, upstream Phase 5 dataset version + fingerprint) to `experiment_log.json`.
5. **Select & register:** best model per §8.3 chosen; artifacts persisted under `models/<experiment_id>/` (immutable, never hand-edited).
6. **Contract:** the selected model is described in `phase6_model_contract.yaml` for Phase 7.
7. **Validate & gate:** `validation` confirms acceptance criteria; non-zero exit on any FATAL.

Re-running with an identical config + seed reproduces the same experiment ID, models, and metrics (§16 reproducibility). Tracking is **file-based JSON** — deliberately not MLflow or a tracking server, because a single engineer on a two-unit dataset does not need managed experiment infrastructure; a fingerprinted JSON log is sufficient, auditable, and dependency-free (interview-defensible right-sizing).

---

## 15. Testing Strategy (Design Item — testing)

| Test type | What it verifies |
|-----------|------------------|
| **Unit (metrics)** | MAE/RMSE/R²/near-failure/signed-error computed correctly on hand-checked fixtures; MAPE guard rejects zero-RUL denominators. |
| **Unit (baselines)** | Mean/median predictors return the **train-fold** statistic only; never see test rows. |
| **Unit (preprocessing)** | Scaler/imputer `fit` uses train-fold rows only; transform applied identically to test fold; no NaN leaks through. |
| **Unit (fold consumption)** | `cross_validation` uses Phase 5 fold roles verbatim; asserts unit-disjointness; never regenerates splits. |
| **Leakage tests** | Injected leakage (scaler fit on full dataset, test unit in train, channel of test bearing in train) is **caught and made FATAL** — the guard actually fires. |
| **Integration** | End-to-end on a small synthetic 2-unit fixture → all artifacts produced, model contract schema valid, predictions grain-aligned. |
| **Reproducibility tests** | Same config + seed → identical metrics, identical model fingerprint, identical experiment ID across two runs. |
| **Regression tests** | Baseline metrics and selected-model metrics stable across runs on frozen inputs (within seed-fixed tolerance). |
| **Contract tests** | `phase6_model_contract.yaml` matches schema; declared prediction columns/dtypes/model type equal the actual registered artifact; entry-side Phase 5 contract validation fails closed on a mutated fixture. |
| **Selection tests** | §8.3 rule picks the correct model on crafted metric tables; baseline-beating gate rejects an under-performing model. |

Quality over quantity: the leakage, reproducibility, and contract tests are the **load-bearing** ones and are prioritized.

---

## 16. Governance (Design Item — governance)

- **Model version** + **experiment ID** (semantic + fingerprint-derived) in registry, manifest, and contract.
- **Configuration fingerprint:** SHA-256 of serialized `model_dev` config (Phase 3–5 pattern).
- **Model fingerprint:** hash of the serialized model artifact, so any two artifacts can be compared without re-training.
- **Seeds:** single global `random_seed` recorded per experiment; threaded into every stochastic component.
- **Environment capture:** library versions (scikit-learn, xgboost, shap, numpy, pandas) recorded in `model_manifest.json` for reproducibility.
- **Data lineage:** manifest records upstream Phase 5 dataset version + config fingerprint + contract `schema_version`, extending the Phase 3→4→5→6 chain.
- **Git commit** short SHA captured (existing helper).
- **Atomic writes:** every artifact written to `*.tmp` then `Path.replace()`; never partial files.
- **Immutability:** artifacts regenerated wholesale each run; header banners state "generated — do not hand-edit."

**Reproducibility statement:** given **all four** of the following identical inputs:

1. **Phase 5 inputs** — `training_dataset.parquet`, `fold_assignments.parquet`, `phase5_dataset_contract.yaml` (byte-identical)
2. **Configuration** — `model_dev` block in `config/config.yaml` (identical config fingerprint)
3. **Software versions** — scikit-learn, xgboost, shap, numpy, pandas, Python interpreter (identical version strings as recorded in `model_manifest.json`)
4. **Random seed** — `random_seed` value in config (identical)

Phase 6 produces **identical outputs within floating-point tolerance**:

| Output | Reproducibility guarantee |
|--------|---------------------------|
| Per-fold and summary metrics (MAE, RMSE, R², etc.) | Bit-identical (deterministic computation on seeded models) |
| Model selection decision | Identical (same metrics → same hierarchy outcome) |
| Experiment ID | Identical (derived from config fingerprint + seed) |
| Model fingerprint | Identical within the same library versions; may differ across library builds (bounded by environment capture) |
| Prediction values | Identical within IEEE 754 floating-point tolerance (≤ 1e-9 relative difference) |
| Feature importance rankings | Identical (permutation importance is seeded; SHAP TreeExplainer is deterministic for tree models) |
| Narrative reports (.md) | Identical (generated from deterministic metrics) |

**Why this holds:** folds are consumed (not sampled), all stochastic estimators are seeded at construction, preprocessing is a deterministic in-fold fit, and artifacts are written deterministically with fixed sort keys. The explicit **software version** requirement bounds the one source of non-determinism that seeds cannot control (library-internal numerical changes across versions).

**Verification:** reproducibility is a **tested acceptance criterion** (§15 — reproducibility tests). Two runs with the same four inputs must produce identical metrics and identical experiment IDs. A reproducibility test failure is FATAL.

---

## 17. Phase 6 → Phase 7 Contract (Design Item — contract)

`reports/phase6_model_contract.yaml` — the **only** machine-readable interface Phase 7 may consume.

**Specified content — complete artifact enumeration:**

The contract enumerates **every artifact transferred** and its role. Phase 7 may consume only what is listed here.

| Contract section | Key fields | Purpose |
|-----------------|------------|---------|
| **`schema_version`** | Semantic version (e.g., `1.0`) | Phase 7 entry gate; fail-closed on mismatch |
| **`producer`** | `phase`, `module`, `generated_at`, `git_commit`, `config_fingerprint` | Provenance envelope — same pattern as Phase 5 |
| **`experiment`** | `experiment_id`, `random_seed`, `config_fingerprint` | Reproducibility anchor — links the contract to a specific, repeatable experiment run |
| **`model`** | `type` (e.g., `xgboost_regressor`), `path` to `refit_model.pkl`, `model_fingerprint`, `training_population` (`rtf_units: [bearing_3, bearing_4]`), `preprocessing` (scaler type + imputer type + `fit_scope: train_fold_only`) | The trained artifact + its preprocessing contract |
| **`prediction_schema`** | `input_features` (the 14 Phase 5 features, exact names/dtypes/order), `output` = `predicted_rul_minutes` (float, ≥ 0), `grain` = `[bearing_id, channel, timestamp]`, `units` = `minutes` | Schema that Phase 7 must supply/consume — the machine-readable prediction interface |
| **`performance`** | Per-fold MAE/RMSE/R², near-failure MAE, mean signed error, late-prediction rate, two-fold summary (mean + spread), `n2_caveat` (mandatory boolean + caveat text) | Honest performance evidence; the `n2_caveat` field is **required, not optional** — a contract without it is invalid |
| **`feature_importance`** | Top-ranked features by permutation importance + SHAP (global), per-fold rankings | Interpretability evidence for Phase 7 decision-making |
| **`error_analysis_summary`** | Bias direction (over/under-predict), near-failure accuracy, largest-residual characteristics | Risk profile for maintenance threshold calibration |
| **`consumer_expectations`** | (a) Feed features in the exact contract schema; (b) do not retrain or re-tune the model; (c) treat quoted performance as an n=2 out-of-sample estimate, not a fleet SLA; (d) apply decision thresholds **on top of** raw RUL predictions (thresholds are Phase 7's responsibility, not baked into the model); (e) honour the late-prediction risk direction (§10.2) when setting alarm margins; (f) do not use SHAP outputs for any model modification | Phase 7 obligations — machine-readable and fail-closed |
| **`limitations`** | `n_rtf_units: 2`, `operating_regime: constant`, `failure_modes: [inner_race, rolling_element]`, `extrapolation_permitted: false` | Explicit scope boundary — no fleet claims |
| **`versioning`** | `model_version`, `experiment_id`, `upstream_dataset_version`, `upstream_contract_schema_version` | Full lineage chain back through Phase 5 |

**Referenced artifacts (paths declared in contract):**

| Artifact | Contract field | Consumer role |
|----------|---------------|---------------|
| `models/<experiment_id>/refit_model.pkl` | `model.path` | **Primary**: the model Phase 7 loads for prediction |
| `models/<experiment_id>/model.pkl` | `model.fold_models_path` | Provenance: the per-fold models used for honest evaluation |
| `reports/model_predictions.parquet` | `predictions.path` | Held-out LOBO predictions (grain-aligned) for Phase 7 to compare against |
| `reports/experiment_log.json` | `experiment.log_path` | Full experiment history for audit |
| `reports/model_manifest.json` | `manifest.path` | Selected-model provenance + environment |
| `config/config.yaml` (model_dev section) | `experiment.config_fingerprint` | Reproducibility: re-run with this config + seed to regenerate |

**Ownership & validation:** Phase 6 owns and writes the contract; Phase 7 validates `schema_version` + `prediction_schema` column presence + `model.path` existence on entry and **fails closed** on any mismatch. The contract is the **single source of truth** for the Phase 6 → 7 interface — Phase 7 never reads Phase 6 `.md` reports for control flow.

---

## 18. Configuration Strategy (Design Item — config)

New `model_dev` block in `config/config.yaml` (values illustrative; **no hardcoding in modules**):
- `random_seed: 42`
- `feature_columns:` (loaded from the Phase 5 contract, not re-declared — single source of truth)
- `channel_handling: {mode: per_channel_rows, aggregate: false}`
- `baselines: {mean: true, median: true, linear: true}`
- `candidates: {random_forest: true, xgboost: true, lightgbm: false, catboost: false}`
- `model_params: {random_forest: {...defaults...}, xgboost: {...defaults...}}`
- `preprocessing: {scaler: standard, imputer: median, fit_scope: train_fold_only}`
- `hyperparameter_search: {enabled: false, method: within_train_unit_temporal, early_life_fraction: 0.3, grid: {...}, early_stopping_rounds: 50}`
- `metrics: {primary: mae, near_failure_threshold_minutes: <cfg>, report_asymmetry: true}`
- `selection: {metric: mae, must_beat_baselines: true, tie_breaker: simpler_model}`
- `refit_for_handoff: {enabled: true, population: all_rtf}`
- `output_paths: {model_dir, model_predictions, model_contract, evaluation_report, model_selection_report, feature_importance_report, experiment_log, model_manifest, model_validation_report}`

RTF/feature membership is **inherited from the Phase 5 contract**, not re-declared, preventing drift.

---

## 19. Validation Strategy (Design Item — validation)

Three severities: **FATAL** (stops execution, non-zero exit), **WARNING** (logged, recorded, continues), **INFO**.

| Rule | Severity | Rationale |
|------|----------|-----------|
| Phase 5 contract `schema_version` / columns mismatch on entry | FATAL | Fail-closed handoff; consuming a drifted dataset is unsafe |
| Any preprocessing fit uses rows outside the train fold | FATAL | Leakage (L-07 inheritance) |
| Test bearing appears in a training fold | FATAL | Unit-disjointness violation (L-01/L-03 inheritance) |
| Prediction contains NaN / negative RUL | FATAL | Invalid output; RUL ≥ 0 by definition |
| Selected model fails to beat naive baselines on either fold | FATAL | A model worse than "predict the mean" must not be handed off |
| Model contract schema ≠ registered artifact | FATAL | Contract integrity |
| Missing seed / config fingerprint in experiment log | FATAL | Reproducibility governance |
| Large RMSE/MAE gap (few catastrophic misses) | WARNING | Flag near-failure instability for human review |
| High two-fold metric spread (one failure mode much worse) | WARNING | Model may not generalize across failure modes; surface, don't block |
| Native vs. permutation importance divergence | WARNING | Possible biased importance; note in report |
| NaN fraction imputed per fold | INFO | Documents imputation impact |

`model_validation_report.md` enumerates every rule with observed values. Exit code is non-zero if any FATAL fires (**new `EXIT_ERROR_MODEL_DEV`, proposed 70**, consistent with the 10/20/30/40/50/60 convention).

---

## 20. Failure Handling (Design Item — failure)

- **Missing/invalid input** (dataset absent, contract mismatch, schema drift) → FATAL, `EXIT_ERROR_MODEL_DEV`, clear message; **no model artifact or contract written**.
- **Leakage detected** (preprocessing scope, unit overlap) → pipeline halts before model registration; no model can reach Phase 7 from a leaky fit.
- **Model fails baseline gate** → selection halts; no contract emitted (Phase 7 cannot consume an unvalidated model).
- **SHAP/importance failure** (e.g., explainer error) → WARNING, degrade gracefully to permutation-only importance; does not block model handoff (interpretability is required but a single explainer's failure should not sink a validated model — the failure is logged loudly).
- **Atomic writes** guarantee readers never see half-written artifacts.
- **Warnings** are surfaced in reports and logs but do not stop the run.

---

## 21. Acceptance Criteria

Phase 6 is accepted when **all** hold:
1. Phase 5 contract validated on entry; pipeline fails closed on any mismatch.
2. All baselines (mean, median, linear) evaluated under LOBO and reported.
3. All in-scope candidates trained with **in-fold-only** preprocessing; no leakage check fires.
4. Selected model beats both naive baselines on **both** folds; selection rationale documented.
5. Metrics reported **per fold + two-fold spread** with the n=2 methodology caveat everywhere.
6. Permutation + SHAP importance produced (global + local), computed on held-out folds.
7. Model registered with version, config + model fingerprint, seed, environment capture, and full Phase 3→6 lineage.
8. `phase6_model_contract.yaml` validates and its declared prediction schema equals the registered artifact.
9. Refit-for-handoff model produced on all RTF data.
10. Full test suite (unit + integration + leakage + reproducibility + contract + selection) passes; Phase 3/4/5 suites remain green (no regressions).

---

## 22. Architecture Assumptions

1. Phase 5 outputs (`training_dataset.parquet`, `fold_assignments.parquet`, contract) are frozen, correct, and leakage-free.
2. The 14 feature columns and RTF/censored partition are authoritative from the Phase 5 contract; Phase 6 never redefines them.
3. Both channels of a bearing share the bearing's RUL label and trajectory (not independent samples).
4. n=2 RTF units → all results are methodology validation, not fleet-level accuracy.
5. Continuous features only → no categorical-specialized model is warranted.

## 23. Out of Scope

- Recomputing labels, splits, leakage checks, or features (frozen upstream).
- Deep learning / sequence models (insufficient trajectories).
- LightGBM / CatBoost / stacking (no validatable benefit here).
- Maintenance thresholds, alarm policy, cost optimization, scheduling (Phase 7).
- Dashboards / visualization product (Phase 7+).
- Deployment / serving / streaming inference (Phase 8+).
- Multi-dataset generalization (IMS Set 2/3, other bearings/regimes).
- Managed experiment-tracking infrastructure (MLflow server, model store) — file-based governance is deliberate.

## 24. Data Lineage

```
Phase 3 health + gating ─┐
Phase 4 feature_store + feature_manifest ─┐
Phase 5 training_dataset + phase5_dataset_contract ─┐
                                                    └─> Phase 6
   experiments → selected/refit model → phase6_model_contract.yaml → Phase 7
```
Each hop records the upstream version + config fingerprint in `model_manifest.json`, giving an auditable chain from raw acquisition to trained model.

---

## 25. Risk Register

| ID | Risk | Likelihood | Impact | Mitigation |
|----|------|-----------|--------|------------|
| R-01 | n=2 units → non-generalizable model | Certain | High | LOBO CV; n=2 caveat in every report + contract; framed as methodology validation |
| R-02 | Analyst tunes hyperparameters and overfits/leaks | Medium | High | Default no-tuning; optional path uses within-train-unit slice only, clearly labelled weaker evidence |
| R-03 | Preprocessing fit on full data leaks test distribution | Medium | High | In-fold-only fit enforced structurally + FATAL validation check + leakage test |
| R-04 | Channel rows treated as independent → over-optimistic metrics | Medium | Medium | Metrics interpreted at unit grain; honesty guardrail states 1 test bearing per fold |
| R-05 | Model worse than baseline handed to Phase 7 | Low | High | Baseline-beating gate (FATAL) before contract emission |
| R-06 | Contract drift between Phase 6 and 7 | Medium | Medium | Contract tests + `schema_version` fail-closed gate |
| R-07 | Over-prediction (late warning) unsafe for maintenance | Medium | High | Signed-error / late-rate metric reported; risk direction stated in contract for Phase 7 |
| R-08 | Refit model performance misread as independently validated | Medium | Medium | Contract labels refit performance as the n=2 LOBO estimate, not a new validation |
| R-09 | Serialized-model non-reproducibility across library builds | Low | Low | Model fingerprint + metrics are the reproducibility check; library versions captured |

---

## 26. Implementation Notes (for the future implementation phase — not this one)

- Reuse existing helpers: config loader, logger, `_get_git_commit`, `_compute_config_version`, atomic `Path.replace()` write pattern.
- Add `EXIT_ERROR_MODEL_DEV` (proposed **70**) to `constants.py`.
- Load `feature_columns` and RTF/censored partition **from the Phase 5 contract**, never re-declare them, to prevent drift.
- Preprocessing (scaler, imputer) must be a pipeline object **fit inside each fold** — structurally impossible to see the held-out bearing.
- Thread the single `random_seed` into every stochastic estimator and split operation.
- Keep experiment tracking file-based (JSON) — no tracking server.
- SHAP TreeExplainer for tree models; permutation importance for all; compute both on held-out folds.
- Do **not** introduce LightGBM/CatBoost/deep models or new features.

---

## 27. Final Self-Review

1. **Would you approve this architecture before implementation?** Yes.
2. **Is it scientifically sound?** Yes — baselines precede advanced models so every delta is attributable; the LOBO protocol and leakage-free in-fold preprocessing are inherited and re-asserted; MAPE is correctly rejected for a zero-reaching target; and the n=2 limitation governs every reported number rather than being hidden.
3. **Is it reproducible?** Yes — folds are consumed not sampled, all stochastic components are seeded, config + model fingerprints and library versions are captured, and reproducibility is a tested acceptance criterion.
4. **Is it maintainable?** Yes — single-responsibility modules, config-driven, contract-governed, atomic writes, full test taxonomy, consistent with Phase 3–5 conventions and exit-code scheme.
5. **Is it interview-defensible?** Yes — every decision (baseline-first, no-tuning-on-n=2, LOBO, MAE primary, MAPE rejected, LightGBM/CatBoost/DL excluded with reasons, file-based tracking, refit-for-handoff) has an explicit, comparative justification.
6. **Would a Principal ML Engineer approve this architecture?** Yes — it refuses to manufacture an optimistic accuracy number, prevents leakage structurally, is honest about n=2, is fully traceable, and right-sizes infrastructure to a single engineer.

### Verdict

**APPROVE ARCHITECTURE**

**Reasons:** The design centres on the governing constraint (two run-to-failure units), enforces a disciplined baseline-first model progression so results are attributable, inherits and structurally re-asserts the Phase 5 leakage-free LOBO protocol, chooses evaluation metrics correct for a zero-reaching RUL target while rejecting MAPE for principled reasons, treats interpretability and reproducibility as first-class tested deliverables, and hands Phase 7 a versioned, fingerprinted, machine-readable model contract with explicit consumer obligations and an honest n=2 caveat. It reflects industrial engineering quality without over-engineering, and every decision is interview-defensible.

**STOP — architecture only. No implementation begun.**

---

## 28. Changelog

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-07-04 | Initial Phase 6 architecture (Model Development & Evaluation). Architecture only. |
| 1.1 | 2026-07-04 | Documentation refinements per architecture review, prior to freeze: (1) added §7a Baseline Engineering Purpose Table with per-model engineering question/pass/fail; (2) added §7b Model Promotion Criteria with 5-rule promotion gate + demotion rule; (3) expanded §8.3 into a hierarchical "Best Model" definition with 5-priority selection rule + hard gates; (4) added §10.5 Overfitting Policy with identification signals, monitoring strategy, and rejection criteria; (5) added §10.6 Error Analysis with 6 decomposition dimensions; (6) added §11a SHAP Governance Policy — SHAP for interpretation only, never for model selection, with permitted/prohibited use table; (7) strengthened §16 Reproducibility with explicit 4-input requirement, per-output guarantee table, and floating-point tolerance specification; (8) expanded §17 Phase 6→7 Contract with complete artifact enumeration table and referenced-artifact paths. No change to architecture, modules, data flow, pipeline, or artifacts. |
