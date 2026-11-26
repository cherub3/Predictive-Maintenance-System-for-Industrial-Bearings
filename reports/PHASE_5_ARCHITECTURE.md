# Phase 5 Architecture — RUL Target Construction & Leakage Prevention Layer

**Status:** ARCHITECTURE ONLY — NO IMPLEMENTATION
**Phase:** 5 — RUL Target Construction & Leakage Prevention
**Architecture Version:** 1.0
**Date:** 2026-07-04
**Authoring Roles:** Principal ML Architect · Principal Reliability Engineer · Principal Data Scientist · Principal Software Architect
**Upstream Contract:** `reports/feature_manifest.json`, `data/processed/feature_store.parquet` (17,248 rows × 17 cols, frozen)
**Downstream Consumer:** Phase 6 — Model Development

---

## 1. Executive Summary

Phase 5 converts the frozen Phase 4 **feature store** into a **machine-learning-ready, leakage-free, split-assigned training dataset** carrying a scientifically defensible **Remaining Useful Life (RUL)** target. It is the sole dataset Phase 6 may consume.

The architecture is dominated by one physical fact about NASA IMS Set 1: of the four instrumented bearings, **only two run to functional failure** — `bearing_3` (inner-race defect) and `bearing_4` (rolling-element defect). `bearing_1` and `bearing_2` survive the test and are **right-censored** (no failure event). A defined RUL target therefore exists **only for the two run-to-failure (RTF) units**. Every downstream decision — labelling, splitting, leakage control, evaluation honesty — flows from this.

Because the supervised RUL problem has an effective sample size of **n = 2 units**, the architecture deliberately rejects any splitting scheme that could manufacture optimistic results (random row splits, time splits inside a unit). It adopts **unit-wise Leave-One-Bearing-Out (LOBO)** as the only honest evaluation protocol, and it treats leakage prevention as a first-class, independently-validated deliverable rather than a byproduct.

The design targets **industrial engineering quality without enterprise over-engineering**: config-driven, fully traceable, atomically written, contract-governed, and interview-defensible at each decision.

---

## 2. Objectives

1. Define RUL precisely, with stated physical meaning, assumptions, and limitations.
2. Construct RUL labels **only** for units with a real failure event; censor the rest explicitly.
3. Guarantee temporal integrity (per-unit chronology, no future information).
4. Provide a **complete, enumerated, independently-validated leakage-prevention framework**.
5. Produce a splitting strategy correct for predictive maintenance, with alternatives compared and rejected.
6. Emit versioned, fingerprinted, lineage-linked artifacts.
7. Publish a machine-readable **Phase 5 → Phase 6 contract** that is the single source of truth for the modelling phase.

---

## 3. Responsibilities

**In scope (Phase 5 owns):**
- Failure-event indexing per unit.
- RUL label generation and label dictionary.
- Temporal validation of the feature store.
- Leakage identification, prevention, and reporting.
- Split (fold) assignment at unit granularity.
- Assembly of the training dataset (features + label + split assignment + lineage keys).
- Dataset manifest, governance metadata, and the Phase 6 contract.

**Explicitly NOT Phase 5's responsibility (belongs to Phase 6):**
- Feature scaling / normalization of *inputs* (must be fit on train folds inside Phase 6 to avoid leakage).
- Model selection, training, hyperparameter search, metrics.
- Imputation strategy for model inputs (Phase 5 only *documents* NaN provenance from Phase 4 trend windows).

Keeping input-scaling out of Phase 5 is a deliberate leakage-avoidance decision (see §11, L-07).

**Responsibility boundary (Phase 5 vs. Phase 6):**

| Activity | Phase 5 | Phase 6 |
|----------|:-------:|:-------:|
| Failure indexing / censoring classification | ✅ | ❌ |
| RUL label generation | ✅ | ❌ |
| Temporal integrity validation | ✅ | ❌ |
| Leakage checks (L-01…L-09) | ✅ | ❌ |
| Dataset splitting (LOBO fold assignment) | ✅ | ❌ |
| Dataset manifest / Phase 6 contract | ✅ | ❌ |
| Feature scaling / normalization | ❌ | ✅ |
| Imputation of NaN feature values | ❌ | ✅ (Phase 5 documents provenance only) |
| Model selection & training | ❌ | ✅ |
| Hyperparameter tuning | ❌ | ✅ |
| Metric computation & reporting | ❌ | ✅ |
| Explainability (e.g. SHAP) analysis | ❌ | ✅ |

This table is the authoritative responsibility boundary; if a future task doesn't clearly fall on one side, it belongs to whichever phase can perform it **without** touching data outside its own train-fold view (Phase 5 never sees a "train/test" model-fitting context; Phase 6 must never recompute labels, splits, or leakage checks).

---

## 4. Inputs

| Input | Source | Purpose | Validation on entry |
|-------|--------|---------|---------------------|
| `data/processed/feature_store.parquet` | Phase 4 | Feature matrix (17,248 × 17) | Row count, schema, dtypes, key uniqueness |
| `reports/feature_manifest.json` | Phase 4 | Upstream version + gate lineage | Version + config fingerprint recorded |
| `reports/phase3_gating_decisions.yaml` | Phase 3 | Gate provenance (read-through only) | Presence + schema_version |
| `config/config.yaml` (`rul_target` section) | Project | All Phase 5 parameters | Required keys present, types valid |

Phase 5 **never** parses human-readable reports for control flow (same governance rule enforced in Phase 4: machine-readable contracts only).

---

## 5. Outputs (Artifacts)

| Artifact | Format | Consumer | Purpose |
|----------|--------|----------|---------|
| `data/processed/rul_labels.parquet` | Parquet | Internal / audit | Per-row RUL label + failure index + censoring flag |
| `data/processed/training_dataset.parquet` | Parquet | **Phase 6 (primary)** | Features + label + split assignment + lineage keys |
| `data/processed/splits/fold_assignments.parquet` | Parquet | Phase 6 | Row → fold/role mapping (LOBO) |
| `reports/phase5_dataset_contract.yaml` | YAML | **Phase 6 (machine-readable contract)** | Schema, units, RTF/censored unit lists, versions |
| `reports/dataset_manifest.json` | JSON | Governance | Full provenance, fingerprints, counts |
| `reports/label_dictionary.json` | JSON | Docs / Phase 6 | Label column semantics, units, ranges |
| `reports/leakage_report.md` | Markdown | Human review | Every leakage source + how prevented + evidence |
| `reports/split_summary.md` | Markdown | Human review | Fold composition, unit balance, rationale |
| `reports/validation_report.md` | Markdown | Human review | Every validation rule + pass/warn/fail |

Contract vs. narrative separation mirrors Phase 3/4: `.yaml`/`.json` are machine inputs; `.md` are human-only and must never be parsed downstream.

---

## 6. RUL Definition (Design Item A)

### 6.1 Selected definition
> **RUL(t) for a run-to-failure unit is the remaining operating time from timestamp `t` until the unit's functional failure event `t_fail`, expressed primarily in elapsed minutes and secondarily in remaining acquisition cycles (files).**
>
> `RUL(t) = (t_fail − t)` in minutes, with `RUL(t_fail) = 0`.

- **Primary unit: minutes (physical time).** Chosen because IMS acquisition intervals are **non-uniform** (~5 min early, ~10 min late). A cycle/file count would silently mix two time scales; minutes are physically invariant. This is the same reasoning that fixed the Phase 4 trend-slope time basis.
- **Secondary unit: remaining cycles (file count).** Retained for interpretability and cross-study comparability, explicitly labelled as interval-dependent.
- **Optional derived: fractional life** `RUL_frac(t) = RUL(t) / total_life` ∈ [0, 1], config-gated, default **off**.

### 6.2 Physical meaning
RUL is the reliability-engineering time-to-failure of the specific bearing under the test's constant load/speed regime. It is a property of a **degradation trajectory**, not of an isolated sample.

### 6.3 Engineering assumptions

**Explicit failure event definition** (what "failure" means in this dataset):
- **Failure is defined as the dataset termination point of a run-to-failure unit** — the **last valid acquisition file** recorded for that bearing before the test was stopped. IMS Set 1 tests were run *until* physical failure was detected (per NASA's experiment documentation) and then halted; the final file is treated as coincident with the failure event, not an arbitrary cutoff.
- **Failure is treated as instantaneous at that final timestamp for labelling purposes** (`RUL(t_fail) = 0`), even though the underlying physical degradation is progressive. This is a deliberate simplification: the *degradation* is progressive (reflected continuously in the feature trajectory), but the *event* used to anchor the RUL countdown is the discrete last-observed timestamp, since no finer-grained failure timestamp exists in the dataset.
- Failure is **monotone and terminal**: once failed, the unit does not recover, so RUL cannot increase over time within a unit.
- Load, speed, and environment are constant across the run (true for IMS Set 1).

This definition is intentionally the simplest one consistent with the available evidence — it does not require detecting a degradation "onset" point (that risk is isolated to the optional, off-by-default knee-capping flag in §6.5).

### 6.4 Dataset assumptions & the censoring decision (**central**)
- IMS Set 1 contains **exactly two RTF units**: `bearing_3`, `bearing_4`. RUL is defined for these only.
- `bearing_1`, `bearing_2` are **right-censored**: they survived; their true `t_fail` is unknown and strictly greater than the observation window. Assigning them any finite RUL would be **fabricated supervision** and a leakage/validity violation.
- Decision: censored units carry `is_censored = True`, `rul_minutes = NULL`, and are **excluded from supervised split roles**. They are retained in `rul_labels.parquet` for audit and as an optional unlabeled reference, never in the labelled training folds.

### 6.5 Alternatives considered
| Alternative | Description | Why rejected (as primary) |
|-------------|-------------|---------------------------|
| Piecewise-linear / knee-capped RUL | Flat RUL until degradation onset, linear thereafter | Requires a health-onset detector (extra model + assumptions); deferred as **optional config flag**, off by default, to keep Phase 5 assumption-light and interview-defensible |
| Fractional life ∈ [0,1] | Normalize by total life | Discards absolute time needed for maintenance lead-time decisions (Phase 7); kept as optional derived column |
| Binary "healthy / near-failure" classification | Threshold on RUL | Throws away regression signal; reframes the stated project objective (RUL *prediction*) |
| Discrete cycle-only RUL | Count remaining files | Non-uniform intervals corrupt the target; kept only as secondary |

### 6.6 Limitations (stated, not hidden)
- **n = 2 RTF units, one instance per failure mode** → no statistical generalization; results are demonstrative, not production-validated. This is documented in `dataset_manifest.json` and the Phase 6 contract so no downstream reader over-claims.
- Linear time-to-failure assumes the whole run is "consumable" life; if true degradation onset is late, early RUL values are optimistic — mitigated by the **optional** knee-capping flag.

---

## 6a. Statistical Limitations

Stated explicitly and up front, because it governs how every downstream number must be read:

- **Only two complete run-to-failure trajectories exist** in this dataset (`bearing_3`: inner-race defect, `bearing_4`: rolling-element defect). `bearing_1`/`bearing_2` are censored survivors and contribute no failure label.
- **Any performance metric computed under LOBO CV (§10) will have high variance** — it is the average (or spread) of exactly two evaluation runs, each testing on a single trajectory. A metric reported from n=2 has no meaningful confidence interval in the classical statistical sense.
- **Metrics from this dataset should be interpreted qualitatively** (does the model track the correct trend and rough magnitude of degradation?) **rather than as precise, production-grade accuracy estimates** (e.g., "±X minutes RMSE" should not be read as a guarantee that would hold on a fleet).
- **This dataset is appropriate for demonstrating a methodology** — an RUL definition, a leakage-free pipeline, a defensible evaluation protocol — **not for estimating fleet-wide predictive performance.** Two failure modes on one bearing family under one fixed operating regime cannot be extrapolated to different bearing types, loads, speeds, or failure modes without new data.
- **Any conclusion drawn in Phase 6 or later is constrained by this dataset's size** and must be scoped accordingly; claims of generalizable accuracy beyond these two units are out of scope and not supported by the evidence.

This is not treated as a weakness to obscure — it is treated as a known, load-bearing constraint that the architecture (RUL definition, censoring treatment, LOBO protocol) is explicitly designed around, and it is precisely why the project's value is assessed on methodology, not on a headline accuracy number.

## 6b. Methodology Validation vs. Production Validation

The objective of Phase 5 (and the project's model-development phases that consume it) is to **validate the engineering methodology** — the RUL definition, the feature engineering pipeline, the leakage-prevention strategy, the evaluation protocol, and the maintenance decision framework built on top of them — **not to claim production-ready predictive accuracy on a fleet of bearings.**

Concretely:
- A demonstrated ability to build a leakage-free, unit-disjoint, temporally-valid RUL dataset **is** the deliverable of this phase.
- A LOBO CV score obtained from it is a **sanity check that the methodology works end-to-end**, not a production SLA.
- Statements like "this model achieves X RUL accuracy" must always be qualified with "under LOBO evaluation on the two available NASA IMS Set 1 run-to-failure units," never presented as a fleet-level guarantee.

This distinction is stated here explicitly so that the two-failure-unit constraint (§6a) is read as an accepted scope boundary of a methodology demonstration, not as an unacknowledged weakness of a production claim.

---

## 7. Label Construction (Design Item B)

**Pipeline (design, not code):**
1. **Failure indexing** (`failure_indexer`): for each RTF unit, `t_fail = max(timestamp)` of that unit's valid rows. Cross-checked (warning-level) against Phase 3 health evidence (kurtosis peak / RMS runaway) to confirm the end-of-run is a genuine degradation terminus; a mismatch raises a **warning**, not a stop (Phase 3 remains authoritative on health, Phase 5 owns the label).
2. **Per-row RUL** (`label_generator`): `rul_minutes = (t_fail − t)` over that unit's chronologically ordered rows; `rul_cycles = index distance to failure row`.
3. **Censoring**: non-RTF units → `is_censored=True`, labels NULL.
4. **Optional normalization**: `rul_frac` and/or knee-cap `R_early` applied only if enabled in config.
5. **Label attaches at the (bearing_id, channel, timestamp) grain** — identical to the feature-store grain — so no row multiplication occurs.

**Justification of key choices:** minutes over cycles (§6.1); `t_fail` = last valid acquisition (§6.3 assumption 2); RTF-only labelling (§6.4). Every RTF unit's `rul_minutes` is, by construction, **monotonically non-increasing in time and non-negative**, which becomes a hard validation invariant (§11).

---

## 8. Temporal Integrity (Design Item C)

Owned by `temporal_validator`, run **before** labelling:
- **Chronology preservation:** operate per `bearing_id`; sort by `timestamp`; never reorder across units.
- **Time ordering:** assert strictly increasing timestamps within each (bearing_id, channel); equal/decreasing → fatal.
- **Timestamp validation:** dtype = `datetime64[ns]`; no NaT; range within the frozen run window (2003-10-22 → 2003-11-25).
- **Sequence integrity:** each unit has the full expected acquisition count; gaps are permitted but **recorded** (they are already reflected as NaNs in Phase 4 `trend_*` and must not be silently reinterpolated).
- **Future-information protection:** the label uses `t_fail` which is a per-unit constant known only at the unit level; it is **never** derived from a global maximum across units, and no feature is recomputed here.

---

## 9. Leakage Prevention Framework (Design Item D)

Leakage prevention is a **standalone, independently-validated deliverable** (`leakage_validator` + `leakage_report.md`), not an implicit side effect. Each identified source has a prevention mechanism and a machine-checkable assertion.

| ID | Leakage source | Mechanism of harm | Prevention (design) | Automated check |
|----|----------------|-------------------|---------------------|-----------------|
| L-01 | **Future samples in a split** | Adjacent timestamps in train & test share near-identical degradation state | Split at **whole-unit** granularity (LOBO); no unit appears in two roles | No `bearing_id` in >1 role |
| L-02 | **Rolling-window features spanning split boundary** | Phase 4 `trend_*` windows mix rows that a row-split would separate | Unit-wise split makes window boundaries irrelevant (whole trajectory stays together) | Window never crosses a role boundary (guaranteed by L-01) |
| L-03 | **Bearing overlap across folds** | Same physical unit trains and tests the model | Group key = `bearing_id`; enforced disjoint | Set-intersection of unit IDs per role = ∅ |
| L-04 | **Duplicate timestamps** | Same acquisition counted twice, inflates apparent data / cross-role bleed | Uniqueness enforced on (bearing_id, channel, timestamp) | Duplicate key count = 0 |
| L-05 | **Checkpoint leakage** | Phase 3 Tier-2 checkpoint files reused as if independent | Phase 5 consumes the full feature store, not checkpoints; checkpoints excluded by construction | No join to `checkpoint_selection.csv` |
| L-06 | **Derived/label leakage** | Label built from features, or `t_fail` from future features | Label depends only on `timestamp` + unit identity, never on feature values | Label columns independent of feature columns (provenance assertion) |
| L-07 | **Normalization/statistics leakage** | Scaling fit on full dataset leaks test distribution | Phase 5 computes **no global input statistics**; scaling deferred to Phase 6 fit-on-train | No scaler/global-mean artifact emitted |
| L-08 | **Censored-unit contamination** | Fabricated RUL for surviving units enters supervision | Censored units excluded from labelled roles (§6.4) | No `is_censored=True` row in train/val/test roles |
| L-09 | **Target normalization leakage** | `total_life` for `rul_frac` drawn from test unit | `total_life` is a per-unit constant internal to that unit; never pooled across units | Per-unit normalization only |

`leakage_report.md` documents each row above with the concrete evidence (counts, set operations) produced during the run. Any failed check is **fatal**.

---

## 10. Dataset Splitting Strategy (Design Item E)

### 10.1 Selected strategy
**Unit-wise Leave-One-Bearing-Out (LOBO) cross-validation over the RTF units**, i.e. **GroupKFold with group = `bearing_id` restricted to {bearing_3, bearing_4}**:
- Fold 1: train = `bearing_3`, test = `bearing_4`
- Fold 2: train = `bearing_4`, test = `bearing_3`

Validation, when a held-out val set is needed for early stopping, is carved **temporally from within the training unit's early life** (past-only) so it never sees the training unit's failure region — configurable, off by default. There is **no fixed single hold-out test set**, because with two units a single split is statistically meaningless; **LOBO CV is the reported protocol**.

### 10.2 Why this is correct for predictive maintenance
The deployment question is "given a *new, unseen* bearing, predict its RUL." Only unit-disjoint evaluation answers that. Row- or time-splits answer the easier, irrelevant question "given other samples *from the same bearing*, interpolate its RUL" — which leaks trajectory identity and reports inflated scores.

### 10.3 Evaluation Strategy Justification

The question LOBO is designed to answer is:

> **"Can a model trained on one failed bearing's degradation trajectory generalize to predict the RUL of a completely unseen failed bearing?"**

This is the actual deployment question for predictive maintenance — a model is trained on historical failures and must generalize to a bearing it has never seen operate. Every alternative below was evaluated against this question, not against raw achievable accuracy.

| Alternative | Why considered | Why rejected |
|-------------|-----------------|---------------|
| **Random split (row-level)** | Simplest, maximizes apparent training data | **L-01 leakage.** Adjacent timestamps from the same trajectory land in both train and test; the model interpolates within a degradation curve it has partially seen. Scores are near-perfect and meaningless — this doesn't test generalization to a new bearing at all. |
| **Random timestamp split** | Attempts to split "by time" rather than by row | Same failure as random row split: timestamps from `bearing_3` still appear in both roles unless units are also separated. Without a group key, temporal randomization does not prevent unit bleed. |
| **Time-based split within a bearing** (e.g. first 70% train, last 30% test, per unit) | Mimics a "predict the future of a run in progress" framing | Trains and tests on the *same* trajectory. The model can memorize unit-specific baseline/noise characteristics; it never has to generalize across bearings. Also answers a different, narrower question ("extrapolate this bearing's own curve") than the stated project objective ("predict RUL for a new bearing"). |
| **Last 20% holdout (single split, within or across units without grouping)** | Common default in tabular ML | If within-unit: same leakage as above. If across units without a group key: risk of a unit's rows spanning the boundary, plus a single point estimate carries no sense of variance across failure modes. |
| **Expanding window** (grow the training window forward in time, evaluate on the next block) | Natural for streaming/production monitoring problems | Requires *multiple* sequential failure events per unit to be meaningful; IMS Set 1 gives one continuous run per unit ending in a single terminal failure, so there is no "next window" after failure — the technique doesn't map onto a run-to-failure dataset with only two units. |
| **k-fold ignoring groups** | Standard default CV | Violates L-03 directly; a bearing's rows are scattered across folds, guaranteeing leakage identical to random split. |
| **Single fixed train/test (e.g. train on b3 only, test on b4 only)** | Simple, unit-disjoint, no leakage | Valid but wasteful and fragile: it discards half of the already-scarce RTF evidence for training in each direction, and reports one point estimate with no sense of the variance between failure modes (inner-race vs rolling-element). |

**Why LOBO despite the small training population per fold:** every alternative that permits more training data per fold does so by either (a) violating unit-disjointness (leakage) or (b) trivializing the task into within-trajectory extrapolation. LOBO is the only protocol that is simultaneously (1) leakage-free by construction, (2) faithful to the actual generalization question, and (3) uses **both** available failure modes as test cases across its two folds rather than sacrificing one permanently to a single fixed holdout. The small per-fold training population is a property of the *dataset* (n=2), not a flaw introduced by the split — every alternative either hides this limitation behind leakage or discards data without fixing it.

### 10.4 Honesty guardrail
`split_summary.md` states explicitly that n = 2 → LOBO CV is demonstrative. This prevents Phase 6 from reporting a single optimistic number as if validated. See also §6a (Statistical Limitations) and §6b (Methodology vs. Production Validation).

---

## 11. Module Architecture (Design Item G)

All modules are **design specifications only** in this phase. Grain = (bearing_id, channel, timestamp) throughout. Dependencies flow downward; no cycles.

| Module | Responsibility | Inputs | Outputs | Depends on |
|--------|----------------|--------|---------|-----------|
| `config` (loader, existing) | Provide `rul_target` params | `config.yaml` | typed config dict | — |
| `temporal_validator` | Enforce §8 integrity | feature store | pass/fail + findings | config |
| `failure_indexer` | Compute `t_fail` per RTF unit; classify censoring | feature store, config (RTF unit list) | per-unit failure index table | temporal_validator |
| `label_generator` | Produce `rul_minutes`, `rul_cycles`, optional `rul_frac` | failure index, feature store | `rul_labels.parquet` | failure_indexer |
| `leakage_validator` | Run all L-01…L-09 checks | labels, splits, feature store | `leakage_report.md`, fatal/pass | split_generator, label_generator |
| `split_generator` | LOBO fold assignment (group=bearing_id, RTF only) | labels, config | `fold_assignments.parquet`, `split_summary.md` | label_generator |
| `dataset_builder` | Join features + labels + splits + lineage keys | all above | `training_dataset.parquet` | leakage_validator (must pass first) |
| `label_dictionary` | Emit label semantics/units/ranges | labels | `label_dictionary.json` | label_generator |
| `dataset_manifest` | Provenance, fingerprints, counts | all artifacts | `dataset_manifest.json` | dataset_builder |
| `phase6_contract` | Emit machine-readable contract | manifest, dataset | `phase5_dataset_contract.yaml` | dataset_manifest |
| `validation` | Aggregate all rules → `validation_report.md`, gate exit | all | report + exit code | all |
| `main` | Orchestrate ordered pipeline, logging, exit codes | config | pipeline run | all |
| `tests/` | §13 test suite | — | — | all |

**Ordering invariant enforced by `main`:** temporal_validator → failure_indexer → label_generator → split_generator → **leakage_validator (gate)** → dataset_builder → dictionary/manifest/contract → validation. `dataset_builder` **must not run** until leakage checks pass.

---

## 12. Processing Pipeline & Data Flow

```
feature_store.parquet ─┬─> temporal_validator ──> (integrity OK)
                       │                              │
feature_manifest.json ─┘                              v
                                              failure_indexer ──> per-unit t_fail + censoring
                                                     │
                                                     v
                                              label_generator ──> rul_labels.parquet + label_dictionary.json
                                                     │
                                                     v
                                              split_generator ──> fold_assignments.parquet + split_summary.md
                                                     │
                                                     v
                                     ┌────── leakage_validator (GATE) ──> leakage_report.md
                                     │ (fatal on any fail)
                                     v
                              dataset_builder ──> training_dataset.parquet
                                     │
                                     v
                   dataset_manifest.json + phase5_dataset_contract.yaml
                                     │
                                     v
                              validation ──> validation_report.md ──> exit code
```

---

## 13. Validation Strategy (Design Item H)

Three severities: **FATAL** (stops execution, non-zero exit), **WARNING** (logged, recorded, continues), **INFO**.

| Rule | Severity | Rationale |
|------|----------|-----------|
| Negative RUL on any RTF row | FATAL | Violates definition; indicates ordering/index bug |
| RUL not monotonically non-increasing within RTF unit | FATAL | Degradation is terminal; non-monotonicity = bug |
| Missing label on an RTF row | FATAL | Every RTF row must be labelled |
| Duplicate label key (bearing_id, channel, timestamp) | FATAL | L-04 |
| Future timestamp (> run window) or NaT | FATAL | §8 |
| Chronology violation (non-increasing) | FATAL | §8 |
| Split overlap by `bearing_id` | FATAL | L-01/L-03 |
| Censored unit in a labelled role | FATAL | L-08 |
| Row-count contract (training rows = labelled RTF rows) | FATAL | Governance/count integrity |
| `t_fail` disagrees with Phase 3 health terminus | WARNING | Label authority is Phase 5; flag for human review |
| Acquisition gap inside a unit | WARNING | Already reflected as Phase 4 NaNs; record, don't repair |
| Fraction of NaN trend features per fold | INFO | Informs Phase 6 imputation choice |

`validation_report.md` enumerates every rule with observed values. Exit code is non-zero if any FATAL fires (new `EXIT_ERROR_RUL_TARGET`, e.g. 60, consistent with the 40/50 convention).

---

## 14. Testing Strategy (Design Item I)

| Test type | What it verifies |
|-----------|------------------|
| **Unit** | `rul_minutes = t_fail − t`; monotonic non-increasing; `RUL(t_fail)=0`; censored units → NULL; cycle vs minute consistency on uniform-interval fixtures |
| **Unit (indexing)** | `failure_indexer` picks last valid row; correct RTF vs censored classification from config |
| **Unit (splits)** | LOBO produces disjoint unit sets; only RTF units assigned roles; each fold covers all RTF units across folds |
| **Validation tests** | Each FATAL rule *fires* on a crafted violating fixture (negative RUL, duplicate key, split overlap, censored-in-train, future timestamp) — tests that the guard actually triggers |
| **Integration** | End-to-end on a small synthetic 2-unit fixture → all artifacts produced, contract schema valid |
| **Regression** | Label counts, RTF/censored partition, and fold composition stable across runs on frozen inputs |
| **Contract tests** | `phase5_dataset_contract.yaml` matches schema; declared columns/units/versions equal actual `training_dataset.parquet` |
| **Leakage tests** | Injected leakage (same unit in two roles, row-split) is caught and made FATAL |

---

## 15. Governance (Design Item J)

- **Dataset version** + **label version** (semantic, e.g. `1.0`) in manifest and contract.
- **Configuration fingerprint:** SHA-256 of serialized `rul_target` config (reusing the Phase 3/4 `_compute_config_version` pattern).
- **Git commit** short SHA captured (reusing existing helper).
- **Data lineage:** manifest records upstream `feature_manifest` version + its config fingerprint + `phase3_gating_decisions.yaml` schema version, forming an unbroken chain Phase 3 → 4 → 5.
- **Reproducibility:** deterministic ordering (sort keys fixed); no randomness in labelling or LOBO assignment (folds are enumerated, not sampled); any optional sampling seeded and recorded.
- **Atomic writes:** all artifacts written to `*.tmp` then `Path.replace()` (the cross-platform-safe pattern adopted in Phase 3), never partial files.
- **Immutability:** artifacts regenerated wholesale each run; never hand-edited (header banner states this, as in the Phase 3 contract).

**Reproducibility statement:** given (a) identical Phase 4 inputs (`feature_store.parquet`, `feature_manifest.json`) and (b) an identical `rul_target` configuration block, Phase 5 produces **byte-for-byte identical outputs** on every run. This holds because: label generation is a pure function of timestamps and unit identity (no randomness); LOBO fold assignment is enumerated over a fixed, config-declared unit list (not sampled); and all artifacts are written deterministically (fixed sort keys, atomic replace). The configuration fingerprint recorded in `dataset_manifest.json` lets any two runs be compared and confirmed identical without re-executing the pipeline.

---

## 16. Phase 5 → Phase 6 Contract (Design Item K)

`reports/phase5_dataset_contract.yaml` — the **only** machine-readable interface Phase 6 may consume.

**Specified content:**
- `schema_version`, `producer` (phase, module, generated_at, git_commit, config_version) — same envelope as Phase 3 contract.
- `dataset`: path to `training_dataset.parquet`, row count, grain keys `[bearing_id, channel, timestamp]`.
- `label_spec`: `rul_minutes` (float, ≥0, primary), `rul_cycles` (int, secondary), optional `rul_frac`; units and valid ranges.
- `unit_partition`: `rtf_units: [bearing_3, bearing_4]`, `censored_units: [bearing_1, bearing_2]`.
- `split_spec`: protocol = `LOBO_cv`, enumerated folds with train/test unit lists; validation policy.
- `feature_columns`: explicit list (the 14 `td_/fd_/trend_/env_` columns) — Phase 6 must not invent features here.
- `nan_provenance`: which columns carry legitimate NaNs (Phase 4 `trend_*` window startup / gap nulling) and the count, so Phase 6 chooses imputation deliberately.
- `consumer_expectations`: Phase 6 must (a) fit any scaler on train folds only, (b) honour unit-disjoint folds, (c) not pool `total_life` across units, (d) report LOBO CV, not a single split.
- `versioning`: dataset_version, label_version, upstream feature_manifest_version.

**Ownership & validation:** Phase 5 owns and writes it; Phase 6 validates `schema_version` and column presence on entry and **fails closed** if mismatched.

---

## 17. Configuration Strategy (Design Item — config)

New `rul_target` block in `config/config.yaml` (values illustrative; **no hardcoding in modules**):
- `rtf_units: [bearing_3, bearing_4]`
- `censored_units: [bearing_1, bearing_2]`
- `rul_primary_unit: minutes`
- `emit_rul_cycles: true`
- `emit_rul_frac: false`
- `knee_capping: {enabled: false, r_early_minutes: null}`
- `split: {protocol: lobo_cv, val_from_train_early_fraction: 0.0}`
- `run_window: {start: 2003-10-22, end: 2003-11-25}`
- `output_paths: {rul_labels, training_dataset, fold_assignments, dataset_contract, dataset_manifest, label_dictionary, leakage_report, split_summary, validation_report}`

RTF/censored membership lives in config (auditable, not buried in code), matching the project's config-driven principle.

---

## 18. Failure Handling (Design Item — Failure)

- **Missing/invalid input** (feature store absent, wrong row count, schema drift) → FATAL, `EXIT_ERROR_RUL_TARGET`, clear message; no partial artifacts.
- **Leakage gate failure** → pipeline halts *before* `dataset_builder`; no training dataset is written, so a leaky dataset can never reach Phase 6.
- **Validation FATAL** → non-zero exit; artifacts from earlier steps remain for diagnosis but the contract is **not** emitted (Phase 6 cannot consume an unvalidated dataset).
- **Atomic writes** guarantee readers never see half-written files.
- **Warnings** are surfaced in reports and logs but do not stop the run.

---

## 19. Acceptance Criteria

Phase 5 is accepted when **all** hold:
1. Every RTF row has a non-negative, monotonically non-increasing `rul_minutes`; `RUL(t_fail)=0`.
2. Censored units carry NULL labels and appear in **no** labelled split role.
3. All nine leakage checks (L-01…L-09) pass and are evidenced in `leakage_report.md`.
4. Splits are unit-disjoint (LOBO); `split_summary.md` documents composition + n=2 honesty caveat.
5. `training_dataset.parquet` row count equals labelled RTF row count and matches the contract.
6. `phase5_dataset_contract.yaml` validates and its declared schema equals the actual dataset.
7. Full test suite (unit + integration + validation + contract + leakage) passes; Phase 3/4 suites remain green (no regressions).
8. Manifest carries complete lineage + config fingerprint + git commit.

---

## 20. Architecture Assumptions

1. IMS Set 1 RTF units are exactly `bearing_3` and `bearing_4`; `bearing_1/2` are censored (config-declared, human-verifiable against Phase 3 findings).
2. Last valid acquisition of an RTF unit = operational failure point.
3. Constant operating regime (load/speed) across the run.
4. Feature store grain and dtypes are frozen and correct (Phase 4 accepted).

## 21. Architecture Scope

RUL definition, labelling, temporal validation, leakage prevention, unit-wise splitting, dataset assembly, governance metadata, and the Phase 6 contract for NASA IMS Set 1.

## 22. Out of Scope

- Input feature scaling/normalization (Phase 6, fit-on-train).
- Model training, tuning, metrics (Phase 6).
- Maintenance decision thresholds / dashboard (Phase 7+).
- Multi-dataset generalization (IMS Set 2/3, other bearings).
- Survival-analysis modelling of censored units (documented as future work; Phase 5 only preserves them).
- Streaming/online inference infrastructure.

## 23. Data Lineage

```
Phase 1 metadata ─┐
Phase 3 health + phase3_gating_decisions.yaml ─┐
Phase 4 feature_store.parquet + feature_manifest.json ─┐
                                                       └─> Phase 5
   rul_labels → training_dataset → phase5_dataset_contract.yaml → Phase 6
```
Each hop records the upstream version + config fingerprint in `dataset_manifest.json`, giving an auditable chain from raw acquisition to training dataset.

---

## 24. Risk Register

| ID | Risk | Likelihood | Impact | Mitigation |
|----|------|-----------|--------|------------|
| R-01 | n=2 RTF units → non-generalizable model | Certain | High | Documented in contract + split_summary; LOBO CV; framed as demonstrative |
| R-02 | Linear RUL over-credits early life | Medium | Medium | Optional knee-capping flag; limitation stated |
| R-03 | Analyst reintroduces leakage in Phase 6 (scaling on full data) | Medium | High | Contract `consumer_expectations` + Phase 6 fail-closed validation |
| R-04 | `t_fail` misidentified (test artifact at end of run) | Low | High | Warning-level cross-check vs Phase 3 health terminus; human review of `validation_report.md` |
| R-05 | Censored units silently labelled | Low | High | L-08 FATAL check + config-declared partition |
| R-06 | Contract drift between Phase 5 and 6 | Medium | Medium | Contract tests + `schema_version` gate |
| R-07 | Non-uniform intervals corrupt cycle-based reasoning | Medium | Medium | Minutes are primary; cycles labelled interval-dependent |

---

## 25. Changelog

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-07-04 | Initial Phase 5 architecture (RUL target construction & leakage prevention). Architecture only. |
| 1.1 | 2026-07-04 | Documentation refinements per architecture review, prior to freeze: (1) expanded LOBO justification into a dedicated Evaluation Strategy Justification comparing six alternatives against the explicit generalization question; (2) added §6a Statistical Limitations; (3) added §6b Methodology Validation vs. Production Validation; (4) made the failure-event definition (last valid acquisition, instantaneous-at-that-point) explicit in §6.3; (5) added a Phase 5/Phase 6 responsibility table to §3; (6) added an explicit reproducibility statement to §15. No change to architecture, algorithms, data flow, or artifacts. |

---

## 26. Implementation Notes (for the future implementation phase — not this one)

- Reuse existing helpers: config loader, logger, `_get_git_commit`, `_compute_config_version`, atomic `Path.replace()` write pattern.
- Add `EXIT_ERROR_RUL_TARGET` (proposed 60) to `constants.py`.
- Keep every threshold/parameter in the `rul_target` config block; modules read, never hardcode.
- Leakage gate must be a hard barrier before `dataset_builder` — structurally impossible to emit a leaky dataset.
- Do **not** compute any global input statistics in Phase 5.

---

## 27. Final Self-Review

1. **Would you approve this architecture before implementation?** Yes.
2. **Is the phase scientifically correct?** Yes — RUL is defined only for run-to-failure units; censored units are explicitly excluded rather than fabricated; time-based labelling respects non-uniform intervals.
3. **Is it maintainable?** Yes — single-responsibility modules, config-driven, contract-governed, atomic writes, full test taxonomy, consistent with Phase 3/4 conventions.
4. **Is it interview-defensible?** Yes — every decision (minutes vs cycles, LOBO vs random, censoring, deferring scaling to Phase 6) has an explicit, comparative justification.
5. **Does it properly prevent data leakage?** Yes — nine enumerated sources, unit-wise splitting that structurally eliminates the dominant risks, a hard leakage gate before dataset assembly, and consumer-side expectations enforced in the Phase 6 contract.
6. **Would this satisfy a Principal ML Engineer reviewing it?** Yes — it is honest about n=2, prevents optimistic evaluation, and is fully traceable.

### Verdict

**APPROVE ARCHITECTURE**

**Reasons:** The design correctly centres on the governing physical constraint (only two run-to-failure units), refuses to fabricate labels for censored bearings, selects the one leakage-free evaluation protocol appropriate to predictive maintenance (unit-wise LOBO), treats leakage prevention as an independently-validated gate rather than an afterthought, and hands Phase 6 a versioned, fingerprinted, machine-readable contract with explicit consumer obligations. It reflects industrial engineering quality without over-engineering, and every decision is interview-defensible.

**STOP — architecture only. No implementation begun.**
