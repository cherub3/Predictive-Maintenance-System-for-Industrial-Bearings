# Literature Benchmark — NASA IMS Bearing Dataset RUL Prediction

**Purpose:** Compare this project's Phase 6 LOBO evaluation results against published
work that uses the same NASA IMS (Intelligent Maintenance Systems, University of
Cincinnati) bearing dataset, and explain — honestly, not defensively — where the
methodology agrees, where it diverges, and why.

This document is a narrative reference for human review. It is not consumed by any
pipeline phase and carries no schema or contract obligations.

---

## 1. Published Work Surveyed

| Approach | Reported Metric | Notes |
|---|---|---|
| Stacked Autoencoder (SAE) health indicator + LSTM | RMSE ≈ 0.033, MAE ≈ 0.024 (normalized) | HI constructed via SAE bottleneck, RUL regressed via LSTM |
| CNN + Bi-LSTM with domain adaptation | NRMSE ≈ 0.187 | Compared against plain DCNN/I-DCNN baselines |
| CNN–Mogrifier LSTM | ~35% RMSE reduction vs. CNN-LSTM baseline | IMS Data Challenge variant |
| Deep BiLSTM (vs. SVM / RNN / single-layer BiLSTM / LSTM) | RMSE ≈ 0.028 (normalized) | Ablation-style comparison across recurrent architectures |
| Fractal dimension + 1D-CNN | Qualitatively better than LSTM/ELM on IMS | No single normalized-error headline figure available from the search excerpt |

*(Search performed 2026-07-31; full citations were not independently verified by
DOI/author cross-check beyond the search snippets themselves — treat as representative
of the field's general approach and result magnitude, not as a curated, peer-reviewed
literature review.)*

## 2. The Core Methodology Difference: Normalized Fractional Life vs. Raw Minutes

**Every surveyed paper reports error in normalized, unitless terms** — RUL scaled to
`[0, 1]` fractional life, or a "score" metric, rather than raw physical time. This
project deliberately does the opposite: Phase 5 §6.1 selects **raw minutes** as the
primary RUL unit specifically because "fractional life discards absolute time needed
for maintenance lead-time decisions" (Phase 7 acts on physical minutes-to-failure, not
a unitless fraction).

**Consequence:** the reported MAE/RMSE figures in this project (thousands of minutes)
and the published figures (hundredths on a `[0,1]` scale) are **not directly comparable
without a normalization step this project does not perform by design**. This is not a
performance gap to explain away — it is a deliberate, documented scope choice (Phase 5
§6.5 keeps `rul_frac` as an optional, off-by-default derived column specifically so
Phase 6/7 are never tempted to report or optimize against a normalized target that
would be less useful for the actual maintenance-scheduling decision). Any future
comparison against published normalized metrics would first require computing
`rul_frac = rul_minutes / total_life` per unit and re-evaluating — not yet done here.

## 3. The Harder Generalization Test: Cross-Unit LOBO vs. Within-Trajectory Splits

A recurring pattern in IMS-bearing RUL literature is to construct a health indicator
or degradation curve for **a single bearing's own run-to-failure trajectory**, then
split that one trajectory in time (early life = train, late life = test) to evaluate
the RUL regressor. This answers a *different, easier* question than the one this
project's architecture deliberately targets (Phase 5 §10.3): **"given other samples
from the same bearing's own trajectory, interpolate its remaining life"** rather than
**"given a model trained on entirely different bearings, predict RUL for a bearing the
model has never seen operate."**

This project's Leave-One-Bearing-Out (LOBO) design is the harder, more operationally
realistic test — it is exactly the question a real deployment must answer (a new
bearing, never before observed, needs an RUL estimate from a model trained on past
failures). Some published work does use proper unit-disjoint evaluation; where it does
not, direct comparison would overstate the difficulty of what this project achieves
relative to what within-trajectory papers achieve, in the opposite direction from the
normalization gap in §2 — i.e., the two methodology differences partially offset,
and neither is corrected for below.

## 4. Sample Size: n=2 vs. Typical Published Corpora

Published deep-learning approaches (CNN/LSTM/attention architectures) generally
require and use **all four IMS Set 1 bearings**, or pool Set 1/2/3 together, to obtain
enough degradation trajectories to fit models with far more parameters than this
project's baseline/tree-ensemble candidates. This project's Phase 5 architecture
deliberately restricts to Set 1's **2 confirmed run-to-failure units** and explicitly
excludes deep learning (Phase 6 §8.2) precisely because "requires far more trajectories
than n=2 to avoid gross overfitting; would be scientific malpractice on this dataset."
Published deep-learning results are not directly comparable on sample-size grounds
alone — they are typically evaluated with a materially larger, and sometimes
within-trajectory (§3), effective training/evaluation corpus.

**This is the single most concrete, actionable finding from this comparison**: it is
also exactly what the Phase 3 (multi-dataset expansion) work-in-progress in this
project is addressing — bringing in NASA IMS Set 2 and Set 3 raises the RTF sample
from n=2 to n=4, closing part of this gap without requiring deep learning.

## 5. Honest Summary

| Dimension | This project | Typical published work |
|---|---|---|
| RUL units | Raw minutes (decision-actionable) | Normalized `[0,1]` fractional life |
| Evaluation protocol | Leave-One-Bearing-Out, cross-unit (harder) | Often within-trajectory time-split (easier), though not universally |
| RTF sample size | n=2 (Set 1 only, as of this writing) | Typically all 4 Set-1 bearings and/or Set 2/3 pooled |
| Model family | Baselines + tree ensembles (RF/XGBoost); deep learning explicitly excluded per §8.2 | CNN/LSTM/attention/autoencoder architectures |
| Headline result | Median baseline in production; RF/XGBoost beat it on mean & near-failure MAE but can't be promoted under n=2 (ENGINEERING_DECISIONS.md ED-004) | Tuned deep models reporting low normalized error on their chosen split |

**Conclusion:** this project's numbers are not directly comparable to the papers
surveyed, for three compounding, documented reasons (units, split difficulty, sample
size) that cut in different directions. The published literature confirms that
degradation-sensitive architectures (CNN/LSTM) *can* extract strong RUL signal from
this same physical dataset when given more trajectories and, in some cases, an easier
within-trajectory evaluation — which corroborates, rather than contradicts, this
project's own ED-004 finding that its features already carry real degradation signal
(Random Forest beats the median baseline) and that the binding constraint is
specifically the n=2 sample size, not a defect in the feature engineering.

---

_This document is a human-readable research reference. It is not a Phase deliverable,
carries no schema_version, and is not consumed by any pipeline module._
