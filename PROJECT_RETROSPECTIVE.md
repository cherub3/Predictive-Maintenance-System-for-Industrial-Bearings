# Project Retrospective

*Written in first person, as the engineer(s) who actually did the work — not the
"Principal Engineer" persona used in the frozen architecture documents. This is
the messy, real, chronological story: what we got wrong, what we found, what we
fixed, and why the final answer is more honest (and more interesting) than the
first one.*

---

## Where this started

The pipeline was originally built against **NASA IMS Bearing Set 1 only** — 4
bearings, 2 of them run-to-failure (`bearing_3`, inner race; `bearing_4`,
rolling element). Every phase — signal analysis, feature engineering, RUL
labeling, LOBO cross-validation, model selection, business rules — was designed
and frozen against that single dataset.

That's a real constraint: **n=2**. Two failure trajectories is not enough to
train anything flexible without massive overfitting risk, and not enough to
report a confidence interval on anything. The project's own architecture
documents said this out loud, which is good. But an early review of the
project (looking at it as a hiring manager might) surfaced something worse
than the small sample: the *reported reason* for the small-n outcome was
wrong.

## Finding a bug in our own conclusion

The Phase 6 model-selection report said, in effect, "no model beats the median
baseline — the features carry no usable signal." That's a strong, falsifiable
claim, and it was false. Re-deriving the actual per-fold numbers showed Random
Forest and XGBoost **beating** the median baseline by a wide margin — 12.7%
lower mean MAE, ~78% lower MAE in the safety-critical near-failure region. They
weren't promoted for a *mechanical* reason: with only two RTF units, LOBO CV
produces exactly two folds, each training on a single trajectory. A flexible
model can nearly interpolate one trajectory (train MAE → 0), which trips the
overfitting gate by construction — and with only two folds, winning decisively
on one and losing narrowly on the other is enough to fail the "beat baseline
on every fold" rule, regardless of whether the model is actually any good.

That's `ENGINEERING_DECISIONS.md` ED-004. It's the single most important
finding in this project, because it's the difference between "the ML doesn't
work" and "the ML works, and here is the exact, falsifiable mechanism blocking
its promotion." We rewrote the model-selection narrative, the dashboard (which
had gone stale and still said the old, wrong thing — a bug in itself, caught
by literally clicking through the dashboard), and the final report to state
this precisely.

## Deciding to get more data

ED-004 names its own resolution: get more failure trajectories. NASA's IMS
dataset has two more sets — Set 2 (`bearing_1`, outer race) and Set 3
(`bearing_3`, outer race) — that were never ingested. That's not a hard
technical problem, but it *is* a real one: the whole pipeline was written
assuming one dataset, one channel layout (8 channels, paired X/Y), one raw
directory. Adding two more datasets with a different channel layout
(single-axis, 4 channels) touches every phase.

We planned this as an incremental, risk-ordered expansion — config registry
first, then Phase 1 through 7 one at a time, verifying against real files at
each step rather than refactoring blind — specifically because file-parsing
assumptions are exactly the kind of thing that fails silently. That caution
paid off immediately.

## A second bug, this time in the data itself

Once Set 3 was downloaded, checking it against its own bundled README turned
up a real discrepancy: the README claims the test ends April 4, 2004, but the
actual files run to April 18. Plotting RMS and kurtosis across that extra two
weeks showed a textbook failure signature — a slow climb, then a kurtosis
spike into double digits, then the sensor going silent — sitting entirely
inside the "extra" data the README doesn't document. Trusting the README
blindly would have truncated the dataset *before* the actual failure event and
mislabeled a genuinely-degrading bearing as artificially censored. That's
`ENGINEERING_DECISIONS.md` ED-005.

This is the kind of thing that's easy to miss if you don't actually look at
the raw signal before trusting a distributor's documentation — and a good
reminder that "the README says so" isn't the same as "the data says so."

## The multi-dataset refactor, and the bug class that kept recurring

The core design decision for adding datasets was to introduce a qualified
`unit_id = f"{dataset_id}::{bearing_id}"` grain key everywhere bearing identity
matters, instead of adding `dataset_id` as a second key to every groupby/merge
call site. The reasoning: `bearing_id` alone is **not unique** across
datasets — Set 1's `bearing_3` (inner race, 2003) and Set 3's `bearing_3`
(outer race, 2004) are physically unrelated bearings that happen to share a
name.

That sounds like an edge case until you actually go looking for where bare
`bearing_id` was still being used as if it were a unique key. It wasn't an
edge case — it was almost everywhere: Phase 5's failure indexer, label
generator, and leakage validator; Phase 6's fold-based train/test filtering,
feature importance, and SHAP; Phase 7's channel aggregation and final
recommendation output. Every one of these would have either crashed, silently
produced garbage (blending two physically unrelated bearings' predictions
into one aggregate), or — worst of all — quietly leaked one dataset's test
unit into another dataset's training fold, because a naive `isin()` filter on
bare `bearing_id` doesn't know the difference.

Finding and fixing this systematically, phase by phase, verifying against a
real end-to-end run each time rather than trusting that a design principle
had actually been applied everywhere, was most of the actual engineering work
in this expansion. The final Phase 7 recommendations output correctly shows
four distinct, disambiguated units — `set1::bearing_3`, `set1::bearing_4`,
`set2::bearing_1`, `set3::bearing_3` — with no collisions, which is the
concrete, checkable proof that this class of bug is actually fixed and not
just less likely to trigger.

## What we found at n=4

With four RTF units instead of two, LOBO CV now runs four folds. The honest
result (`ENGINEERING_DECISIONS.md` ED-006):

- **XGBoost's advantage over the baseline widened**, not shrank — ~24% lower
  mean MAE, up to ~90% lower near-failure MAE in some folds. The signal in the
  features was real at n=2 and it's still real, more convincingly, at n=4.
- **The demotion fallback moved one rung more conservative**, from median to
  mean, because median itself now fails to beat mean on the largest fold
  (`set3::bearing_3`). That's a genuinely new finding, not a bug: Set 3's
  bearing has a different life scale than Set 1's bearings, and the flat
  median generalizes worse to it than the flat mean does.
- Every model shows extreme negative R² on the shortest fold
  (`set2::bearing_1`, a ~9.8-day run) — a real, physical outlier trajectory
  that makes every-fold comparison genuinely hard across three datasets with
  different operating durations.

We did not adjust the promotion thresholds to force a "nicer" headline. The
every-fold-beat rule and overfitting gate are frozen architectural decisions;
applying them faithfully to new data and reporting what comes out, including
an outcome that's *more* conservative than before, is the point of having
frozen rules in the first place.

## What's still not done

- The promotion gates are strict by design, and XGBoost's clear advantage
  still isn't enough to clear them at n=4. A fold-structure review (should the
  every-fold-beat rule be evaluated differently when folds have very different
  sizes and life scales?) is a legitimate next question — not something we
  resolved here, since it would mean touching the frozen architecture rather
  than applying it.
- Risk thresholds are cited against reliability-engineering literature but
  still explicitly flagged as pending domain-expert validation
  before any real deployment decision.
- This was the first session with meaningful iteration; there is still no git
  history prior to it. We'd recommend initializing a repository now and
  committing incrementally from this point forward, so future iteration is
  visible the way this retrospective describes it, rather than requiring
  another written reconstruction.

---

*This document reflects the state of the project as of the multi-dataset
(Set 1+2+3) expansion. It will go stale the moment the next real change lands;
treat it as a snapshot of how we got here, not a permanent description of
where the project is.*
