"""Model Promotion Criteria (architecture §7b) and Best Model selection
hierarchy (architecture §8.3), plus refit-for-handoff (architecture §8.4).

This module NEVER imports `shap_analysis` and takes only an `EvaluationResult`
(metrics) as input — SHAP outputs cannot reach model selection because there
is no parameter through which they could be passed (architecture §11a
governance, enforced structurally, not by convention).
"""

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.model_dev.baseline_models import fit_linear_baseline, fit_mean_baseline, fit_median_baseline
from src.model_dev.cross_validation import CVResults, CANDIDATE_ORDER
from src.model_dev.evaluation import EvaluationResult, ModelEvaluation
from src.model_dev.model_trainer import train_candidate
from src.model_dev.preprocessing import FittedPreprocessor, fit_train_preprocessor, transform_features

NON_BASELINE_CANDIDATES = ["linear"] + CANDIDATE_ORDER


@dataclass
class PromotionDecision:
    """§7b promotion outcome for one model relative to its immediate ladder predecessor."""

    model_name: str
    predecessor: Optional[str]
    promoted: bool
    reasons: List[str] = field(default_factory=list)


@dataclass
class PromotionLedger:
    """All promotion decisions across the ladder (Mean -> Median -> Linear -> RF -> XGBoost)."""

    decisions: Dict[str, PromotionDecision] = field(default_factory=dict)


@dataclass
class SelectionResult:
    """Outcome of the §8.3 Best Model selection hierarchy."""

    selected_model: str
    hard_gate_survivors: List[str]
    rationale: List[str]
    resolved_at_priority: int
    resolved_by: str


def _fold_maes(model_eval: ModelEvaluation) -> Dict[str, float]:
    return {fold_id: fm.metrics.mae for fold_id, fm in model_eval.fold_metrics.items()}


def _fold_near_failure_maes(model_eval: ModelEvaluation) -> Dict[str, float]:
    return {fold_id: fm.metrics.near_failure_mae for fold_id, fm in model_eval.fold_metrics.items()}


def _is_overfit_flagged(model_eval: ModelEvaluation, threshold: float) -> bool:
    """§10.5: flagged if train/test MAE ratio is below threshold on EITHER fold.

    A NaN ratio (test_mae == 0, i.e. a degenerate perfect-test fold) is treated as
    flagged rather than silently ignored — `NaN < threshold` is False, so without
    this guard a degenerate fold would escape the overfitting check entirely.
    """
    for fm in model_eval.fold_metrics.values():
        ratio = fm.metrics.overfitting_ratio
        if math.isnan(ratio) or ratio < threshold:
            return True
    return False


def _is_stable(model_eval: ModelEvaluation, instability_threshold: float) -> bool:
    return model_eval.stability_ratio <= instability_threshold


def apply_promotion_criteria(
    evaluation_result: EvaluationResult, config: Dict[str, Any], logger: logging.Logger
) -> PromotionLedger:
    """Apply the 5-rule promotion gate (architecture §7b) at every rung of the ladder.

    Rules, ALL must hold for `promoted=True`:
      1. Baseline exceeded on both folds (candidate MAE < predecessor MAE, per fold).
      2. Improvement is meaningful (mean MAE improvement >= equivalence_margin_minutes).
      3. Complexity justified (improvement >= margin * complexity_justification_multiplier,
         OR the near-failure MAE improved — evidence the added complexity captures
         degradation structure the simpler model missed).
      4. No evidence of overfitting (§10.5 ratio check on both folds).
      5. Reproducible (structural guarantee — deterministic seeded pipeline; always True).

    Args:
        evaluation_result: Output of `evaluation.evaluate_cv_results`.
        config: Full project configuration dictionary.
        logger: Pipeline logger.

    Returns:
        PromotionLedger with one PromotionDecision per non-floor rung.
    """
    promotion_cfg = config.get("model_dev", {}).get("promotion", {})
    overfitting_cfg = config.get("model_dev", {}).get("overfitting_policy", {})
    margin = promotion_cfg.get("equivalence_margin_minutes", 30.0)
    multiplier = promotion_cfg.get("complexity_justification_multiplier", 1.5)
    overfit_threshold = overfitting_cfg.get("train_test_ratio_threshold", 0.3)

    ledger = PromotionLedger()
    order = evaluation_result.model_order

    for i in range(1, len(order)):
        candidate_name = order[i]
        predecessor_name = order[i - 1]
        candidate = evaluation_result.model_evaluations[candidate_name]
        predecessor = evaluation_result.model_evaluations[predecessor_name]

        reasons: List[str] = []
        passed_all = True

        # Rule 1: baseline exceeded on both folds
        cand_maes = _fold_maes(candidate)
        pred_maes = _fold_maes(predecessor)
        common_folds = set(cand_maes) & set(pred_maes)
        beats_both_folds = all(cand_maes[f] < pred_maes[f] for f in common_folds)
        if beats_both_folds:
            reasons.append(f"Rule 1 PASS: beats {predecessor_name} on all {len(common_folds)} fold(s)")
        else:
            reasons.append(f"Rule 1 FAIL: does not beat {predecessor_name} on every fold")
            passed_all = False

        # Rule 2: improvement is meaningful
        mean_improvement = predecessor.mean_mae - candidate.mean_mae
        meaningful = mean_improvement >= margin
        if meaningful:
            reasons.append(f"Rule 2 PASS: mean MAE improved by {mean_improvement:.2f} min (>= margin {margin})")
        else:
            reasons.append(f"Rule 2 FAIL: mean MAE improvement {mean_improvement:.2f} min < margin {margin}")
            passed_all = False

        # Rule 3: complexity justified
        cand_nf = _fold_near_failure_maes(candidate)
        pred_nf = _fold_near_failure_maes(predecessor)
        common_nf_folds = set(cand_nf) & set(pred_nf)
        near_failure_improvement = (
            float(np.mean([pred_nf[f] - cand_nf[f] for f in common_nf_folds])) if common_nf_folds else float("nan")
        )
        complexity_justified = (mean_improvement >= margin * multiplier) or (near_failure_improvement > 0)
        if complexity_justified:
            reasons.append(
                f"Rule 3 PASS: complexity justified (improvement={mean_improvement:.2f} min, "
                f"near-failure improvement={near_failure_improvement:.2f} min)"
            )
        else:
            reasons.append(
                f"Rule 3 FAIL: complexity not justified (improvement={mean_improvement:.2f} min "
                f"< margin*multiplier={margin * multiplier:.2f}, near-failure improvement="
                f"{near_failure_improvement:.2f} min <= 0)"
            )
            passed_all = False

        # Rule 4: no evidence of overfitting
        overfit_flagged = _is_overfit_flagged(candidate, overfit_threshold)
        if not overfit_flagged:
            reasons.append(f"Rule 4 PASS: no overfitting flag (ratio threshold {overfit_threshold})")
        else:
            reasons.append(f"Rule 4 FAIL: overfitting flagged on at least one fold (ratio threshold {overfit_threshold})")
            passed_all = False

        # Rule 5: reproducible — structural guarantee (seeded, deterministic pipeline)
        reasons.append("Rule 5 PASS: reproducible by construction (seeded, deterministic pipeline)")

        ledger.decisions[candidate_name] = PromotionDecision(
            model_name=candidate_name, predecessor=predecessor_name, promoted=passed_all, reasons=reasons
        )
        logger.info(
            "Promotion decision [%s vs %s]: %s",
            candidate_name, predecessor_name, "PROMOTED" if passed_all else "REJECTED",
        )

    return ledger


def select_best_model(
    evaluation_result: EvaluationResult,
    promotion_ledger: PromotionLedger,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> SelectionResult:
    """Apply the §8.3 Best Model selection hierarchy.

    Hard gates (must pass before the hierarchy is consulted):
      - Beats both naive baselines (mean, median) on both folds.
      - Individually satisfies the §7b promotion criteria vs. its ladder predecessor.
      - Reproducible (structural guarantee).

    Hierarchy (applied in strict order, each only consulted if the prior
    criterion leaves more than one candidate "effectively tied"):
      1. Lowest mean MAE.
      2. Stable performance across both folds (preferred among near-ties).
      3. No overfitting evidence.
      4. Simpler model preferred when effectively tied.
      5. More interpretable model as final tiebreak.

    Args:
        evaluation_result: Output of `evaluation.evaluate_cv_results`.
        promotion_ledger: Output of `apply_promotion_criteria`.
        config: Full project configuration dictionary.
        logger: Pipeline logger.

    Returns:
        SelectionResult naming the selected model and the full rationale trace. If
        no non-baseline candidate clears the hard gates, the §7b demotion rule
        ("the simpler predecessor is selected") is applied and a baseline
        (median or mean) is returned as the selected model — this is a valid,
        honestly-documented outcome, not a pipeline failure.

    Raises:
        ValueError: Only in the degenerate case where the single remaining
            near-tied survivor is overfit-flagged and no alternative exists.
    """
    selection_cfg = config.get("model_dev", {}).get("selection", {})
    overfitting_cfg = config.get("model_dev", {}).get("overfitting_policy", {})
    instability_threshold = selection_cfg.get("instability_ratio_threshold", 2.0)
    overfit_threshold = overfitting_cfg.get("train_test_ratio_threshold", 0.3)
    margin = config.get("model_dev", {}).get("promotion", {}).get("equivalence_margin_minutes", 30.0)

    order = evaluation_result.model_order
    baseline_names = {"mean", "median"}
    candidate_names = [m for m in order if m not in baseline_names]

    rationale: List[str] = []

    # Hard gates
    survivors: List[str] = []
    for model_name in candidate_names:
        model_eval = evaluation_result.model_evaluations[model_name]
        mean_eval = evaluation_result.model_evaluations.get("mean")
        median_eval = evaluation_result.model_evaluations.get("median")

        beats_mean = all(
            _fold_maes(model_eval)[f] < _fold_maes(mean_eval)[f] for f in _fold_maes(model_eval)
        ) if mean_eval else True
        beats_median = all(
            _fold_maes(model_eval)[f] < _fold_maes(median_eval)[f] for f in _fold_maes(model_eval)
        ) if median_eval else True

        promoted = promotion_ledger.decisions.get(model_name)
        promoted_ok = promoted.promoted if promoted else True  # first candidate always has a promotion decision

        if beats_mean and beats_median and promoted_ok:
            survivors.append(model_name)
            rationale.append(f"HARD GATE PASS: {model_name} beats both baselines and satisfies promotion criteria")
        else:
            rationale.append(
                f"HARD GATE FAIL: {model_name} excluded "
                f"(beats_mean={beats_mean}, beats_median={beats_median}, promoted={promoted_ok})"
            )

    if not survivors:
        # Demotion rule (architecture §7b, frozen text): "if a more complex model fails
        # any of these criteria, the simpler predecessor is selected." No non-baseline
        # candidate individually cleared the hard gate — walk the ladder frontier
        # (Mean -> Median -> Linear -> RF -> XGBoost) and stop at the last rung that
        # was actually promoted over its own predecessor. This always terminates in at
        # least "mean" (the floor never requires promotion over anything) and therefore
        # always yields a valid, honestly-documented selection instead of a pipeline crash.
        last_valid = order[0]
        for model_name in order[1:]:
            decision = promotion_ledger.decisions.get(model_name)
            if decision is not None and decision.promoted:
                last_valid = model_name
            else:
                break
        rationale.append(
            "HARD GATES: no non-baseline candidate beat BOTH naive baselines on EVERY fold "
            "while also satisfying its own §7b promotion criteria. Applying the demotion "
            f"rule: the ladder-valid frontier stops at '{last_valid}'. NOTE: this is NOT "
            "necessarily evidence that the features lack degradation signal — check whether "
            "the flexible models (random_forest, xgboost) achieved lower MEAN test MAE than "
            "the median baseline despite failing the every-fold-beat rule; if so, they lose "
            "on some fold(s) rather than all of them, and any training fold with few RTF "
            "units risks the overfitting policy flagging it by construction (a flexible "
            "learner can more easily interpolate a small number of trajectories). The "
            "conservative baseline is retained pending collection of additional trajectories "
            "or a fold-by-fold review. See ENGINEERING_DECISIONS.md for the fold-count-specific "
            "analysis at the time of the most recent training run."
        )
        return SelectionResult(
            selected_model=last_valid, hard_gate_survivors=[], rationale=rationale,
            resolved_at_priority=0, resolved_by="demotion_fallback",
        )

    # Priority 1: lowest mean MAE
    survivors_sorted = sorted(survivors, key=lambda m: evaluation_result.model_evaluations[m].mean_mae)
    best_mae = evaluation_result.model_evaluations[survivors_sorted[0]].mean_mae
    near_tied = [m for m in survivors_sorted if evaluation_result.model_evaluations[m].mean_mae - best_mae <= margin]
    rationale.append(f"PRIORITY 1 (lowest MAE): survivors ranked {survivors_sorted}; near-tied group={near_tied}")

    if len(near_tied) == 1:
        selected = near_tied[0]
        model_eval = evaluation_result.model_evaluations[selected]
        if _is_overfit_flagged(model_eval, overfit_threshold):
            rationale.append(f"PRIORITY 3 (overfitting check on clear winner): {selected} flagged — re-evaluating without it")
            remaining = [m for m in survivors_sorted if m != selected]
            if not remaining:
                raise ValueError(f"Clear MAE winner {selected} is overfit-flagged and no alternative survivor remains.")
            return select_best_model(
                EvaluationResult(
                    model_evaluations={k: v for k, v in evaluation_result.model_evaluations.items() if k != selected},
                    model_order=[m for m in evaluation_result.model_order if m != selected],
                ),
                promotion_ledger, config, logger,
            )
        rationale.append(f"RESOLVED at priority 1: {selected} is the clear lowest-MAE survivor")
        return SelectionResult(
            selected_model=selected, hard_gate_survivors=survivors, rationale=rationale,
            resolved_at_priority=1, resolved_by="lowest_mae",
        )

    # Priority 2: stability among near-tied
    stable = [m for m in near_tied if _is_stable(evaluation_result.model_evaluations[m], instability_threshold)]
    rationale.append(f"PRIORITY 2 (stability, threshold={instability_threshold}): stable subset={stable}")
    if len(stable) == 1:
        rationale.append(f"RESOLVED at priority 2: {stable[0]} is the only stable near-tied survivor")
        return SelectionResult(
            selected_model=stable[0], hard_gate_survivors=survivors, rationale=rationale,
            resolved_at_priority=2, resolved_by="stability",
        )
    pool = stable if stable else near_tied

    # Priority 3: no overfitting evidence
    not_overfit = [m for m in pool if not _is_overfit_flagged(evaluation_result.model_evaluations[m], overfit_threshold)]
    rationale.append(f"PRIORITY 3 (no overfitting): subset={not_overfit}")
    if len(not_overfit) == 1:
        rationale.append(f"RESOLVED at priority 3: {not_overfit[0]} is the only non-overfit candidate")
        return SelectionResult(
            selected_model=not_overfit[0], hard_gate_survivors=survivors, rationale=rationale,
            resolved_at_priority=3, resolved_by="no_overfitting",
        )
    pool = not_overfit if not_overfit else pool

    # Priority 4/5: simpler + more interpretable — both resolve to "earliest in ladder order"
    pool_sorted_by_ladder = sorted(pool, key=lambda m: order.index(m))
    selected = pool_sorted_by_ladder[0]
    rationale.append(
        f"RESOLVED at priority 4/5 (simpler/more interpretable tiebreak): {selected} "
        f"is earliest in the architecture ladder among remaining tied candidates {pool_sorted_by_ladder}"
    )
    return SelectionResult(
        selected_model=selected, hard_gate_survivors=survivors, rationale=rationale,
        resolved_at_priority=4, resolved_by="simpler_more_interpretable",
    )


def refit_best_model(
    selected_model_name: str,
    training_dataset: pd.DataFrame,
    feature_columns: List[str],
    cv_results: CVResults,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> Tuple[Any, Optional[FittedPreprocessor]]:
    """Refit the selected model type + configuration on ALL RTF data (architecture §8.4).

    The LOBO evaluation gives the honest performance estimate; this artifact
    is the one actually handed to Phase 7, trained on every available failure
    trajectory. Its expected performance is the LOBO estimate quoted in the
    contract — never re-validated here.

    Args:
        selected_model_name: Model name chosen by `select_best_model`.
        training_dataset: Full RTF-only training dataset (all bearings).
        feature_columns: Feature column names.
        cv_results: CVResults, used only to recover the hyperparameters that
            were actually used during CV (for candidate models).
        config: Full project configuration dictionary.
        logger: Pipeline logger.

    Returns:
        Tuple of (refit_model, refit_preprocessor). preprocessor is None for
        the constant mean/median baselines (no feature dependency).
    """
    seed = config.get("model_dev", {}).get("random_seed", 42)
    y_all = training_dataset["rul_minutes"].to_numpy(dtype=float)

    if selected_model_name == "mean":
        return fit_mean_baseline(y_all), None
    if selected_model_name == "median":
        return fit_median_baseline(y_all), None

    preprocessor = fit_train_preprocessor(training_dataset, feature_columns, config)
    X_all = transform_features(training_dataset, preprocessor)

    if selected_model_name == "linear":
        return fit_linear_baseline(X_all, y_all), preprocessor

    if selected_model_name in CANDIDATE_ORDER:
        # Recover the hyperparameters actually used in CV (identical across folds by default).
        any_fold_result = next(iter(cv_results.results[selected_model_name].values()))
        params = any_fold_result.hp_record.selected_params if any_fold_result.hp_record else {}
        refit_model = train_candidate(selected_model_name, X_all, y_all, params, seed)
        logger.info("Refit-for-handoff: %s trained on all %d RTF rows", selected_model_name, len(training_dataset))
        return refit_model, preprocessor

    raise ValueError(f"Unrecognized model for refit: {selected_model_name}")


def generate_model_selection_report(
    evaluation_result: EvaluationResult,
    promotion_ledger: PromotionLedger,
    selection_result: SelectionResult,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> str:
    """Render `model_selection_report.md` documenting promotion + selection decisions."""
    lines = [
        "# Phase 6 — Model Selection Report",
        "",
        "## Model Promotion Criteria (architecture §7b)",
        "",
    ]
    for model_name, decision in promotion_ledger.decisions.items():
        lines.append(f"### {decision.predecessor} → {model_name}")
        lines.append(f"**Outcome:** {'PROMOTED' if decision.promoted else 'REJECTED'}")
        lines.append("")
        for reason in decision.reasons:
            lines.append(f"- {reason}")
        lines.append("")

    lines += ["---", "", "## Best Model Selection Hierarchy (architecture §8.3)", ""]
    for step in selection_result.rationale:
        lines.append(f"- {step}")
    lines += [
        "",
        f"**Selected model:** `{selection_result.selected_model}`",
        f"**Resolved at priority:** {selection_result.resolved_at_priority} ({selection_result.resolved_by})",
        f"**Hard-gate survivors considered:** {selection_result.hard_gate_survivors}",
        "",
        "---",
        "",
        "_Generated by Phase 6 pipeline. Do not edit manually._",
    ]

    report_md = "\n".join(lines)
    logger.info("Model selection report generated — selected: %s", selection_result.selected_model)
    return report_md


def save_model_selection_report(report_md: str, output_path: Path, logger: logging.Logger) -> None:
    """Atomically write `model_selection_report.md`."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".md.tmp")
    try:
        tmp_path.write_text(report_md, encoding="utf-8")
        tmp_path.replace(output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    logger.info("Model selection report saved: %s", output_path)
