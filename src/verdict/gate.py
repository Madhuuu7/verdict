"""The gate: run both variants, judge, calibrate, test, correct, decide.

The order matters and each step exists because of a specific way the naive
version gets it wrong.

1. **Run both variants over the same cases.** Same suite, same case ids, so
   every comparison downstream can be paired.
2. **Judge every output, and calibrate the judge first.** If the judge does
   not agree with people nearly as well as another person would, the gate
   returns ``abstain`` and tests nothing. A number produced by an instrument
   that has not been checked is worse than no number, because it will be
   acted on.
3. **Test each metric with a paired test appropriate to its type** -- McNemar
   for pass/fail, signed-rank for the rubric, paired bootstrap for latency and
   for the confidence interval on the effect.
4. **Correct for multiplicity across the whole family**, metrics and slices
   together. A run of this gate on the bundled suite makes 15 comparisons;
   uncorrected, an unchanged pipeline trips at least one of them 54% of the
   time.
5. **Decide sequentially on the primary metric**, with a confidence
   sequence, so stopping early does not inflate the error rate.

The error budget is spent in two explicit halves, and this is the part worth
arguing about. The gate gets two chances to be wrong: it can stop early and be
wrong, or it can run the whole suite and be wrong at the end. Spending the
full ``alpha`` on each would allow up to ``2 * alpha`` in total, so the
sequential stop is tested at ``alpha / 2`` and the end-of-suite family is
corrected at ``alpha / 2``. Crude -- it is a Bonferroni split across two looks
-- but it is honest about the fact that there are two, and the combined null
error rate is measured in ``scripts/validate.py`` rather than assumed.

The decision has four values and ``inconclusive`` is a real one. A gate forced
to answer yes or no on 40 cases will answer wrongly; one that can say "this
suite does not resolve the question" sends you to add cases instead of
shipping on noise.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Literal, Sequence

import numpy as np

from .calibration import Calibration, annotator_ceiling, calibrate, constant_judge_kappa
from .cases import Case, Suite
from .judges import Judge, Judgement, PASS_AT, label_subset
from .pipelines import Output, Pipeline
from .stats.multiplicity import Adjusted, Correction, benjamini_hochberg, holm
from .stats.paired import (
    TestResult,
    mcnemar,
    paired_bootstrap,
    wilcoxon_signed_rank,
)
from .stats.sequential import (
    Decision,
    SequentialResult,
    confidence_sequence,
    empirical_bernstein,
    fixed_horizon,
    peeking,
)

MetricKind = Literal["binary", "ordinal", "continuous"]


class Verdict(str, Enum):
    PASS = "pass"
    REGRESSION = "regression"
    INCONCLUSIVE = "inconclusive"
    #: The judge could not be certified, so nothing was tested.
    ABSTAIN = "abstain"


@dataclass
class GateConfig:
    alpha: float = 0.05
    #: Non-inferiority margin on the primary metric, in rubric points. A
    #: candidate within this much of the baseline is not a regression. Zero
    #: would mean testing a point null, which no real pipeline ever satisfies
    #: and which turns the gate into a coin flip at large n.
    tolerance: float = 0.05
    #: Fraction of the suite that gets human labels.
    label_fraction: float = 0.5
    #: Required share of the inter-annotator ceiling.
    kappa_threshold: float = 0.75
    correction: Literal["bh", "holm"] = "bh"
    bootstrap_draws: int = 4000
    #: Sample size the sub-Gaussian comparison sequence is tuned tightest
    #: at. The gate itself uses the variance-adaptive boundary, which needs no
    #: tuning; this only affects the sub-Gaussian curve shown beside it.
    tuned_for: int | None = None
    min_n: int = 20
    #: Slice the suite by tag as well as testing it whole.
    slice_by_tag: bool = True
    #: Tags with fewer cases than this are not tested separately -- a slice of
    #: nine cases cannot support a comparison and only adds to the
    #: multiplicity burden.
    min_slice: int = 25
    seed: int = 0


@dataclass
class MetricReport:
    label: str
    kind: MetricKind
    higher_is_better: bool
    baseline_mean: float
    candidate_mean: float
    test: TestResult
    adjusted: Adjusted | None = None
    n: int = 0

    @property
    def delta(self) -> float:
        return self.candidate_mean - self.baseline_mean

    @property
    def worse(self) -> bool:
        """Did the candidate move in the bad direction at all?"""
        return self.delta < 0 if self.higher_is_better else self.delta > 0

    @property
    def flagged(self) -> bool:
        """Significant after correction *and* in the bad direction.

        Both halves are needed. A significant improvement is not a regression,
        and a gate that blocks on any significant difference blocks on good
        news too -- which is how teams learn to ignore it.
        """
        if self.adjusted is None:
            return False
        return self.adjusted.rejected and self.worse

    @property
    def flagged_uncorrected(self) -> bool:
        return self.test.p_value <= 0.05 and self.worse


@dataclass
class GateReport:
    verdict: Verdict
    suite: str
    n_cases: int
    calibration: Calibration
    ceiling: Calibration
    constant_judge_kappa: float
    metrics: list[MetricReport]
    correction: Correction
    sequential: SequentialResult
    #: The same data under the invalid procedure, for contrast in the console.
    sequential_peeking: SequentialResult
    sequential_fixed: SequentialResult
    #: And under the honest-but-unusable Hoeffding proxy, which is the reason
    #: the gate uses a variance-adaptive boundary instead.
    sequential_subgaussian: SequentialResult | None = None
    primary: str = "rubric score"
    elapsed_s: float = 0.0
    reason: str = ""
    config: GateConfig = field(default_factory=GateConfig)

    @property
    def flagged(self) -> list[MetricReport]:
        return [m for m in self.metrics if m.flagged]

    @property
    def flagged_uncorrected(self) -> list[MetricReport]:
        return [m for m in self.metrics if m.flagged_uncorrected]

    @property
    def comparisons(self) -> int:
        return len(self.metrics)

    @property
    def resolvable_margin(self) -> float:
        """How small a margin this suite could actually have certified.

        The single most useful number when the verdict is ``inconclusive``,
        and the one a bare "inconclusive" hides. It is the distance from the
        observed effect to the bottom of the end-of-suite interval, so it says
        "with these cases, this judge and this much noise, you can conclude
        non-inferiority to within this many rubric points -- and no finer".

        Shrinking it means more cases or a quieter judge, and the gate is
        reporting which one you need rather than leaving it to be guessed.
        """
        lo = self.sequential_fixed.ci[0]
        return abs(lo) if lo < 0 else 0.0

    def family_error_if_uncorrected(self) -> float:
        return self.correction.family_error_if_uncorrected()

    def summary(self) -> str:
        if self.verdict is Verdict.ABSTAIN:
            return f"ABSTAIN -- {self.reason}"
        head = {
            Verdict.PASS: "PASS",
            Verdict.REGRESSION: "REGRESSION",
            Verdict.INCONCLUSIVE: "INCONCLUSIVE",
        }[self.verdict]
        bits = [
            f"{head} on {self.n_cases} cases",
            f"{self.comparisons} comparisons, {len(self.flagged)} flagged after "
            f"{self.correction.method}",
            f"primary effect {self.sequential.effect:+.3f} "
            f"[{self.sequential.ci[0]:+.3f}, {self.sequential.ci[1]:+.3f}]",
        ]
        if self.sequential.stopped_early:
            bits.append(
                f"stopped at {self.sequential.n_used}/{self.sequential.n_available} "
                f"({self.sequential.saved_fraction:.0%} saved)"
            )
        return " | ".join(bits)


# --------------------------------------------------------------------------
# running the suite
# --------------------------------------------------------------------------


@dataclass
class Run:
    """One variant's outputs and judgements over a suite, in case order."""

    name: str
    cases: list[Case]
    outputs: list[Output]
    judgements: list[Judgement]

    @property
    def grades(self) -> np.ndarray:
        return np.array([j.grade for j in self.judgements], dtype=float)

    @property
    def passes(self) -> np.ndarray:
        return np.array([j.grade >= PASS_AT for j in self.judgements], dtype=bool)

    @property
    def refusals(self) -> np.ndarray:
        return np.array([o.refused for o in self.outputs], dtype=bool)

    @property
    def latencies(self) -> np.ndarray:
        return np.array([o.latency_ms for o in self.outputs], dtype=float)

    def grades_by_id(self) -> dict[str, int]:
        return {c.case_id: j.grade for c, j in zip(self.cases, self.judgements)}


def execute(suite: Suite, pipeline: Pipeline, judge: Judge, name: str) -> Run:
    cases = list(suite)
    outputs = [pipeline(c) for c in cases]
    judgements = [judge(c, o) for c, o in zip(cases, outputs)]
    return Run(name=name, cases=cases, outputs=outputs, judgements=judgements)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def _binary_metric(
    label: str, base: np.ndarray, cand: np.ndarray, *, higher_is_better: bool
) -> MetricReport:
    return MetricReport(
        label=label,
        kind="binary",
        higher_is_better=higher_is_better,
        baseline_mean=float(np.mean(base)),
        candidate_mean=float(np.mean(cand)),
        test=mcnemar(base, cand),
        n=int(base.size),
    )


def _ordinal_metric(
    label: str, base: np.ndarray, cand: np.ndarray, *, higher_is_better: bool = True
) -> MetricReport:
    """Signed-rank for the p-value, bootstrap for the interval.

    Two tests on one metric, deliberately: the signed-rank test is the right
    null hypothesis test for ordinal paired data, and it does not hand back a
    confidence interval on the mean difference, which is the number a reviewer
    actually wants to see. The p-value reported is the signed-rank one; the
    interval comes from the bootstrap.
    """
    signed = wilcoxon_signed_rank(base, cand)
    boot = paired_bootstrap(base, cand, draws=2000)
    test = TestResult(
        name="wilcoxon+bootstrap",
        statistic=signed.statistic,
        p_value=signed.p_value,
        effect=boot.effect,
        ci=boot.ci,
        n=signed.n,
        informative=signed.informative,
        detail=f"{signed.detail}; interval from {boot.detail}",
    )
    return MetricReport(
        label=label,
        kind="ordinal",
        higher_is_better=higher_is_better,
        baseline_mean=float(np.mean(base)),
        candidate_mean=float(np.mean(cand)),
        test=test,
        n=int(base.size),
    )


def _continuous_metric(
    label: str,
    base: np.ndarray,
    cand: np.ndarray,
    *,
    higher_is_better: bool,
    draws: int,
    seed: int,
) -> MetricReport:
    return MetricReport(
        label=label,
        kind="continuous",
        higher_is_better=higher_is_better,
        baseline_mean=float(np.mean(base)),
        candidate_mean=float(np.mean(cand)),
        test=paired_bootstrap(base, cand, draws=draws, seed=seed),
        n=int(base.size),
    )


def build_metrics(
    baseline: Run, candidate: Run, suite: Suite, config: GateConfig
) -> list[MetricReport]:
    """Every comparison the gate will make, before any correction.

    This function is where the multiplicity comes from, so it is worth seeing
    all at once: four suite-level metrics plus one per qualifying tag.
    """
    metrics = [
        _ordinal_metric("rubric score", baseline.grades, candidate.grades),
        _binary_metric(
            "pass rate", baseline.passes, candidate.passes, higher_is_better=True
        ),
        _binary_metric(
            "refusal rate",
            baseline.refusals,
            candidate.refusals,
            higher_is_better=False,
        ),
        _continuous_metric(
            "mean latency",
            baseline.latencies,
            candidate.latencies,
            higher_is_better=False,
            draws=config.bootstrap_draws,
            seed=config.seed,
        ),
    ]

    if config.slice_by_tag:
        index = {c.case_id: i for i, c in enumerate(baseline.cases)}
        for tag in suite.tags:
            ids = [c.case_id for c in suite.slice(tag)]
            if len(ids) < config.min_slice:
                continue
            rows = np.array([index[i] for i in ids if i in index], dtype=int)
            metrics.append(
                _ordinal_metric(
                    f"rubric score [{tag}]",
                    baseline.grades[rows],
                    candidate.grades[rows],
                )
            )
    return metrics


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------


def run_gate(
    suite: Suite,
    baseline_pipeline: Pipeline,
    candidate_pipeline: Pipeline,
    judge: Judge,
    annotators: Sequence[Callable[[Case, Output], Judgement]],
    config: GateConfig | None = None,
) -> GateReport:
    """Run the whole thing and return a report that explains itself."""
    config = config or GateConfig()
    started = time.perf_counter()

    baseline = execute(suite, baseline_pipeline, judge, "baseline")
    candidate = execute(suite, candidate_pipeline, judge, "candidate")

    # -- calibration, on the baseline's outputs -------------------------
    # Deliberately the baseline's: the judge is being checked against human
    # labels for outputs of the pipeline currently in production, which is
    # the only set a team can realistically keep labelled.
    if len(annotators) < 2:
        raise ValueError(
            "two annotators are required: without a second one there is no "
            "inter-annotator ceiling, and the judge would be held to an "
            "absolute bar that may be unreachable"
        )
    labels = [
        label_subset(
            baseline.cases,
            baseline.outputs,
            annotator,
            fraction=config.label_fraction,
        )
        for annotator in annotators[:2]
    ]
    a_grades = {k: v.grade for k, v in labels[0].items()}
    b_grades = {k: v.grade for k, v in labels[1].items()}
    ceiling = annotator_ceiling(a_grades, b_grades)
    cal = calibrate(
        a_grades,
        baseline.grades_by_id(),
        ceiling=ceiling.weighted_kappa,
        threshold=config.kappa_threshold,
        seed=config.seed,
    )
    constant = constant_judge_kappa(a_grades)

    metrics = build_metrics(baseline, candidate, suite, config)
    family = [(m.label, m.test.p_value) for m in metrics]
    # Half the budget: the other half pays for the right to stop early.
    family_alpha = config.alpha / 2.0
    correction = (
        benjamini_hochberg(family, family_alpha)
        if config.correction == "bh"
        else holm(family, family_alpha)
    )
    by_label = {a.label: a for a in correction.results}
    for m in metrics:
        m.adjusted = by_label.get(m.label)

    # -- the sequential decision on the primary metric ------------------
    diffs = candidate.grades - baseline.grades
    half = config.alpha / 2.0
    # The rubric is 1-5, so a difference is bounded by 4. That bound feeds the
    # empirical-Bernstein boundary's linear term; its square-root term uses
    # the variance it observes, which is what makes it usable -- the honest
    # sub-Gaussian proxy for a bounded difference is sigma = 4, and that
    # boundary is about four times wider than a fixed-sample interval and
    # resolves nothing on a suite this size.
    seq = empirical_bernstein(
        diffs, bound=4.0, alpha=half, tolerance=config.tolerance, min_n=config.min_n
    )
    seq_peek = peeking(
        diffs, alpha=config.alpha, tolerance=config.tolerance, min_n=config.min_n
    )
    seq_fixed = fixed_horizon(diffs, alpha=half, tolerance=config.tolerance)
    seq_subgaussian = confidence_sequence(
        diffs,
        sigma=4.0,
        alpha=half,
        tolerance=config.tolerance,
        tuned_for=config.tuned_for or len(diffs),
        min_n=config.min_n,
    )

    # -- the verdict ----------------------------------------------------
    flagged = [m for m in metrics if m.flagged]
    if not cal.trustworthy:
        verdict = Verdict.ABSTAIN
        reason = cal.verdict()
    elif seq.decision is Decision.REGRESSION:
        # Resolved before the suite ran out; the remaining cases were not
        # judged and did not need to be.
        verdict = Verdict.REGRESSION
        reason = (
            f"stopped at case {seq.n_used} of {seq.n_available}: the always-valid "
            f"interval [{seq.ci[0]:+.3f}, {seq.ci[1]:+.3f}] is entirely below the "
            f"{-config.tolerance:+.2f} margin"
        )
    elif seq.decision is Decision.PASS:
        verdict = Verdict.PASS
        reason = (
            f"stopped at case {seq.n_used} of {seq.n_available}: the always-valid "
            f"interval [{seq.ci[0]:+.3f}, {seq.ci[1]:+.3f}] clears the "
            f"{-config.tolerance:+.2f} margin"
        )
    elif flagged or seq_fixed.decision is Decision.REGRESSION:
        # The sequence never resolved, so the whole suite ran and the
        # end-of-suite family decides -- at the half of alpha reserved for it.
        verdict = Verdict.REGRESSION
        parts = []
        if seq_fixed.decision is Decision.REGRESSION:
            parts.append(
                f"the primary metric ends below the {-config.tolerance:+.2f} margin "
                f"({seq_fixed.effect:+.3f} [{seq_fixed.ci[0]:+.3f}, "
                f"{seq_fixed.ci[1]:+.3f}])"
            )
        if flagged:
            parts.append(
                f"flagged after {correction.method}: "
                + ", ".join(m.label for m in flagged)
            )
        reason = "; ".join(parts)
    elif seq_fixed.decision is Decision.PASS:
        verdict = Verdict.PASS
        reason = (
            f"the primary metric stays within the {-config.tolerance:+.2f} margin "
            f"over all {len(diffs)} cases and nothing survived the "
            f"{correction.method} correction"
        )
    else:
        verdict = Verdict.INCONCLUSIVE
        reason = (
            f"{len(diffs)} cases do not resolve the question: the interval "
            f"[{seq_fixed.ci[0]:+.3f}, {seq_fixed.ci[1]:+.3f}] still straddles the "
            f"{-config.tolerance:+.2f} margin"
        )

    return GateReport(
        verdict=verdict,
        suite=suite.name,
        n_cases=len(suite),
        calibration=cal,
        ceiling=ceiling,
        constant_judge_kappa=constant,
        metrics=metrics,
        correction=correction,
        sequential=seq,
        sequential_peeking=seq_peek,
        sequential_fixed=seq_fixed,
        sequential_subgaussian=seq_subgaussian,
        elapsed_s=time.perf_counter() - started,
        reason=reason,
        config=config,
    )
