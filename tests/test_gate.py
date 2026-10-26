"""The gate end to end: one verdict per regression shape, and why.

The useful property of these tests is not that the gate flags things. It is
*which* comparison catches each shape, because that is what says the metric
set and the slicing are earning their place:

=========================  ===============================================
one slice broken           only ``rubric score [math]`` -- the suite mean
                           barely moves, so slicing is the only thing that
                           sees it
slower, same quality       only ``mean latency``
more refusals              ``refusal rate``, and the rubric with it
identical pipelines        nothing, on 14 comparisons
a candidate that improved  nothing, although 11 comparisons are significant
=========================  ===============================================

That last row is the one that keeps a gate switched on. A gate that blocks on
any significant difference blocks on good news too, and a team that has been
blocked by good news twice stops reading the output.
"""

from __future__ import annotations

import numpy as np
import pytest

from verdict import (
    GateConfig,
    HumanLabeller,
    RubricJudge,
    Verdict,
    execute,
    run_gate,
    synthetic_suite,
    variant_pair,
)
from verdict.stats.sequential import Decision


# -- running a variant ----------------------------------------------------


def test_execute_keeps_cases_outputs_and_judgements_aligned(suite, judge):
    base, _ = variant_pair("none")
    run = execute(suite, base, judge, "baseline")
    assert len(run.cases) == len(run.outputs) == len(run.judgements) == len(suite)
    assert [c.case_id for c in run.cases] == [c.case_id for c in suite]
    assert run.grades.shape == (len(suite),)
    assert run.passes.dtype == bool
    assert run.refusals.dtype == bool
    assert run.latencies.min() > 0
    by_id = run.grades_by_id()
    assert by_id[run.cases[0].case_id] == run.judgements[0].grade


def test_pass_rate_is_the_rubric_threshold_applied_to_the_grades(suite, judge):
    base, _ = variant_pair("none")
    run = execute(suite, base, judge, "baseline")
    assert np.array_equal(run.passes, run.grades >= 4)


# -- one verdict per shape ------------------------------------------------


def test_identical_pipelines_are_not_called_a_regression(report_identical):
    """The false-alarm case, on 14 comparisons. The gate reports
    inconclusive rather than pass, because 600 cases cannot certify
    non-inferiority to within 0.05 rubric points -- and it says how fine a
    margin they *could* certify.
    """
    assert report_identical.verdict is not Verdict.REGRESSION
    assert report_identical.flagged == []
    assert report_identical.comparisons == 14
    assert report_identical.resolvable_margin > 0.05


def test_a_wide_enough_margin_lets_identical_pipelines_pass(suite, judge, annotators):
    """Same data, a 0.3 margin instead of 0.05: the gate now resolves and
    stops early. What changed is the question, not the evidence."""
    base, cand = variant_pair("none")
    report = run_gate(
        suite, base, cand, judge, annotators, GateConfig(tolerance=0.3)
    )
    assert report.verdict is Verdict.PASS
    assert report.sequential.decision is Decision.PASS
    assert report.sequential.stopped_early


def test_a_clear_quality_drop_is_caught_and_stops_the_suite_early(report_clear_drop):
    """The saving that pays for the sequential machinery: the verdict is
    settled at case 225 of 600, so 375 cases were never judged -- 62% of the
    model calls a fixed-horizon gate would have spent.
    """
    assert report_clear_drop.verdict is Verdict.REGRESSION
    assert report_clear_drop.sequential.decision is Decision.REGRESSION
    assert report_clear_drop.sequential.stopped_early
    assert report_clear_drop.sequential.saved_fraction > 0.5
    assert "stopped at case" in report_clear_drop.reason
    assert report_clear_drop.sequential.ci[1] < -report_clear_drop.config.tolerance


def test_a_broken_slice_is_caught_by_the_slice_and_nothing_else(report_slice_broken):
    """The regression that slicing exists for. Damage concentrated on one tag
    is diluted to almost nothing in the suite mean; the only comparison that
    survives correction is that tag's own.
    """
    assert report_slice_broken.verdict is Verdict.REGRESSION
    assert [m.label for m in report_slice_broken.flagged] == ["rubric score [math]"]
    whole = next(
        m for m in report_slice_broken.metrics if m.label == "rubric score"
    )
    assert not whole.flagged


def test_a_latency_regression_is_caught_without_touching_quality(report_latency):
    assert report_latency.verdict is Verdict.REGRESSION
    assert [m.label for m in report_latency.flagged] == ["mean latency"]
    latency = next(m for m in report_latency.metrics if m.label == "mean latency")
    assert latency.candidate_mean > latency.baseline_mean
    assert not latency.higher_is_better


def test_a_refusal_regression_shows_up_on_the_refusal_rate(suite, judge, annotators):
    base, cand = variant_pair("refusal", severity=0.12)
    report = run_gate(suite, base, cand, judge, annotators)
    assert report.verdict is Verdict.REGRESSION
    labels = [m.label for m in report.flagged]
    assert "refusal rate" in labels
    refusals = next(m for m in report.metrics if m.label == "refusal rate")
    assert refusals.worse
    assert refusals.candidate_mean > refusals.baseline_mean


def test_an_improvement_is_significant_and_not_a_regression(report_improvement):
    """11 of 14 comparisons reach significance and the gate flags none of
    them, because every one moved in the good direction. A gate that blocks
    on good news is a gate the team turns off.
    """
    significant = [m for m in report_improvement.metrics if m.adjusted.rejected]
    assert len(significant) > 8
    assert report_improvement.flagged == []
    assert report_improvement.verdict is Verdict.PASS


# -- the judge is checked before anything is tested -----------------------


def test_an_uncertifiable_judge_makes_the_gate_abstain(report_abstain):
    """240 cases is 133 labels here, and that is not enough to certify this
    judge: the interval around its agreement is wider than the margin it
    needs to clear. So the gate refuses to rule on a real quality drop, and
    says why.

    This is the behaviour to defend in a review. The alternative -- ruling on
    an instrument that has not been checked -- produces a confident number
    that will be acted on, which is worse than no number.
    """
    assert report_abstain.verdict is Verdict.ABSTAIN
    assert not report_abstain.calibration.trustworthy
    assert report_abstain.calibration.n < 200
    assert "abstain" in report_abstain.reason
    assert "kappa" in report_abstain.reason

    # Everything was still computed, so the console can show what the gate
    # *would* have flagged next to its refusal to rule -- and there is
    # something there, since this candidate really is worse.
    assert report_abstain.comparisons == 14
    assert min(m.test.p_value for m in report_abstain.metrics) < 0.05
    assert report_abstain.sequential_fixed.decision is Decision.REGRESSION


def test_the_ceiling_is_measured_not_assumed(report_identical):
    cal, ceiling = report_identical.calibration, report_identical.ceiling
    assert 0.5 < ceiling.weighted_kappa < 0.95
    assert cal.ceiling == pytest.approx(ceiling.weighted_kappa)
    assert cal.required == pytest.approx(
        max(cal.floor, cal.threshold * ceiling.weighted_kappa)
    )
    # And the sanity check that makes kappa legible in the console.
    assert abs(report_identical.constant_judge_kappa) < 1e-9


def test_one_annotator_is_not_enough(suite, judge):
    """Without a second annotator there is no ceiling, and the judge would be
    held to an absolute bar that may be unreachable -- the design that was
    tried first and discarded."""
    base, cand = variant_pair("none")
    with pytest.raises(ValueError, match="two annotators"):
        run_gate(suite, base, cand, judge, [HumanLabeller(name="only-one")])


# -- multiplicity and the error budget ------------------------------------


def test_the_family_is_corrected_at_half_the_budget(report_clear_drop):
    """The gate can be wrong twice -- by stopping early, or at the end -- so
    each look gets alpha/2. Bonferroni across two looks: crude, and honest
    about the fact that there are two.
    """
    config = report_clear_drop.config
    assert report_clear_drop.correction.alpha == pytest.approx(config.alpha / 2)
    assert report_clear_drop.sequential.alpha == pytest.approx(config.alpha / 2)
    assert report_clear_drop.sequential_fixed.alpha == pytest.approx(config.alpha / 2)
    # The invalid comparison is deliberately run at the full alpha: it is
    # shown to make a point, not to decide anything.
    assert report_clear_drop.sequential_peeking.alpha == pytest.approx(config.alpha)


def test_correction_removes_flags_the_uncorrected_gate_would_have_raised(
    report_slice_broken,
):
    assert len(report_slice_broken.flagged) <= len(
        report_slice_broken.flagged_uncorrected
    )
    assert report_slice_broken.family_error_if_uncorrected() > 0.25


def test_slicing_is_what_creates_the_multiplicity(suite, judge, annotators):
    """Four suite-level metrics on their own, fourteen with the slices. The
    correction exists because of this function."""
    base, cand = variant_pair("none")
    whole = run_gate(
        suite, base, cand, judge, annotators, GateConfig(slice_by_tag=False)
    )
    sliced = run_gate(suite, base, cand, judge, annotators, GateConfig())
    assert whole.comparisons == 4
    assert sliced.comparisons == 14
    assert whole.family_error_if_uncorrected() < sliced.family_error_if_uncorrected()


def test_slices_too_small_to_support_a_test_are_left_out(suite, judge, annotators):
    """A nine-case slice cannot support a comparison and only adds to the
    multiplicity burden, so it is dropped rather than tested badly."""
    base, cand = variant_pair("none")
    permissive = run_gate(suite, base, cand, judge, annotators, GateConfig(min_slice=1))
    strict = run_gate(suite, base, cand, judge, annotators, GateConfig(min_slice=100))
    assert permissive.comparisons > strict.comparisons
    assert strict.comparisons >= 4


def test_holm_is_offered_and_is_stricter_than_the_default(suite, judge, annotators):
    base, cand = variant_pair("tag", severity=0.06)
    bh = run_gate(suite, base, cand, judge, annotators, GateConfig(correction="bh"))
    holm = run_gate(suite, base, cand, judge, annotators, GateConfig(correction="holm"))
    assert holm.correction.method == "holm"
    assert {m.label for m in holm.flagged} <= {m.label for m in bh.flagged}


def test_every_metric_carries_the_threshold_it_was_judged_against(report_clear_drop):
    """What makes a correction explainable to whoever the gate just blocked:
    not only the adjusted p-value but the bar it was compared to."""
    for metric in report_clear_drop.metrics:
        assert metric.adjusted is not None
        assert metric.adjusted.label == metric.label
        assert 0 < metric.adjusted.threshold <= report_clear_drop.config.alpha / 2
        assert metric.adjusted.p_raw == pytest.approx(metric.test.p_value)


# -- the contrast the console shows ---------------------------------------


def test_peeking_would_have_called_it_far_earlier_than_the_valid_rule(
    report_clear_drop,
):
    """Both procedures see the same data. The invalid one stops at case 24
    and the always-valid one at 225 -- which is the cost of validity, shown
    rather than asserted. The 24 is not a free lunch: the same procedure
    fires on unchanged pipelines a quarter of the time (test_sequential.py).
    """
    peek = report_clear_drop.sequential_peeking
    valid = report_clear_drop.sequential
    assert peek.decision is Decision.REGRESSION
    assert peek.n_used < valid.n_used


def test_the_honest_subgaussian_boundary_resolves_nothing_at_this_scale(
    report_clear_drop, report_identical, report_latency
):
    """Why the gate uses a variance-adaptive boundary. A rubric difference is
    bounded by 4, so the only honest Hoeffding proxy is sigma = 4, and on 600
    cases that interval is wide enough to be useless -- it reports
    inconclusive even for the drop every other procedure calls obvious.
    """
    for report in (report_clear_drop, report_identical, report_latency):
        assert report.sequential_subgaussian.decision is Decision.INCONCLUSIVE
    subgaussian_width = (
        report_clear_drop.sequential_subgaussian.ci[1]
        - report_clear_drop.sequential_subgaussian.ci[0]
    )
    adaptive_width = (
        report_clear_drop.sequential.ci[1] - report_clear_drop.sequential.ci[0]
    )
    assert subgaussian_width > adaptive_width


def test_resolvable_margin_says_how_fine_a_question_the_suite_can_answer(
    report_identical,
):
    """The number a bare "inconclusive" hides: with these cases, this judge
    and this much noise, non-inferiority is certifiable to within this many
    rubric points and no finer. It makes the outcome actionable -- add cases
    or quieten the judge -- instead of a shrug.
    """
    margin = report_identical.resolvable_margin
    assert margin == pytest.approx(abs(report_identical.sequential_fixed.ci[0]))
    assert margin > report_identical.config.tolerance


def test_a_resolved_suite_reports_no_outstanding_margin(suite, judge, annotators):
    base, cand = variant_pair("none")
    report = run_gate(suite, base, cand, judge, annotators, GateConfig(tolerance=0.5))
    assert report.verdict is Verdict.PASS


# -- report mechanics -----------------------------------------------------


def test_the_summary_names_the_verdict_the_counts_and_the_effect(
    report_clear_drop, report_abstain
):
    summary = report_clear_drop.summary()
    assert summary.startswith("REGRESSION on 600 cases")
    assert "14 comparisons" in summary
    assert "benjamini-hochberg" in summary
    assert "primary effect" in summary
    assert "saved" in summary
    assert report_abstain.summary().startswith("ABSTAIN -- ")


def test_metric_direction_is_recorded_per_metric(report_identical):
    kinds = {m.label: (m.kind, m.higher_is_better) for m in report_identical.metrics}
    assert kinds["rubric score"] == ("ordinal", True)
    assert kinds["pass rate"] == ("binary", True)
    assert kinds["refusal rate"] == ("binary", False)
    assert kinds["mean latency"] == ("continuous", False)


def test_the_ordinal_metric_reports_a_signed_rank_p_and_a_bootstrap_interval(
    report_clear_drop,
):
    """Two tests on one metric, deliberately: the signed-rank test is the
    right null for ordinal paired data and gives no interval on the mean
    difference, which is the number a reviewer wants to see."""
    rubric = next(m for m in report_clear_drop.metrics if m.label == "rubric score")
    assert rubric.test.name == "wilcoxon+bootstrap"
    assert rubric.test.ci is not None
    assert "non-tied pairs" in rubric.test.detail
    assert "percentile interval" in rubric.test.detail


def test_the_same_inputs_give_the_same_report(suite, judge, annotators):
    base, cand = variant_pair("tag", severity=0.06)
    first = run_gate(suite, base, cand, judge, annotators)
    second = run_gate(suite, base, cand, judge, annotators)
    assert first.verdict is second.verdict
    assert first.reason == second.reason
    assert [m.test.p_value for m in first.metrics] == [
        m.test.p_value for m in second.metrics
    ]


def test_the_report_knows_how_long_it_took(report_identical):
    assert report_identical.elapsed_s > 0
    assert report_identical.n_cases == 600
    assert report_identical.suite == "synthetic"


def test_a_suite_too_small_to_slice_still_produces_a_report(judge, annotators):
    """The gate must survive a suite that resolves nothing, because that is
    what a team's first eval suite looks like."""
    tiny = synthetic_suite(30, name="tiny")
    base, cand = variant_pair("quality", severity=0.3)
    report = run_gate(tiny, base, cand, judge, annotators, GateConfig(min_n=10))
    assert report.verdict in set(Verdict)
    assert report.comparisons == 4  # no slice reaches min_slice
    assert report.n_cases == 30
