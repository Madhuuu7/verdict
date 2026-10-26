"""Calibrating the judge: the statistic, the interval, and the bar.

The tests fall into three groups.

*Is kappa doing what kappa is for?* A judge that answers "4" to everything
agrees with a human on more than half the cases and knows nothing. Raw
agreement rewards it; kappa must score it at zero. That is the test that makes
the choice of statistic defensible rather than decorative.

*Is the weighting doing what the weighting is for?* The rubric is ordinal, so
confusing a 4 with a 5 is a rounding difference and confusing a 1 with a 5
means the judge is not reading the answer. A judge that is off by exactly one
grade everywhere must score high on weighted kappa and poorly on unweighted.

*Is the bar honest?* The threshold is a fraction of the inter-annotator
ceiling, tested on the interval's lower bound rather than the point estimate.
The last test measures how many labels that actually takes, which is the
number a team has to budget for.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from verdict.calibration import (
    Calibration,
    _kappa_from_matrix,
    annotator_ceiling,
    calibrate,
    confusion_matrix,
    constant_judge_kappa,
    quadratic_weights,
)
from verdict.cases import synthetic_suite
from verdict.judges import GRADES, HumanLabeller, RubricJudge
from verdict.pipelines import SimulatedPipeline

try:
    from sklearn.metrics import cohen_kappa_score
except ImportError:  # pragma: no cover - not a dependency
    cohen_kappa_score = None

needs_sklearn = pytest.mark.skipif(
    cohen_kappa_score is None, reason="scikit-learn not installed"
)


def labels(values):
    return {f"c{i}": v for i, v in enumerate(values)}


# -- is kappa doing what kappa is for? ------------------------------------


def test_perfect_agreement_is_one():
    grades = labels([1, 2, 3, 4, 5] * 20)
    cal = calibrate(grades, grades, draws=200)
    assert cal.kappa == pytest.approx(1.0)
    assert cal.weighted_kappa == pytest.approx(1.0)
    assert cal.exact_agreement == 1.0
    assert cal.bias == 0.0


def test_a_constant_judge_scores_zero_despite_agreeing_half_the_time():
    """The test that justifies kappa over raw agreement.

    Most answers on a real suite are good, so a judge that says "4" to
    everything agrees with the humans on a large share of cases while
    carrying no information whatsoever. Raw agreement calls that a decent
    judge; kappa calls it zero, which is the correct answer.
    """
    rng = np.random.default_rng(3)
    human = labels(rng.choice([3, 4, 5], size=300, p=[0.2, 0.45, 0.35]))
    constant = {k: 4 for k in human}
    cal = calibrate(human, constant, draws=200)
    assert cal.exact_agreement > 0.4
    assert abs(cal.kappa) < 1e-9
    assert abs(constant_judge_kappa(human)) < 1e-9


def test_an_inverted_judge_scores_below_zero():
    """Worse than chance has to be representable, or the statistic is only
    measuring how hard the judge is trying."""
    human = labels([1, 2, 4, 5] * 40)
    inverted = labels([5, 4, 2, 1] * 40)
    cal = calibrate(human, inverted, draws=200)
    assert cal.kappa < 0
    assert cal.weighted_kappa < 0


def test_kappa_is_not_fooled_by_a_lucky_marginal_match():
    """Same marginal distribution, no case-level relationship: shuffling the
    judge's grades destroys the agreement without changing the histogram, and
    kappa must fall to about zero while the marginals stay identical."""
    rng = np.random.default_rng(5)
    values = rng.integers(1, 6, size=400)
    human = labels(values)
    shuffled = labels(rng.permutation(values))
    cal = calibrate(human, shuffled, draws=200)
    assert abs(cal.kappa) < 0.1
    assert sorted(human.values()) == sorted(shuffled.values())


def test_degenerate_marginals_return_zero_rather_than_dividing_by_zero():
    both_constant = labels([4] * 50)
    assert _kappa_from_matrix(
        confusion_matrix(
            np.array(list(both_constant.values())),
            np.array(list(both_constant.values())),
        ),
        None,
    ) == 0.0


def test_an_empty_matrix_is_zero():
    assert _kappa_from_matrix(np.zeros((5, 5)), None) == 0.0


@needs_sklearn
def test_unweighted_kappa_matches_scikit_learn():
    rng = np.random.default_rng(7)
    human = rng.integers(1, 6, size=300)
    judge = np.clip(human + rng.integers(-1, 2, size=300), 1, 5)
    cal = calibrate(labels(human), labels(judge), draws=10)
    assert cal.kappa == pytest.approx(cohen_kappa_score(human, judge), abs=1e-9)


@needs_sklearn
def test_weighted_kappa_matches_scikit_learn():
    rng = np.random.default_rng(11)
    human = rng.integers(1, 6, size=300)
    judge = np.clip(human + rng.integers(-1, 2, size=300), 1, 5)
    cal = calibrate(labels(human), labels(judge), draws=10)
    assert cal.weighted_kappa == pytest.approx(
        cohen_kappa_score(human, judge, weights="quadratic"), abs=1e-9
    )


# -- is the weighting doing what the weighting is for? --------------------


def test_quadratic_weights_penalise_distance_not_mere_disagreement():
    w = quadratic_weights(5)
    assert w[0, 0] == 0.0
    assert w[0, 1] == pytest.approx(1 / 16)
    assert w[0, 4] == pytest.approx(1.0)
    assert np.allclose(w, w.T)


def test_an_off_by_one_judge_is_ordered_right_and_quantised_wrong():
    """The gap between weighted and unweighted kappa is itself diagnostic: a
    judge like this is reading the answers correctly and rounding
    differently, which is usually fine, and unweighted kappa alone would
    reject it.
    """
    human = labels([1, 2, 3, 4] * 60)
    off_by_one = labels([2, 3, 4, 5] * 60)
    cal = calibrate(human, off_by_one, draws=200)
    assert cal.weighted_kappa > 0.6
    assert cal.kappa < 0.2
    assert cal.within_one == 1.0
    assert cal.exact_agreement == 0.0


def test_a_wildly_wrong_judge_fails_on_both_statistics():
    human = labels([1, 5] * 80)
    wrong = labels([5, 1] * 80)
    cal = calibrate(human, wrong, draws=200)
    assert cal.weighted_kappa < 0
    assert cal.within_one == 0.0


def test_bias_is_signed_and_reported_with_an_interval():
    human = labels([3] * 100)
    generous = labels([4] * 100)
    harsh = labels([2] * 100)
    assert calibrate(human, generous, draws=50).bias == pytest.approx(1.0)
    assert calibrate(human, harsh, draws=50).bias == pytest.approx(-1.0)

    rng = np.random.default_rng(13)
    noisy_human = labels(rng.integers(1, 6, size=200))
    noisy_judge = labels([min(5, g + 1) for g in noisy_human.values()])
    cal = calibrate(noisy_human, noisy_judge, draws=200)
    lo, hi = cal.bias_ci
    assert lo < cal.bias < hi
    assert lo > 0  # a generous judge, and the interval says so


def test_the_confusion_matrix_is_human_rows_and_judge_columns():
    """Orientation is the sort of thing that is wrong half the time and
    invisible until someone reads a row as a column."""
    human = labels([1, 1, 1])
    judge = labels([5, 5, 5])
    cal = calibrate(human, judge, draws=10)
    matrix = np.array(cal.confusion)
    assert matrix[0, 4] == 3
    assert matrix[4, 0] == 0
    assert matrix.shape == (len(GRADES), len(GRADES))


# -- is the bar honest? ---------------------------------------------------


def test_the_bar_is_a_fraction_of_the_ceiling_not_an_absolute_number():
    """The first design used an absolute kappa bar, and it was unreachable no
    matter how good the judge got, because the reference's own noise was the
    ceiling. Same judge, two ceilings, two answers.
    """
    cal = Calibration(
        n=200, kappa=0.5, kappa_ci=(0.45, 0.55),
        weighted_kappa=0.60, weighted_kappa_ci=(0.55, 0.65),
        exact_agreement=0.5, within_one=0.9, bias=0.1, bias_ci=(0.0, 0.2),
        ceiling=0.72, threshold=0.75,
    )
    assert cal.required == pytest.approx(0.54)
    assert cal.relative == pytest.approx(0.60 / 0.72)
    assert cal.trustworthy

    strict = replace(cal, ceiling=0.95)
    assert strict.required == pytest.approx(0.7125)
    assert not strict.trustworthy


def test_a_low_ceiling_cannot_certify_a_judge_through_the_floor():
    """Two annotators who agree on nothing must not be able to certify a bad
    judge by setting the bar beneath it."""
    cal = Calibration(
        n=200, kappa=0.2, kappa_ci=(0.15, 0.25),
        weighted_kappa=0.25, weighted_kappa_ci=(0.20, 0.30),
        exact_agreement=0.3, within_one=0.6, bias=0.0, bias_ci=(-0.1, 0.1),
        ceiling=0.20, threshold=0.75, floor=0.40,
    )
    assert cal.required == pytest.approx(0.40)
    assert not cal.trustworthy


def test_certification_is_tested_on_the_lower_bound_not_the_estimate():
    """With a few dozen labels the point estimate cannot tell 0.58 from 0.68,
    so a judge whose estimate clears the bar on a wide interval is not
    certified. Same point estimate, different interval width, different
    answer."""
    common = dict(
        n=40, kappa=0.5, kappa_ci=(0.3, 0.7), weighted_kappa=0.60,
        exact_agreement=0.5, within_one=0.9, bias=0.0, bias_ci=(-0.1, 0.1),
        ceiling=0.72, threshold=0.75,
    )
    wide = Calibration(**common, weighted_kappa_ci=(0.40, 0.80))
    tight = Calibration(**common, weighted_kappa_ci=(0.56, 0.64))
    assert not wide.trustworthy
    assert tight.trustworthy
    assert "abstain" in wide.verdict()
    assert "good enough to rule" in tight.verdict()


def test_no_labels_means_the_gate_abstains():
    cal = calibrate({}, {"c0": 4})
    assert cal.n == 0
    assert not cal.trustworthy
    assert "no case was labelled by both" in cal.note
    assert "unverified" in cal.verdict()


def test_only_the_shared_cases_are_compared():
    human = labels([4, 4, 4])
    judge = {"c1": 4, "c2": 4, "c9": 1}
    cal = calibrate(human, judge, draws=10)
    assert cal.n == 2


def test_the_interval_is_reproducible_for_a_seed():
    rng = np.random.default_rng(17)
    human = labels(rng.integers(1, 6, size=120))
    judge = labels([min(5, max(1, g + int(rng.integers(-1, 2)))) for g in human.values()])
    first = calibrate(human, judge, draws=300, seed=4)
    second = calibrate(human, judge, draws=300, seed=4)
    assert first.weighted_kappa_ci == second.weighted_kappa_ci


def test_an_interval_needs_at_least_three_labels():
    cal = calibrate(labels([4, 5]), labels([4, 4]), draws=100)
    assert np.isnan(cal.weighted_kappa_ci[0])
    assert not cal.trustworthy


# -- how many labels this actually takes ----------------------------------


def test_the_judge_sits_near_the_ceiling_but_needs_hundreds_of_labels():
    """The measurement a team has to budget for, on the bundled suite.

    The simulated judge holds 0.8-0.9 of the inter-annotator ceiling at every
    sample size -- it is about as good as a second annotator from 80 labels
    onwards -- and is still not *certifiable* at 160, because the bootstrap
    interval is wider than the margin between the judge and the bar. On this
    setup the lower bound crosses the required value between 320 and 480
    labels.

    That gap between "good enough" and "provably good enough" is the point of
    testing the lower bound, and it is the number that makes the gate's
    abstain outcome a real one rather than a formality.
    """
    suite = list(synthetic_suite(800))
    pipeline = SimulatedPipeline(name="p")
    judge = RubricJudge()
    outputs = [pipeline(c) for c in suite]

    def grades(labeller):
        return {c.case_id: labeller(c, o).grade for c, o in zip(suite, outputs)}

    judge_grades = {c.case_id: judge(c, o).grade for c, o in zip(suite, outputs)}
    a = grades(HumanLabeller(name="annotator-a"))
    b = grades(HumanLabeller(name="annotator-b"))

    curve = []
    for n in (80, 160, 320, 640):
        ids = [c.case_id for c in suite[:n]]
        ceiling = annotator_ceiling(
            {k: a[k] for k in ids}, {k: b[k] for k in ids}, draws=300
        ).weighted_kappa
        cal = calibrate(
            {k: a[k] for k in ids},
            {k: judge_grades[k] for k in ids},
            ceiling=ceiling,
            draws=300,
        )
        curve.append((n, cal))

    # Close to a second annotator throughout, and never mistaken for one.
    for n, cal in curve:
        assert 0.75 <= cal.relative <= 0.95, n

    # The interval narrows with labels, which is why waiting helps.
    widths = [c.weighted_kappa_ci[1] - c.weighted_kappa_ci[0] for _, c in curve]
    assert widths[0] > widths[-1] * 1.5

    # Not certifiable early, certifiable late.
    assert not curve[1][1].trustworthy  # 160 labels
    assert curve[-1][1].trustworthy  # 640 labels


def test_the_annotator_ceiling_is_itself_an_estimate_with_an_interval():
    """The judge is compared against a measured ceiling, not a constant, and
    a ceiling from 60 cases is uncertain enough to matter."""
    suite = list(synthetic_suite(300))
    pipeline = SimulatedPipeline(name="p")
    outputs = [pipeline(c) for c in suite]
    a = {c.case_id: HumanLabeller(name="a")(c, o).grade for c, o in zip(suite, outputs)}
    b = {c.case_id: HumanLabeller(name="b")(c, o).grade for c, o in zip(suite, outputs)}

    ceiling = annotator_ceiling(a, b, draws=400)
    lo, hi = ceiling.weighted_kappa_ci
    assert 0.5 < ceiling.weighted_kappa < 0.95
    assert lo < ceiling.weighted_kappa < hi
    assert hi - lo > 0.02
