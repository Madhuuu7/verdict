"""Stopping early, and the four procedures that do it well or badly.

Every error rate in this file is measured. That is the only way to test a
transcribed boundary: the empirical-Bernstein constants (1.7, 1.56, 0.72,
10.4) come from a paper, and a typo in any of them would still produce an
interval that looks plausible on a chart. What it would not do is hold its
false-alarm rate under thousands of null experiments, so that is what gets
checked.

The trial counts here are kept small enough to run in a few seconds -- the
full 10,000-trial study lives in ``scripts/validate.py`` and its numbers go in
the README. These are the same measurements at lower resolution, which is
enough to catch a broken constant but not enough to quote.

Under a null (``effect = 0``) with ``tolerance = 0``, the truth sits exactly
on the margin, so *any* definite verdict is an error -- PASS and REGRESSION
both. That is the quantity called the error rate throughout.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from verdict.stats.distributions import normal_ppf
from verdict.stats.sequential import (
    Decision,
    confidence_sequence,
    empirical_bernstein,
    fixed_horizon,
    mixture_radius,
    mixture_rho,
    peeking,
    simulate_error_rate,
    two_sided_p_from_mean,
)


def error_rate(stats: dict[str, float]) -> float:
    """Any definite verdict under a null is a mistake."""
    return stats["flagged_regression"] + stats["passed"]


# -- the boundary itself ---------------------------------------------------


def test_rho_scales_inversely_with_the_sample_size_it_is_tuned_for():
    assert mixture_rho(0.05, 100) == pytest.approx(mixture_rho(0.05, 200) * 2)
    assert mixture_rho(0.05, 50) > 0


def test_rho_rejects_a_nonsense_target():
    for bad in (0, -10):
        with pytest.raises(ValueError):
            mixture_rho(0.05, bad)


def test_radius_shrinks_with_n_and_grows_with_sigma():
    rho = mixture_rho(0.05, 200)
    widths = [mixture_radius(n, 1.0, 0.05, rho) for n in (10, 50, 100, 400, 1000)]
    assert all(b < a for a, b in zip(widths, widths[1:]))
    assert mixture_radius(100, 2.0, 0.05, rho) == pytest.approx(
        2.0 * mixture_radius(100, 1.0, 0.05, rho)
    )


def test_radius_is_infinite_before_any_data():
    assert mixture_radius(0, 1.0, 0.05, 0.1) == math.inf


def test_the_time_uniform_interval_is_wider_than_a_fixed_sample_one():
    """The price of optional stopping, as a number.

    At the sample size the boundary is tuned for, the always-valid radius is
    about 1.3x the fixed-n radius. That factor is what buys the right to look
    after every case, and quoting it is more honest than saying sequential
    testing is free.
    """
    n, sigma, alpha = 200, 1.0, 0.05
    sequential = mixture_radius(n, sigma, alpha, mixture_rho(alpha, n))
    fixed = normal_ppf(1 - alpha / 2) * sigma / math.sqrt(n)
    assert 1.1 < sequential / fixed < 1.6


# -- peeking: the measurement that justifies the whole module -------------


def test_peeking_inflates_the_error_rate_far_past_alpha():
    """A fixed-sample interval recomputed after every case.

    300 null trials at n = 200. The nominal rate is 5%; peeking lands an
    order of magnitude above it, because the interval was built for one look
    and is being used for 200. ``scripts/validate.py`` runs this at 1,500
    trials and n = 600 for the README.
    """
    stats = simulate_error_rate(
        peeking, trials=300, n=200, effect=0.0, noise=0.8, seed=5,
        tolerance=0.0, min_n=10,
    )
    assert error_rate(stats) > 0.25


def test_fixed_horizon_holds_its_size_on_the_same_data():
    """The control: one look at the end, 300 null trials, and the error rate
    sits at the 5% it promises. Same data-generating process as the peeking
    test above, so the difference between the two is the peeking alone."""
    stats = simulate_error_rate(
        fixed_horizon, trials=300, n=200, effect=0.0, noise=0.8, seed=5,
        tolerance=0.0,
    )
    assert error_rate(stats) <= 0.09


def test_the_confidence_sequence_holds_its_size_while_looking_every_time():
    """The same 300 null trials, checked after every observation, and the
    error rate stays at or under alpha. This is the claim the module exists
    to make."""
    stats = simulate_error_rate(
        confidence_sequence, trials=300, n=200, effect=0.0, noise=0.8, seed=5,
        sigma=4.0, tolerance=0.0, min_n=10,
    )
    assert error_rate(stats) <= 0.05


def test_empirical_bernstein_holds_its_size_too():
    """The variance-adaptive boundary is the one the gate actually uses, so
    its false-alarm rate is the one that matters. 600 null trials.

    If any of the transcribed constants were wrong, this is the test that
    would fail -- which is the whole reason the number is measured instead of
    trusted.
    """
    stats = simulate_error_rate(
        empirical_bernstein, trials=600, n=200, effect=0.0, noise=0.8, seed=9,
        bound=4.0, tolerance=0.0, min_n=10,
    )
    assert error_rate(stats) <= 0.05


def test_peeking_is_worse_the_more_often_you_look():
    """Peeking's error rate grows with the number of looks, which is the
    mechanism rather than an accident of one sample size: n = 50 and n = 400
    on identical noise."""
    short = simulate_error_rate(
        peeking, trials=300, n=50, effect=0.0, noise=0.8, seed=11,
        tolerance=0.0, min_n=10,
    )
    long = simulate_error_rate(
        peeking, trials=300, n=400, effect=0.0, noise=0.8, seed=11,
        tolerance=0.0, min_n=10,
    )
    assert error_rate(long) > error_rate(short)


# -- why the gate uses empirical-Bernstein and not Hoeffding --------------


def test_the_honest_subgaussian_proxy_resolves_nothing_on_a_real_regression():
    """Correct and useless, measured.

    A rubric difference is bounded by 4, so the only honest sub-Gaussian
    proxy is sigma = 4. On 200 cases with a genuine -0.25 effect that
    boundary never resolves: the gate would report `inconclusive` for a
    regression a fixed-horizon test calls obvious. This is the measurement
    that chose empirical-Bernstein.
    """
    subgaussian = simulate_error_rate(
        confidence_sequence, trials=200, n=200, effect=-0.25, noise=0.8, seed=13,
        sigma=4.0, tolerance=0.0, min_n=10,
    )
    adaptive = simulate_error_rate(
        empirical_bernstein, trials=200, n=200, effect=-0.25, noise=0.8, seed=13,
        bound=4.0, tolerance=0.0, min_n=10,
    )
    fixed = simulate_error_rate(
        fixed_horizon, trials=200, n=200, effect=-0.25, noise=0.8, seed=13,
        tolerance=0.0,
    )
    assert subgaussian["flagged_regression"] < 0.05
    assert subgaussian["inconclusive"] > 0.9
    assert adaptive["flagged_regression"] > subgaussian["flagged_regression"]
    assert fixed["flagged_regression"] > adaptive["flagged_regression"]


def test_empirical_bernstein_stops_early_on_a_clear_regression():
    """Power costs sample size, and the saving is the point: on a large
    effect the boundary resolves well before the suite runs out."""
    stats = simulate_error_rate(
        empirical_bernstein, trials=100, n=400, effect=-0.6, noise=0.8, seed=17,
        bound=4.0, tolerance=0.0, min_n=10,
    )
    assert stats["flagged_regression"] > 0.9
    assert stats["mean_n"] < 400 * 0.6


# -- decision semantics ---------------------------------------------------


def test_nothing_is_decided_before_min_n():
    """The boundary is enormous for the first few observations; deciding
    there would be valid and useless."""
    diffs = [-5.0] * 100
    result = empirical_bernstein(diffs, bound=4.0, min_n=40)
    assert result.n_used == 40
    assert all(s.decision is Decision.INCONCLUSIVE for s in result.steps[:39])


def test_non_inferiority_is_tested_against_the_margin_not_zero():
    """One dataset, two margins, opposite verdicts.

    A candidate 0.15 rubric points worse is a PASS against a 0.4 margin (it
    is comfortably inside what was declared negligible) and a REGRESSION
    against a margin of 0. Nothing about the data changed: the margin is the
    decision. The gate tests non-inferiority deliberately, because a point
    null is a hypothesis no real pipeline satisfies -- test it and the gate
    becomes a coin flip that blocks every release at large enough n.
    """
    diffs = np.random.default_rng(19).normal(-0.15, 0.2, size=2000)
    lenient = empirical_bernstein(diffs, bound=4.0, tolerance=0.4)
    strict = empirical_bernstein(diffs, bound=4.0, tolerance=0.0)
    assert lenient.decision is Decision.PASS
    assert strict.decision is Decision.REGRESSION
    # And the sample size each verdict needed: a wide margin resolves fast.
    assert lenient.n_used < strict.n_used


def test_an_interval_straddling_the_margin_stays_inconclusive():
    """"This suite does not resolve the question" is a real outcome, and a
    gate that cannot say it will answer wrongly on a small suite."""
    rng = np.random.default_rng(23)
    diffs = rng.normal(-0.05, 1.2, size=40)
    result = empirical_bernstein(diffs, bound=4.0, tolerance=0.05, min_n=20)
    assert result.decision is Decision.INCONCLUSIVE
    assert result.n_used == 40
    assert not result.stopped_early


def test_steps_record_the_running_mean_and_a_shrinking_radius():
    rng = np.random.default_rng(29)
    diffs = rng.normal(0.0, 0.5, size=120)
    result = empirical_bernstein(diffs, bound=4.0, tolerance=0.3, min_n=200)
    assert len(result.steps) == 120
    assert result.steps[-1].mean == pytest.approx(float(np.mean(diffs)))
    # Not monotone step by step (the accumulated variance can jump), but the
    # trend over a decade of n must be downward.
    assert result.steps[-1].radius < result.steps[9].radius / 2


def test_saved_counts_the_cases_that_never_had_to_be_judged():
    """The saving is the money: every case not reached is a model call not
    made. On the most extreme possible evidence -- every difference at the
    bound -- the boundary still needs a couple of observations past min_n,
    because its linear term starts at 1.56 * bound and has to decay.
    """
    diffs = [-4.0] * 200
    result = empirical_bernstein(diffs, bound=4.0, min_n=10)
    assert 10 <= result.n_used <= 20
    assert result.saved == 200 - result.n_used
    assert result.saved_fraction == pytest.approx(result.saved / 200)
    assert result.saved_fraction > 0.9
    assert result.stopped_early


def test_empty_and_degenerate_inputs_are_inconclusive_not_crashes():
    for result in (
        empirical_bernstein([], bound=4.0),
        empirical_bernstein([1.0, 2.0], bound=0.0),
        confidence_sequence([]),
        fixed_horizon([]),
        fixed_horizon([0.5]),
        peeking([]),
    ):
        assert result.decision is Decision.INCONCLUSIVE


def test_confidence_sequence_infers_sigma_from_the_range_when_not_given():
    """Half the observed range is the Hoeffding proxy for bounded data. Using
    the sample standard deviation would be tighter and would quietly void the
    guarantee -- the bound needs a proxy fixed in advance, not one estimated
    from the data it is judging."""
    diffs = list(np.linspace(-2.0, 2.0, 200))
    inferred = confidence_sequence(diffs, tolerance=0.0, min_n=10)
    explicit = confidence_sequence(diffs, sigma=2.0, tolerance=0.0, min_n=10)
    assert inferred.ci == pytest.approx(explicit.ci)


# -- fixed_horizon and the p-value bridge ---------------------------------


def test_fixed_horizon_is_the_textbook_z_interval():
    rng = np.random.default_rng(31)
    diffs = rng.normal(-0.1, 0.4, size=250)
    result = fixed_horizon(diffs, alpha=0.05, tolerance=0.0)
    mean = float(np.mean(diffs))
    se = float(np.std(diffs, ddof=1) / math.sqrt(250))
    z = normal_ppf(0.975)
    assert result.effect == pytest.approx(mean)
    assert result.ci[0] == pytest.approx(mean - z * se)
    assert result.ci[1] == pytest.approx(mean + z * se)
    assert result.n_used == 250
    assert not result.stopped_early


def test_two_sided_p_matches_the_interval_it_came_from():
    """The bridge between the sequential machinery and the FDR correction:
    the p-value must cross 0.05 exactly where the 95% interval touches zero,
    or the two halves of the gate would disagree about the same data."""
    sd, n = 0.4, 250
    se = sd / math.sqrt(n)
    at_boundary = normal_ppf(0.975) * se
    assert two_sided_p_from_mean(-at_boundary, sd, n) == pytest.approx(0.05, abs=1e-9)
    assert two_sided_p_from_mean(-at_boundary * 1.01, sd, n) < 0.05
    assert two_sided_p_from_mean(-at_boundary * 0.99, sd, n) > 0.05


def test_two_sided_p_is_one_when_there_is_nothing_to_test():
    assert two_sided_p_from_mean(0.5, 0.0, 100) == 1.0
    assert two_sided_p_from_mean(0.5, 0.3, 1) == 1.0


def test_two_sided_p_is_symmetric_in_the_sign_of_the_effect():
    assert two_sided_p_from_mean(0.2, 0.5, 80) == pytest.approx(
        two_sided_p_from_mean(-0.2, 0.5, 80)
    )


# -- reproducibility ------------------------------------------------------


def test_the_same_data_gives_the_same_verdict():
    rng = np.random.default_rng(37)
    diffs = rng.normal(-0.2, 0.6, size=300)
    first = empirical_bernstein(diffs, bound=4.0, tolerance=0.05)
    second = empirical_bernstein(diffs, bound=4.0, tolerance=0.05)
    assert (first.decision, first.n_used, first.ci) == (
        second.decision,
        second.n_used,
        second.ci,
    )


def test_simulate_error_rate_reports_a_full_partition():
    stats = simulate_error_rate(
        empirical_bernstein, trials=50, n=100, effect=0.0, noise=0.5, seed=41,
        bound=4.0, tolerance=0.1, min_n=10,
    )
    total = stats["flagged_regression"] + stats["passed"] + stats["inconclusive"]
    assert total == pytest.approx(1.0)
    assert stats["trials"] == 50
    assert 0 < stats["mean_n"] <= 100
