"""The paired tests, and the measured cost of throwing the pairing away.

The headline test here is ``test_pairing_finds_what_unpaired_calls_noise``:
on data with the structure a real eval suite has -- large shared case
difficulty, small per-case noise, a small true effect -- the paired bootstrap
detects the effect in nearly every trial and the unpaired bootstrap on the
same data detects it in a minority of them. That is the entire argument for
pairing, and it is measured rather than asserted.

The rest pins the parts of each test that are easy to get subtly wrong: which
pairs McNemar is allowed to look at, when it switches to the exact binomial,
Wilcoxon's tie correction, and whether the bootstrap interval actually covers
at its nominal rate.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from verdict.stats.distributions import normal_sf
from verdict.stats.paired import (
    _average_ranks,
    _tie_correction,
    mcnemar,
    paired_bootstrap,
    unpaired_bootstrap,
    wilcoxon_signed_rank,
)

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover
    scipy_stats = None

needs_scipy = pytest.mark.skipif(scipy_stats is None, reason="scipy not installed")


# -- McNemar ---------------------------------------------------------------


def test_mcnemar_ignores_concordant_pairs():
    """The defining property. Cases both variants passed, or both failed,
    carry no information about which is better, so padding the suite with
    hundreds of them must not move the p-value by a hair.
    """
    base = [True] * 8 + [False] * 6
    cand = [False] * 8 + [True] * 6
    small = mcnemar(base, cand)
    padded = mcnemar(base + [True] * 300, cand + [True] * 300)
    assert padded.p_value == pytest.approx(small.p_value)
    assert padded.informative == small.informative == 14
    # The effect *is* diluted by the padding, correctly: it is a rate over
    # the whole suite, not over the discordant pairs.
    assert abs(padded.effect) < abs(small.effect)


def test_mcnemar_with_no_disagreement_says_so():
    result = mcnemar([True, False] * 20, [True, False] * 20)
    assert result.p_value == 1.0
    assert result.informative == 0
    assert result.effect == 0.0
    assert "no discordant pairs" in result.detail


def test_mcnemar_all_one_way_is_the_exact_binomial():
    """Ten discordant pairs, all favouring the candidate: 2/2^10 by hand."""
    base = [True] * 10 + [True] * 5
    cand = [False] * 10 + [True] * 5
    result = mcnemar(base, cand)
    assert result.p_value == pytest.approx(2 / 1024, abs=1e-12)
    assert "exact binomial" in result.detail


def test_mcnemar_switches_to_chi_square_above_the_threshold():
    """Which test ran is reported, because 'p = 0.03 from six discordant
    pairs' deserves to be read differently from the same p from six hundred.
    """
    base = [True] * 40 + [False] * 10
    cand = [False] * 40 + [True] * 10
    assert "chi-square" in mcnemar(base, cand).detail
    assert "exact binomial" in mcnemar(base, cand, exact_below=100).detail


def test_mcnemar_continuity_corrected_statistic_is_the_textbook_one():
    """(|b - c| - 1)^2 / (b + c) with b = 40, c = 10: 29^2/50 = 16.82."""
    base = [True] * 40 + [False] * 10
    cand = [False] * 40 + [True] * 10
    result = mcnemar(base, cand)
    assert result.statistic == pytest.approx((abs(40 - 10) - 1) ** 2 / 50)
    assert result.p_value == pytest.approx(2.0 * normal_sf(math.sqrt(16.82)), rel=1e-3)


def test_mcnemar_effect_is_the_paired_difference_in_pass_rates():
    base = np.array([True] * 30 + [False] * 70)
    cand = np.array([True] * 50 + [False] * 50)
    result = mcnemar(base, cand)
    assert result.effect == pytest.approx(float(cand.mean() - base.mean()))


def test_mcnemar_interval_excludes_zero_exactly_when_it_rejects():
    base = [True] * 40 + [False] * 10
    cand = [False] * 40 + [True] * 10
    result = mcnemar(base, cand)
    assert result.significant(0.05)
    assert result.ci is not None and result.ci[1] < 0


def test_mcnemar_holds_its_size_under_the_null():
    """Two variants with identical pass rates, 3,000 simulated suites: the
    false-alarm rate must sit at or below 5%. The exact binomial makes it
    noticeably conservative at small discordance, which is the expected
    direction for an exact test and worth seeing."""
    rng = np.random.default_rng(31)
    trials = 3000
    flagged = 0
    for _ in range(trials):
        shared = rng.uniform(0, 1, size=120)  # shared case difficulty
        base = rng.uniform(0, 1, size=120) < shared
        cand = rng.uniform(0, 1, size=120) < shared
        if mcnemar(base, cand).p_value <= 0.05:
            flagged += 1
    assert flagged / trials <= 0.05


def test_mcnemar_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        mcnemar([True, False], [True])


@needs_scipy
def test_mcnemar_exact_matches_scipy_binomtest():
    base = [True] * 12 + [False] * 8 + [True] * 40
    cand = [False] * 12 + [True] * 8 + [True] * 40
    expected = scipy_stats.binomtest(8, 20, 0.5).pvalue
    assert mcnemar(base, cand).p_value == pytest.approx(expected, rel=1e-9)


# -- Wilcoxon signed rank --------------------------------------------------


def test_wilcoxon_every_pair_tied():
    scores = [3, 4, 5, 2, 4, 4, 3]
    result = wilcoxon_signed_rank(scores, scores)
    assert result.p_value == 1.0
    assert result.informative == 0
    assert "tied" in result.detail


def test_wilcoxon_refuses_to_rule_on_too_few_non_tied_pairs():
    """Five informative pairs cannot support a normal approximation, and an
    exact signed-rank distribution is more machinery than this gate needs.
    Saying so beats returning a confident number."""
    base = [3] * 40
    cand = [4] * 5 + [3] * 35
    result = wilcoxon_signed_rank(base, cand)
    assert result.p_value == 1.0
    assert result.informative == 5
    assert "too few" in result.detail


def test_wilcoxon_matches_a_hand_computed_example():
    """Eight pairs, every difference +1.

    All |d| are equal, so every rank is the average 4.5: W+ = 36, W- = 0,
    W = 0. mean = m(m+1)/4 = 18. The tie term is sum(t^3 - t) = 8^3 - 8 = 504,
    so var = 8*9*17/24 - 504/48 = 51 - 10.5 = 40.5. With the continuity
    correction z = (0 - 18 + 0.5)/sqrt(40.5) = -2.7497, p = 0.00597.
    """
    base = [1, 2, 3, 4, 5, 6, 7, 8]
    cand = [v + 1 for v in base]
    result = wilcoxon_signed_rank(base, cand)
    assert result.statistic == pytest.approx(0.0)
    assert result.effect == pytest.approx(1.0)
    z = (0 - 18 + 0.5) / math.sqrt(40.5)
    assert result.p_value == pytest.approx(2.0 * normal_sf(abs(z)), rel=1e-12)
    assert result.p_value == pytest.approx(0.00597, abs=5e-5)


def test_wilcoxon_effect_is_the_median_not_the_mean():
    """The signed-rank test is about the median difference, and reporting a
    mean next to it would describe a different null. One large outlier moves
    the mean and must not move the reported effect."""
    base = [3.0] * 20
    cand = [4.0] * 19 + [-20.0]
    result = wilcoxon_signed_rank(base, cand)
    assert result.effect == pytest.approx(1.0)
    assert float(np.mean(np.array(cand) - np.array(base))) < 0


def test_wilcoxon_averages_tied_ranks():
    ranks = _average_ranks(np.array([2.0, 2.0, 5.0, 7.0, 7.0, 7.0]))
    assert list(ranks) == pytest.approx([1.5, 1.5, 3.0, 5.0, 5.0, 5.0])


def test_tie_correction_is_zero_without_ties_and_positive_with_them():
    assert _tie_correction(np.array([1.0, 2.0, 3.0])) == 0.0
    assert _tie_correction(np.array([1.0, 1.0, 3.0])) == pytest.approx(6.0)


def test_the_tie_correction_recovers_power_it_does_not_inflate_error():
    """Measures the direction of the tie correction over 4,000 null suites of
    integer rubric scores, where ties are the rule rather than the exception.

    The corrected variance is *smaller* -- it subtracts sum(t^3 - t)/48 -- so
    the corrected test is the more willing of the two. Over 20,000 trials the
    corrected test sits at 4.8-5.1% across n = 40, 150 and 400 (nominal 5%)
    and the uncorrected one at 4.4-4.8%. So omitting the correction is
    *conservative*, not anti-conservative: it costs power, and a gate that
    quietly loses power is one that lets regressions through.
    """
    rng = np.random.default_rng(1234)

    def p_without_correction(diff):
        nz = diff[diff != 0.0]
        m = nz.size
        ranks = _average_ranks(np.abs(nz))
        w = min(float(ranks[nz > 0].sum()), float(ranks[nz < 0].sum()))
        var = m * (m + 1) * (2 * m + 1) / 24.0
        z = (w - m * (m + 1) / 4.0 + 0.5) / math.sqrt(var)
        return 2.0 * normal_sf(abs(z))

    trials = 4000
    corrected = uncorrected = 0
    for _ in range(trials):
        a = rng.integers(1, 6, size=150)
        b = rng.integers(1, 6, size=150)
        diff = (b - a).astype(float)
        corrected += wilcoxon_signed_rank(a, b).p_value <= 0.05
        uncorrected += p_without_correction(diff) <= 0.05

    # The Monte-Carlo standard error at 4,000 trials is 0.0034, so this is a
    # three-sigma band around the nominal 5%, not a tight bound.
    assert 0.04 <= corrected / trials <= 0.06
    assert uncorrected / trials < corrected / trials


def test_wilcoxon_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        wilcoxon_signed_rank([1, 2, 3], [1, 2])


@needs_scipy
def test_wilcoxon_matches_scipy_normal_approximation():
    rng = np.random.default_rng(41)
    a = rng.integers(1, 6, size=80).astype(float)
    b = (a + rng.integers(-1, 2, size=80)).astype(float)
    expected = scipy_stats.wilcoxon(
        a, b, correction=True, zero_method="wilcox", method="approx"
    ).pvalue
    assert wilcoxon_signed_rank(a, b).p_value == pytest.approx(expected, rel=1e-6)


# -- the paired bootstrap --------------------------------------------------


def test_paired_bootstrap_is_deterministic_for_a_seed():
    rng = np.random.default_rng(43)
    a, b = rng.normal(size=50), rng.normal(size=50)
    first = paired_bootstrap(a, b, draws=500, seed=7)
    second = paired_bootstrap(a, b, draws=500, seed=7)
    assert first.p_value == second.p_value
    assert first.ci == second.ci


def test_paired_bootstrap_p_value_never_hits_zero():
    """A bootstrap cannot evidence p < 1/draws, and reporting 0.0 would be a
    claim the resampling did not make."""
    a = np.zeros(80)
    b = np.full(80, 5.0)
    result = paired_bootstrap(a, b, draws=1000)
    assert result.p_value == pytest.approx(1 / 1000)
    assert result.p_value > 0.0


def test_paired_bootstrap_interval_covers_at_its_nominal_rate():
    """400 suites with a known true effect: the 95% percentile interval
    should contain it about 95% of the time. Measured, because an interval
    that does not cover is worse than no interval -- the gate reports it next
    to the verdict."""
    rng = np.random.default_rng(47)
    truth = -0.2
    covered = 0
    trials = 400
    for _ in range(trials):
        difficulty = rng.normal(0, 1.0, size=80)
        base = difficulty + rng.normal(0, 0.3, size=80)
        cand = difficulty + truth + rng.normal(0, 0.3, size=80)
        lo, hi = paired_bootstrap(base, cand, draws=400, seed=0).ci
        covered += lo <= truth <= hi
    assert 0.90 <= covered / trials <= 0.99


def test_paired_bootstrap_on_an_empty_suite_is_inconclusive_not_a_crash():
    result = paired_bootstrap([], [])
    assert result.p_value == 1.0
    assert result.n == 0


def test_paired_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap([1.0, 2.0], [1.0])


# -- the whole point ------------------------------------------------------


def test_pairing_finds_what_unpaired_calls_noise():
    """The measured argument for pairing.

    The data has the structure every eval suite has: case difficulty varies a
    lot (sd 1.0) and is shared by both variants, per-case noise is small
    (sd 0.3), and the real effect is small (-0.15). An unpaired comparison has
    to treat the difficulty spread as noise; a paired one differences it away.

    150 trials each at n = 120. The gap is large enough that the assertion
    does not need loose bounds.
    """
    rng = np.random.default_rng(53)
    trials, n, effect = 150, 120, -0.15
    paired_hits = unpaired_hits = 0
    for _ in range(trials):
        difficulty = rng.normal(0, 1.0, size=n)
        base = difficulty + rng.normal(0, 0.3, size=n)
        cand = difficulty + effect + rng.normal(0, 0.3, size=n)
        paired_hits += paired_bootstrap(base, cand, draws=600).p_value <= 0.05
        unpaired_hits += unpaired_bootstrap(base, cand, draws=600).p_value <= 0.05

    paired_power = paired_hits / trials
    unpaired_power = unpaired_hits / trials
    assert paired_power > 0.95
    assert unpaired_power < 0.5
    assert paired_power > unpaired_power * 2


def test_pairing_narrows_the_interval_on_the_same_data():
    """The same effect seen as an interval rather than a hit rate: the paired
    interval is several times narrower, and both are centred on the same
    estimate -- pairing changes the precision, not the answer."""
    rng = np.random.default_rng(59)
    difficulty = rng.normal(0, 1.0, size=200)
    base = difficulty + rng.normal(0, 0.3, size=200)
    cand = difficulty - 0.15 + rng.normal(0, 0.3, size=200)

    pair = paired_bootstrap(base, cand, draws=2000)
    unpair = unpaired_bootstrap(base, cand, draws=2000)
    pair_width = pair.ci[1] - pair.ci[0]
    unpair_width = unpair.ci[1] - unpair.ci[0]

    assert pair.effect == pytest.approx(unpair.effect, abs=1e-12)
    assert pair_width < unpair_width / 3


def test_unpaired_bootstrap_is_not_wrong_only_weaker():
    """Keeping the unpaired version honest: it must still hold its size under
    the null, so the comparison above is power against power and not a
    comparison with something broken."""
    rng = np.random.default_rng(61)
    trials = 400
    flagged = 0
    for _ in range(trials):
        difficulty = rng.normal(0, 1.0, size=100)
        base = difficulty + rng.normal(0, 0.3, size=100)
        cand = difficulty + rng.normal(0, 0.3, size=100)
        flagged += unpaired_bootstrap(base, cand, draws=400).p_value <= 0.05
    assert flagged / trials <= 0.08
