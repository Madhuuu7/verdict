"""The six hand-written distribution functions, checked against things that
are known independently of this code.

The trap in testing a hand-rolled normal CDF is to compare it to itself --
assert ``normal_cdf(1.96) == 0.9750021048517795`` with the constant copied out
of a run of the function. Those tests pass on a broken implementation. So
every check here is against one of:

* an identity the function must satisfy regardless of implementation
  (symmetry, round-tripping, the chi-square/normal relation);
* a value computed exactly another way (``math.comb`` for binomial tails);
* SciPy, when it happens to be installed -- it is not a dependency, so those
  checks skip rather than fail.
"""

from __future__ import annotations

import math

import pytest

from verdict.stats.distributions import (
    binomial_pmf,
    binomial_two_sided_p,
    chi2_sf_1df,
    normal_cdf,
    normal_pdf,
    normal_ppf,
    normal_sf,
)

try:  # an optional cross-check, never a dependency
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover - scipy is deliberately absent here
    scipy_stats = None

needs_scipy = pytest.mark.skipif(scipy_stats is None, reason="scipy not installed")


# -- normal_cdf ------------------------------------------------------------


def test_cdf_is_one_half_at_zero():
    assert normal_cdf(0.0) == pytest.approx(0.5, abs=1e-15)


@pytest.mark.parametrize("x", [0.1, 0.5, 1.0, 1.96, 2.5, 4.0, 7.5])
def test_cdf_is_symmetric(x):
    """F(-x) = 1 - F(x) for every x, for any correct normal CDF."""
    assert normal_cdf(-x) + normal_cdf(x) == pytest.approx(1.0, abs=1e-14)


def test_cdf_is_monotone():
    xs = [i / 10.0 for i in range(-60, 61)]
    values = [normal_cdf(x) for x in xs]
    assert all(b >= a for a, b in zip(values, values[1:]))


def test_pdf_integrates_to_the_cdf():
    """Simpson's rule on the pdf must reproduce the cdf.

    An independent check: the two functions share no code beyond the
    constants, so agreement to 1e-12 says both are right.

    The lower limit is -9 rather than -6 because the integral has to account
    for the tail it truncates: P(Z < -6) is 9.9e-10, which swamps Simpson's
    own error here and would make this test look like a CDF bug.
    """
    lo, hi, n = -9.0, 1.3, 20_000
    h = (hi - lo) / n
    total = normal_pdf(lo) + normal_pdf(hi)
    for i in range(1, n):
        total += normal_pdf(lo + i * h) * (4 if i % 2 else 2)
    integral = total * h / 3.0
    assert integral == pytest.approx(normal_cdf(hi), abs=1e-12)


# -- normal_sf -------------------------------------------------------------


def test_sf_keeps_precision_in_the_far_tail():
    """The reason ``normal_sf`` exists rather than ``1 - normal_cdf``.

    At x = 9 the survival function is ~1.1e-19, which is below the resolution
    of ``1 - cdf(9)``: that subtraction returns exactly 0.0. A gate that
    reports p-values must not silently floor them at zero.
    """
    assert 1.0 - normal_cdf(9.0) == 0.0
    assert normal_sf(9.0) > 0.0
    # Mills-ratio bounds: pdf(x)*x/(x^2+1) < sf(x) < pdf(x)/x, exact for all
    # x > 0 and tight enough at x = 9 to pin the value to a few percent.
    pdf = normal_pdf(9.0)
    assert pdf * 9.0 / (81.0 + 1.0) < normal_sf(9.0) < pdf / 9.0


@pytest.mark.parametrize("x", [-3.0, -0.5, 0.0, 0.5, 1.96, 4.0])
def test_sf_complements_cdf(x):
    assert normal_sf(x) + normal_cdf(x) == pytest.approx(1.0, abs=1e-14)


# -- normal_ppf ------------------------------------------------------------


@pytest.mark.parametrize(
    "p",
    [1e-12, 1e-6, 0.001, 0.02424, 0.02425, 0.05, 0.25, 0.5, 0.75, 0.975, 0.9999,
     1 - 1e-9],
)
def test_ppf_round_trips_through_cdf(p):
    """The defining property, over all three branches of Acklam's
    approximation including both breakpoints at 0.02425."""
    assert normal_cdf(normal_ppf(p)) == pytest.approx(p, rel=1e-11, abs=1e-15)


def test_ppf_is_zero_at_one_half():
    assert normal_ppf(0.5) == pytest.approx(0.0, abs=1e-15)


def test_ppf_is_antisymmetric():
    for p in (0.01, 0.1, 0.3, 0.45):
        assert normal_ppf(p) == pytest.approx(-normal_ppf(1.0 - p), rel=1e-11)


def test_ppf_hits_the_published_critical_values():
    """1.959963984540054 is the 97.5th percentile, known to 16 digits.

    This is the most load-bearing constant in the project -- it sets the width
    of every fixed-horizon interval the gate reports -- so it is pinned
    against the published value, not only round-tripped. Acklam's
    approximation alone is good to ~1.15e-9; passing at 1e-12 is what says
    the Halley refinement step is actually running.
    """
    assert normal_ppf(0.975) == pytest.approx(1.959963984540054, abs=1e-12)
    assert normal_ppf(0.995) == pytest.approx(2.5758293035489004, abs=1e-12)
    assert normal_ppf(0.95) == pytest.approx(1.6448536269514722, abs=1e-12)


@pytest.mark.parametrize("p", [0.0, 1.0, -0.1, 1.5])
def test_ppf_rejects_values_outside_the_open_unit_interval(p):
    with pytest.raises(ValueError):
        normal_ppf(p)


@needs_scipy
def test_ppf_matches_scipy():
    for i in range(1, 1000):
        p = i / 1000.0
        assert normal_ppf(p) == pytest.approx(scipy_stats.norm.ppf(p), rel=1e-11)


@needs_scipy
def test_cdf_matches_scipy():
    for i in range(-80, 81):
        x = i / 10.0
        assert normal_cdf(x) == pytest.approx(scipy_stats.norm.cdf(x), rel=1e-12)


# -- chi2_sf_1df -----------------------------------------------------------


def test_chi2_critical_values_are_the_table_ones():
    """3.841458820694124 and 6.634896601021214 are the 95th and 99th
    percentiles of chi-square(1); every statistics table agrees on them."""
    assert chi2_sf_1df(3.841458820694124) == pytest.approx(0.05, abs=1e-12)
    assert chi2_sf_1df(6.634896601021214) == pytest.approx(0.01, abs=1e-12)


def test_chi2_is_the_square_of_a_normal():
    """P(X > x) for chi-square(1) must equal P(|Z| > sqrt(x)), by the
    definition of the distribution rather than by this implementation."""
    for x in (0.5, 1.0, 2.7, 10.0, 30.0):
        both_tails = 2.0 * (1.0 - normal_cdf(math.sqrt(x)))
        assert chi2_sf_1df(x) == pytest.approx(both_tails, rel=1e-10)


def test_chi2_is_one_at_or_below_zero():
    assert chi2_sf_1df(0.0) == 1.0
    assert chi2_sf_1df(-4.0) == 1.0


@needs_scipy
def test_chi2_matches_scipy():
    for x in (0.1, 1.0, 3.84, 10.0, 40.0):
        assert chi2_sf_1df(x) == pytest.approx(scipy_stats.chi2.sf(x, 1), rel=1e-10)


# -- binomial_two_sided_p --------------------------------------------------


def test_binomial_pmf_sums_to_one():
    for n in (1, 5, 20):
        for p in (0.3, 0.5, 0.9):
            assert sum(binomial_pmf(k, n, p) for k in range(n + 1)) == pytest.approx(
                1.0, abs=1e-12
            )


def test_binomial_all_one_way_is_two_over_two_to_the_n():
    """Ten discordant pairs all favouring one variant: the exact two-sided
    p-value is 2/2^10, which can be checked by hand."""
    assert binomial_two_sided_p(10, 10, 0.5) == pytest.approx(2 / 1024, abs=1e-15)
    assert binomial_two_sided_p(0, 10, 0.5) == pytest.approx(2 / 1024, abs=1e-15)
    assert binomial_two_sided_p(6, 6, 0.5) == pytest.approx(2 / 64, abs=1e-15)


def test_binomial_symmetric_under_a_fair_null():
    for n in (7, 12, 25):
        for k in range(n + 1):
            assert binomial_two_sided_p(k, n, 0.5) == pytest.approx(
                binomial_two_sided_p(n - k, n, 0.5), abs=1e-14
            )


def test_binomial_matches_an_independent_enumeration():
    """The method of small likelihoods, enumerated separately with
    ``math.comb`` -- including asymmetric nulls, where the familiar 'double
    the smaller tail' shortcut gives a different answer."""
    for n, k, p in [(10, 7, 0.5), (15, 3, 0.3), (9, 8, 0.75), (20, 10, 0.4)]:
        obs = math.comb(n, k) * p**k * (1 - p) ** (n - k)
        expected = sum(
            pmf
            for i in range(n + 1)
            if (pmf := math.comb(n, i) * p**i * (1 - p) ** (n - i))
            <= obs * (1 + 1e-12)
        )
        assert binomial_two_sided_p(k, n, p) == pytest.approx(expected, rel=1e-12)


def test_binomial_doubling_shortcut_disagrees_on_an_asymmetric_null():
    """Pins the documented choice: for p != 0.5 the exact test is *not* twice
    the smaller tail, and this is which one got implemented."""
    n, k, p = 15, 3, 0.3
    doubled = min(1.0, 2.0 * sum(binomial_pmf(i, n, p) for i in range(k + 1)))
    exact = binomial_two_sided_p(k, n, p)
    assert exact != pytest.approx(doubled, rel=1e-6)
    assert 0.0 < exact <= 1.0


def test_binomial_is_one_with_nothing_to_go_on():
    assert binomial_two_sided_p(0, 0, 0.5) == 1.0


def test_binomial_never_exceeds_one():
    for n in (1, 2, 3, 8, 30):
        for k in range(n + 1):
            assert 0.0 < binomial_two_sided_p(k, n, 0.5) <= 1.0


@pytest.mark.parametrize("p", [0.0, 1.0, -0.2, 2.0])
def test_binomial_rejects_an_impossible_null(p):
    with pytest.raises(ValueError):
        binomial_two_sided_p(1, 5, p)


@needs_scipy
def test_binomial_matches_scipy_binomtest():
    for n, k, p in [(10, 7, 0.5), (25, 9, 0.5), (15, 3, 0.3), (40, 25, 0.6)]:
        expected = scipy_stats.binomtest(k, n, p).pvalue
        assert binomial_two_sided_p(k, n, p) == pytest.approx(expected, rel=1e-9)
