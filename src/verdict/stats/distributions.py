"""The handful of distribution functions the tests need, written out.

SciPy would supply all of this. It is not a dependency here for two reasons,
and the second is the real one:

* it is a ~40 MB wheel to get a normal CDF and a binomial tail, on a project
  whose whole statistical surface is six functions;
* a gate that decides whether code ships should not have an opaque p-value in
  it. Everything below is short enough to read, and each function says which
  identity it uses, so the number the gate reports can be checked by hand.

Accuracy is cross-checked against SciPy in ``tests/test_distributions.py``
when SciPy happens to be installed, and against closed-form values that are
known exactly (symmetry points, small binomials computed with ``math.comb``)
when it is not -- so the tests do not quietly pass by comparing the code to
itself.
"""

from __future__ import annotations

import math

SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)


def normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT2PI


def normal_cdf(x: float) -> float:
    """P(Z <= x) via the error function.

    ``math.erf`` is the C library's, so this is as accurate as the platform's
    libm -- around 1e-16 relative, far beyond what a p-value needs.
    """
    return 0.5 * (1.0 + math.erf(x / SQRT2))


def normal_sf(x: float) -> float:
    """P(Z > x), computed from the other tail to keep precision in it.

    ``1 - cdf(x)`` loses all significance for x beyond about 8, where the
    answer is ~6e-16: the subtraction cancels. Using the symmetry of the
    normal and evaluating the *lower* tail keeps the relative error small.
    """
    return 0.5 * math.erfc(x / SQRT2)


def normal_ppf(p: float) -> float:
    """The inverse CDF, via Acklam's rational approximation plus one Halley
    refinement step.

    Acklam alone is good to ~1.15e-9 relative; the refinement takes it to
    machine precision, which matters because this function sets the width of
    every confidence interval the gate reports.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")

    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)

    p_low, p_high = 0.02425, 1.0 - 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )

    # One Halley step on F(x) - p = 0.
    e = normal_cdf(x) - p
    u = e * SQRT2PI * math.exp(0.5 * x * x)
    return x - u / (1.0 + 0.5 * x * u)


def chi2_sf_1df(x: float) -> float:
    """P(X > x) for chi-square with one degree of freedom.

    A chi-square(1) variate is the square of a standard normal, so its upper
    tail is just both normal tails beyond sqrt(x). Worth writing this way
    rather than reaching for an incomplete gamma: McNemar's statistic is the
    only chi-square in the project and it always has one degree of freedom.
    """
    if x <= 0:
        return 1.0
    return 2.0 * normal_sf(math.sqrt(x))


def binomial_pmf(k: int, n: int, p: float = 0.5) -> float:
    return math.comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))


def binomial_two_sided_p(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value, by the method of small likelihoods.

    The p-value is the total probability of every outcome *no more likely*
    than the one observed. For p=0.5 that is the same as doubling the smaller
    tail, but it stays correct for an asymmetric null, where doubling does
    not. This is the exact test the gate falls back to when a McNemar table
    has too few discordant pairs for the chi-square approximation.
    """
    if n == 0:
        return 1.0
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    observed = binomial_pmf(k, n, p)
    # A relative tolerance, so floating-point noise in two equally likely
    # outcomes does not drop one of them from the sum.
    tol = 1e-12 * max(observed, 1e-300)
    total = sum(
        pmf
        for i in range(n + 1)
        if (pmf := binomial_pmf(i, n, p)) <= observed + tol
    )
    return min(1.0, total)
