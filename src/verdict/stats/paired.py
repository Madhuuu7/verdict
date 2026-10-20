"""Paired tests, because the cases are the same on both sides.

Running variant A and variant B over the *same* suite and then comparing two
independent means throws away the most useful structure in the data. Case
difficulty varies enormously -- some prompts are hard for every version of a
pipeline -- and an unpaired test has to treat that variance as noise. A paired
test differences it away, which is why a pairing can detect an effect that an
unpaired comparison on the same data calls noise.

``tests/test_paired.py`` measures that difference rather than asserting it:
on a suite with heavy case-difficulty variance, the paired test's power is far
above the unpaired test's at the same sample size.

Three tests, picked by what the judge produces:

``mcnemar``
    binary pass/fail. Looks only at the cases where the two variants disagree,
    which is the entire information content of a paired binary comparison.
``wilcoxon_signed_rank``
    ordinal scores (a 1-5 rubric). Does not assume normality, which a rubric
    score flatly is not.
``paired_bootstrap``
    any statistic, including ones with no closed form, and the one that gives
    a confidence interval on the *effect size* rather than only a p-value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .distributions import binomial_two_sided_p, chi2_sf_1df, normal_ppf, normal_sf


@dataclass(frozen=True, slots=True)
class TestResult:
    """One comparison's verdict, with everything needed to argue about it."""

    name: str
    statistic: float
    p_value: float
    effect: float
    #: Confidence interval on the effect, where the test provides one.
    ci: tuple[float, float] | None = None
    n: int = 0
    #: Pairs that carry information -- discordant pairs, or non-zero
    #: differences. A p-value computed from four informative pairs deserves to
    #: be read differently from one computed from four hundred.
    informative: int = 0
    detail: str = ""

    def significant(self, alpha: float = 0.05) -> bool:
        return self.p_value <= alpha


def mcnemar(
    baseline: Sequence[bool], candidate: Sequence[bool], *, exact_below: int = 25
) -> TestResult:
    """Paired binary comparison.

    Only the discordant pairs matter: cases both variants passed, or both
    failed, say nothing about which is better. With ``b`` cases the candidate
    fixed and ``c`` it broke, the null is that a discordance is equally likely
    either way -- a coin flip -- so this is a binomial test on ``b`` out of
    ``b + c``.

    Below ``exact_below`` discordant pairs the chi-square approximation is
    not trustworthy (the usual rule of thumb is b + c >= 25), so the exact
    binomial is used instead. The gate reports which one ran, because "p =
    0.03 from six discordant pairs" is a different sentence from "p = 0.03
    from six hundred".
    """
    a = np.asarray(baseline, dtype=bool)
    b_arr = np.asarray(candidate, dtype=bool)
    if a.shape != b_arr.shape:
        raise ValueError("paired inputs must have the same length")
    n = int(a.size)

    fixed = int(np.sum(~a & b_arr))     # candidate passes where baseline failed
    broke = int(np.sum(a & ~b_arr))     # candidate fails where baseline passed
    discordant = fixed + broke
    effect = (fixed - broke) / n if n else 0.0

    if discordant == 0:
        return TestResult(
            name="mcnemar",
            statistic=0.0,
            p_value=1.0,
            effect=0.0,
            ci=(0.0, 0.0),
            n=n,
            informative=0,
            detail="no discordant pairs: the variants agreed on every case",
        )

    if discordant < exact_below:
        p = binomial_two_sided_p(fixed, discordant, 0.5)
        stat = float(fixed)
        detail = f"exact binomial on {discordant} discordant pairs"
    else:
        # Continuity-corrected chi-square with one degree of freedom.
        stat = (abs(fixed - broke) - 1.0) ** 2 / discordant
        p = chi2_sf_1df(stat)
        detail = f"continuity-corrected chi-square on {discordant} discordant pairs"

    return TestResult(
        name="mcnemar",
        statistic=float(stat),
        p_value=float(p),
        effect=float(effect),
        ci=_wilson_difference_ci(fixed, broke, n),
        n=n,
        informative=discordant,
        detail=detail,
    )


def _wilson_difference_ci(
    fixed: int, broke: int, n: int, alpha: float = 0.05
) -> tuple[float, float]:
    """CI for the difference in pass rates of paired binary data.

    Uses the standard error of the paired difference, which depends only on
    the discordant counts -- the concordant pairs contribute nothing to the
    variance of the difference, the same reason they contribute nothing to the
    test.
    """
    if n == 0:
        return (0.0, 0.0)
    diff = (fixed - broke) / n
    var = (fixed + broke - (fixed - broke) ** 2 / n) / (n * n)
    se = float(np.sqrt(max(var, 0.0)))
    z = normal_ppf(1.0 - alpha / 2.0)
    return (diff - z * se, diff + z * se)


def wilcoxon_signed_rank(
    baseline: Sequence[float], candidate: Sequence[float]
) -> TestResult:
    """Paired ordinal comparison, normal approximation with tie correction.

    Zero differences are dropped (Wilcoxon's original handling) and tied
    absolute differences get average ranks, with the variance corrected for
    ties -- without that correction the test is anti-conservative on rubric
    scores, where ties are the norm rather than the exception.

    The effect reported is the median difference, which is the quantity this
    test is actually about; the mean would describe a different null.
    """
    x = np.asarray(baseline, dtype=float)
    y = np.asarray(candidate, dtype=float)
    if x.shape != y.shape:
        raise ValueError("paired inputs must have the same length")
    n = int(x.size)
    diff = y - x
    nonzero = diff[diff != 0.0]
    m = int(nonzero.size)
    median = float(np.median(diff)) if n else 0.0

    if m == 0:
        return TestResult(
            name="wilcoxon",
            statistic=0.0,
            p_value=1.0,
            effect=0.0,
            n=n,
            informative=0,
            detail="every pair was tied",
        )
    if m < 6:
        # The normal approximation is meaningless this small, and an exact
        # signed-rank distribution is more machinery than this gate needs:
        # say so rather than return a confident number.
        return TestResult(
            name="wilcoxon",
            statistic=0.0,
            p_value=1.0,
            effect=median,
            n=n,
            informative=m,
            detail=f"only {m} non-tied pairs: too few to test, treated as inconclusive",
        )

    ranks = _average_ranks(np.abs(nonzero))
    w_plus = float(np.sum(ranks[nonzero > 0]))
    w_minus = float(np.sum(ranks[nonzero < 0]))
    w = min(w_plus, w_minus)

    mean_w = m * (m + 1) / 4.0
    tie_term = _tie_correction(np.abs(nonzero))
    var_w = m * (m + 1) * (2 * m + 1) / 24.0 - tie_term / 48.0
    if var_w <= 0:
        return TestResult(
            name="wilcoxon", statistic=w, p_value=1.0, effect=median, n=n,
            informative=m, detail="zero variance after tie correction",
        )
    z = (w - mean_w + 0.5) / float(np.sqrt(var_w))  # continuity correction
    p = 2.0 * normal_sf(abs(z))

    return TestResult(
        name="wilcoxon",
        statistic=float(w),
        p_value=float(min(1.0, p)),
        effect=median,
        n=n,
        informative=m,
        detail=f"normal approximation, {m} non-tied pairs, tie-corrected variance",
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, which is what the tie correction assumes."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=float)
    sorted_vals = values[order]
    i = 0
    while i < sorted_vals.size:
        j = i
        while j + 1 < sorted_vals.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _tie_correction(values: np.ndarray) -> float:
    _, counts = np.unique(values, return_counts=True)
    return float(np.sum(counts ** 3 - counts))


@dataclass
class BootstrapResult:
    effect: float
    ci: tuple[float, float]
    p_value: float
    draws: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))


def paired_bootstrap(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    draws: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> TestResult:
    """Resample the *pairs*, not the two sides independently.

    Resampling each side separately would break the pairing and reintroduce
    the case-difficulty variance the pairing exists to remove. Resampling
    case indices keeps each case's two scores together.

    The p-value comes from the bootstrap distribution of the mean difference
    by the percentile method: double the smaller tail mass on the far side of
    zero. The interval is a percentile interval, which is adequate here
    because the statistic is a mean difference and so close to symmetric; a
    BCa interval would matter for a skewed statistic and is noted in the
    README's limits rather than pretended to.
    """
    x = np.asarray(baseline, dtype=float)
    y = np.asarray(candidate, dtype=float)
    if x.shape != y.shape:
        raise ValueError("paired inputs must have the same length")
    n = int(x.size)
    if n == 0:
        return TestResult("paired_bootstrap", 0.0, 1.0, 0.0, (0.0, 0.0), 0, 0, "empty")

    diff = y - x
    observed = float(np.mean(diff))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(draws, n))
    means = diff[idx].mean(axis=1)

    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    # Two-sided percentile p-value, with the +1 that keeps it from ever being
    # exactly zero -- a bootstrap cannot evidence p < 1/draws.
    below = float(np.mean(means <= 0.0))
    above = float(np.mean(means >= 0.0))
    p = min(1.0, 2.0 * min(below, above) + 1.0 / draws)

    return TestResult(
        name="paired_bootstrap",
        statistic=observed,
        p_value=p,
        effect=observed,
        ci=(lo, hi),
        n=n,
        informative=int(np.sum(diff != 0.0)),
        detail=f"{draws:,} paired resamples, percentile interval",
    )


def unpaired_bootstrap(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    draws: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> TestResult:
    """The same statistic with the pairing thrown away.

    Kept deliberately, and only used by ``tests/test_paired.py`` and
    ``scripts/validate.py``, so the cost of ignoring the pairing can be
    measured rather than asserted.
    """
    x = np.asarray(baseline, dtype=float)
    y = np.asarray(candidate, dtype=float)
    observed = float(np.mean(y) - np.mean(x))
    rng = np.random.default_rng(seed)
    xs = x[rng.integers(0, x.size, size=(draws, x.size))].mean(axis=1)
    ys = y[rng.integers(0, y.size, size=(draws, y.size))].mean(axis=1)
    means = ys - xs
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    p = min(
        1.0,
        2.0 * min(float(np.mean(means <= 0.0)), float(np.mean(means >= 0.0)))
        + 1.0 / draws,
    )
    return TestResult(
        name="unpaired_bootstrap",
        statistic=observed,
        p_value=p,
        effect=observed,
        ci=(lo, hi),
        n=int(x.size),
        informative=int(x.size),
        detail=f"{draws:,} independent resamples per side (pairing discarded)",
    )
