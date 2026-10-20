"""Stopping early without cheating.

The practical problem: judging a suite costs model calls, and the free tiers
this project is built against are rate limited. You want to stop as soon as
the answer is clear. The statistical problem: a fixed-sample test is only
valid if you look once. Checking a t-test after every new case and stopping
the moment p < 0.05 is the single most common way to get a confident wrong
answer -- under a null where nothing changed at all, that procedure rejects
far more often than 5% of the time, because you have given yourself hundreds
of chances to be unlucky.

``scripts/validate.py`` measures the inflation instead of citing it, over
10,000 simulated null experiments, and the number goes in the README.

The fix used here is a **confidence sequence**: an interval for the running
mean that is valid *at every sample size simultaneously*, so you may look as
often as you like and stop whenever you want. The price is width -- it is
wider than a fixed-n interval at the horizon it was tuned for, which is the
honest cost of optional stopping and is also measured.

The boundary is the normal-mixture one (Robbins 1970; Howard et al. 2021,
"Time-uniform Chernoff bounds"): for observations that are sub-Gaussian with
variance proxy ``sigma``, the radius after t observations is

    r(t) = sigma * sqrt( 2 (t*rho + 1) / (t^2 * rho) * log( sqrt(t*rho + 1) / alpha ) )

``rho`` tunes which t the boundary is tightest at. It is set from
``tuned_for``, the sample size you expect to need, because a boundary tuned
for 10 cases is needlessly wide at 300 and vice versa.

The decision rule is **non-inferiority**, not a point null, which is what a
regression gate actually wants to ask:

* the interval lies entirely below ``-tolerance`` -> REGRESSION, stop;
* the interval lies entirely above ``-tolerance`` -> PASS, stop (the
  candidate is at worst negligibly worse);
* otherwise keep sampling, and at ``max_n`` report INCONCLUSIVE.

"Inconclusive" being a real outcome is deliberate. A gate that must answer
yes or no on 30 cases will answer wrongly; one that can say "this suite does
not resolve the question" sends you to add cases instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence

import numpy as np

from .distributions import normal_ppf, normal_sf


class Decision(str, Enum):
    PASS = "pass"
    REGRESSION = "regression"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class Step:
    """The state of the evidence after one more observation."""

    n: int
    mean: float
    lower: float
    upper: float
    radius: float
    decision: Decision


@dataclass
class SequentialResult:
    decision: Decision
    n_used: int
    n_available: int
    effect: float
    ci: tuple[float, float]
    tolerance: float
    alpha: float
    stopped_early: bool
    steps: list[Step] = field(default_factory=list)

    @property
    def saved(self) -> int:
        return self.n_available - self.n_used

    @property
    def saved_fraction(self) -> float:
        return self.saved / self.n_available if self.n_available else 0.0


def mixture_rho(alpha: float, tuned_for: int) -> float:
    """Choose the boundary's tuning constant for a target sample size.

    Solves the standard choice that minimises the boundary's width at
    ``t = tuned_for``: rho = (2 log(1/alpha) + log(1 + 2 log(1/alpha))) /
    tuned_for. Any positive rho gives a valid sequence -- rho only moves where
    the boundary is tightest, never whether it holds -- which is worth knowing
    before trusting a tuning constant chosen by eye.
    """
    if tuned_for <= 0:
        raise ValueError("tuned_for must be positive")
    la = 2.0 * math.log(1.0 / alpha)
    return (la + math.log(1.0 + la)) / tuned_for


def mixture_radius(n: int, sigma: float, alpha: float, rho: float) -> float:
    """Half-width of the time-uniform interval for the mean after n points."""
    if n <= 0:
        return math.inf
    inner = n * rho + 1.0
    return sigma * math.sqrt(
        (2.0 * inner / (n * n * rho)) * math.log(math.sqrt(inner) / alpha)
    )


def confidence_sequence(
    differences: Sequence[float],
    *,
    sigma: float | None = None,
    alpha: float = 0.05,
    tolerance: float = 0.0,
    tuned_for: int | None = None,
    min_n: int = 10,
) -> SequentialResult:
    """Walk the paired differences, stopping as soon as the answer is settled.

    ``sigma`` is the sub-Gaussian variance proxy. If omitted it is taken as
    half the observed range, which is the Hoeffding bound for bounded data --
    valid for rubric scores and pass/fail indicators, both bounded by
    construction. Using the *sample* standard deviation instead would be
    tighter and would quietly invalidate the guarantee, since the bound needs
    a proxy fixed in advance, not one estimated from the same data.

    ``min_n`` exists because the boundary is enormous for the first few
    observations; checking before then wastes nothing but produces confusing
    output.
    """
    diffs = np.asarray(list(differences), dtype=float)
    total = int(diffs.size)
    if total == 0:
        return SequentialResult(
            Decision.INCONCLUSIVE, 0, 0, 0.0, (-math.inf, math.inf), tolerance, alpha, False
        )

    if sigma is None:
        spread = float(np.max(diffs) - np.min(diffs))
        sigma = max(spread / 2.0, 1e-9)
    rho = mixture_rho(alpha, tuned_for or total)

    steps: list[Step] = []
    running = 0.0
    decision = Decision.INCONCLUSIVE
    used = total
    lower = -math.inf
    upper = math.inf
    mean = 0.0

    for i, d in enumerate(diffs, start=1):
        running += float(d)
        mean = running / i
        radius = mixture_radius(i, sigma, alpha, rho)
        lower, upper = mean - radius, mean + radius
        step_decision = Decision.INCONCLUSIVE
        if i >= min_n:
            if upper < -tolerance:
                step_decision = Decision.REGRESSION
            elif lower > -tolerance:
                step_decision = Decision.PASS
        steps.append(Step(i, mean, lower, upper, radius, step_decision))
        if step_decision is not Decision.INCONCLUSIVE:
            decision = step_decision
            used = i
            break

    return SequentialResult(
        decision=decision,
        n_used=used,
        n_available=total,
        effect=mean,
        ci=(lower, upper),
        tolerance=tolerance,
        alpha=alpha,
        stopped_early=used < total,
        steps=steps,
    )


def empirical_bernstein(
    differences: Sequence[float],
    *,
    bound: float,
    alpha: float = 0.05,
    tolerance: float = 0.0,
    min_n: int = 10,
) -> SequentialResult:
    """A confidence sequence that adapts to the variance it observes.

    The sub-Gaussian boundary above needs a variance proxy fixed in advance,
    and for a 1-5 rubric the only *honest* proxy is the range: differences
    live in [-4, 4], so Hoeffding gives sigma = 4. That boundary is correct
    and useless -- on 600 cases it is about four times wider than a
    fixed-sample interval, so the gate never resolves anything and reports
    ``inconclusive`` for regressions a fixed-horizon test calls obvious.

    The real distribution of rubric differences has a standard deviation
    nearer 0.8 than 4, and an empirical-Bernstein boundary gets to use that:
    it accumulates the observed variance

        V_t = sum_i (X_i - mu_hat_{i-1})^2

    with *predictable* centering -- each term centred on the mean of the
    observations strictly before it, so nothing is estimated from the point it
    is being used to judge, which is what keeps the bound valid under optional
    stopping.

    The boundary is the stitched empirical-Bernstein one from Howard, Ramdas,
    McAuliffe and Sekhon (2021), "Time-uniform, nonasymptotic confidence
    sequences", with their recommended constants:

        u(V) = 1.7 sqrt(V (loglog(2V) + 0.72 log(10.4/alpha)))
               + 1.56 c (loglog(2V) + 0.72 log(10.4/alpha))

    where ``c`` bounds |X|. Transcribing published constants is a place to be
    wrong, so the coverage of this implementation is **measured** rather than
    trusted: ``scripts/validate.py`` runs it over thousands of null
    experiments and reports the false-alarm rate next to alpha. If a constant
    here were wrong, that number would exceed alpha and the script says so.
    """
    diffs = np.asarray(list(differences), dtype=float)
    total = int(diffs.size)
    if total == 0 or bound <= 0:
        return SequentialResult(
            Decision.INCONCLUSIVE, 0, total, 0.0, (-math.inf, math.inf), tolerance,
            alpha, False,
        )

    log_term_const = 0.72 * math.log(10.4 / alpha)
    steps: list[Step] = []
    running = 0.0
    variance = 0.0
    decision = Decision.INCONCLUSIVE
    used = total
    lower, upper = -math.inf, math.inf
    mean = 0.0

    for i, x in enumerate(diffs, start=1):
        prev_mean = mean  # predictable: the mean of everything before x
        running += float(x)
        mean = running / i
        variance += (float(x) - prev_mean) ** 2

        # loglog(2V) needs 2V > e for the inner log to be positive; below that
        # the boundary is dominated by the linear term anyway.
        v = max(variance, 1e-12)
        inner = max(math.log(2.0 * v), 1.0)
        poly = math.log(inner) + log_term_const
        radius = (1.7 * math.sqrt(v * poly) + 1.56 * bound * poly) / i

        lower, upper = mean - radius, mean + radius
        step_decision = Decision.INCONCLUSIVE
        if i >= min_n:
            if upper < -tolerance:
                step_decision = Decision.REGRESSION
            elif lower > -tolerance:
                step_decision = Decision.PASS
        steps.append(Step(i, mean, lower, upper, radius, step_decision))
        if step_decision is not Decision.INCONCLUSIVE:
            decision = step_decision
            used = i
            break

    return SequentialResult(
        decision=decision,
        n_used=used,
        n_available=total,
        effect=mean,
        ci=(lower, upper),
        tolerance=tolerance,
        alpha=alpha,
        stopped_early=used < total,
        steps=steps,
    )


def fixed_horizon(
    differences: Sequence[float], *, alpha: float = 0.05, tolerance: float = 0.0
) -> SequentialResult:
    """The honest comparison: one test, at the end, on all the data.

    A z-test on the paired mean difference. This is what the confidence
    sequence is paying for the right to replace, so it is implemented
    identically in every other respect -- same tolerance, same alpha, same
    non-inferiority framing -- and used by ``scripts/validate.py`` as the
    baseline for both error rate and sample size.
    """
    diffs = np.asarray(list(differences), dtype=float)
    n = int(diffs.size)
    if n < 2:
        return SequentialResult(
            Decision.INCONCLUSIVE, n, n, 0.0, (-math.inf, math.inf), tolerance, alpha, False
        )
    mean = float(np.mean(diffs))
    se = float(np.std(diffs, ddof=1) / math.sqrt(n))
    z = normal_ppf(1.0 - alpha / 2.0)
    lo, hi = mean - z * se, mean + z * se
    if hi < -tolerance:
        decision = Decision.REGRESSION
    elif lo > -tolerance:
        decision = Decision.PASS
    else:
        decision = Decision.INCONCLUSIVE
    return SequentialResult(
        decision=decision,
        n_used=n,
        n_available=n,
        effect=mean,
        ci=(lo, hi),
        tolerance=tolerance,
        alpha=alpha,
        stopped_early=False,
        steps=[],
    )


def peeking(
    differences: Sequence[float],
    *,
    alpha: float = 0.05,
    tolerance: float = 0.0,
    min_n: int = 10,
) -> SequentialResult:
    """The wrong way, implemented so its error rate can be measured.

    A fixed-sample z-interval recomputed after every observation, stopping the
    first time it clears the margin. This is what everybody does, it looks
    identical to the correct version in a dashboard, and it is wrong: the
    interval is valid for one look and it is being used for n of them.

    Kept in the library rather than in a test so that ``scripts/validate.py``
    and the console can both run it head to head against
    ``confidence_sequence`` on the same data.
    """
    diffs = np.asarray(list(differences), dtype=float)
    total = int(diffs.size)
    z = normal_ppf(1.0 - alpha / 2.0)
    steps: list[Step] = []
    decision = Decision.INCONCLUSIVE
    used = total
    lo = -math.inf
    hi = math.inf
    mean = 0.0

    for i in range(1, total + 1):
        window = diffs[:i]
        mean = float(np.mean(window))
        if i < 2:
            continue
        se = float(np.std(window, ddof=1) / math.sqrt(i))
        lo, hi = mean - z * se, mean + z * se
        step_decision = Decision.INCONCLUSIVE
        if i >= min_n:
            if hi < -tolerance:
                step_decision = Decision.REGRESSION
            elif lo > -tolerance:
                step_decision = Decision.PASS
        steps.append(Step(i, mean, lo, hi, z * se, step_decision))
        if step_decision is not Decision.INCONCLUSIVE:
            decision = step_decision
            used = i
            break

    return SequentialResult(
        decision=decision,
        n_used=used,
        n_available=total,
        effect=mean,
        ci=(lo, hi),
        tolerance=tolerance,
        alpha=alpha,
        stopped_early=used < total,
        steps=steps,
    )


def two_sided_p_from_mean(mean: float, sd: float, n: int) -> float:
    """z-test p-value for a mean difference, for feeding the FDR correction.

    The multiplicity correction needs p-values, and a confidence sequence
    produces intervals. This converts the fixed-horizon view of the same data
    into a p-value so the two mechanisms compose -- with the consequence,
    stated in the README, that the family-wise correction is applied to
    fixed-horizon p-values while the stopping decision is the always-valid
    one.
    """
    if n < 2 or sd <= 0:
        return 1.0
    z = mean / (sd / math.sqrt(n))
    return float(min(1.0, 2.0 * normal_sf(abs(z))))


def simulate_error_rate(
    procedure,
    *,
    trials: int = 2000,
    n: int = 200,
    effect: float = 0.0,
    noise: float = 1.0,
    seed: int = 0,
    **kwargs,
) -> dict[str, float]:
    """Run a stopping procedure on synthetic data and count its mistakes.

    With ``effect=0`` this measures the false-alarm rate, which should sit at
    or below alpha for a valid procedure and will not for ``peeking``. With a
    negative effect it measures power. Returns the rates and the average
    sample size, because the whole argument for sequential testing is a trade
    between those two.
    """
    rng = np.random.default_rng(seed)
    flagged = 0
    passed = 0
    inconclusive = 0
    used = 0
    for _ in range(trials):
        diffs = rng.normal(effect, noise, size=n)
        result = procedure(diffs, **kwargs)
        used += result.n_used
        if result.decision is Decision.REGRESSION:
            flagged += 1
        elif result.decision is Decision.PASS:
            passed += 1
        else:
            inconclusive += 1
    return {
        "trials": float(trials),
        "flagged_regression": flagged / trials,
        "passed": passed / trials,
        "inconclusive": inconclusive / trials,
        "mean_n": used / trials,
        "n_available": float(n),
    }
