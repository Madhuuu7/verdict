"""Calibrating the judge before letting it rule.

This is the part most eval pipelines skip, and skipping it is what makes an
eval gate untrustworthy. The judge is an instrument; the questions are how
much it agrees with people, whether it is systematically generous, and where
it goes wrong.

Why Cohen's kappa rather than raw agreement: on a five-point rubric where most
answers are good, a judge that says "4" to everything agrees with a human
maybe 55% of the time while carrying no information at all. Kappa subtracts
the agreement you would get by chance from the marginal distributions, so the
constant judge scores about zero, which is the correct answer.

And why **quadratic-weighted** kappa as the headline: the rubric is ordinal.
Confusing a 4 with a 5 is a rounding difference; confusing a 1 with a 5 means
the judge is not reading the answer. Unweighted kappa treats those as the same
mistake. Both are reported, because the gap between them is itself
diagnostic -- a judge with high weighted and low unweighted kappa is ordering
correctly and quantising differently, which is usually fine.

The bootstrap interval matters more than it looks. Weighted kappa on 60
labelled cases has a 95% interval roughly +/- 0.12 wide, so a point estimate
of 0.62 against a 0.60 threshold is not evidence of anything. The gate
compares the *lower bound* against the threshold for exactly that reason.

## The threshold is relative, because the reference is not perfect

An absolute bar -- "kappa must clear 0.6" -- was the first design here and it
was wrong. Human labels carry their own noise, and no judge can agree with a
noisy reference more than a second annotator would. Measured on the bundled
suite, two simulated annotators reach weighted kappa 0.73 with each other;
against an absolute 0.6 bar, driving the judge's noise down to nearly zero
still left it failing, because the ceiling was never the judge.

So calibration takes an **inter-annotator ceiling** and asks the question that
has an answer: is the judge nearly as good as another person? The default bar
is 0.75 of the ceiling, with a small absolute floor so that two annotators who
agree on nothing cannot certify a judge by having a low ceiling. The 0.75 is a
convention, not a law, and it is a parameter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .judges import GRADES


@dataclass(frozen=True, slots=True)
class Calibration:
    """How well the judge matches the human labels."""

    n: int
    kappa: float
    kappa_ci: tuple[float, float]
    weighted_kappa: float
    weighted_kappa_ci: tuple[float, float]
    exact_agreement: float
    within_one: float
    #: Mean (judge - human). Positive means the judge is generous.
    bias: float
    bias_ci: tuple[float, float]
    #: Rows = human grade, columns = judge grade.
    confusion: list[list[int]] = field(default_factory=list)
    #: Weighted kappa between two human annotators on the same cases: the best
    #: any judge could do against this reference. None when unknown, in which
    #: case the absolute floor is the only bar.
    ceiling: float | None = None
    #: Required fraction of the ceiling.
    threshold: float = 0.75
    #: Absolute minimum, so two annotators who barely agree cannot certify a
    #: judge by setting a low ceiling.
    floor: float = 0.40
    note: str = ""

    @property
    def relative(self) -> float | None:
        """Judge agreement as a fraction of the inter-annotator ceiling."""
        if not self.ceiling:
            return None
        return self.weighted_kappa / self.ceiling

    @property
    def required(self) -> float:
        """The weighted kappa this judge actually has to clear."""
        if self.ceiling:
            return max(self.floor, self.threshold * self.ceiling)
        return self.floor

    @property
    def trustworthy(self) -> bool:
        """Whether the gate may rule on this judge's scores.

        Tested on the *lower* confidence bound, not the point estimate: with a
        few dozen labels the estimate alone cannot tell 0.58 from 0.68.
        """
        if self.n == 0:
            return False
        return self.weighted_kappa_ci[0] >= self.required

    def verdict(self) -> str:
        if self.n == 0:
            return "no human labels: the judge is unverified, so the gate abstains"
        lo, hi = self.weighted_kappa_ci
        head = (
            f"weighted kappa {self.weighted_kappa:.2f} (95% CI {lo:.2f}-{hi:.2f}) "
            f"on {self.n} labels"
        )
        if self.ceiling:
            head += f", {self.relative:.0%} of the {self.ceiling:.2f} annotator ceiling"
        if self.trustworthy:
            return head + ": good enough to rule"
        return head + f": below the {self.required:.2f} bar, so the gate abstains"


def confusion_matrix(human: np.ndarray, judge: np.ndarray) -> np.ndarray:
    k = len(GRADES)
    m = np.zeros((k, k), dtype=int)
    for h, j in zip(human, judge):
        m[int(h) - 1, int(j) - 1] += 1
    return m


def _kappa_from_matrix(m: np.ndarray, weights: np.ndarray | None) -> float:
    """Cohen's kappa, optionally weighted, straight from the definition.

    kappa = 1 - (observed disagreement) / (expected disagreement), where
    expectation uses the product of the two marginals. With all-zero weights
    on the diagonal and ones elsewhere this is the unweighted statistic.
    """
    total = m.sum()
    if total == 0:
        return 0.0
    observed = m / total
    rows = observed.sum(axis=1, keepdims=True)
    cols = observed.sum(axis=0, keepdims=True)
    expected = rows @ cols

    if weights is None:
        k = m.shape[0]
        weights = 1.0 - np.eye(k)

    num = float((weights * observed).sum())
    den = float((weights * expected).sum())
    if den == 0.0:
        # No expected disagreement at all: the marginals are degenerate (every
        # label the same grade). Kappa is undefined; 0.0 is the honest answer
        # rather than a division that raises.
        return 0.0
    return 1.0 - num / den


def quadratic_weights(k: int) -> np.ndarray:
    """(i - j)^2 / (k - 1)^2 -- the standard ordinal penalty."""
    idx = np.arange(k)
    return ((idx[:, None] - idx[None, :]) ** 2) / ((k - 1) ** 2)


def _bootstrap_ci(
    human: np.ndarray,
    judge: np.ndarray,
    statistic,
    *,
    draws: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile interval by resampling the labelled *pairs*."""
    n = human.size
    if n < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(draws, n))
    values = np.empty(draws, dtype=float)
    for d in range(draws):
        pick = idx[d]
        values[d] = statistic(human[pick], judge[pick])
    return (
        float(np.quantile(values, alpha / 2.0)),
        float(np.quantile(values, 1.0 - alpha / 2.0)),
    )


def calibrate(
    human_grades: dict[str, int],
    judge_grades: dict[str, int],
    *,
    ceiling: float | None = None,
    threshold: float = 0.75,
    floor: float = 0.40,
    draws: int = 2000,
    seed: int = 0,
) -> Calibration:
    """Compare the judge against the human labels on their shared cases."""
    shared = sorted(set(human_grades) & set(judge_grades))
    if not shared:
        return Calibration(
            n=0,
            kappa=0.0,
            kappa_ci=(float("nan"), float("nan")),
            weighted_kappa=0.0,
            weighted_kappa_ci=(float("nan"), float("nan")),
            exact_agreement=0.0,
            within_one=0.0,
            bias=0.0,
            bias_ci=(float("nan"), float("nan")),
            confusion=[[0] * len(GRADES) for _ in GRADES],
            ceiling=ceiling,
            threshold=threshold,
            floor=floor,
            note="no case was labelled by both",
        )

    human = np.array([human_grades[c] for c in shared], dtype=int)
    judge = np.array([judge_grades[c] for c in shared], dtype=int)
    k = len(GRADES)
    qw = quadratic_weights(k)

    plain = lambda h, j: _kappa_from_matrix(confusion_matrix(h, j), None)  # noqa: E731
    weighted = lambda h, j: _kappa_from_matrix(confusion_matrix(h, j), qw)  # noqa: E731

    diff = judge - human
    bias = float(np.mean(diff))
    se = float(np.std(diff, ddof=1) / np.sqrt(diff.size)) if diff.size > 1 else 0.0

    return Calibration(
        n=len(shared),
        kappa=plain(human, judge),
        kappa_ci=_bootstrap_ci(human, judge, plain, draws=draws, seed=seed),
        weighted_kappa=weighted(human, judge),
        weighted_kappa_ci=_bootstrap_ci(
            human, judge, weighted, draws=draws, seed=seed + 1
        ),
        exact_agreement=float(np.mean(human == judge)),
        within_one=float(np.mean(np.abs(diff) <= 1)),
        bias=bias,
        bias_ci=(bias - 1.959963985 * se, bias + 1.959963985 * se),
        confusion=confusion_matrix(human, judge).tolist(),
        ceiling=ceiling,
        threshold=threshold,
        floor=floor,
    )


def annotator_ceiling(
    a_grades: dict[str, int], b_grades: dict[str, int], *, draws: int = 2000
) -> Calibration:
    """Agreement between two human annotators: the bar a judge is held to.

    Returned as a full ``Calibration`` so the console can show the ceiling's
    own confidence interval. A ceiling measured on 60 cases is itself
    uncertain, and the judge is being compared against an estimate rather than
    a constant -- the sort of thing that is easy to forget once the number has
    been printed once.
    """
    return calibrate(a_grades, b_grades, draws=draws)


def constant_judge_kappa(human_grades: dict[str, int], grade: int = 4) -> float:
    """What a judge that says the same thing every time would score.

    Printed next to the real kappa in the console. It is the sanity check that
    makes kappa legible: raw agreement for this judge is often above 50%,
    and its kappa is ~0. Seeing both numbers side by side is what stops
    someone reading 55% agreement as 'mostly right'.
    """
    if not human_grades:
        return 0.0
    judge = {c: grade for c in human_grades}
    return calibrate(human_grades, judge, draws=1).kappa
