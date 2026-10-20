"""Correcting for the fact that a gate tests many things at once.

A realistic suite is not one comparison. It is several metrics -- pass rate,
rubric score, refusal rate, latency -- times several slices of the suite, and
every one of them gets a p-value. At alpha = 0.05 with twelve comparisons, the
chance of at least one false alarm under a completely unchanged pipeline is
1 - 0.95^12 = 46%. A gate that cries wolf on half of all clean runs is a gate
the team turns off, and then it is worse than no gate, because the absence of
an alarm no longer means anything.

Two corrections, and the choice between them is a choice about what the gate
is for:

``holm``
    Controls the probability of *any* false alarm (FWER). Correct but strict;
    with twelve comparisons the smallest p-value must clear 0.0042.
``benjamini_hochberg``
    Controls the expected *proportion* of false alarms among the comparisons
    flagged (FDR). The right default here: when the gate flags three metrics,
    the useful guarantee is "most of these are real", not "there is a 95%
    chance none of them is spurious".

Both are step procedures on the sorted p-values, and both are implemented so
the adjusted values they return are monotone -- an adjusted p-value that is
smaller for a larger raw p-value would be nonsense to display next to it, and
that monotonicity step is the part people leave out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class Adjusted:
    """One comparison after correction."""

    label: str
    p_raw: float
    p_adjusted: float
    rejected: bool
    #: The threshold this comparison was actually judged against, which is
    #: what makes a correction explainable to whoever the gate just blocked.
    threshold: float


@dataclass(frozen=True, slots=True)
class Correction:
    method: str
    alpha: float
    results: list[Adjusted]

    @property
    def rejected(self) -> list[Adjusted]:
        return [r for r in self.results if r.rejected]

    @property
    def naive_rejected(self) -> list[Adjusted]:
        """What an uncorrected gate would have flagged.

        Reported alongside the corrected set, because the difference between
        them is the whole argument for doing this.
        """
        return [r for r in self.results if r.p_raw <= self.alpha]

    def family_error_if_uncorrected(self) -> float:
        """P(at least one false alarm) with no correction, under the null."""
        m = len(self.results)
        return 1.0 - (1.0 - self.alpha) ** m if m else 0.0


def benjamini_hochberg(
    labelled: Sequence[tuple[str, float]], alpha: float = 0.05
) -> Correction:
    """FDR control at level alpha (Benjamini-Hochberg, 1995).

    Sort the p-values ascending, find the largest k with p_(k) <= k/m * alpha,
    and reject everything up to it. Note *largest*: stopping at the first
    failure is the common misimplementation and it is strictly less powerful.

    Assumes the p-values are independent or positively dependent, which holds
    for metrics computed from a shared suite -- they move together, not
    against each other. Arbitrary dependence needs the BY variant with its
    log(m) penalty, and that is noted in the README rather than silently
    assumed away.
    """
    m = len(labelled)
    if m == 0:
        return Correction("benjamini-hochberg", alpha, [])

    order = sorted(range(m), key=lambda i: labelled[i][1])
    sorted_p = [labelled[i][1] for i in order]

    # Largest k (1-based) satisfying the BH condition.
    k_max = 0
    for k in range(1, m + 1):
        if sorted_p[k - 1] <= k / m * alpha:
            k_max = k

    # Adjusted p-values, enforced monotone from the top down so that the
    # column reads sensibly next to the raw one.
    adjusted = [0.0] * m
    running = 1.0
    for k in range(m, 0, -1):
        running = min(running, sorted_p[k - 1] * m / k)
        adjusted[k - 1] = min(1.0, running)

    out: list[Adjusted] = [None] * m  # type: ignore[list-item]
    for rank, i in enumerate(order, start=1):
        out[i] = Adjusted(
            label=labelled[i][0],
            p_raw=labelled[i][1],
            p_adjusted=adjusted[rank - 1],
            rejected=rank <= k_max,
            threshold=rank / m * alpha,
        )
    return Correction("benjamini-hochberg", alpha, out)


def holm(labelled: Sequence[tuple[str, float]], alpha: float = 0.05) -> Correction:
    """FWER control (Holm, 1979): step down, stop at the first failure.

    Here stopping at the first failure *is* the procedure -- the opposite of
    BH -- because once a hypothesis survives, nothing after it may be
    rejected either.
    """
    m = len(labelled)
    if m == 0:
        return Correction("holm", alpha, [])

    order = sorted(range(m), key=lambda i: labelled[i][1])
    sorted_p = [labelled[i][1] for i in order]

    reject_until = -1
    for k in range(m):
        if sorted_p[k] <= alpha / (m - k):
            reject_until = k
        else:
            break

    adjusted = [0.0] * m
    running = 0.0
    for k in range(m):
        running = max(running, sorted_p[k] * (m - k))
        adjusted[k] = min(1.0, running)

    out: list[Adjusted] = [None] * m  # type: ignore[list-item]
    for rank, i in enumerate(order):
        out[i] = Adjusted(
            label=labelled[i][0],
            p_raw=labelled[i][1],
            p_adjusted=adjusted[rank],
            rejected=rank <= reject_until,
            threshold=alpha / (m - rank),
        )
    return Correction("holm", alpha, out)
