"""The statistics the gate runs on, written out rather than imported.

Four pieces, each answering a way a naive eval gate gets the wrong answer:

``paired``        the suite is the same on both sides, so difference it away
``multiplicity``  many metrics means many chances to cry wolf
``sequential``    stopping when the answer looks clear is cheating unless the
                  interval was built for it
``distributions`` the six functions the above need, so SciPy is not a
                  dependency and no p-value in the gate is opaque
"""

from .distributions import (
    binomial_two_sided_p,
    chi2_sf_1df,
    normal_cdf,
    normal_ppf,
    normal_sf,
)
from .multiplicity import Adjusted, Correction, benjamini_hochberg, holm
from .paired import (
    TestResult,
    mcnemar,
    paired_bootstrap,
    unpaired_bootstrap,
    wilcoxon_signed_rank,
)
from .sequential import (
    Decision,
    SequentialResult,
    Step,
    confidence_sequence,
    fixed_horizon,
    peeking,
    simulate_error_rate,
    two_sided_p_from_mean,
)

__all__ = [
    "Adjusted",
    "Correction",
    "Decision",
    "SequentialResult",
    "Step",
    "TestResult",
    "benjamini_hochberg",
    "binomial_two_sided_p",
    "chi2_sf_1df",
    "confidence_sequence",
    "fixed_horizon",
    "holm",
    "mcnemar",
    "normal_cdf",
    "normal_ppf",
    "normal_sf",
    "paired_bootstrap",
    "peeking",
    "simulate_error_rate",
    "two_sided_p_from_mean",
    "unpaired_bootstrap",
    "wilcoxon_signed_rank",
]
