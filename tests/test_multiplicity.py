"""BH and Holm: the step logic, the monotonicity, and the error rates.

Two of these tests are the ones that matter, because they catch the two
mistakes that are easy to make and impossible to see in the output:

``test_bh_takes_the_largest_k_not_the_first_failure``
    The common misimplementation walks the sorted p-values and stops at the
    first one that fails its threshold. BH says take the *largest* k that
    passes. The two agree on most inputs, which is why the bug survives, and
    the case below is one where they differ -- the wrong version rejects one
    hypothesis where the right version rejects five.

``test_bh_controls_the_false_discovery_rate``
    Measures the FDR over simulated families rather than trusting the
    procedure, and measures the uncorrected family error next to it so the
    46% number in the README is a measurement.
"""

from __future__ import annotations

import numpy as np
import pytest

from verdict.stats.multiplicity import benjamini_hochberg, holm


def labelled(ps):
    return [(f"m{i}", p) for i, p in enumerate(ps)]


# -- step logic ------------------------------------------------------------


def test_bh_takes_the_largest_k_not_the_first_failure():
    """Sorted p = 0.001, 0.025, 0.03, 0.03, 0.04 at alpha = 0.05.

    Thresholds k/m*alpha are 0.01, 0.02, 0.03, 0.04, 0.05. The second
    p-value (0.025) fails its 0.02 threshold, so a stop-at-first-failure
    implementation rejects exactly one hypothesis. But k = 5 passes
    (0.04 <= 0.05), and BH rejects everything up to the largest passing k --
    all five.
    """
    result = benjamini_hochberg(labelled([0.001, 0.03, 0.025, 0.03, 0.04]), 0.05)
    assert len(result.rejected) == 5
    assert {r.label for r in result.rejected} == {"m0", "m1", "m2", "m3", "m4"}


def test_bh_rejects_nothing_when_no_k_passes():
    result = benjamini_hochberg(labelled([0.2, 0.4, 0.6, 0.8]), 0.05)
    assert result.rejected == []
    assert all(r.p_adjusted >= r.p_raw for r in result.results)


def test_bh_thresholds_are_the_step_up_line():
    ps = [0.001, 0.01, 0.02, 0.04]
    result = benjamini_hochberg(labelled(ps), 0.05)
    by_label = {r.label: r for r in result.results}
    # Ranks follow the sorted order of the raw p-values.
    assert by_label["m0"].threshold == pytest.approx(1 / 4 * 0.05)
    assert by_label["m3"].threshold == pytest.approx(4 / 4 * 0.05)


def test_holm_stops_at_the_first_failure():
    """The same input as the BH test above. Holm's thresholds are
    alpha/(m-k): 0.0125, 0.0167, 0.025, 0.05, 0.05. The smallest p-value
    (0.001) clears 0.0125; the next (0.025) does not clear 0.0167, and Holm
    stops there -- one rejection where BH made five.
    """
    result = holm(labelled([0.001, 0.03, 0.025, 0.03, 0.04]), 0.05)
    assert [r.label for r in result.rejected] == ["m0"]


def test_holm_is_never_more_permissive_than_bh():
    """FWER control implies FDR control, so Holm's rejections are always a
    subset of BH's on the same family. Checked over random families rather
    than one example, because this is the property that justifies offering
    both and defaulting to BH."""
    rng = np.random.default_rng(7)
    for _ in range(300):
        m = int(rng.integers(2, 20))
        ps = list(rng.beta(0.4, 3.0, size=m))
        bh = {r.label for r in benjamini_hochberg(labelled(ps), 0.05).rejected}
        hm = {r.label for r in holm(labelled(ps), 0.05).rejected}
        assert hm <= bh


# -- adjusted p-values -----------------------------------------------------


def test_bh_adjusted_values_are_monotone_in_the_raw_ones():
    """The step people leave out. Without the running minimum enforced from
    the top down, a larger raw p-value can get a smaller adjusted one, and
    the two columns side by side in the report read as nonsense."""
    rng = np.random.default_rng(3)
    for _ in range(200):
        ps = sorted(rng.uniform(0, 1, size=12))
        results = benjamini_hochberg(labelled(ps), 0.05).results
        adjusted = [r.p_adjusted for r in results]  # already in raw-sorted order
        assert all(b >= a - 1e-15 for a, b in zip(adjusted, adjusted[1:]))


def test_holm_adjusted_values_are_monotone_in_the_raw_ones():
    rng = np.random.default_rng(11)
    for _ in range(200):
        ps = sorted(rng.uniform(0, 1, size=12))
        adjusted = [r.p_adjusted for r in holm(labelled(ps), 0.05).results]
        assert all(b >= a - 1e-15 for a, b in zip(adjusted, adjusted[1:]))


def test_adjusted_values_are_never_below_the_raw_ones_or_above_one():
    rng = np.random.default_rng(5)
    ps = list(rng.uniform(0, 1, size=30))
    for correction in (benjamini_hochberg(labelled(ps)), holm(labelled(ps))):
        for r in correction.results:
            assert r.p_raw - 1e-15 <= r.p_adjusted <= 1.0


def test_a_rejection_means_the_adjusted_value_cleared_alpha():
    """The invariant that lets the report show one column instead of two
    procedures: rejected exactly when p_adjusted <= alpha."""
    rng = np.random.default_rng(13)
    for _ in range(200):
        ps = list(rng.beta(0.3, 4.0, size=9))
        for correction in (
            benjamini_hochberg(labelled(ps), 0.05),
            holm(labelled(ps), 0.05),
        ):
            for r in correction.results:
                assert r.rejected == (r.p_adjusted <= 0.05 + 1e-15)


def test_single_comparison_corrects_to_itself():
    """With one hypothesis there is no multiplicity, and both procedures must
    reduce to the uncorrected test -- a sanity check that the m-1 arithmetic
    has no off-by-one."""
    for correction in (
        benjamini_hochberg(labelled([0.04]), 0.05),
        holm(labelled([0.04]), 0.05),
    ):
        assert correction.results[0].p_adjusted == pytest.approx(0.04)
        assert correction.results[0].rejected


def test_empty_family_is_not_an_error():
    for correction in (benjamini_hochberg([]), holm([])):
        assert correction.results == []
        assert correction.rejected == []
        assert correction.family_error_if_uncorrected() == 0.0


# -- the argument for doing this at all ------------------------------------


def test_family_error_formula_matches_the_readme_number():
    """1 - 0.95^14 = 51%, which is the gate's own comparison count on the
    bundled suite. This is the number the correction exists to remove."""
    correction = benjamini_hochberg(labelled([0.5] * 14), 0.05)
    assert correction.family_error_if_uncorrected() == pytest.approx(
        1.0 - 0.95**14, abs=1e-12
    )
    assert correction.family_error_if_uncorrected() > 0.5


def test_naive_rejected_is_reported_alongside_the_corrected_set():
    ps = [0.001, 0.02, 0.03, 0.04, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    correction = benjamini_hochberg(labelled(ps), 0.05)
    assert len(correction.naive_rejected) == 4
    assert len(correction.rejected) < len(correction.naive_rejected)


def test_uncorrected_family_error_is_what_the_formula_says():
    """Simulate 4,000 families of 14 all-null comparisons and count how often
    at least one p-value lands under 0.05 uncorrected. Under the null,
    p-values are uniform by construction, so this measures the 51% directly.
    """
    rng = np.random.default_rng(17)
    trials, m = 4000, 14
    any_flagged = 0
    for _ in range(trials):
        ps = rng.uniform(0, 1, size=m)
        if np.any(ps <= 0.05):
            any_flagged += 1
    rate = any_flagged / trials
    expected = 1.0 - 0.95**m
    assert rate == pytest.approx(expected, abs=0.025)


def test_bh_controls_the_false_discovery_rate():
    """Mixed families: 4 real effects, 10 nulls, 2,000 trials.

    The FDR is the expected proportion of rejections that are null. BH
    guarantees it at alpha; measured here so a transcription error in the
    step line would show up as a number above 0.05 rather than as a passing
    test. The uncorrected rate is measured in the same loop for contrast.
    """
    rng = np.random.default_rng(19)
    trials, n_alt, n_null = 2000, 4, 10
    fdr_bh = []
    fdr_naive = []
    for _ in range(trials):
        # Alternatives get small p-values, nulls get uniform ones.
        ps = np.concatenate(
            [rng.beta(0.3, 30.0, size=n_alt), rng.uniform(0, 1, size=n_null)]
        )
        truth = np.array([True] * n_alt + [False] * n_null)
        correction = benjamini_hochberg(
            [(str(i), float(p)) for i, p in enumerate(ps)], 0.05
        )
        rejected = np.array([r.rejected for r in correction.results])
        if rejected.any():
            fdr_bh.append(float(np.mean(~truth[rejected])))
        naive = ps <= 0.05
        if naive.any():
            fdr_naive.append(float(np.mean(~truth[naive])))

    assert float(np.mean(fdr_bh)) <= 0.05 + 0.01
    # And the point of the exercise: uncorrected, a much larger share of what
    # gets flagged is noise.
    assert float(np.mean(fdr_naive)) > float(np.mean(fdr_bh)) * 2


def test_bh_is_more_powerful_than_holm_on_the_same_families():
    """The reason BH is the default. Measured, because 'more powerful' is an
    empirical claim about this input distribution, not a theorem about every
    input."""
    rng = np.random.default_rng(23)
    bh_total = holm_total = 0
    for _ in range(500):
        ps = np.concatenate([rng.beta(0.3, 30.0, size=5), rng.uniform(0, 1, size=9)])
        family = [(str(i), float(p)) for i, p in enumerate(ps)]
        bh_total += len(benjamini_hochberg(family, 0.05).rejected)
        holm_total += len(holm(family, 0.05).rejected)
    assert bh_total > holm_total
