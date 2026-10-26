"""Shared fixtures.

The gate runs are session-scoped on purpose: a single ``run_gate`` over the
bundled suite does two pipeline passes, ~500 judgements and several thousand
bootstrap resamples, and a dozen tests want to assert different things about
the *same* report. Rebuilding it per test would turn a two-second suite into a
thirty-second one and would prove nothing extra -- the gate is deterministic,
so every test is looking at the same numbers either way.
"""

from __future__ import annotations

import pytest

from verdict import (
    GateConfig,
    HumanLabeller,
    RubricJudge,
    execute,
    run_gate,
    synthetic_suite,
    variant_pair,
)


@pytest.fixture(scope="session")
def suite():
    return synthetic_suite(240)


@pytest.fixture(scope="session")
def small_suite():
    return synthetic_suite(60, name="small")


@pytest.fixture(scope="session")
def judge():
    return RubricJudge()


@pytest.fixture(scope="session")
def annotators():
    # Two labellers with different names, which is what makes them two
    # *people*: seeding both on a shared literal is the bug that produced an
    # inter-annotator ceiling of exactly 1.000.
    return (HumanLabeller(name="annotator-a"), HumanLabeller(name="annotator-b"))


@pytest.fixture(scope="session")
def identical_runs(suite, judge):
    """Baseline and candidate with no regression injected."""
    base, cand = variant_pair("none")
    return (
        execute(suite, base, judge, "baseline"),
        execute(suite, cand, judge, "candidate"),
    )


@pytest.fixture(scope="session")
def regressed_runs(suite, judge):
    base, cand = variant_pair("quality", severity=0.12)
    return (
        execute(suite, base, judge, "baseline"),
        execute(suite, cand, judge, "candidate"),
    )


def gate_for(preset: str, severity: float, suite, judge, annotators, **overrides):
    base, cand = variant_pair(preset, severity=severity)
    config = GateConfig(**overrides)
    return run_gate(suite, base, cand, judge, annotators, config)


@pytest.fixture(scope="session")
def report_identical(suite, judge, annotators):
    return gate_for("none", 0.0, suite, judge, annotators)


@pytest.fixture(scope="session")
def report_clear_drop(suite, judge, annotators):
    return gate_for("quality", 0.12, suite, judge, annotators)


@pytest.fixture(scope="session")
def report_slice_broken(suite, judge, annotators):
    return gate_for("tag", 0.06, suite, judge, annotators)


@pytest.fixture(scope="session")
def report_latency(suite, judge, annotators):
    return gate_for("latency", 0.25, suite, judge, annotators)
