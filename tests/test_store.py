"""Gate history in SQLite.

A single run answers "did this change regress anything". The question a team
actually has is "is this metric drifting", and that one needs the runs kept.
So the tests here are about the two things that make the history worth
keeping rather than a log: that a run and its metrics are never stored apart,
and that the judge's identity is stored next to the numbers it produced.

``test_a_failed_metric_insert_leaves_no_run_behind``
    A run row whose metrics are missing is worse than no run at all -- the
    history would show a verdict with nothing behind it. The insert is one
    transaction and the test breaks the second half of it on purpose.

``test_the_judge_is_stored_next_to_the_numbers_it_produced``
    The most confusing failure in eval history is a judge swap that shifts
    every score at once and reads as a pipeline regression. Without the
    instrument recorded alongside the measurement it cannot be told apart.
"""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from verdict.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "history.db")
    yield s
    s.close()


# -- recording ------------------------------------------------------------


def test_a_run_and_its_metrics_round_trip(store, report_clear_drop):
    run_id = store.record(report_clear_drop, label="release-42")
    assert run_id > 0

    row = store.run(run_id)
    assert row is not None
    assert row["label"] == "release-42"
    assert row["verdict"] == "regression"
    assert row["n_cases"] == 600
    assert row["comparisons"] == len(report_clear_drop.metrics)
    assert row["flagged"] == len(report_clear_drop.flagged)
    assert row["reason"] == report_clear_drop.reason
    assert len(row["metrics"]) == len(report_clear_drop.metrics)


def test_the_stored_effect_is_the_end_of_suite_one_with_its_interval(
    store, report_clear_drop
):
    """The point estimate alone is unreadable across releases -- every run
    looks like a change. The interval is what lets the history show whether
    anything moved beyond its own uncertainty."""
    run_id = store.record(report_clear_drop)
    row = store.run(run_id)
    fixed = report_clear_drop.sequential_fixed
    assert row["effect"] == pytest.approx(fixed.effect)
    assert row["ci_lo"] == pytest.approx(fixed.ci[0])
    assert row["ci_hi"] == pytest.approx(fixed.ci[1])
    assert row["ci_lo"] < row["effect"] < row["ci_hi"]


def test_the_judge_is_stored_next_to_the_numbers_it_produced(
    store, report_clear_drop
):
    run_id = store.record(report_clear_drop)
    assert store.run(run_id)["judge"] == "rubric-judge"
    assert store.history()[0].judge == "rubric-judge"
    # And the instrument's own agreement, so a kappa step change is visible
    # in the same row as the metric step change.
    assert store.run(run_id)["kappa"] == pytest.approx(
        report_clear_drop.calibration.weighted_kappa
    )
    assert store.run(run_id)["ceiling"] == pytest.approx(
        report_clear_drop.ceiling.weighted_kappa
    )


def test_the_label_defaults_to_the_verdict(store, report_identical):
    run_id = store.record(report_identical)
    assert store.run(run_id)["label"] == report_identical.verdict.value


def test_the_config_is_stored_so_an_old_verdict_can_be_read_in_context(
    store, report_clear_drop
):
    """A verdict means nothing without the margin and alpha it was reached
    under, and both are configurable."""
    import json

    store.record(report_clear_drop)
    with sqlite3.connect(store.path) as conn:
        raw = conn.execute("SELECT config FROM runs").fetchone()[0]
    config = json.loads(raw)
    assert config["tolerance"] == report_clear_drop.config.tolerance
    assert config["alpha"] == report_clear_drop.config.alpha
    assert config["correction"] == report_clear_drop.config.correction


def test_every_metric_keeps_its_raw_and_adjusted_p_value(store, report_clear_drop):
    run_id = store.record(report_clear_drop)
    stored = {m["label"]: m for m in store.run(run_id)["metrics"]}
    for metric in report_clear_drop.metrics:
        row = stored[metric.label]
        assert row["p_raw"] == pytest.approx(metric.test.p_value)
        assert row["p_adjusted"] == pytest.approx(metric.adjusted.p_adjusted)
        assert bool(row["flagged"]) == metric.flagged
        assert row["baseline"] == pytest.approx(metric.baseline_mean)
        assert row["candidate"] == pytest.approx(metric.candidate_mean)


def test_metrics_come_back_most_significant_first(store, report_clear_drop):
    run_id = store.record(report_clear_drop)
    ps = [m["p_raw"] for m in store.run(run_id)["metrics"]]
    assert ps == sorted(ps)


def test_a_metric_without_an_interval_stores_null_not_zero(store, report_identical):
    """Zero would read as a point estimate with no uncertainty, which is a
    different and wrong claim from 'this test provides no interval'."""
    run_id = store.record(report_identical)
    # Keyed by label, because the rows come back ordered by p-value.
    by_label = {m["label"]: m for m in store.run(run_id)["metrics"]}
    for metric in report_identical.metrics:
        if metric.test.ci is None:
            assert by_label[metric.label]["ci_lo"] is None


# -- atomicity ------------------------------------------------------------


def test_a_failed_metric_insert_leaves_no_run_behind(
    store, report_clear_drop, monkeypatch
):
    """One transaction, not two statements. Breaking the metric insert must
    roll the run row back with it.

    The break goes in through a proxy connection rather than a patched
    method, because ``sqlite3.Connection`` attributes are read-only. The
    proxy forwards everything else, so the real BEGIN and ROLLBACK still
    reach the database -- which is the thing under test.
    """
    store.record(report_clear_drop)
    assert len(store.history()) == 1

    class _FailsOnMetrics:
        def __init__(self, conn):
            self._real = conn

        def __getattr__(self, name):
            return getattr(self._real, name)

        def executemany(self, *args, **kwargs):
            raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(store, "_conn", _FailsOnMetrics(store._conn))
    with pytest.raises(sqlite3.OperationalError):
        store.record(report_clear_drop, label="doomed")
    monkeypatch.undo()

    assert len(store.history()) == 1
    assert [r.label for r in store.history()] != ["doomed"]


def test_metrics_go_with_the_run_when_it_is_deleted(store, report_clear_drop):
    """Foreign keys are on and the cascade is declared, so a deleted run
    cannot leave orphaned metric rows behind to be joined against later."""
    run_id = store.record(report_clear_drop)
    with sqlite3.connect(store.path) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        remaining = conn.execute(
            "SELECT COUNT(*) FROM metrics WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
    assert remaining == 0


def test_concurrent_writers_do_not_lose_or_interleave_runs(store, report_identical):
    """The store is handed to a FastAPI threadpool, so a lock and WAL are not
    optional. Eight threads, eight distinct runs, every one with its full set
    of metrics."""
    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(
            pool.map(lambda i: store.record(report_identical, f"run-{i}"), range(8))
        )
    assert len(set(ids)) == 8
    assert len(store.history()) == 8
    for run_id in ids:
        assert len(store.run(run_id)["metrics"]) == len(report_identical.metrics)


def test_the_journal_is_in_wal_mode(store):
    with sqlite3.connect(store.path) as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


# -- reading it back ------------------------------------------------------


def test_history_is_newest_first_and_respects_the_limit(store, report_identical):
    for i in range(5):
        store.record(report_identical, f"run-{i}")
    newest_first = [r.label for r in store.history()]
    assert newest_first[0] == "run-4"
    assert newest_first[-1] == "run-0"
    assert len(store.history(limit=2)) == 2


def test_history_rows_carry_what_a_trend_line_needs(store, report_clear_drop):
    store.record(report_clear_drop, "release-7")
    row = store.history()[0]
    assert row.label == "release-7"
    assert row.verdict == "regression"
    assert row.ci[0] < row.effect < row.ci[1]
    assert row.n_cases == 600
    assert row.n_used == report_clear_drop.sequential.n_used
    assert row.stopped_early is True
    assert row.created_ms > 0


def test_one_metric_across_runs_comes_back_oldest_first(
    store, report_identical, report_clear_drop
):
    """A trend line reads left to right, so the query reverses the
    newest-first ordering it had to use for the LIMIT."""
    store.record(report_identical, "clean")
    store.record(report_clear_drop, "regressed")
    series = store.metric_history("rubric score")
    assert [row["label"] for row in series] == ["clean", "regressed"]
    assert series[0]["flagged"] is False
    assert series[1]["flagged"] is True
    assert series[1]["effect"] < series[0]["effect"]
    assert len(series[0]["ci"]) == 2


def test_metric_history_of_something_never_recorded_is_empty(store, report_identical):
    store.record(report_identical)
    assert store.metric_history("rubric score [nonexistent]") == []


def test_an_unknown_run_is_none_not_an_exception(store):
    assert store.run(9999) is None


def test_a_fresh_store_on_the_same_file_sees_the_earlier_runs(
    tmp_path, report_identical
):
    """Two processes, one file: the schema is created idempotently and the
    history survives a restart, which is the only reason to use a file at
    all."""
    path = tmp_path / "shared.db"
    first = Store(path)
    first.record(report_identical, "before-restart")
    first.close()

    second = Store(path)
    try:
        assert [r.label for r in second.history()] == ["before-restart"]
        second.record(report_identical, "after-restart")
        assert len(second.history()) == 2
    finally:
        second.close()
