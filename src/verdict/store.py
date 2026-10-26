"""Gate history, in SQLite.

A single gate run answers "did this change regress anything". The question a
team actually has is "is this metric drifting", and that one needs the runs
kept. Two columns make the difference between a log and something useful:

``effect`` and the interval around it
    A sequence of point estimates is unreadable -- every release looks like a
    change. Keeping the interval means the history can show whether successive
    releases moved anything beyond their own uncertainty.
``judge`` and ``kappa``
    So that a step change in the metrics can be checked against a step change
    in the *instrument*. The most confusing failure in eval history is a
    judge upgrade that shifts every score at once, and without the judge's
    identity and agreement stored next to the numbers it looks like the
    pipeline regressed.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .gate import GateReport

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    created_ms  INTEGER NOT NULL,
    suite       TEXT NOT NULL,
    label       TEXT NOT NULL,
    verdict     TEXT NOT NULL,
    reason      TEXT NOT NULL,
    n_cases     INTEGER NOT NULL,
    comparisons INTEGER NOT NULL,
    flagged     INTEGER NOT NULL,
    judge       TEXT NOT NULL,
    kappa       REAL NOT NULL,
    kappa_lo    REAL NOT NULL,
    ceiling     REAL NOT NULL,
    effect      REAL NOT NULL,
    ci_lo       REAL NOT NULL,
    ci_hi       REAL NOT NULL,
    n_used      INTEGER NOT NULL,
    stopped_early INTEGER NOT NULL,
    elapsed_s   REAL NOT NULL,
    config      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS metrics (
    run_id      INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    label       TEXT NOT NULL,
    kind        TEXT NOT NULL,
    baseline    REAL NOT NULL,
    candidate   REAL NOT NULL,
    effect      REAL NOT NULL,
    p_raw       REAL NOT NULL,
    p_adjusted  REAL NOT NULL,
    flagged     INTEGER NOT NULL,
    ci_lo       REAL,
    ci_hi       REAL,
    PRIMARY KEY (run_id, label)
);

CREATE INDEX IF NOT EXISTS runs_recent ON runs (created_ms DESC);
CREATE INDEX IF NOT EXISTS metrics_label ON metrics (label, run_id);
"""


@dataclass(frozen=True, slots=True)
class HistoryRow:
    run_id: int
    created_ms: int
    label: str
    suite: str
    verdict: str
    effect: float
    ci: tuple[float, float]
    kappa: float
    #: The judge that produced the grades, so a step change in the metrics can
    #: be checked against a change in the instrument.
    judge: str
    flagged: int
    n_cases: int
    n_used: int
    stopped_early: bool


class Store:
    def __init__(self, path: str | Path = "verdict.db"):
        self.path = str(path)
        self._conn = sqlite3.connect(
            self.path, isolation_level=None, timeout=30.0, check_same_thread=False
        )
        self._lock = threading.RLock()
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)

    def record(self, report: GateReport, label: str = "") -> int:
        """Store a run and its metrics in one transaction.

        One transaction because a run whose metrics are missing is worse than
        no run: the history would show a verdict with nothing behind it.
        """
        cal = report.calibration
        seq = report.sequential
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                cur = self._conn.execute(
                    "INSERT INTO runs (created_ms, suite, label, verdict, reason,"
                    " n_cases, comparisons, flagged, judge, kappa, kappa_lo, ceiling,"
                    " effect, ci_lo, ci_hi, n_used, stopped_early, elapsed_s, config)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        int(time.time() * 1000),
                        report.suite,
                        label or report.verdict.value,
                        report.verdict.value,
                        report.reason,
                        report.n_cases,
                        report.comparisons,
                        len(report.flagged),
                        report.judge,
                        cal.weighted_kappa,
                        cal.weighted_kappa_ci[0],
                        cal.ceiling or 0.0,
                        report.sequential_fixed.effect,
                        report.sequential_fixed.ci[0],
                        report.sequential_fixed.ci[1],
                        seq.n_used,
                        int(seq.stopped_early),
                        report.elapsed_s,
                        json.dumps(report.config.__dict__, default=str, sort_keys=True),
                    ),
                )
                run_id = int(cur.lastrowid)
                self._conn.executemany(
                    "INSERT INTO metrics (run_id, label, kind, baseline, candidate,"
                    " effect, p_raw, p_adjusted, flagged, ci_lo, ci_hi)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    [
                        (
                            run_id,
                            m.label,
                            m.kind,
                            m.baseline_mean,
                            m.candidate_mean,
                            m.test.effect,
                            m.test.p_value,
                            m.adjusted.p_adjusted if m.adjusted else m.test.p_value,
                            int(m.flagged),
                            m.test.ci[0] if m.test.ci else None,
                            m.test.ci[1] if m.test.ci else None,
                        )
                        for m in report.metrics
                    ],
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        return run_id

    def history(self, limit: int = 50) -> list[HistoryRow]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT run_id, created_ms, label, suite, verdict, effect, ci_lo,"
                " ci_hi, kappa, flagged, n_cases, n_used, stopped_early, judge"
                " FROM runs ORDER BY created_ms DESC, run_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            HistoryRow(
                run_id=r[0],
                created_ms=r[1],
                label=r[2],
                suite=r[3],
                verdict=r[4],
                effect=r[5],
                ci=(r[6], r[7]),
                kappa=r[8],
                flagged=r[9],
                n_cases=r[10],
                n_used=r[11],
                stopped_early=bool(r[12]),
                judge=r[13],
            )
            for r in rows
        ]

    def metric_history(self, label: str, limit: int = 50) -> list[dict[str, Any]]:
        """One metric across runs, oldest first, for a trend line."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT r.run_id, r.created_ms, r.label, m.effect, m.ci_lo, m.ci_hi,"
                " m.p_adjusted, m.flagged"
                " FROM metrics m JOIN runs r ON r.run_id = m.run_id"
                " WHERE m.label = ? ORDER BY r.created_ms DESC LIMIT ?",
                (label, limit),
            ).fetchall()
        return [
            {
                "run_id": r[0],
                "created_ms": r[1],
                "label": r[2],
                "effect": r[3],
                "ci": [r[4], r[5]],
                "p_adjusted": r[6],
                "flagged": bool(r[7]),
            }
            for r in reversed(rows)
        ]

    def run(self, run_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT run_id, created_ms, suite, label, verdict, reason, n_cases,"
                " comparisons, flagged, judge, kappa, kappa_lo, ceiling, effect,"
                " ci_lo, ci_hi, n_used, stopped_early, elapsed_s"
                " FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            metrics = self._conn.execute(
                "SELECT label, kind, baseline, candidate, effect, p_raw, p_adjusted,"
                " flagged, ci_lo, ci_hi FROM metrics WHERE run_id = ? ORDER BY p_raw",
                (run_id,),
            ).fetchall()
        keys = (
            "run_id created_ms suite label verdict reason n_cases comparisons flagged"
            " judge kappa kappa_lo ceiling effect ci_lo ci_hi n_used stopped_early"
            " elapsed_s"
        ).split()
        out = dict(zip(keys, row))
        out["metrics"] = [
            dict(
                zip(
                    "label kind baseline candidate effect p_raw p_adjusted flagged"
                    " ci_lo ci_hi".split(),
                    m,
                )
            )
            for m in metrics
        ]
        return out

    def close(self) -> None:
        with self._lock:
            self._conn.close()
