"""The HTTP surface: run the gate, read the history, serve the console.

Three decisions worth stating, because each one is a trade.

**The gate runs synchronously, in the threadpool.** A run on 600 cases takes
about a second: two pipeline passes, 1,200 judgements, and several thousand
bootstrap resamples, all CPU. Declaring the handler ``def`` rather than
``async def`` hands it to Starlette's threadpool instead of blocking the event
loop, which is the whole fix needed at this scale. A job queue would be the
right answer for a real suite against a real model, where a run is minutes of
rate-limited API calls, and that is noted in the README's limits rather than
pretended at here.

**The step curves are downsampled server-side.** A 600-case run produces four
sequences of up to 600 steps each. Sending 2,400 points so a chart can draw
400 pixels is wasteful, so each curve is thinned to at most ``MAX_STEPS``
points with the decision point always kept -- the one point whose exact
position is the claim being made.

**Every response carries the config it was produced under.** A verdict with
no margin and no alpha next to it cannot be read later, and the history is the
reason this is a service rather than a script.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__
from ..calibration import Calibration
from ..cases import load_jsonl, synthetic_suite
from ..gate import GateConfig, GateReport, MetricReport, Verdict, run_gate
from ..judges import HumanLabeller, RubricJudge
from ..pipelines import PRESETS, variant_pair
from ..stats.multiplicity import Correction
from ..stats.sequential import SequentialResult
from ..store import Store

ROOT = Path(__file__).resolve().parents[3]
WEB = ROOT / "web"
VALIDATION = ROOT / "reports" / "validation.json"

#: Points per sequence curve after thinning. 240 is about one per pixel on a
#: console-width chart; more is bytes nobody can see.
MAX_STEPS = 240

#: Below this many human labels the judge cannot be certified and the gate
#: abstains on every input. Reported to the console so it can say why, rather
#: than leaving a user to conclude the gate is broken.
LABELS_TO_CERTIFY = 280


# -- requests -------------------------------------------------------------


class GateRequest(BaseModel):
    """What to run. Every field has the gate's own default behind it."""

    preset: str = Field(
        default="clear quality drop", description="a key of PRESETS"
    )
    n_cases: int = Field(default=600, ge=20, le=4000)
    suite: str | None = Field(
        default=None, description="path to a JSONL suite; overrides n_cases"
    )
    tolerance: float = Field(default=0.05, ge=0.0, le=2.0)
    alpha: float = Field(default=0.05, gt=0.0, lt=0.5)
    label_fraction: float = Field(default=0.5, gt=0.0, le=1.0)
    kappa_threshold: float = Field(default=0.75, gt=0.0, le=1.0)
    correction: Literal["bh", "holm"] = "bh"
    slice_by_tag: bool = True
    min_slice: int = Field(default=25, ge=2)
    min_n: int = Field(default=20, ge=2)
    judge_noise: float = Field(
        default=0.35, ge=0.0, le=3.0, description="the judge's own precision"
    )
    judge_bias: float = Field(
        default=0.18, ge=-2.0, le=2.0, description="positive means generous"
    )
    annotator_noise: float = Field(default=0.42, ge=0.0, le=3.0)
    record: bool = Field(default=True, description="store the run in the history")
    label: str = ""


# -- serialisation --------------------------------------------------------


def thin(steps: list[Any], decision_at: int) -> list[dict[str, float | str]]:
    """Downsample a step curve, always keeping the first, last and decision
    points -- the decision point being the only one whose exact position is
    a claim.

    The decision step is found by its ``n``, not by subtracting one from
    ``n_used``: ``peeking`` records nothing for n = 1 (a single observation
    has no sample standard deviation), so a curve's index and its case number
    are not the same thing. Indexing by position put the last step out of
    range on exactly that procedure.
    """
    if not steps:
        return []
    last = len(steps) - 1
    keep = {0, last}
    keep.update(i for i, step in enumerate(steps) if step.n == decision_at)
    # Round the stride up, so MAX_STEPS is a cap rather than roughly it:
    # 600 steps at a floored stride of 2 would send 300 points.
    stride = max(1, -(-len(steps) // MAX_STEPS))
    keep.update(range(0, len(steps), stride))
    return [
        {
            "n": steps[i].n,
            "mean": steps[i].mean,
            "lower": _finite(steps[i].lower),
            "upper": _finite(steps[i].upper),
            "decision": steps[i].decision.value,
        }
        for i in sorted(keep)
        if 0 <= i <= last
    ]


def _finite(x: float) -> float | None:
    """JSON has no infinity. The first few steps of a confidence sequence
    genuinely are unbounded, and ``null`` says that; 0 would be a lie and
    ``Infinity`` is invalid JSON that some parsers accept and others reject.
    """
    return None if x in (float("inf"), float("-inf")) or x != x else x


def sequence_json(result: SequentialResult, *, steps: bool = True) -> dict[str, Any]:
    return {
        "decision": result.decision.value,
        "n_used": result.n_used,
        "n_available": result.n_available,
        "effect": result.effect,
        "ci": [_finite(result.ci[0]), _finite(result.ci[1])],
        "tolerance": result.tolerance,
        "alpha": result.alpha,
        "stopped_early": result.stopped_early,
        "saved": result.saved,
        "saved_fraction": result.saved_fraction,
        "steps": thin(result.steps, result.n_used) if steps else [],
    }


def calibration_json(cal: Calibration) -> dict[str, Any]:
    return {
        "n": cal.n,
        "kappa": cal.kappa,
        "kappa_ci": list(cal.kappa_ci),
        "weighted_kappa": cal.weighted_kappa,
        "weighted_kappa_ci": [_finite(x) for x in cal.weighted_kappa_ci],
        "exact_agreement": cal.exact_agreement,
        "within_one": cal.within_one,
        "bias": cal.bias,
        "bias_ci": list(cal.bias_ci),
        "confusion": cal.confusion,
        "ceiling": cal.ceiling,
        "relative": cal.relative,
        "required": cal.required,
        "threshold": cal.threshold,
        "floor": cal.floor,
        "trustworthy": cal.trustworthy,
        "verdict": cal.verdict(),
        "note": cal.note,
    }


def metric_json(metric: MetricReport) -> dict[str, Any]:
    return {
        "label": metric.label,
        "kind": metric.kind,
        "higher_is_better": metric.higher_is_better,
        "baseline": metric.baseline_mean,
        "candidate": metric.candidate_mean,
        "delta": metric.delta,
        "worse": metric.worse,
        "n": metric.n,
        "test": {
            "name": metric.test.name,
            "statistic": metric.test.statistic,
            "p_value": metric.test.p_value,
            "effect": metric.test.effect,
            "ci": list(metric.test.ci) if metric.test.ci else None,
            "informative": metric.test.informative,
            "detail": metric.test.detail,
        },
        "p_adjusted": metric.adjusted.p_adjusted if metric.adjusted else None,
        "threshold": metric.adjusted.threshold if metric.adjusted else None,
        "rejected": metric.adjusted.rejected if metric.adjusted else False,
        "flagged": metric.flagged,
        "flagged_uncorrected": metric.flagged_uncorrected,
    }


def correction_json(correction: Correction) -> dict[str, Any]:
    return {
        "method": correction.method,
        "alpha": correction.alpha,
        "rejected": [a.label for a in correction.rejected],
        "naive_rejected": [a.label for a in correction.naive_rejected],
        "family_error_if_uncorrected": correction.family_error_if_uncorrected(),
    }


def report_json(report: GateReport, run_id: int | None = None) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "verdict": report.verdict.value,
        "reason": report.reason,
        "summary": report.summary(),
        "suite": report.suite,
        "n_cases": report.n_cases,
        "judge": report.judge,
        "primary": report.primary,
        "comparisons": report.comparisons,
        "flagged": [m.label for m in report.flagged],
        "flagged_uncorrected": [m.label for m in report.flagged_uncorrected],
        "resolvable_margin": report.resolvable_margin,
        "elapsed_s": report.elapsed_s,
        "calibration": calibration_json(report.calibration),
        "ceiling": calibration_json(report.ceiling),
        "constant_judge_kappa": report.constant_judge_kappa,
        "metrics": [metric_json(m) for m in report.metrics],
        "correction": correction_json(report.correction),
        "sequential": sequence_json(report.sequential),
        "sequential_peeking": sequence_json(report.sequential_peeking),
        "sequential_fixed": sequence_json(report.sequential_fixed, steps=False),
        "sequential_subgaussian": (
            sequence_json(report.sequential_subgaussian)
            if report.sequential_subgaussian
            else None
        ),
        "config": {
            "alpha": report.config.alpha,
            "tolerance": report.config.tolerance,
            "label_fraction": report.config.label_fraction,
            "kappa_threshold": report.config.kappa_threshold,
            "correction": report.config.correction,
            "slice_by_tag": report.config.slice_by_tag,
            "min_slice": report.config.min_slice,
            "min_n": report.config.min_n,
        },
        "labels_to_certify": LABELS_TO_CERTIFY,
    }


# -- the app --------------------------------------------------------------


def create_app(db_path: str | Path = "verdict.db") -> FastAPI:
    app = FastAPI(
        title="verdict",
        version=__version__,
        summary="A regression gate for LLM pipelines: a calibrated judge, "
        "paired tests, FDR control and valid early stopping.",
    )
    store = Store(db_path)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "runs": len(store.history(limit=1000)),
            "labels_to_certify": LABELS_TO_CERTIFY,
        }

    @app.get("/api/presets")
    def presets() -> list[dict[str, Any]]:
        """The regression shapes the console offers, each a different way for
        a pipeline to get worse."""
        described = {
            "identical": "nothing changed; the false-alarm case",
            "small quality drop": "every answer slightly worse",
            "clear quality drop": "every answer clearly worse",
            "one slice broken": "one tag much worse, the rest untouched",
            "more refusals": "the model starts declining things it used to answer",
            "slower, same quality": "identical answers, much slower",
            "worse on hard cases only": "the median holds, the tail falls apart",
        }
        return [
            {
                "name": name,
                "shape": shape,
                "severity": severity,
                "description": described.get(name, ""),
            }
            for name, (shape, severity) in PRESETS.items()
        ]

    @app.post("/api/gate")
    def gate(request: GateRequest) -> dict[str, Any]:
        """Run the whole gate and return a report that explains itself.

        Synchronous on purpose: see the module docstring.
        """
        if request.preset not in PRESETS:
            raise HTTPException(
                status_code=422,
                detail=f"unknown preset {request.preset!r}; "
                f"try one of {sorted(PRESETS)}",
            )
        shape, severity = PRESETS[request.preset]

        if request.suite:
            path = Path(request.suite)
            if not path.is_file():
                raise HTTPException(status_code=422, detail=f"no suite at {path}")
            try:
                suite = load_jsonl(path)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        else:
            suite = synthetic_suite(request.n_cases)

        baseline, candidate = variant_pair(shape, severity=severity)
        judge = RubricJudge(noise=request.judge_noise, bias=request.judge_bias)
        annotators = (
            HumanLabeller(name="annotator-a", noise=request.annotator_noise),
            HumanLabeller(name="annotator-b", noise=request.annotator_noise),
        )
        config = GateConfig(
            alpha=request.alpha,
            tolerance=request.tolerance,
            label_fraction=request.label_fraction,
            kappa_threshold=request.kappa_threshold,
            correction=request.correction,
            slice_by_tag=request.slice_by_tag,
            min_slice=request.min_slice,
            min_n=request.min_n,
        )
        report = run_gate(suite, baseline, candidate, judge, annotators, config)

        run_id = None
        if request.record:
            run_id = store.record(report, label=request.label or request.preset)
        return report_json(report, run_id)

    @app.get("/api/history")
    def history(limit: int = Query(default=50, ge=1, le=500)) -> list[dict[str, Any]]:
        return [
            {
                "run_id": row.run_id,
                "created_ms": row.created_ms,
                "label": row.label,
                "suite": row.suite,
                "verdict": row.verdict,
                "effect": row.effect,
                "ci": list(row.ci),
                "kappa": row.kappa,
                "judge": row.judge,
                "flagged": row.flagged,
                "n_cases": row.n_cases,
                "n_used": row.n_used,
                "stopped_early": row.stopped_early,
            }
            for row in store.history(limit=limit)
        ]

    @app.get("/api/runs/{run_id}")
    def run(run_id: int) -> dict[str, Any]:
        row = store.run(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"no run {run_id}")
        return row

    @app.get("/api/metrics/{label}/history")
    def metric_history(
        label: str, limit: int = Query(default=50, ge=1, le=500)
    ) -> list[dict[str, Any]]:
        """One metric across runs, oldest first: the drift question, which is
        the one a single run cannot answer."""
        return store.metric_history(label, limit=limit)

    @app.get("/api/validation")
    def validation() -> JSONResponse:
        """The simulation study's output, if it has been run.

        The console shows these next to a live run so the error rates it
        claims are the measured ones rather than the documented ones.
        """
        if not VALIDATION.is_file():
            raise HTTPException(
                status_code=404,
                detail="no validation report yet; run scripts/validate.py",
            )
        return JSONResponse(json.loads(VALIDATION.read_text(encoding="utf-8")))

    @app.get("/")
    def console() -> FileResponse:
        index = WEB / "index.html"
        if not index.is_file():
            raise HTTPException(status_code=404, detail="console not built")
        return FileResponse(index)

    if WEB.is_dir():
        app.mount("/web", StaticFiles(directory=WEB), name="web")

    app.state.store = store
    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
