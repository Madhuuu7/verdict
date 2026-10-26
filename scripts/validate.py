"""The simulation study behind every number in the README.

Run it and the README's claims are reproducible on your machine:

    python scripts/validate.py              # the full study, ~3 minutes
    python scripts/validate.py --quick      # a tenth of the trials, ~20s
    python scripts/validate.py --json out.json

Six studies, each answering a question the gate's design depends on:

1. **peeking** -- how badly does re-testing after every case inflate the
   error rate, and do the always-valid alternatives hold their size?
2. **power** -- what does that validity cost, in detections and in cases?
3. **pairing** -- how much does pairing on the case buy on realistic data?
4. **multiplicity** -- how often does an unchanged pipeline trip at least one
   of 14 comparisons, uncorrected and corrected?
5. **calibration** -- how close is the judge to the annotator ceiling, and
   how many labels does it take to *prove* it?
6. **presets** -- for each regression shape, what verdict does the gate reach
   and which comparison caught it?

Every figure printed here is a measurement under a known truth, which is the
only way to state an error rate at all: a real model cannot tell you whether
there was really no regression. That is why the pipelines are simulated, and
why the README says so.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verdict import (  # noqa: E402
    GateConfig,
    HumanLabeller,
    PRESETS,
    RubricJudge,
    SimulatedPipeline,
    Verdict,
    annotator_ceiling,
    calibrate,
    run_gate,
    synthetic_suite,
    variant_pair,
)
from verdict.judges import label_subset  # noqa: E402
from verdict.stats.multiplicity import benjamini_hochberg, holm  # noqa: E402
from verdict.stats.paired import paired_bootstrap, unpaired_bootstrap  # noqa: E402
from verdict.stats.sequential import (  # noqa: E402
    confidence_sequence,
    empirical_bernstein,
    fixed_horizon,
    peeking,
    simulate_error_rate,
)

ALPHA = 0.05


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


# -- 1. the cost of peeking -----------------------------------------------


def study_peeking(trials: int, n: int) -> dict[str, Any]:
    """Null experiments: nothing changed, so any definite verdict is wrong.

    The tolerance is 0, which puts the truth exactly on the margin and makes
    both PASS and REGRESSION mistakes. That is the strictest reading and the
    one that isolates the stopping rule from everything else.
    """
    rule(f"1. stopping rules under the null ({trials} trials, n={n}, noise 0.8)")
    shared = dict(trials=trials, n=n, effect=0.0, noise=0.8, seed=7, tolerance=0.0)
    procedures = {
        "fixed horizon (one look)": (fixed_horizon, {}),
        "peeking after every case": (peeking, {"min_n": 10}),
        "sub-Gaussian sequence (sigma=4)": (
            confidence_sequence, {"sigma": 4.0, "min_n": 10}
        ),
        "empirical-Bernstein sequence": (
            empirical_bernstein, {"bound": 4.0, "min_n": 10}
        ),
    }
    out = {}
    print(f"{'procedure':34s} {'error rate':>11s} {'mean n':>8s}")
    for label, (proc, kwargs) in procedures.items():
        stats = simulate_error_rate(proc, **shared, **kwargs)
        error = stats["flagged_regression"] + stats["passed"]
        out[label] = {"error_rate": error, "mean_n": stats["mean_n"]}
        print(f"{label:34s} {pct(error):>11s} {stats['mean_n']:8.0f}")
    print(f"\n  nominal alpha is {pct(ALPHA)}.")
    print("  Peeking is the procedure everybody writes: it looks identical in a")
    print("  dashboard and it is wrong, because the interval is valid for one")
    print("  look and is being used for n of them.")
    return out


# -- 2. what validity costs -----------------------------------------------


def study_power(trials: int, n: int) -> dict[str, Any]:
    rule(f"2. power at a real regression ({trials} trials, n={n}, effect -0.25)")
    shared = dict(trials=trials, n=n, effect=-0.25, noise=0.8, seed=11, tolerance=0.0)
    procedures = {
        "fixed horizon (one look)": (fixed_horizon, {}),
        "sub-Gaussian sequence (sigma=4)": (
            confidence_sequence, {"sigma": 4.0, "min_n": 10}
        ),
        "empirical-Bernstein sequence": (
            empirical_bernstein, {"bound": 4.0, "min_n": 10}
        ),
    }
    out = {}
    print(f"{'procedure':34s} {'detected':>9s} {'mean n':>8s} {'cases saved':>12s}")
    for label, (proc, kwargs) in procedures.items():
        stats = simulate_error_rate(proc, **shared, **kwargs)
        saved = 1.0 - stats["mean_n"] / n
        out[label] = {
            "power": stats["flagged_regression"],
            "mean_n": stats["mean_n"],
            "saved_fraction": saved,
        }
        print(
            f"{label:34s} {pct(stats['flagged_regression']):>9s} "
            f"{stats['mean_n']:8.0f} {pct(saved):>12s}"
        )
    print("\n  The sub-Gaussian row is the honest Hoeffding proxy for a bounded")
    print("  rubric difference, and it is correct and useless: it resolves")
    print("  almost nothing. That measurement is why the gate uses the")
    print("  variance-adaptive boundary instead.")
    return out


# -- 3. pairing -----------------------------------------------------------


def study_pairing(trials: int, n: int) -> dict[str, Any]:
    """Case difficulty is shared; per-case noise is not. An unpaired test has
    to treat the difficulty spread as noise."""
    rule(f"3. pairing vs not ({trials} trials, n={n}, effect -0.15)")
    rng = np.random.default_rng(13)
    paired_hits = unpaired_hits = 0
    paired_widths: list[float] = []
    unpaired_widths: list[float] = []
    for _ in range(trials):
        difficulty = rng.normal(0, 1.0, size=n)
        base = difficulty + rng.normal(0, 0.3, size=n)
        cand = difficulty - 0.15 + rng.normal(0, 0.3, size=n)
        p = paired_bootstrap(base, cand, draws=800)
        u = unpaired_bootstrap(base, cand, draws=800)
        paired_hits += p.p_value <= ALPHA
        unpaired_hits += u.p_value <= ALPHA
        paired_widths.append(p.ci[1] - p.ci[0])
        unpaired_widths.append(u.ci[1] - u.ci[0])

    out = {
        "paired_power": paired_hits / trials,
        "unpaired_power": unpaired_hits / trials,
        "paired_ci_width": float(np.mean(paired_widths)),
        "unpaired_ci_width": float(np.mean(unpaired_widths)),
    }
    print(f"  paired bootstrap   detected {pct(out['paired_power'])} "
          f"| mean CI width {out['paired_ci_width']:.3f}")
    print(f"  unpaired bootstrap detected {pct(out['unpaired_power'])} "
          f"| mean CI width {out['unpaired_ci_width']:.3f}")
    print(f"\n  Same data, same estimate, "
          f"{out['unpaired_ci_width'] / out['paired_ci_width']:.1f}x the interval.")
    return out


# -- 4. multiplicity ------------------------------------------------------


def study_multiplicity(trials: int, comparisons: int) -> dict[str, Any]:
    """Under the null, p-values are uniform, so this is measurable directly.

    The gate spends alpha/2 on the end-of-suite family -- the other half pays
    for the right to stop early -- so both levels are reported.
    """
    rule(f"4. {comparisons} comparisons on an unchanged pipeline ({trials} families)")
    rng = np.random.default_rng(17)
    naive_at_full = naive_at_half = bh_any = holm_any = 0
    for _ in range(trials):
        ps = rng.uniform(0, 1, size=comparisons)
        family = [(f"m{i}", float(p)) for i, p in enumerate(ps)]
        naive_at_full += bool(np.any(ps <= ALPHA))
        naive_at_half += bool(np.any(ps <= ALPHA / 2))
        bh_any += bool(benjamini_hochberg(family, ALPHA / 2).rejected)
        holm_any += bool(holm(family, ALPHA / 2).rejected)

    out = {
        "comparisons": comparisons,
        "uncorrected_alpha": naive_at_full / trials,
        "uncorrected_half_alpha": naive_at_half / trials,
        "benjamini_hochberg": bh_any / trials,
        "holm": holm_any / trials,
        "formula_alpha": 1.0 - (1.0 - ALPHA) ** comparisons,
    }
    print(f"  uncorrected at alpha={ALPHA}       {pct(out['uncorrected_alpha'])}"
          f"   (formula: {pct(out['formula_alpha'])})")
    print(f"  uncorrected at alpha/2         {pct(out['uncorrected_half_alpha'])}")
    print(f"  Benjamini-Hochberg at alpha/2  {pct(out['benjamini_hochberg'])}")
    print(f"  Holm at alpha/2                {pct(out['holm'])}")
    print("\n  A gate that cries wolf on a third of clean runs is a gate the team")
    print("  turns off, and then the absence of an alarm means nothing either.")
    return out


# -- 5. calibration -------------------------------------------------------


def study_calibration(sizes: tuple[int, ...]) -> dict[str, Any]:
    """How close the judge is to the ceiling, and when that becomes provable.

    The judge is compared against annotator A; the ceiling is A against B on
    the same cases. Both are measured on the same labelled subset, as the
    gate does it.
    """
    rule("5. judge calibration and the label budget")
    suite = list(synthetic_suite(max(sizes) * 2))
    pipeline = SimulatedPipeline(name="baseline")
    outputs = [pipeline(c) for c in suite]
    judge = RubricJudge()
    judge_grades = {c.case_id: judge(c, o).grade for c, o in zip(suite, outputs)}

    def annotator(name: str) -> dict[str, int]:
        labeller = HumanLabeller(name=name)
        return {c.case_id: labeller(c, o).grade for c, o in zip(suite, outputs)}

    a, b = annotator("annotator-a"), annotator("annotator-b")

    rows = []
    print(f"{'labels':>7s} {'ceiling':>8s} {'judge':>7s} {'lower':>7s} "
          f"{'needs':>7s} {'of ceiling':>11s} {'certified':>10s}")
    for size in sizes:
        ids = [c.case_id for c in suite[:size]]
        ceiling = annotator_ceiling(
            {k: a[k] for k in ids}, {k: b[k] for k in ids}
        ).weighted_kappa
        cal = calibrate(
            {k: a[k] for k in ids},
            {k: judge_grades[k] for k in ids},
            ceiling=ceiling,
        )
        rows.append(
            {
                "labels": size,
                "ceiling": ceiling,
                "weighted_kappa": cal.weighted_kappa,
                "lower_bound": cal.weighted_kappa_ci[0],
                "required": cal.required,
                "relative": cal.relative,
                "certified": cal.trustworthy,
                "bias": cal.bias,
            }
        )
        print(
            f"{size:7d} {ceiling:8.3f} {cal.weighted_kappa:7.3f} "
            f"{cal.weighted_kappa_ci[0]:7.3f} {cal.required:7.3f} "
            f"{cal.relative:11.0%} {str(cal.trustworthy):>10s}"
        )

    certified = [r["labels"] for r in rows if r["certified"]]
    constant = calibrate({k: a[k] for k in list(a)[: sizes[-1]]},
                         {k: 4 for k in list(a)[: sizes[-1]]}).kappa
    print(f"\n  a judge that answers 4 to everything scores kappa {constant:.3f}")
    print(f"  certified from {certified[0] if certified else 'never'} labels on")
    print("  The judge holds ~0.85 of the ceiling at every size. What takes")
    print("  hundreds of labels is not the agreement -- it is proving it, since")
    print("  the bootstrap interval has to be narrower than the margin between")
    print("  the judge and its bar.")
    return {"curve": rows, "constant_judge_kappa": constant,
            "certified_from": certified[0] if certified else None}


# -- 6. the gate per regression shape ------------------------------------


def study_presets(n_cases: int) -> dict[str, Any]:
    rule(f"6. the gate on each regression shape ({n_cases} cases)")
    suite = synthetic_suite(n_cases)
    judge = RubricJudge()
    annotators = (HumanLabeller(name="annotator-a"), HumanLabeller(name="annotator-b"))
    out = {}
    print(f"{'preset':26s} {'verdict':13s} {'n used':>7s} {'saved':>7s}  caught by")
    for name, (shape, severity) in PRESETS.items():
        base, cand = variant_pair(shape, severity=severity)
        report = run_gate(suite, base, cand, judge, annotators, GateConfig())
        caught = [m.label for m in report.flagged]
        out[name] = {
            "shape": shape,
            "severity": severity,
            "verdict": report.verdict.value,
            "n_used": report.sequential.n_used,
            "saved_fraction": report.sequential.saved_fraction,
            "flagged": caught,
            "flagged_uncorrected": [m.label for m in report.flagged_uncorrected],
            "comparisons": report.comparisons,
            "effect": report.sequential_fixed.effect,
            "resolvable_margin": report.resolvable_margin,
        }
        print(
            f"{name:26s} {report.verdict.value:13s} {report.sequential.n_used:7d} "
            f"{pct(report.sequential.saved_fraction):>7s}  "
            f"{', '.join(caught) if caught else '(nothing)'}"
        )
    print("\n  Each shape is caught by a different comparison, which is the")
    print("  argument for the metric set and for slicing: drop the slices and")
    print("  the broken-slice row goes unflagged.")
    return out


# -- main -----------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="a tenth of the trials")
    parser.add_argument("--json", type=Path, default=Path("reports/validation.json"))
    parser.add_argument("--skip", nargs="*", default=[], help="study numbers to skip")
    args = parser.parse_args()

    scale = 10 if args.quick else 1
    started = time.perf_counter()
    print(__doc__.split("\n\n")[0])
    print(f"\nalpha = {ALPHA}, {'quick' if args.quick else 'full'} run")

    results: dict[str, Any] = {"alpha": ALPHA, "quick": args.quick}
    skip = set(args.skip)

    if "1" not in skip:
        results["peeking"] = study_peeking(1500 // scale, 600)
    if "2" not in skip:
        results["power"] = study_power(1000 // scale, 600)
    if "3" not in skip:
        results["pairing"] = study_pairing(300 // scale, 120)
    if "4" not in skip:
        results["multiplicity"] = study_multiplicity(4000 // scale, 14)
    if "5" not in skip:
        sizes = (80, 160, 280, 400, 600) if not args.quick else (80, 280, 600)
        results["calibration"] = study_calibration(sizes)
    if "6" not in skip:
        results["presets"] = study_presets(600)

    results["elapsed_s"] = time.perf_counter() - started
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {args.json} in {results['elapsed_s']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
