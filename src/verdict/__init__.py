"""verdict -- a regression gate for LLM pipelines.

Four things a naive eval gate gets wrong, and what this does instead:

================================  ==========================================
it trusts the judge               calibrates it against human labels first,
                                  relative to the inter-annotator ceiling,
                                  and abstains when the judge is not good
                                  enough to rule
it compares two means             pairs on the case, so shared case
                                  difficulty stops being noise
it tests fifteen things at 0.05   controls the false discovery rate across
                                  the whole family
it peeks at the p-value           uses a confidence sequence, valid at every
                                  sample size, so stopping early is allowed
================================  ==========================================

The error rates of all four are measured, not asserted; see
``scripts/validate.py`` and the README.
"""

from .calibration import Calibration, annotator_ceiling, calibrate
from .cases import Case, Suite, load_jsonl, synthetic_suite
from .gate import GateConfig, GateReport, MetricReport, Run, Verdict, execute, run_gate
from .judges import HumanLabeller, Judge, Judgement, LlmJudge, RubricJudge
from .pipelines import (
    PRESETS,
    HttpPipeline,
    Output,
    Pipeline,
    SimulatedPipeline,
    variant_pair,
)

__version__ = "0.1.0"

__all__ = [
    "PRESETS",
    "Calibration",
    "Case",
    "GateConfig",
    "GateReport",
    "HttpPipeline",
    "HumanLabeller",
    "Judge",
    "Judgement",
    "LlmJudge",
    "MetricReport",
    "Output",
    "Pipeline",
    "Run",
    "RubricJudge",
    "SimulatedPipeline",
    "Suite",
    "Verdict",
    "annotator_ceiling",
    "calibrate",
    "execute",
    "load_jsonl",
    "run_gate",
    "synthetic_suite",
    "variant_pair",
]
