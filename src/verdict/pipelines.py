"""The thing under test, and the simulated versions of it.

A pipeline is anything that turns a ``Case`` into an ``Output``. For real use
that is ``HttpPipeline`` pointed at a model endpoint. For everything else in
this project it is ``SimulatedPipeline``, and the reason is not convenience:

**the gate's error rates can only be measured against a known truth.** To say
"this gate flags a regression that is not there 1.5% of the time", you need
runs where you know there is no regression. A real model cannot give you that.
A simulated pipeline with a dialled-in quality offset can, and the statistical
machinery being measured is identical either way -- it only ever sees numbers.

Each simulated output carries a hidden ``quality`` in [0, 1], built from the
case's difficulty and the pipeline's own offset, plus per-case noise. The
judge and the human labeller each read that latent quality through their own
noise, which is what makes the calibration numbers (kappa, bias) real
measurements of a real disagreement rather than constants typed into a test.

Regressions are injectable and each one is a different *shape* of regression,
because a gate that only catches uniform quality drops is not much of a gate:

``quality``   everything gets slightly worse
``tag``       one slice gets much worse, the rest is unchanged
``refusal``   the model starts declining things it used to answer
``latency``   quality is identical and the thing got slower
``tail``      the median is unchanged and the worst cases get much worse
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from .cases import Case

Regression = Literal["none", "quality", "tag", "refusal", "latency", "tail"]


@dataclass(frozen=True, slots=True)
class Output:
    text: str
    latency_ms: float = 0.0
    refused: bool = False
    #: Latent quality in [0, 1]. Present for simulated pipelines, absent for
    #: real ones -- the gate never reads it, only the judge and the simulated
    #: human labeller do.
    quality: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _unit(seed: int, salt: int = 0) -> float:
    """A deterministic pseudo-uniform in [0, 1) from an integer seed."""
    x = (seed ^ (salt * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 31
    return (x >> 11) / float(1 << 53)


def _normal(seed: int, salt: int = 0) -> float:
    """Box-Muller from two deterministic uniforms, clamped away from zero."""
    u1 = max(_unit(seed, salt * 2 + 1), 1e-12)
    u2 = _unit(seed, salt * 2 + 2)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


@dataclass
class SimulatedPipeline:
    """A reproducible stand-in with a known, dialled-in quality.

    ``base_quality`` is the quality of an average-difficulty case; harder
    cases come out worse. ``noise`` is the per-case variation that makes two
    pipelines of equal quality still disagree on individual cases, which is
    what the paired tests are there to see through.
    """

    name: str
    base_quality: float = 0.72
    noise: float = 0.11
    refusal_rate: float = 0.04
    latency_ms: float = 820.0
    latency_spread: float = 160.0
    regression: Regression = "none"
    #: Strength of the injected regression, in latent-quality units.
    severity: float = 0.0
    #: Which slice ``regression="tag"`` damages.
    target_tag: str = "math"

    def __call__(self, case: Case) -> Output:
        seed = case.seed(self.name)
        quality = self.base_quality - 0.35 * (case.difficulty - 0.5)
        quality += self.noise * _normal(seed, 1)

        refusal_rate = self.refusal_rate
        latency = self.latency_ms + self.latency_spread * _normal(seed, 2)

        if self.regression == "quality":
            quality -= self.severity
        elif self.regression == "tag" and self.target_tag in case.tags:
            # Concentrated damage: large on one slice, invisible on the mean
            # of the whole suite if the slice is small. This is the regression
            # that slicing exists to catch.
            quality -= self.severity * 3.0
        elif self.regression == "refusal":
            refusal_rate = min(1.0, self.refusal_rate + self.severity)
        elif self.regression == "latency":
            latency *= 1.0 + 4.0 * self.severity
        elif self.regression == "tail":
            # Only the already-hard cases degrade, so the median moves barely
            # at all. A gate watching only the mean will miss this.
            if case.difficulty > 0.7:
                quality -= self.severity * 4.0

        # Salt 97, not 3. _normal(seed, 1) consumes salts 3 and 4 for its two
        # uniforms, so drawing the refusal at salt 3 made a refusal happen
        # exactly when the quality noise was in its lower tail -- a hidden
        # correlation between two things the simulation treats as independent,
        # and the sort of bug that makes simulated results subtly wrong rather
        # than obviously broken.
        refused = _unit(seed, 97) < refusal_rate
        quality = min(1.0, max(0.0, quality))
        if refused:
            quality = 0.0

        text = (
            f"[{self.name}] declined: outside what I can advise on."
            if refused
            else f"[{self.name}] answer for {case.case_id} (q={quality:.3f})"
        )
        return Output(
            text=text,
            latency_ms=max(1.0, latency),
            refused=refused,
            quality=quality,
            meta={"pipeline": self.name},
        )


class HttpPipeline:
    """A real pipeline: an OpenAI-shaped chat endpoint.

    Works against OpenAI, Groq, OpenRouter, Together or a local llama.cpp
    server. Never run against a paid endpoint here -- there is no key -- so it
    is listed under the README's known limits rather than claimed as tested.
    """

    def __init__(
        self,
        name: str,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        system: str = "You are a careful assistant. Answer concisely.",
        timeout: float = 60.0,
    ):
        self.name = name
        self.model = model or os.environ.get("VERDICT_MODEL", "gpt-4o-mini")
        self.base_url = (
            base_url or os.environ.get("VERDICT_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("VERDICT_API_KEY", "")
        self.system = system
        self.timeout = timeout

    def __call__(self, case: Case) -> Output:
        import time

        import httpx

        if not self.api_key:
            raise RuntimeError("VERDICT_API_KEY is not set")
        started = time.perf_counter()
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system},
                    {"role": "user", "content": case.prompt},
                ],
                "temperature": 0,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        return Output(
            text=text,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            # A refusal heuristic, and a shallow one: good enough to make the
            # refusal metric mean something, not good enough to be a classifier.
            refused=any(
                p in text.lower()
                for p in ("i can't", "i cannot", "i'm unable", "i am unable")
            ),
            meta={"model": self.model},
        )


Pipeline = Callable[[Case], Output]


def variant_pair(
    regression: Regression = "none", severity: float = 0.06
) -> tuple[SimulatedPipeline, SimulatedPipeline]:
    """A baseline and a candidate differing only in the injected regression.

    Same base quality, same noise, same seeds: the only difference between
    the two is the regression under test, which is what makes a measured
    false-alarm rate meaningful when ``regression="none"``.
    """
    baseline = SimulatedPipeline(name="baseline")
    candidate = SimulatedPipeline(
        name="candidate", regression=regression, severity=severity
    )
    return baseline, candidate


#: The catalogue the console and the CLI offer.
PRESETS: dict[str, tuple[Regression, float]] = {
    "identical": ("none", 0.0),
    "small quality drop": ("quality", 0.04),
    "clear quality drop": ("quality", 0.12),
    "one slice broken": ("tag", 0.06),
    "more refusals": ("refusal", 0.12),
    "slower, same quality": ("latency", 0.25),
    "worse on hard cases only": ("tail", 0.05),
}
