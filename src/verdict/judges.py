"""Judges, and the human labels that keep them honest.

An LLM judge is a measuring instrument, and every instrument has a bias and a
precision. The usual eval setup never measures either: it asks the judge for a
score, averages the scores, and treats the average as the truth. Then a gate
built on it blocks a release because the judge drifted.

So a judge here is never trusted on its own authority. ``calibration.py``
scores it against human labels first -- agreement, bias, per-grade confusion
-- and the gate refuses to rule at all if the agreement is too low to support
a decision. "The judge is not good enough to answer this" is a legitimate
outcome and a far more useful one than a confident number.

Three implementations:

``RubricJudge``
    A deterministic simulated judge: it reads the output's latent quality
    through its own bias and noise. Its disagreement with the simulated human
    labeller is therefore a real disagreement with a known cause, which is
    what makes the measured kappa a measurement rather than a constant.
``LlmJudge``
    A real judge over an OpenAI-shaped endpoint, with a rubric prompt and a
    strict parse. Untested against a paid endpoint; see the README's limits.
``human_labels``
    The reference labels. In the synthetic setting these come from the same
    latent quality read through *human* noise, because real human labels are
    not perfect either -- a judge cannot be expected to agree with its
    reference more than two humans agree with each other.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from .cases import Case
from .pipelines import Output, _normal, _unit

#: The rubric. Five points, because a 1-10 scale gives the illusion of
#: resolution a judge does not have, and a binary one throws away the
#: ordering that the signed-rank test can use.
GRADES = (1, 2, 3, 4, 5)
PASS_AT = 4


@dataclass(frozen=True, slots=True)
class Judgement:
    grade: int
    passed: bool
    rationale: str = ""

    @staticmethod
    def from_grade(grade: int, rationale: str = "") -> "Judgement":
        grade = max(GRADES[0], min(GRADES[-1], int(grade)))
        return Judgement(grade=grade, passed=grade >= PASS_AT, rationale=rationale)


class Judge(Protocol):
    name: str

    def __call__(self, case: Case, output: Output) -> Judgement: ...


@dataclass
class RubricJudge:
    """A simulated judge with a known bias and a known precision.

    ``bias`` shifts every grade (a judge that likes verbose answers), and
    ``noise`` is how much it wobbles on the same output. Both default to
    values that produce the kind of middling agreement real judges show --
    good enough to be useful, not good enough to trust blindly.
    """

    name: str = "rubric-judge"
    bias: float = 0.18
    noise: float = 0.35
    #: Judges are famously soft on refusals -- a declined answer reads as
    #: cautious rather than useless. Modelled explicitly because it is the
    #: failure mode that lets a refusal regression slip through a judge-only
    #: gate.
    refusal_leniency: float = 1.4

    def __call__(self, case: Case, output: Output) -> Judgement:
        quality = 0.5 if output.quality is None else output.quality
        seed = case.seed(self.name + (output.meta.get("pipeline") or ""))
        latent = quality * 4.0 + 1.0 + self.bias + self.noise * _normal(seed, 11)
        if output.refused:
            latent += self.refusal_leniency
        grade = int(round(max(1.0, min(5.0, latent))))
        return Judgement.from_grade(
            grade, rationale=f"rubric {grade}/5 on {case.case_id}"
        )


class LlmJudge:
    """A real judge: one model call per output, strict parse, no retries.

    The prompt asks for the grade on its own line. A response that cannot be
    parsed raises rather than defaulting to a middling grade -- a silent
    default would quietly fill the dataset with 3s and flatten the very
    signal the gate is looking for.
    """

    name = "llm-judge"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ):
        self.model = model or os.environ.get("VERDICT_JUDGE_MODEL", "gpt-4o-mini")
        self.base_url = (
            base_url or os.environ.get("VERDICT_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("VERDICT_API_KEY", "")
        self.timeout = timeout

    PROMPT = (
        "You are grading one answer against a task. Use this rubric:\n"
        "5 = correct, complete, well-targeted\n"
        "4 = correct with a minor gap\n"
        "3 = partially correct\n"
        "2 = mostly wrong but on topic\n"
        "1 = wrong, empty, or an unjustified refusal\n\n"
        "TASK:\n{prompt}\n\nREFERENCE (may be empty):\n{reference}\n\n"
        "ANSWER:\n{answer}\n\n"
        "Reply with exactly two lines:\nGRADE: <1-5>\nWHY: <one sentence>"
    )

    def __call__(self, case: Case, output: Output) -> Judgement:
        import httpx

        if not self.api_key:
            raise RuntimeError("VERDICT_API_KEY is not set")
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": self.PROMPT.format(
                            prompt=case.prompt,
                            reference=case.reference,
                            answer=output.text,
                        ),
                    }
                ],
                "temperature": 0,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        match = re.search(r"GRADE:\s*([1-5])", text)
        if not match:
            raise ValueError(f"judge did not return a grade: {text[:160]!r}")
        why = re.search(r"WHY:\s*(.+)", text)
        return Judgement.from_grade(
            int(match.group(1)), rationale=(why.group(1).strip() if why else "")
        )


@dataclass
class HumanLabeller:
    """The reference labels, with human noise included on purpose.

    Setting ``noise`` to zero would make the judge's measured agreement an
    overstatement of how good any judge can be, because it would be competing
    against an oracle rather than against a person. Two trained annotators on
    a five-point rubric typically land around kappa 0.6-0.8 with each other;
    a judge that scores 0.7 against a noisy reference is doing about as well
    as a second human would.
    """

    name: str = "human"
    noise: float = 0.42

    def __call__(self, case: Case, output: Output) -> Judgement:
        quality = 0.5 if output.quality is None else output.quality
        # Seeded on this annotator's own name, so two HumanLabellers really
        # are two people: a shared literal here made the inter-annotator
        # ceiling come out at exactly 1.000, which is the tell that the
        # "second annotator" was a copy of the first.
        seed = case.seed(self.name + (output.meta.get("pipeline") or ""))
        latent = quality * 4.0 + 1.0 + self.noise * _normal(seed, 23)
        grade = int(round(max(1.0, min(5.0, latent))))
        return Judgement.from_grade(grade, rationale="human label")


def label_subset(
    cases: Sequence[Case],
    outputs: Sequence[Output],
    labeller: Callable[[Case, Output], Judgement],
    *,
    fraction: float = 0.25,
    seed_salt: str = "label-subset",
) -> dict[str, Judgement]:
    """Label a reproducible subset, the way a real team would.

    Nobody hand-labels a whole suite every release; they label a sample and
    use it to check the judge. The subset is chosen deterministically from the
    case ids so that two runs of the gate calibrate against the same cases --
    otherwise the kappa would move between runs for a reason unrelated to the
    judge.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    cut = fraction * (1 << 53)
    out: dict[str, Judgement] = {}
    for case, output in zip(cases, outputs):
        if _unit(case.seed(seed_salt)) * (1 << 53) < cut:
            out[case.case_id] = labeller(case, output)
    return out
