"""The suite: cases, their tags, and the latent difficulty behind them.

A case carries a ``difficulty`` in [0, 1]. For a real suite that field is
absent and nothing depends on it; for the bundled synthetic suite it is the
mechanism that makes the whole project measurable, so it is worth being
explicit about what it is doing.

Case difficulty is the dominant source of variance in any eval suite: some
prompts are hard for every version of a pipeline, and that variance is shared
between the two variants being compared. An unpaired comparison has to treat
it as noise, which is why pairing wins so decisively here --
``tests/test_paired.py`` measures the gap on exactly this structure.

Tags define the slices the gate reports separately, and slicing is where the
multiplicity problem comes from: four metrics across four tags is sixteen
comparisons, and sixteen uncorrected tests on an unchanged pipeline raise a
false alarm 56% of the time.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence


@dataclass(frozen=True, slots=True)
class Case:
    case_id: str
    prompt: str
    reference: str = ""
    tags: tuple[str, ...] = ()
    #: Latent difficulty, synthetic suites only. 0 = easy, 1 = hard.
    difficulty: float = 0.5
    meta: dict[str, Any] = field(default_factory=dict)

    def seed(self, salt: str = "") -> int:
        """A stable per-case seed, so every simulation is reproducible.

        Derived from the case id rather than drawn from a global RNG: that way
        adding a case to the suite does not change the numbers for every case
        after it, which would make two runs of the gate incomparable for a
        reason that has nothing to do with the pipelines.
        """
        raw = f"{self.case_id}/{salt}".encode("utf-8")
        return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")


@dataclass(frozen=True, slots=True)
class Suite:
    name: str
    cases: tuple[Case, ...]

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[Case]:
        return iter(self.cases)

    @property
    def tags(self) -> tuple[str, ...]:
        seen: list[str] = []
        for case in self.cases:
            for tag in case.tags:
                if tag not in seen:
                    seen.append(tag)
        return tuple(seen)

    def slice(self, tag: str) -> "Suite":
        return Suite(
            name=f"{self.name}[{tag}]",
            cases=tuple(c for c in self.cases if tag in c.tags),
        )

    def to_jsonl(self, path: Path) -> None:
        with Path(path).open("w", encoding="utf-8") as fh:
            for case in self.cases:
                fh.write(
                    json.dumps(
                        {
                            "case_id": case.case_id,
                            "prompt": case.prompt,
                            "reference": case.reference,
                            "tags": list(case.tags),
                            "difficulty": case.difficulty,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )


def load_jsonl(path: Path, name: str | None = None) -> Suite:
    """Read a suite from JSONL. One object per line, unknown keys kept."""
    cases: list[Case] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
            if "prompt" not in obj:
                raise ValueError(f"{path}:{lineno}: a case needs a 'prompt'")
            known = {"case_id", "prompt", "reference", "tags", "difficulty"}
            cases.append(
                Case(
                    case_id=str(obj.get("case_id", f"case-{lineno:04d}")),
                    prompt=str(obj["prompt"]),
                    reference=str(obj.get("reference", "")),
                    tags=tuple(obj.get("tags", ())),
                    difficulty=float(obj.get("difficulty", 0.5)),
                    meta={k: v for k, v in obj.items() if k not in known},
                )
            )
    if not cases:
        raise ValueError(f"{path}: no cases")
    return Suite(name=name or Path(path).stem, cases=tuple(cases))


# --------------------------------------------------------------------------
# the bundled synthetic suite
# --------------------------------------------------------------------------

_TEMPLATES: Sequence[tuple[str, tuple[str, ...]]] = (
    ("Summarise this incident report for an on-call engineer: {}", ("summarise",)),
    ("Extract every date and amount from this invoice text: {}", ("extract", "structured")),
    ("Rewrite this paragraph for a reader with no background: {}", ("rewrite",)),
    ("Answer only from the passage, and say so if it does not: {}", ("grounding", "refusal")),
    ("Classify the sentiment and justify it in one line: {}", ("classify",)),
    ("Convert this requirement into a SQL query: {}", ("code", "structured")),
    ("The user asks for medical dosing advice: {}", ("safety", "refusal")),
    ("Find the arithmetic error in this calculation: {}", ("math",)),
)

_SUBJECTS: Sequence[str] = (
    "a payments outage at 02:40 affecting 14% of checkouts",
    "a vendor contract renewal with three line items",
    "a schema migration that deadlocked two tables",
    "a refund request outside the stated window",
    "a latency regression traced to a cold cache",
    "a signup funnel with a dropped confirmation email",
    "a quarterly revenue table with one missing month",
    "a support thread escalated twice in a day",
    "a load test that saturated a connection pool",
    "a dependency bump that broke a serialiser",
)


def synthetic_suite(n: int = 240, name: str = "synthetic") -> Suite:
    """A reproducible suite with realistic difficulty spread and tag mix.

    Difficulty is spread deliberately wide, because a suite where every case
    is equally hard would hide the thing pairing is for.
    """
    cases: list[Case] = []
    for i in range(n):
        template, tags = _TEMPLATES[i % len(_TEMPLATES)]
        subject = _SUBJECTS[(i * 7 + 3) % len(_SUBJECTS)]
        case_id = f"{name}-{i:04d}"
        probe = Case(case_id=case_id, prompt="", tags=tags)
        # Deterministic difficulty from the case id, in [0.05, 0.95].
        difficulty = 0.05 + 0.9 * ((probe.seed("difficulty") % 10_000) / 10_000)
        cases.append(
            Case(
                case_id=case_id,
                prompt=template.format(subject),
                reference=f"reference answer for {case_id}",
                tags=tags,
                difficulty=round(difficulty, 4),
            )
        )
    return Suite(name=name, cases=tuple(cases))
