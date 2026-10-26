"""The suite, the simulated pipelines and the judges.

Two of these tests are regression tests for bugs that were found while
building, and both bugs had the same character: they made the *simulation*
subtly wrong rather than obviously broken, so every number computed on top of
them looked reasonable.

``test_refusals_are_independent_of_the_quality_noise``
    The refusal draw shared a salt with the first uniform of the quality
    noise, so a refusal happened exactly when the noise was extreme. Two
    things the simulation treats as independent were perfectly coupled, and
    the visible symptom was a false REGRESSION on a pair of identical
    pipelines. The test measures the coupling against the buggy draw computed
    alongside the fixed one, so it would fail again if the salt moved back.

``test_two_annotators_are_two_different_people``
    ``HumanLabeller`` seeded itself on the literal ``"human"`` rather than on
    its own name, so two annotators were the same person and the
    inter-annotator ceiling came out at exactly 1.000 -- which then made every
    judge look worse than it was, measured against an oracle.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from verdict.calibration import annotator_ceiling
from verdict.cases import Case, Suite, load_jsonl, synthetic_suite
from verdict.judges import (
    GRADES,
    PASS_AT,
    HumanLabeller,
    Judgement,
    LlmJudge,
    RubricJudge,
    label_subset,
)
from verdict.pipelines import PRESETS, HttpPipeline, SimulatedPipeline, _unit, variant_pair


# -- cases -----------------------------------------------------------------


def test_case_seeds_are_stable_and_case_specific():
    """Seeds come from the case id through blake2b, not from a global RNG.

    The property that matters: adding a case to the suite must not change the
    numbers for any other case. A shared RNG would reshuffle everything after
    the insertion point and make two runs of the gate incomparable for a
    reason that has nothing to do with the pipelines.
    """
    a = Case(case_id="x-1", prompt="p")
    assert a.seed() == Case(case_id="x-1", prompt="different prompt").seed()
    assert a.seed() != Case(case_id="x-2", prompt="p").seed()
    assert a.seed("judge") != a.seed("human")


def test_synthetic_suite_is_reproducible():
    first, second = synthetic_suite(120), synthetic_suite(120)
    assert [c.case_id for c in first] == [c.case_id for c in second]
    assert [c.difficulty for c in first] == [c.difficulty for c in second]


def test_growing_the_suite_leaves_earlier_cases_untouched():
    small = synthetic_suite(50)
    large = synthetic_suite(400)
    assert [c.difficulty for c in small] == [c.difficulty for c in large][:50]


def test_difficulty_is_spread_wide_on_purpose():
    """A suite where every case is equally hard would hide the variance that
    pairing exists to remove, so the spread is a property worth asserting."""
    diffs = np.array([c.difficulty for c in synthetic_suite(400)])
    assert diffs.min() >= 0.05
    assert diffs.max() <= 0.95
    assert diffs.std() > 0.2


def test_slicing_keeps_only_the_tagged_cases():
    suite = synthetic_suite(240)
    assert "math" in suite.tags
    math = suite.slice("math")
    assert len(math) > 0
    assert all("math" in c.tags for c in math)
    assert math.name.endswith("[math]")


def test_tags_are_listed_once_in_first_seen_order():
    suite = synthetic_suite(80)
    assert len(suite.tags) == len(set(suite.tags))
    assert suite.tags[0] in suite.cases[0].tags


def test_jsonl_round_trip(tmp_path):
    original = synthetic_suite(30)
    path = tmp_path / "suite.jsonl"
    original.to_jsonl(path)
    loaded = load_jsonl(path)
    assert [c.case_id for c in loaded] == [c.case_id for c in original]
    assert [c.tags for c in loaded] == [c.tags for c in original]
    assert [c.difficulty for c in loaded] == [c.difficulty for c in original]


def test_jsonl_keeps_unknown_keys_as_meta(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text(
        json.dumps({"prompt": "hello", "ticket": "ENG-4", "owner": "qa"}) + "\n",
        encoding="utf-8",
    )
    case = load_jsonl(path).cases[0]
    assert case.meta == {"ticket": "ENG-4", "owner": "qa"}
    assert case.case_id == "case-0001"


def test_jsonl_skips_blanks_and_comments(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text(
        "# a comment\n\n" + json.dumps({"prompt": "a"}) + "\n", encoding="utf-8"
    )
    assert len(load_jsonl(path)) == 1


@pytest.mark.parametrize(
    "line, message",
    [
        ('{"reference": "no prompt here"}', "prompt"),
        ("{not json at all", ":1:"),
    ],
)
def test_jsonl_failures_name_the_line(tmp_path, line, message):
    path = tmp_path / "bad.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        load_jsonl(path)
    assert message in str(excinfo.value)


def test_an_empty_suite_file_is_an_error(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n# nothing\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no cases"):
        load_jsonl(path)


# -- simulated pipelines ---------------------------------------------------


def test_the_same_pipeline_gives_the_same_output_every_time():
    suite = synthetic_suite(40)
    pipe = SimulatedPipeline(name="p")
    first = [pipe(c) for c in suite]
    second = [pipe(c) for c in suite]
    assert [o.quality for o in first] == [o.quality for o in second]
    assert [o.latency_ms for o in first] == [o.latency_ms for o in second]
    assert [o.refused for o in first] == [o.refused for o in second]


def test_harder_cases_come_out_worse():
    """The mechanism behind the shared variance: quality falls with
    difficulty, which is what makes both variants agree about which cases are
    hard."""
    suite = synthetic_suite(600)
    pipe = SimulatedPipeline(name="p", noise=0.0, refusal_rate=0.0)
    difficulty = np.array([c.difficulty for c in suite])
    quality = np.array([pipe(c).quality for c in suite])
    assert np.corrcoef(difficulty, quality)[0, 1] < -0.9


def test_refusals_are_independent_of_the_quality_noise():
    """Regression test for the salt collision.

    ``_normal(seed, 1)`` consumes salts 3 and 4 for its two uniforms. Drawing
    the refusal at salt 3 made a refusal happen exactly when the quality
    noise was extreme -- not shifting the mean much, which is why it was easy
    to miss, but inflating the spread of the refused group by half.

    Both draws are computed here, the fixed one and the buggy one, and the
    test asserts the fixed one is uncoupled *and* that the statistic it uses
    would have caught the bug. Measured over 3,000 cases: sd ratio 1.01 for
    the fixed draw, 1.45 for the buggy one.
    """
    suite = synthetic_suite(3000)
    # Same name, so the quality draws are identical and the only difference
    # between these two pipelines is whether a case gets refused.
    quiet = SimulatedPipeline(name="p", refusal_rate=0.0)
    loud = SimulatedPipeline(name="p", refusal_rate=0.5)

    quality = np.array([quiet(c).quality for c in suite])
    refused_now = np.array([loud(c).refused for c in suite])
    refused_buggy = np.array([_unit(c.seed("p"), 3) < 0.5 for c in suite])

    def sd_ratio(mask):
        return float(np.std(quality[mask]) / np.std(quality[~mask]))

    assert 0.95 < sd_ratio(refused_now) < 1.05
    assert sd_ratio(refused_buggy) > 1.3


def test_a_refused_output_has_no_quality_left():
    suite = synthetic_suite(200)
    pipe = SimulatedPipeline(name="p", refusal_rate=1.0)
    outputs = [pipe(c) for c in suite]
    assert all(o.refused for o in outputs)
    assert all(o.quality == 0.0 for o in outputs)
    assert all("declined" in o.text for o in outputs)


def test_quality_stays_inside_the_unit_interval():
    suite = synthetic_suite(300)
    for pipe in (
        SimulatedPipeline(name="generous", base_quality=0.99, noise=0.5),
        SimulatedPipeline(name="poor", base_quality=0.05, noise=0.5),
    ):
        qualities = [pipe(c).quality for c in suite]
        assert min(qualities) >= 0.0
        assert max(qualities) <= 1.0


def test_latency_is_never_zero_or_negative():
    suite = synthetic_suite(300)
    pipe = SimulatedPipeline(name="p", latency_ms=50.0, latency_spread=400.0)
    assert min(pipe(c).latency_ms for c in suite) >= 1.0


# -- the shape of each injected regression --------------------------------


def mean_quality(pipe, suite, tag=None):
    cases = [c for c in suite if tag is None or tag in c.tags]
    return float(np.mean([pipe(c).quality for c in cases]))


def test_a_quality_regression_moves_everything_down():
    suite = synthetic_suite(300)
    base, cand = variant_pair("quality", severity=0.12)
    assert mean_quality(cand, suite) < mean_quality(base, suite) - 0.08


def test_a_tag_regression_damages_one_slice_and_leaves_the_rest_alone():
    """The regression that slicing exists to catch: large on one tag, nearly
    invisible on the suite mean."""
    suite = synthetic_suite(400)
    base, cand = variant_pair("tag", severity=0.06)
    math_drop = mean_quality(base, suite, "math") - mean_quality(cand, suite, "math")
    other_drop = mean_quality(base, suite, "rewrite") - mean_quality(
        cand, suite, "rewrite"
    )
    whole_drop = mean_quality(base, suite) - mean_quality(cand, suite)
    assert math_drop > 0.12
    assert abs(other_drop) < 0.02
    assert whole_drop < math_drop / 4


def test_a_refusal_regression_raises_refusals_without_touching_the_answers():
    suite = synthetic_suite(400)
    base, cand = variant_pair("refusal", severity=0.12)
    base_refusals = float(np.mean([base(c).refused for c in suite]))
    cand_refusals = float(np.mean([cand(c).refused for c in suite]))
    assert cand_refusals > base_refusals + 0.05
    # Among the answers that did come back, quality is untouched.
    kept = [c for c in suite if not cand(c).refused and not base(c).refused]
    before = float(np.mean([base(c).quality for c in kept]))
    after = float(np.mean([cand(c).quality for c in kept]))
    assert after == pytest.approx(before, abs=0.02)


def test_a_latency_regression_changes_only_the_clock():
    suite = synthetic_suite(300)
    base, cand = variant_pair("latency", severity=0.25)
    base_latency = float(np.mean([base(c).latency_ms for c in suite]))
    cand_latency = float(np.mean([cand(c).latency_ms for c in suite]))
    assert cand_latency > base_latency * 1.5
    assert mean_quality(cand, suite) == pytest.approx(mean_quality(base, suite), abs=0.02)


def test_a_tail_regression_spares_the_median():
    """Only the already-hard cases degrade, so the median moves a fraction of
    what the damaged slice does.

    Measured on 1,200 cases: the hard third drops 0.17 in latent quality and
    the median of the whole suite drops 0.025 -- a seventh of it. The hard
    cases are 29% of this suite, so the median sits among cases that were
    never touched.
    """
    suite = synthetic_suite(1200)
    base, cand = variant_pair("tail", severity=0.05)
    base_q = np.array([base(c).quality for c in suite])
    cand_q = np.array([cand(c).quality for c in suite])
    hard = np.array([c.difficulty > 0.7 for c in suite])

    hard_drop = float(np.mean(base_q[hard]) - np.mean(cand_q[hard]))
    median_drop = abs(float(np.median(base_q) - np.median(cand_q)))
    assert hard_drop > 0.1
    assert median_drop < hard_drop / 4


def test_identical_variants_differ_only_by_noise():
    """With no regression injected the two pipelines have the same quality in
    expectation. They are not output-identical -- they draw their own per-case
    noise, as two runs of a real pipeline would -- and that is exactly the
    situation where a false-alarm rate can be measured.
    """
    suite = synthetic_suite(600)
    base, cand = variant_pair("none")
    base_q = np.array([base(c).quality for c in suite])
    cand_q = np.array([cand(c).quality for c in suite])
    assert not np.array_equal(base_q, cand_q)
    assert float(np.mean(cand_q - base_q)) == pytest.approx(0.0, abs=0.02)


def test_every_preset_is_a_known_regression_shape():
    shapes = {"none", "quality", "tag", "refusal", "latency", "tail"}
    for name, (shape, severity) in PRESETS.items():
        assert shape in shapes, name
        assert severity >= 0.0
    assert PRESETS["identical"] == ("none", 0.0)


# -- judges ----------------------------------------------------------------


def test_judgement_clamps_to_the_rubric():
    assert Judgement.from_grade(9).grade == GRADES[-1]
    assert Judgement.from_grade(-4).grade == GRADES[0]
    assert Judgement.from_grade(PASS_AT).passed
    assert not Judgement.from_grade(PASS_AT - 1).passed


def test_the_judge_tracks_quality_without_matching_it():
    """An instrument with a bias and a precision: strongly correlated with
    the latent quality, not equal to it. A judge that matched exactly would
    make the calibration machinery meaningless."""
    suite = synthetic_suite(400)
    pipe = SimulatedPipeline(name="p")
    judge = RubricJudge()
    quality = np.array([pipe(c).quality for c in suite])
    grades = np.array([judge(c, pipe(c)).grade for c in suite])
    correlation = float(np.corrcoef(quality, grades)[0, 1])
    assert 0.6 < correlation < 0.98
    assert set(np.unique(grades)) <= set(GRADES)


def test_the_judge_is_soft_on_refusals():
    """The failure mode that lets a refusal regression through a judge-only
    gate: a declined answer reads as cautious rather than useless, so it
    scores well above the zero quality it actually delivered."""
    suite = synthetic_suite(200)
    judge = RubricJudge()
    refuser = SimulatedPipeline(name="p", refusal_rate=1.0)
    grades = [judge(c, refuser(c)).grade for c in suite]
    assert float(np.mean(grades)) > 2.0


def test_the_judge_is_deterministic_for_a_case_and_output():
    suite = synthetic_suite(50)
    pipe = SimulatedPipeline(name="p")
    judge = RubricJudge()
    first = [judge(c, pipe(c)).grade for c in suite]
    second = [judge(c, pipe(c)).grade for c in suite]
    assert first == second


def test_a_noisier_judge_agrees_with_the_reference_less():
    """Monotonicity the calibration numbers depend on: if a louder judge did
    not score worse, the kappa would not be measuring the judge."""
    suite = synthetic_suite(300)
    pipe = SimulatedPipeline(name="p")
    outputs = [pipe(c) for c in suite]
    human = HumanLabeller(name="annotator-a")
    reference = {c.case_id: human(c, o).grade for c, o in zip(suite, outputs)}

    def agreement(noise):
        judge = RubricJudge(noise=noise)
        return float(
            np.mean(
                [
                    judge(c, o).grade == reference[c.case_id]
                    for c, o in zip(suite, outputs)
                ]
            )
        )

    assert agreement(0.2) > agreement(1.5)


def test_two_annotators_are_two_different_people():
    """Regression test for the ``HumanLabeller`` seed bug.

    Seeding on a shared literal instead of ``self.name`` made the second
    annotator a copy of the first, and the giveaway was an inter-annotator
    ceiling of exactly 1.000 -- which then held every judge to an oracle's
    standard. Two labellers with the same name still agree perfectly, which
    is what shows the name is doing the work.
    """
    suite = synthetic_suite(200)
    pipe = SimulatedPipeline(name="p")
    outputs = [pipe(c) for c in suite]

    def grades(labeller):
        return {c.case_id: labeller(c, o).grade for c, o in zip(suite, outputs)}

    a = grades(HumanLabeller(name="annotator-a"))
    b = grades(HumanLabeller(name="annotator-b"))
    same = grades(HumanLabeller(name="annotator-a"))

    assert a == same
    assert a != b
    ceiling = annotator_ceiling(a, b).weighted_kappa
    assert 0.5 < ceiling < 0.95


def test_label_subset_is_the_same_cases_every_run():
    """Otherwise the kappa moves between runs for a reason unrelated to the
    judge, and the history becomes unreadable."""
    suite = synthetic_suite(400)
    pipe = SimulatedPipeline(name="p")
    outputs = [pipe(c) for c in suite]
    labeller = HumanLabeller(name="annotator-a")
    first = label_subset(list(suite), outputs, labeller, fraction=0.25)
    second = label_subset(list(suite), outputs, labeller, fraction=0.25)
    assert set(first) == set(second)
    assert 0.2 < len(first) / 400 < 0.3


def test_label_subset_grows_with_the_fraction():
    suite = synthetic_suite(400)
    pipe = SimulatedPipeline(name="p")
    outputs = [pipe(c) for c in suite]
    labeller = HumanLabeller(name="annotator-a")
    quarter = label_subset(list(suite), outputs, labeller, fraction=0.25)
    half = label_subset(list(suite), outputs, labeller, fraction=0.5)
    assert len(quarter) < len(half) <= 400
    # A larger fraction must be a superset: the subset is a threshold on a
    # per-case hash, not an independent draw.
    assert set(quarter) <= set(half)


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
def test_label_subset_rejects_an_impossible_fraction(fraction):
    with pytest.raises(ValueError):
        label_subset([], [], HumanLabeller(), fraction=fraction)


# -- the real-endpoint classes, without an endpoint -----------------------


class _FakeResponse:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def test_the_llm_judge_parses_a_well_formed_grade(monkeypatch):
    import httpx

    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: _FakeResponse("GRADE: 4\nWHY: minor gap")
    )
    judge = LlmJudge(api_key="test-key")
    judgement = judge(Case(case_id="c1", prompt="p"), type("O", (), {"text": "a"})())
    assert judgement.grade == 4
    assert judgement.passed
    assert judgement.rationale == "minor gap"


def test_the_llm_judge_raises_rather_than_defaulting_to_a_middling_grade(monkeypatch):
    """A silent default would quietly fill the dataset with 3s and flatten
    the signal the gate is looking for."""
    import httpx

    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: _FakeResponse("I'd rather not grade this.")
    )
    judge = LlmJudge(api_key="test-key")
    with pytest.raises(ValueError, match="did not return a grade"):
        judge(Case(case_id="c1", prompt="p"), type("O", (), {"text": "a"})())


def test_the_real_classes_refuse_to_run_without_a_key(monkeypatch):
    monkeypatch.delenv("VERDICT_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="VERDICT_API_KEY"):
        HttpPipeline(name="real", api_key="")(Case(case_id="c", prompt="p"))
    with pytest.raises(RuntimeError, match="VERDICT_API_KEY"):
        LlmJudge(api_key="")(Case(case_id="c", prompt="p"), type("O", (), {"text": ""})())


def test_the_http_pipeline_detects_a_refusal_in_the_text(monkeypatch):
    import httpx

    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: _FakeResponse("I cannot advise on dosing.")
    )
    output = HttpPipeline(name="real", api_key="k")(Case(case_id="c", prompt="p"))
    assert output.refused
    assert output.quality is None  # a real pipeline has no latent quality
    assert output.latency_ms > 0.0
