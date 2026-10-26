"""The HTTP surface.

Mostly contract tests -- status codes, shapes, the config echoed back -- but
three of them are about things that went wrong or would:

``test_step_curves_are_capped_and_keep_the_decision_point``
    The curve thinner indexed steps by position and subtracted one from
    ``n_used`` to find the decision point. ``peeking`` records no step for
    n = 1, so position and case number differ by one on that procedure alone,
    and the last index fell off the end. It is the kind of off-by-one that
    only fires on one of four curves.

``test_infinities_leave_as_null``
    The first steps of a confidence sequence genuinely have an unbounded
    interval. ``json.dumps`` writes ``Infinity``, which is not JSON: some
    parsers accept it, ``JSON.parse`` does not.

``test_a_thin_suite_abstains_and_says_how_many_labels_it_wanted``
    The response has to explain an abstain, or the console looks broken.
"""

from __future__ import annotations

import json
import math

import pytest
from fastapi.testclient import TestClient

from verdict.api.app import LABELS_TO_CERTIFY, MAX_STEPS, create_app


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    db = tmp_path_factory.mktemp("api") / "history.db"
    app = create_app(db)
    with TestClient(app) as test_client:
        yield test_client
    app.state.store.close()


@pytest.fixture(scope="module")
def slice_run(client):
    """One real gate run, reused: it costs about a second."""
    response = client.post(
        "/api/gate", json={"preset": "one slice broken", "n_cases": 600}
    )
    assert response.status_code == 200
    return response.json()


# -- the small endpoints --------------------------------------------------


def test_health_reports_the_version_and_the_label_bar(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["version"]
    assert body["labels_to_certify"] == LABELS_TO_CERTIFY


def test_presets_are_listed_with_their_shape_and_severity(client):
    presets = client.get("/api/presets").json()
    names = {p["name"] for p in presets}
    assert "identical" in names
    assert "one slice broken" in names
    for preset in presets:
        assert preset["shape"] in {
            "none", "quality", "tag", "refusal", "latency", "tail"
        }
        assert preset["description"]
    assert next(p for p in presets if p["name"] == "identical")["severity"] == 0.0


# -- running the gate -----------------------------------------------------


def test_a_broken_slice_comes_back_flagged_by_its_slice_alone(slice_run):
    assert slice_run["verdict"] == "regression"
    assert slice_run["flagged"] == ["rubric score [math]"]
    assert slice_run["comparisons"] == 14
    assert slice_run["judge"] == "rubric-judge"
    assert slice_run["run_id"] == 1
    assert slice_run["summary"].startswith("REGRESSION")


def test_every_metric_carries_its_test_and_its_adjusted_threshold(slice_run):
    metrics = {m["label"]: m for m in slice_run["metrics"]}
    rubric = metrics["rubric score"]
    assert rubric["kind"] == "ordinal"
    assert rubric["higher_is_better"] is True
    assert rubric["test"]["name"] == "wilcoxon+bootstrap"
    assert len(rubric["test"]["ci"]) == 2
    assert 0 < metrics["rubric score [math]"]["threshold"] <= 0.025
    assert metrics["rubric score [math]"]["rejected"] is True
    # The direction is reported separately from the significance, because a
    # significant improvement is not a regression.
    assert metrics["mean latency"]["higher_is_better"] is False


def test_the_response_carries_the_config_it_was_produced_under(slice_run):
    config = slice_run["config"]
    assert config["alpha"] == 0.05
    assert config["tolerance"] == 0.05
    assert config["correction"] == "bh"
    # And the family was corrected at half of it.
    assert slice_run["correction"]["alpha"] == pytest.approx(config["alpha"] / 2)


def test_the_uncorrected_comparison_is_returned_for_contrast(slice_run):
    assert len(slice_run["flagged"]) <= len(slice_run["flagged_uncorrected"])
    assert slice_run["correction"]["family_error_if_uncorrected"] > 0.25
    assert slice_run["constant_judge_kappa"] == pytest.approx(0.0, abs=1e-9)


def test_step_curves_are_capped_and_keep_the_decision_point(slice_run):
    for key in ("sequential", "sequential_peeking", "sequential_subgaussian"):
        curve = slice_run[key]
        steps = curve["steps"]
        assert 0 < len(steps) <= MAX_STEPS, key
        ns = [s["n"] for s in steps]
        assert ns == sorted(ns), key
        assert ns[0] == 1 or ns[0] == 2  # peeking has no step at n = 1
        # The decision point survives thinning.
        assert curve["n_used"] in ns, key
    # The end-of-suite test has no curve to draw.
    assert slice_run["sequential_fixed"]["steps"] == []


def test_infinities_leave_as_null(slice_run):
    """``Infinity`` is not valid JSON, and 0 would be a different claim."""
    raw = json.dumps(slice_run)
    assert "Infinity" not in raw
    assert "NaN" not in raw
    first = slice_run["sequential_subgaussian"]["steps"][0]
    assert first["lower"] is None or math.isfinite(first["lower"])


def test_identical_pipelines_are_not_flagged(client):
    body = client.post(
        "/api/gate", json={"preset": "identical", "n_cases": 600, "record": False}
    ).json()
    assert body["verdict"] != "regression"
    assert body["flagged"] == []
    assert body["run_id"] is None  # record=False
    assert body["resolvable_margin"] > 0


def test_a_wider_margin_turns_the_same_data_into_a_pass(client):
    body = client.post(
        "/api/gate",
        json={
            "preset": "identical",
            "n_cases": 600,
            "tolerance": 0.3,
            "record": False,
        },
    ).json()
    assert body["verdict"] == "pass"
    assert body["sequential"]["stopped_early"] is True
    assert body["sequential"]["saved"] > 0


def test_a_thin_suite_abstains_and_says_how_many_labels_it_wanted(client):
    """240 cases is ~116 labels, which cannot certify the judge. The response
    must explain that, or the console looks broken."""
    body = client.post(
        "/api/gate", json={"preset": "clear quality drop", "n_cases": 240}
    ).json()
    assert body["verdict"] == "abstain"
    assert body["calibration"]["trustworthy"] is False
    assert body["calibration"]["n"] < LABELS_TO_CERTIFY
    assert "abstain" in body["reason"]
    assert body["labels_to_certify"] == LABELS_TO_CERTIFY
    # Everything was still computed, so the console can show what it would
    # have flagged next to the refusal to rule.
    assert len(body["metrics"]) == 14


def test_the_calibration_block_carries_the_ceiling_and_the_bar(slice_run):
    cal = slice_run["calibration"]
    assert cal["n"] > LABELS_TO_CERTIFY - 50
    assert 0 < cal["required"] < 1
    assert cal["ceiling"] is not None
    assert 0.5 < cal["relative"] < 1.2
    assert cal["trustworthy"] is True
    assert len(cal["confusion"]) == 5
    assert slice_run["ceiling"]["n"] == cal["n"]


def test_holm_is_available_and_stricter(client):
    bh = client.post(
        "/api/gate",
        json={"preset": "one slice broken", "correction": "bh", "record": False},
    ).json()
    holm = client.post(
        "/api/gate",
        json={"preset": "one slice broken", "correction": "holm", "record": False},
    ).json()
    assert holm["correction"]["method"] == "holm"
    assert set(holm["flagged"]) <= set(bh["flagged"])


def test_turning_off_slicing_removes_the_multiplicity(client):
    body = client.post(
        "/api/gate",
        json={"preset": "identical", "slice_by_tag": False, "record": False},
    ).json()
    assert body["comparisons"] == 4
    assert body["correction"]["family_error_if_uncorrected"] < 0.2


# -- validation -----------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"preset": "no such preset"},
        {"n_cases": 5},
        {"n_cases": 99999},
        {"alpha": 0},
        {"alpha": 0.9},
        {"tolerance": -1},
        {"label_fraction": 0},
        {"correction": "bonferroni"},
        {"suite": "does/not/exist.jsonl"},
    ],
)
def test_bad_requests_are_rejected_with_422(client, payload):
    assert client.post("/api/gate", json=payload).status_code == 422


def test_an_unparseable_suite_file_is_a_422_not_a_500(client, tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"no_prompt": 1}\n', encoding="utf-8")
    response = client.post("/api/gate", json={"suite": str(bad)})
    assert response.status_code == 422
    assert "prompt" in response.json()["detail"]


def test_a_suite_file_can_be_run_instead_of_the_synthetic_one(client, tmp_path):
    from verdict.cases import synthetic_suite

    path = tmp_path / "suite.jsonl"
    synthetic_suite(300, name="from-file").to_jsonl(path)
    body = client.post(
        "/api/gate", json={"suite": str(path), "record": False}
    ).json()
    assert body["suite"] == "suite"
    assert body["n_cases"] == 300


# -- history --------------------------------------------------------------


def test_history_is_newest_first_and_carries_the_judge(client, slice_run):
    rows = client.get("/api/history").json()
    assert len(rows) >= 2
    assert rows[0]["created_ms"] >= rows[-1]["created_ms"]
    assert all(row["judge"] == "rubric-judge" for row in rows)
    assert all(len(row["ci"]) == 2 for row in rows)


def test_a_run_can_be_fetched_with_its_metrics(client, slice_run):
    row = client.get(f"/api/runs/{slice_run['run_id']}").json()
    assert row["verdict"] == "regression"
    assert len(row["metrics"]) == 14
    assert row["judge"] == "rubric-judge"


def test_an_unknown_run_is_a_404(client):
    assert client.get("/api/runs/424242").status_code == 404


def test_one_metric_across_runs_comes_back_oldest_first(client, slice_run):
    series = client.get("/api/metrics/rubric score/history").json()
    assert len(series) >= 2
    assert series[0]["created_ms"] <= series[-1]["created_ms"]
    assert all("ci" in row for row in series)


def test_a_metric_nobody_recorded_is_an_empty_series(client):
    assert client.get("/api/metrics/not a metric/history").json() == []


def test_the_history_limit_is_bounded(client):
    assert client.get("/api/history?limit=1").status_code == 200
    assert len(client.get("/api/history?limit=1").json()) == 1
    assert client.get("/api/history?limit=0").status_code == 422
    assert client.get("/api/history?limit=99999").status_code == 422


# -- the console ----------------------------------------------------------


def test_the_console_is_served_at_the_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_the_openapi_schema_describes_every_endpoint(client):
    schema = client.get("/openapi.json").json()
    for path in (
        "/api/health",
        "/api/presets",
        "/api/gate",
        "/api/history",
        "/api/runs/{run_id}",
        "/api/metrics/{label}/history",
        "/api/validation",
    ):
        assert path in schema["paths"], path
