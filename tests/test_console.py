"""The console, checked against the API's real output.

``scripts/check_console.mjs`` does the work: it evaluates the page's pure
helpers with no DOM and asserts every id the script reaches for exists in the
markup. This module's job is to hand it a *live* report rather than its canned
one, for each of the three verdicts that render differently -- a flagged
regression, a clean inconclusive, and an abstain where nothing was tested.

A report shape that drifts from what the page expects is the failure this
catches: the Python tests would stay green while the console silently stopped
showing a column.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from verdict.api.app import create_app

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "scripts" / "check_console.mjs"

node = shutil.which("node")
needs_node = pytest.mark.skipif(node is None, reason="node is not installed")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    app = create_app(tmp_path_factory.mktemp("console") / "history.db")
    with TestClient(app) as test_client:
        yield test_client
    app.state.store.close()


def run_harness(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [node, str(HARNESS), *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=120,
    )


@needs_node
def test_the_harness_passes_on_its_own_fixture():
    result = run_harness()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "console check passed" in result.stdout


@needs_node
@pytest.mark.parametrize(
    "preset, n_cases, expected",
    [
        ("clear quality drop", 600, "regression"),
        ("identical", 600, "inconclusive"),
        ("clear quality drop", 240, "abstain"),
    ],
)
def test_the_console_handles_each_verdict_the_api_can_return(
    client, tmp_path, preset, n_cases, expected
):
    report = client.post(
        "/api/gate",
        json={"preset": preset, "n_cases": n_cases, "record": False},
    ).json()
    assert report["verdict"] == expected

    path = tmp_path / f"report-{expected}.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    result = run_harness("--report", str(path))
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_page_is_self_contained():
    """No CDN, no font host, no analytics. A review gate that phones out
    while you read its verdict is not a thing to ship, and the demo has to
    work offline. The harness asserts this too; it is cheap enough to assert
    from both sides."""
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    assert "http://" not in html
    assert "https://" not in html
    assert "<link" not in html


def test_the_console_asks_for_fields_the_request_model_accepts():
    """The page posts a JSON body by hand. If a field name drifts from
    GateRequest, FastAPI quietly ignores it and the control stops working
    with no error anywhere."""
    from verdict.api.app import GateRequest

    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    body = html.split("JSON.stringify({")[1].split("})")[0]
    sent = {
        line.split(":")[0].strip()
        for line in body.splitlines()
        if ":" in line and not line.strip().startswith("//")
    }
    assert sent, "the console posts nothing"
    assert sent <= set(GateRequest.model_fields), sent - set(GateRequest.model_fields)
