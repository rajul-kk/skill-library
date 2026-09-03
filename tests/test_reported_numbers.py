"""The tables in README.md and paper/paper.md must match results/cpu.

Numbers in prose drift the moment an experiment is re-run with different seeds.
This parses the tables back out and checks them against the results file, so a
re-run that changes the outcome fails here instead of shipping a paper that
disagrees with its own data.
"""
import json
import re
from pathlib import Path

import pytest

RESULTS = Path("results/cpu/results.json")
ROW = re.compile(r"^\|([^|]+)\|(.+)\|\s*$")

KEY_BY_ROW = {
    "(a) full in-context library": "in_context",
    "(a) in-context": "in_context",
    "(b) sequential lora, no replay": "seq_no_replay",
    "(b) sequential, no replay": "seq_no_replay",
    "(c) isolated per-skill adapters (peam-style)": "isolated",
    "(c) isolated (peam-style)": "isolated",
    "(d) one-shot text-to-lora (latentskill-style)": "single_shot",
    "(d) one-shot (latentskill-style)": "single_shot",
    "ours -- cumulative lora + replay": "ours",
    "ours": "ours",
}


@pytest.fixture(scope="module")
def results():
    if not RESULTS.exists():
        pytest.skip("run scripts/reproduce.py first")
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def parse_table(path):
    """Pull (method_key, [cells]) out of the first results table in a document."""
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = ROW.match(line.strip())
        if not match:
            continue
        label = match.group(1).replace("*", "").strip().lower()
        key = KEY_BY_ROW.get(label)
        if key is None:
            continue
        cells = [c.strip().replace("**", "") for c in match.group(2).split("|")]
        rows[key] = cells
    return rows


def as_number(cell: str) -> float:
    cleaned = cell.replace(",", "").replace("M", "").replace(" ", "")
    return float(cleaned)


@pytest.mark.parametrize("document", ["README.md", "paper/paper.md"])
def test_every_method_appears_in_the_results_table(document, results):
    rows = parse_table(document)
    assert set(rows) == set(results["methods"]), f"{document} table is incomplete"


@pytest.mark.parametrize("document", ["README.md", "paper/paper.md"])
def test_reported_success_rates_match_the_results_file(document, results):
    for key, cells in parse_table(document).items():
        reported = as_number(cells[0])
        actual = results["methods"][key]["final_success_rate"]
        assert reported == pytest.approx(actual, abs=0.005), f"{document}: {key}"


@pytest.mark.parametrize("document", ["README.md", "paper/paper.md"])
def test_reported_oldest_skill_retention_matches(document, results):
    for key, cells in parse_table(document).items():
        reported = as_number(cells[1])
        actual = results["methods"][key]["final_earliest_3"]
        assert reported == pytest.approx(actual, abs=0.005), f"{document}: {key}"


@pytest.mark.parametrize("document", ["README.md", "paper/paper.md"])
def test_reported_token_costs_match(document, results):
    for key, cells in parse_table(document).items():
        reported = as_number(cells[2])
        actual = results["methods"][key]["tokens"]["mean_prompt_tokens"]
        assert reported == pytest.approx(actual, abs=1.0), f"{document}: {key}"


@pytest.mark.parametrize("document", ["README.md", "paper/paper.md"])
def test_reported_storage_matches(document, results):
    for key, cells in parse_table(document).items():
        actual = results["methods"][key]["final_stored_params"]
        reported = as_number(cells[3])
        expected = actual / 1e6 if "M" in cells[3] else actual
        assert reported == pytest.approx(expected, abs=0.05), f"{document}: {key}"


def test_the_headline_claim_still_holds(results):
    """Ours matches isolation on retention at a fraction of the storage."""
    ours = results["methods"]["ours"]
    isolated = results["methods"]["isolated"]
    assert ours["final_success_rate"] >= isolated["final_success_rate"] - 0.02
    assert ours["final_stored_params"] * 10 < isolated["final_stored_params"]


def test_the_control_still_forgets(results):
    naive = results["methods"]["seq_no_replay"]
    assert naive["final_success_rate"] < 0.3
    curve = naive["normalized_forgetting_curve"]
    assert curve[max(curve, key=int)] < 0.2
