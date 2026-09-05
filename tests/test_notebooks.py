"""Notebooks have to at least be syntactically valid and import what they use."""
import ast
import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted(Path("notebooks").glob("*.ipynb"))


def cells(path, kind="code"):
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == kind]


def test_there_are_notebooks():
    assert len(NOTEBOOKS) == 4


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid_json_with_the_expected_shape(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert nb["cells"]
    assert all(c["cell_type"] in ("code", "markdown") for c in nb["cells"])


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_every_code_cell_parses(path):
    for source in cells(path):
        ast.parse(source)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_starts_with_a_title(path):
    first = json.loads(path.read_text(encoding="utf-8"))["cells"][0]
    assert first["cell_type"] == "markdown"
    assert "".join(first["source"]).lstrip().startswith("# ")


def imported_names(path):
    """Every `from x import a, b` pair used in a notebook."""
    pairs = []
    for source in cells(path):
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    pairs.append((node.module, alias.name))
    return pairs


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_project_imports_resolve(path):
    import importlib

    roots = {"env", "library", "sleep", "agent", "baselines", "analysis"}
    for module, name in imported_names(path):
        if module.split(".")[0] not in roots:
            continue
        imported = importlib.import_module(module)
        assert hasattr(imported, name), f"{module}.{name} missing (used in {path.name})"


def test_the_smoke_notebook_covers_the_resume_path():
    source = "\n".join(cells("notebooks/01_setup_and_smoke.ipynb"))
    assert ".resume()" in source
    assert "knows(" in source


def test_the_main_run_notebook_is_resumable_and_budgeted():
    source = "\n".join(cells("notebooks/02_main_run.ipynb"))
    assert "max_nights" in source
    assert ".resume()" in source
    assert "empty_cache" in source, "should free the GPU between methods"


def test_the_figures_notebook_needs_no_gpu():
    source = "\n".join(cells("notebooks/04_figures.ipynb"))
    assert "load_base_model" not in source
    assert "cuda" not in source
