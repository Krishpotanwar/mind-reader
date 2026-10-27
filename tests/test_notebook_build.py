import ast
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from tools.check_notebook_clean import problems

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebook" / "MindReader.ipynb"
METRICS = ROOT / "build" / "metrics.json"


def two_cell_notebook():
    """Cell 1 is tagged live and has an output. Cell 2 holds a camera frame (a JPEG)."""
    live = {"cell_type": "code", "metadata": {"tags": ["live"]}, "source": "", "execution_count": 1,
            "outputs": [{"output_type": "stream", "name": "stdout", "text": "camera on\n"}]}
    frame = {"cell_type": "code", "metadata": {}, "source": "", "execution_count": 2,
             "outputs": [{"output_type": "display_data", "metadata": {}, "data": {"image/jpeg": "AAAA", "text/plain": "<image>"}}]}
    return {"cells": [live, frame]}


def run_check(path):
    return subprocess.run([sys.executable, str(ROOT / "tools" / "check_notebook_clean.py"), str(path)], capture_output=True, text=True)


def test_check_notebook_clean_flags_live_outputs(tmp_path):
    notebook = two_cell_notebook()
    found = problems(notebook)
    assert len(found) == 2 and "cell 1" in found[0] and "cell 2" in found[1]
    path = tmp_path / "dirty.ipynb"
    path.write_text(json.dumps(notebook))
    dirty = run_check(path)
    assert dirty.returncode == 1 and dirty.stdout.splitlines() == found
    for cell in notebook["cells"]:
        cell["outputs"] = []
    assert problems(notebook) == []
    path.write_text(json.dumps(notebook))
    clean = run_check(path)
    assert clean.returncode == 0 and clean.stdout == ""


def test_check_notebook_clean_finds_a_jpeg_hidden_in_text():
    html = {"cell_type": "code", "metadata": {}, "source": "", "execution_count": 1,
            "outputs": [{"output_type": "display_data", "metadata": {}, "data": {"text/html": '<img src="data:image/jpeg;base64,/9j/AAAA">'}}]}
    printed = {"cell_type": "code", "metadata": {}, "source": "", "execution_count": 2,
               "outputs": [{"output_type": "stream", "name": "stdout", "text": "data:image/jpeg;base64,/9j/AAAA\n"}]}
    found = problems({"cells": [html, printed]})
    assert len(found) == 2 and "cell 1" in found[0] and "cell 2" in found[1]


def test_tools_is_a_regular_package():
    import tools  # the notebook imports tools.rps_facts; a regular package wins over any other "tools" on Colab's sys.path

    assert tools.__file__ is not None


@pytest.fixture(scope="module")
def built():
    """Build the notebook once (this runs every cell) and return the executed notebook."""
    METRICS.unlink(missing_ok=True)  # a stale file from an earlier build must not make a test pass
    run = subprocess.run(["bash", "tools/build_notebook.sh"], cwd=ROOT, env={**os.environ, "PYTHON": sys.executable},
                         capture_output=True, text=True, timeout=900)
    assert run.returncode == 0, f"build failed\n{run.stdout[-3000:]}\n{run.stderr[-3000:]}"
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def part(notebook, number):
    """The cells from the `# Part <number>:` heading up to the next `# Part ` heading (or the end)."""
    cells, inside = [], False
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "markdown" and source.startswith("# Part "):
            inside = source.startswith(f"# Part {number}:")
        if inside:
            cells.append(cell)
    return cells


def shown_text(cells):
    """Everything the cells printed or displayed: stdout, plain-text results and markdown."""
    text = []
    for cell in cells:
        for output in cell.get("outputs", []):
            if output["output_type"] == "stream":
                text.append("".join(output["text"]) if output["name"] == "stdout" else "")
            else:
                data = output.get("data", {})
                text.append("".join(data.get("text/markdown", data.get("text/plain", ""))))
    return "\n".join(text)


def pictures(cells):
    return [output for cell in cells for output in cell.get("outputs", []) if "image/png" in output.get("data", {})]


def test_part_2_checks_the_data_and_draws_the_plots(built):
    cells = part(built, 2)
    text = shown_text(cells)
    assert "7,148" in text and "21,456" in text  # the hook's n, and the number of rows the brain learns from
    assert re.findall(r"(\d+) players on both sides", text) == ["0"] * 5  # GroupKFold keeps every player on one side, in all 5 folds
    random_split = float(re.search(r"random split of the rounds.*?: +([\d.]+)%", text).group(1))
    by_player = float(re.search(r"split by player.*?: +([\d.]+)%", text).group(1))
    assert 0 <= random_split - by_player <= 2  # a random split flatters the score a little; with 3 features the leak is small


def seconds(cells):
    """How long the code cells took, from the timing nbclient writes into each cell's metadata."""
    total = 0.0
    for cell in cells:
        if cell["cell_type"] == "code":
            timing = cell["metadata"]["execution"]
            total += (datetime.fromisoformat(timing["shell.execute_reply"]) - datetime.fromisoformat(timing["iopub.execute_input"])).total_seconds()
    return total


def test_part_2_tunes_the_models_and_tells_the_perceptron_story(built):
    cells = part(built, 2)
    text = shown_text(cells)
    for name in ["dt", "logreg", "svm"]:
        assert re.search(rf"^{name}: best \{{.*\}}, accuracy [\d.]+% \u00b1 [\d.]+%", text, re.MULTILINE), name  # GridSearchCV: best params, mean +- std
    assert re.search(r"^svm: .*first 20 players", text, re.MULTILINE)  # the SVM is tuned on a subset (D26)
    assert re.findall(r"^perceptron seed (\d): [\d.]+%", text, re.MULTILINE) == ["0", "1", "2", "3", "4"]
    assert re.search(r"^logreg: [\d.]+% \u00b1 [\d.]+% over 5 folds", text, re.MULTILINE)  # next to the perceptron seeds
    assert re.search(r"^brains ready: dt, logreg, svm, nb, perceptron, personal, random", text, re.MULTILINE)
    assert re.search(r"^best brain: (dt|svm|logreg|nb|perceptron)$", text, re.MULTILINE)
    ties = re.findall(r"(\d+) of (\d+) settings tie for best", text)  # the best setting may be one of several equal ones
    assert [n for _, n in ties] == ["15", "3", "3"] and all(1 <= int(k) <= int(n) for k, n in ties)
    repeat = re.search(r"^always-repeat: ([\d.]+)% \u00b1 ([\d.]+)% over 5 folds", text, re.MULTILINE)  # on the same test players as the models
    assert repeat and float(repeat.group(1)) == pytest.approx(44.0, abs=0.5)
    wins = re.search(r"^dt beats always-repeat in (\d) of 5 folds$", text, re.MULTILINE)
    assert wins and 0 <= int(wins.group(1)) <= 5
    live = json.loads(METRICS.read_text(encoding="utf-8"))["brain_table_live_svm"]  # the 20-player SVM that plays live, on the other 52 players
    assert set(live) == {"accuracy", "always_repeat", "n"}
    assert live["n"] == 21456 - 20 * 298  # 20 players x 298 model rows are its training rows
    assert round(live["always_repeat"], 3) == 0.432 and 0.42 < live["accuracy"] < 0.46


def test_part_2_early_rounds_slope_new_sample_and_final_comparison(built):
    cells = part(built, 2)
    text = shown_text(cells)
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    expected = {"2-10": (0.349, 0.392, 0.400, 648), "2-20": (0.370, 0.415, 0.424, 1368)}  # D7: always-rock, always-repeat, Personal, n
    assert set(metrics["brain_early"]) == set(expected)
    for window, (rock, repeat, personal, n) in expected.items():
        early = metrics["brain_early"][window]
        assert set(early) == {"always_rock", "always_repeat", "personal", "n", "dt", "n_dt"}
        assert early["n"] == n and early["n_dt"] == n - 72  # the tree cannot predict round 2
        assert early["always_rock"] == pytest.approx(rock, abs=0.005)
        assert early["always_repeat"] == pytest.approx(repeat, abs=0.005)
        assert early["personal"] == pytest.approx(personal, abs=0.005)
        assert 0 < early["dt"] < 1
    personal, repeat = metrics["personal_slope_per_100"], metrics["repeat_slope_per_100"]
    assert 0 < repeat < personal  # players also get more repetitive, so always-repeat rises too; Personal rises faster
    printed = re.findall(r"^(Personal brain|always-repeat): +([+-][\d.]+) points per 100 rounds", text, re.MULTILINE)
    assert printed == [("Personal brain", f"{personal:+.1f}"), ("always-repeat", f"{repeat:+.1f}")]
    gain = re.search(r"own gain over always-repeat: ([+-][\d.]+) points per 100 rounds", text)
    assert gain and float(gain.group(1)) == pytest.approx(personal - repeat, abs=0.06)
    assert re.search(r"^the \w+ brain expects you to throw (rock|paper|scissors) next", text, re.MULTILINE)
    assert re.search(r"^CHANAKYA plays (rock|paper|scissors)", text, re.MULTILINE)
    with_output = [cell for cell in cells if cell["cell_type"] == "code" and cell.get("outputs")]
    last = shown_text(with_output[-1:])  # Part 2 ends with the results table and the 2x2 comparison figure
    assert len(pictures(with_output[-1:])) == 1
    assert all(re.search(rf"^{name}\s+\d", last, re.MULTILINE) for name in ["dt", "svm", "logreg", "nb", "perceptron", "luck", "always-rock", "always-repeat"])
    assert not [cell for cell in cells if cell["cell_type"] == "code"][-1].get("outputs")  # the metrics-writing cell after it is silent


def test_notebook_builds_parts_0_2(built):
    outputs = [output for cell in built["cells"] for output in cell.get("outputs", [])]
    assert not [output for output in outputs if output["output_type"] == "error"]
    assert not [output for output in outputs if output["output_type"] == "stream" and output["name"] == "stderr"]  # no warning boxes in the demo
    assert (ROOT / "notebook" / "MindReader.html").exists()
    hook = shown_text(part(built, 0))
    assert "52.7%" in hook and "7,148" in hook
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    table = metrics["brain_table"]
    assert set(table) == {"dt", "svm", "logreg", "nb", "perceptron", "luck", "always_rock", "always_repeat"}
    assert table["dt"] >= 0.44
    assert round(table["always_rock"], 3) == 0.356 and round(table["always_repeat"], 3) == 0.440 and round(table["luck"], 3) == 0.333
    assert round(metrics["facts"]["repeat_after_loss"], 3) == 0.527
    assert seconds(part(built, 2)) <= 90  # fresh Colab "Run all" budget for the brain (D26)
    # 3 data plots, bar chart, confusion matrix, ROC, learning curve, Perceptron seeds, slope, and the final comparison figure
    assert len(pictures(part(built, 2))) == 10


def test_part_2_shows_five_models_next_to_three_baselines(built):
    cells = part(built, 2)
    text = shown_text(cells)
    for name in ["dt", "svm", "logreg", "nb", "perceptron", "luck", "always-rock", "always-repeat"]:
        assert re.search(rf"^{name}\s+\d", text, re.MULTILINE), name  # a row of the comparison table
    assert "precision" in text and "recall" in text and "f1" in text
    assert re.search(r"accuracy right after a loss \(n = [\d,]+\)", text)
    assert "(7,130 of the 7,148 in the hook: the model needs 2 rounds of history)" in text  # 18 players lost round 1, so 18 losses lack a history
    agreement = re.search(r"dt and svm made the same prediction on ([\d.]+)% of the rounds; the (\d+)-situation lookup table agrees with dt on "
                          r"([\d.]+)% and scores ([\d.]+)% itself", text)
    assert agreement and agreement.groups()[:3] == ("100.0", "27", "100.0")  # 3 x 3 x 3 situations; the table gives dt's answers
    assert float(agreement.group(4)) / 100 == pytest.approx(json.loads(METRICS.read_text(encoding="utf-8"))["brain_table"]["dt"], abs=0.0006)


def module_level_names(source):
    """(imported names, names assigned or defined) at the top level of a cell; function bodies and comprehensions have their own scope."""
    imported, assigned = set(), set()

    def visit(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            assigned.add(node.name)
            return
        if isinstance(node, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            return
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imported.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            assigned.add(node.id)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(ast.parse(source))
    return imported, assigned


def test_no_cell_reuses_an_imported_name(built):
    """Re-running a cell must not break a later one: `outcome` and `clone` were each both imported and used as a variable."""
    imported, assigned = set(), set()
    for cell in built["cells"]:
        if cell["cell_type"] == "code":
            found_imports, found_assigned = module_level_names("".join(cell["source"]))
            imported |= found_imports
            assigned |= found_assigned
    assert imported & assigned == set()
