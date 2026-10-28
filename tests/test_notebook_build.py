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


def test_part1_metrics(built):
    cells = part(built, 1)
    assert cells, "the notebook has no Part 1"
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    assert metrics["find_s_clean"] == ["up", "up", "down", "down"]  # index, middle, ring, pinky of a scissors hand
    assert metrics["find_s_noisy"] == ["up", "up", "?", "?"]  # noise turns ring and pinky into "?"
    assert metrics["eyes_detection_rate"] == pytest.approx(369 / 372)  # test split: 369 of the 372 pictures have a hand
    models = ["nb", "dt", "logreg", "svm_linear", "svm_rbf", "perceptron"]
    table = metrics["eyes_table"]
    assert set(table) == set(models) | {"luck"} and table["luck"] == pytest.approx(1 / 3)
    for value in [table[name] for name in models] + [metrics["eyes_test_acc"], metrics["eyes_test_acc_mirrored"]]:
        assert abs(value * 372 - round(value * 372)) < 1e-6 and value <= 369 / 372 + 1e-9  # hits out of all 372 test pictures, and 3 of them have no hand (D5)
    assert metrics["eyes_test_acc"] >= 0.95 and metrics["eyes_test_acc_mirrored"] >= 0.95  # D2: the target, as the pictures are and mirrored
    assert table["nb"] < table["dt"] < table["logreg"]  # the measured order: Naive Bayes, then the tree, then the linear models
    for name, hits in {"nb": 333, "dt": 351, "logreg": 369, "svm_linear": 369, "svm_rbf": 369, "perceptron": 369}.items():
        assert table[name] == pytest.approx(hits / 372, abs=0.01), name  # within 0.01 of the measured hits out of 372 (a drop in one tied top model would not change the order above)
    live = re.search(r"^Live eyes model: (\w+) \(official test accuracy ([\d.]+)% of 372\)$", shown_text(cells), re.MULTILINE)
    capable = [name for name in models if name != "perceptron"]  # D3: the live model needs predict_proba, which the Perceptron lacks
    best = [name for name in capable if table[name] == max(table[other] for other in capable)]
    top = max(table[name] for name in models)
    top_models = [name for name in models if table[name] == top]
    assert f"{len(top_models)} of the {len(models)} models tie at the top, {len(best)} of them with probabilities" in shown_text(cells)  # prose counts come from the table
    assert live and live.group(1) == best[0]  # the best model with probabilities; on a tie the first one in table order
    assert float(live.group(2)) == pytest.approx(table[best[0]] * 100, abs=0.05)
    assert metrics["eyes_test_acc"] == table[live.group(1)]  # the live model's own score on the pictures as they are
    assert seconds(cells) <= 60  # Part 1's share of the fresh Colab "Run all" budget (D26)


def test_part1_story(built):
    headings = ["".join(cell["source"]).split(":")[0] for cell in built["cells"]
                if cell["cell_type"] == "markdown" and "".join(cell["source"]).startswith("# Part ")]
    assert headings[:3] == ["# Part 0", "# Part 1", "# Part 2"] and headings.count("# Part 1") == 1  # one Part 1, between Parts 0 and 2; later parts may follow
    cells = part(built, 1)
    text = shown_text(cells)
    outputs = [output for cell in cells for output in cell.get("outputs", [])]
    assert not [output for output in outputs if output["output_type"] == "error"]
    assert not [output for output in outputs if output["output_type"] == "stream" and output["name"] == "stderr"]  # no warning boxes
    assert "Detection (train): 2,407 of 2,520 pictures have a hand (95.5%)" in text
    assert "Detection (test): 369 of 372 pictures have a hand (99.2%)" in text
    assert "duplicate rows (looking only at the 65 size and landmark columns, not the file name): 0" in text  # the file name is unique in every row, so it must not be part of the check
    markdown = "\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "markdown")
    assert "different hand models" not in (markdown + text).lower()  # D6: the notebook measures the distances instead of claiming that
    assert "never used to choose" not in markdown  # step 10 picks the live model by test accuracy: the test pictures are used once to choose, and never fitted or tuned on
    near = re.search(r"^Nearest-neighbour distance \(median\): training to training ([\d.]+), test to training ([\d.]+)$", text, re.MULTILINE)
    assert near and 0.3 < float(near.group(1)) < 2 and float(near.group(2)) > 2 * float(near.group(1))  # test pictures sit far farther away; a self-distance would give 0.00
    close = re.search(r"^closer than 1\.0: (\d+)% of the training pictures to another training picture, (\d+)% of the test pictures to a training picture$", text, re.MULTILINE)
    assert close and 50 <= int(close.group(1)) <= 80 and int(close.group(2)) == 0  # about two thirds of the training pictures have a near-copy, no test picture has
    assert "fit: 1,925 pictures + 1,925 mirrored twins = 3,850 rows | validation: 482 pictures" in text  # the lab split of the 2,407 training pictures; twins on the fit side only
    area = re.search(r"^micro-average AUC = ([\d.]+)$", text, re.MULTILINE)
    assert area and float(area.group(1)) >= 0.95  # one-vs-rest ROC with label_binarize; a misaligned class order would print about 0.47
    assert all(word in text for word in ["precision", "recall", "macro avg"])  # the classification report of the live model
    for name in ["SVM", "tree"]:
        assert re.search(rf"^Best parameters \({name}\): \{{.*\}}.* [\d.]+ ± [\d.]+", text, re.MULTILINE), name  # GridSearchCV: best settings, mean +- std
    folds = re.findall(r"^Cross-validation (\w+): ([\d.]+) ± ([\d.]+)$", text, re.MULTILINE)  # cross_val_score for every model
    assert [name for name, _, _ in folds] == ["nb", "dt", "logreg", "svm_linear", "svm_rbf", "perceptron"]
    assert all(float(mean) >= 0.99 for _, mean, _ in folds)  # the prose says every model is at the top, because near-copies land in different folds
    assert "Predict one new sample: testscissors01-00.png (true: scissors)" in text
    assert re.search(r"^Live path \(predict_gesture\): scissors ", text, re.MULTILINE)  # the path the live game takes: mirror average of predict_proba
    mirror_average = re.search(r"^Mirror-averaged accuracy \(predict_gesture\): ([\d.]+)% of 372$", text, re.MULTILINE)
    assert mirror_average and float(mirror_average.group(1)) >= 95  # D2: the 95% target, out of all 372
    # countplot, ratio histograms, ratio boxplots, heatmap, pairplot, nearest-neighbour boxplot, accuracy bar chart, confusion matrix, ROC, tree, learning curve, comparison figure
    assert len(pictures(cells)) == 12
    with_output = [cell for cell in cells if cell["cell_type"] == "code" and cell.get("outputs")]
    assert pictures(with_output[-1:]) and "comparison_figure(" in "".join(with_output[-1]["source"])  # Part 1 ends with the comparison figure (D31)


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
