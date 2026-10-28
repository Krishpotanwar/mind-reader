import ast
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import jupytext
import pandas as pd
import pytest

from mindreader.game import GameEngine
from tools.check_notebook_clean import problems

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebook" / "MindReader.ipynb"
METRICS = ROOT / "build" / "metrics.json"
SNAPSHOT = ROOT / "notebook" / "metrics_snapshot.json"


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


def test_part_3_live_cells_are_tagged_and_do_nothing_in_the_build(built):
    headings = ["".join(cell["source"]).split(":")[0] for cell in built["cells"]
                if cell["cell_type"] == "markdown" and "".join(cell["source"]).startswith("# Part ")]
    assert headings[:4] == ["# Part 0", "# Part 1", "# Part 2", "# Part 3"]
    cells = part(built, 3)
    live = [cell for cell in cells if "live" in cell["metadata"].get("tags", [])]
    assert len(live) == 6  # quick start, the game, practice, the eyes test, the pilot export, shut down
    assert all(cell["cell_type"] == "code" and not cell.get("outputs") for cell in live)  # D24: no outputs; with MR_SKIP_LIVE=1 they do nothing
    assert not [cell for cell in cells if cell["cell_type"] == "code" and cell.get("outputs")]  # and nothing else in Part 3 prints
    source = "\n".join("".join(cell["source"]) for cell in live)
    assert "show_widget(\"stage\"" in source and "show_widget(\"practice\"" in source and "show_widget(\"eyes_test\")" in source
    assert "register(ENGINE, READER, LIVE_EYES)" in source and "def quick_start" in source
    assert "image/jpeg" not in json.dumps(built)  # no camera frame ends up in the saved notebook


# ---- Part 3's quick start and shut-down cells, read from the notebook source (no build needed) ----

SOURCE = ROOT / "notebook" / "MindReader.py"
BRAIN_NAMES = ["dt", "logreg", "svm", "nb", "perceptron", "personal", "random"]


def run_quick_start_cell(monkeypatch, live=True, **names):
    """Run the quick-start cell in a namespace that already holds `names` (what Parts 1 and 2 would have left); returns the namespace."""
    monkeypatch.chdir(ROOT)
    cell = next(cell.source for cell in jupytext.read(SOURCE).cells if cell.cell_type == "code" and "def quick_start" in cell.source)
    namespace = {"LIVE": live, **names}
    exec(compile(cell, "quick_start_cell", "exec"), namespace)
    return namespace


SVM_WARNING = "ignore:The `probability` parameter was deprecated:FutureWarning"  # the same one the Setup cell hides


@pytest.mark.filterwarnings(SVM_WARNING)
def test_quick_start_trains_the_eyes_when_only_they_are_missing(monkeypatch):
    brains = object()
    namespace = run_quick_start_cell(monkeypatch, BRAINS=brains, BEST_BRAIN="svm")
    assert namespace["BRAINS"] is brains and namespace["BEST_BRAIN"] == "svm"  # Part 2 had run: its brains stay
    assert hasattr(namespace["LIVE_EYES"], "predict_proba")


@pytest.mark.filterwarnings(SVM_WARNING)
def test_quick_start_trains_the_brains_when_only_they_are_missing(monkeypatch):
    eyes = object()
    namespace = run_quick_start_cell(monkeypatch, LIVE_EYES=eyes)
    assert namespace["LIVE_EYES"] is eyes  # Part 1 had run: the eyes model it tested is not swapped
    assert list(namespace["BRAINS"]) == BRAIN_NAMES and namespace["BEST_BRAIN"] == "dt"


def test_quick_start_does_nothing_when_nothing_is_missing(monkeypatch, capsys):
    names = {"LIVE_EYES": object(), "BRAINS": object(), "BEST_BRAIN": "svm"}
    namespace = run_quick_start_cell(monkeypatch, **names)
    assert all(namespace[name] is value for name, value in names.items())
    assert "Parts 1 and 2 have run" in capsys.readouterr().out


def test_quick_start_cell_does_nothing_in_the_build(monkeypatch, capsys):
    namespace = run_quick_start_cell(monkeypatch, live=False)
    assert "LIVE_EYES" not in namespace and "BRAINS" not in namespace and capsys.readouterr().out == ""


def test_part_3_shut_down_text_says_what_an_open_page_does():
    text = "\n".join(cell.source for cell in jupytext.read(SOURCE).cells if cell.cell_type == "markdown" and "## 6. Shut down" in cell.source)
    assert text and "stops answering" not in text  # a closed hand reader raises nothing: it just finds no hand
    assert "can't see your hand" in text


# ---- Task 9: Part 0's picture, the pilot cells, Part 4 and the time budgets ----

def code_source(marker):
    """The source of the first code cell of the notebook that contains `marker`."""
    return next(cell.source for cell in jupytext.read(SOURCE).cells if cell.cell_type == "code" and marker in cell.source)


def run_cell(monkeypatch, folder, source, **names):
    """Run a cell's source in a namespace that already holds `names`, from inside `folder`. Returns the namespace."""
    monkeypatch.chdir(folder)
    namespace = dict(names)
    exec(compile(source, "cell", "exec"), namespace)
    return namespace


def finished_practice_engine(moves=(0, 1, 2), rounds=3):
    from test_game import Stay, play

    engine = GameEngine({"stay": Stay()}, practice_rounds=rounds)
    play(engine, engine.start("Asha Verma", "practice", "stay")["match_id"], list(moves))
    return engine


def test_the_build_leaves_a_snapshot_of_the_fresh_metrics_for_part_0(built):
    assert SNAPSHOT.read_bytes() == METRICS.read_bytes()  # tools/build_notebook.sh copies it after every successful build
    assert json.loads(SNAPSHOT.read_text(encoding="utf-8")) == json.loads(METRICS.read_text(encoding="utf-8"))


def test_part_0_shows_results_at_a_glance_in_under_two_seconds(built):
    part_0 = part(built, 0)
    cell = next(cell for cell in part_0 if "glance_figure" in "".join(cell["source"]))
    assert "notebook/metrics_snapshot.json" in "".join(cell["source"])  # Part 2 has not run yet when Part 0 does: it draws the last build's numbers
    assert len(pictures([cell])) == 1 and "No saved results" not in shown_text([cell])
    assert seconds([cell]) < 2  # D26: the figure costs almost nothing
    assert len(pictures(part_0)) == 1  # and it is the only picture in Part 0


def test_the_glance_cell_prints_one_note_when_there_is_no_snapshot(monkeypatch, tmp_path, capsys):
    shown = []
    run_cell(monkeypatch, tmp_path, code_source("glance_figure"), os=os, json=json, plt=SimpleNamespace(show=lambda: shown.append(1)))
    assert capsys.readouterr().out.strip().splitlines() == ["No saved results yet: run the whole notebook once, then this picture appears here."]
    assert shown == []  # and no empty figure


def test_the_glance_cell_draws_the_snapshot_it_finds(monkeypatch, tmp_path, capsys):
    (tmp_path / "notebook").mkdir()
    (tmp_path / "notebook" / "metrics_snapshot.json").write_text(SNAPSHOT.read_text(encoding="utf-8"), encoding="utf-8")
    shown = []
    run_cell(monkeypatch, tmp_path, code_source("glance_figure"), os=os, json=json, plt=SimpleNamespace(show=lambda: shown.append(1)))
    assert shown == [1] and capsys.readouterr().out == ""


# --- the pilot: consent first ---

def test_part_3_pilot_text_asks_for_consent_in_plain_words():
    cells = jupytext.read(SOURCE).cells
    text = next(cell.source for cell in cells if cell.cell_type == "markdown" and "## 5. " in cell.source and "ilot" in cell.source)
    flat = " ".join(text.replace("# ", " ").split())
    for promise in ["moves only", "no pictures, no names", "every finished practice match", "data/pilot/", "anonymous id (P01, P02", "unless you tick the box",
                    "must not have heard the pitch"]:  # what is saved and where, the tick, and D29
        assert promise in flat, promise
    camera_text = "This game sends camera frames to this Colab session on Google's servers only to read your hand. Nothing is saved."
    assert camera_text not in text  # D22's camera consent stays in the widget, word for word


def test_part_3_pilot_cell_is_a_live_form_that_is_unticked_and_passes_the_tick_on():
    cell = next(cell for cell in jupytext.read(SOURCE).cells if cell.cell_type == "code" and "export_pilot" in cell.source)
    assert cell.metadata.get("tags") == ["live"]
    lines = cell.source.splitlines()
    assert lines[0] == 'SAVE_MY_THROWS = False  #@param {type:"boolean"}'  # a Colab form box, unticked, at the top level
    assert "consent=SAVE_MY_THROWS" in cell.source and "if LIVE:" in lines[1:]  # the export sits behind the live switch and gets the tick
    export_line = next(line for line in lines if "export_pilot(" in line and "import" not in line)
    assert export_line.startswith("        ")  # inside `if LIVE:` and the try block: nothing is exported at the top level


def test_pilot_cell_requires_consent_in_the_notebook(monkeypatch, tmp_path, capsys):
    engine = finished_practice_engine()
    source = code_source("export_pilot")
    run_cell(monkeypatch, tmp_path, source, LIVE=True, ENGINE=engine)  # the box as it is shipped: unticked
    assert not (tmp_path / "data").exists()  # no file, not even the folder
    assert "Not saved" in capsys.readouterr().out
    run_cell(monkeypatch, tmp_path, source.replace("SAVE_MY_THROWS = False", "SAVE_MY_THROWS = True", 1), LIVE=False, ENGINE=engine)
    assert not (tmp_path / "data").exists() and capsys.readouterr().out == ""  # the build (MR_SKIP_LIVE=1) never saves, ticked or not
    run_cell(monkeypatch, tmp_path, source.replace("SAVE_MY_THROWS = False", "SAVE_MY_THROWS = True", 1), LIVE=True, ENGINE=engine)
    (saved,) = (tmp_path / "data" / "pilot").glob("pilot_*.csv")  # ticked: one file
    out = capsys.readouterr().out
    assert f"data/pilot/{saved.name}" in out and "send this file to Krish" in out
    text = saved.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "id,game,period,choice,other.choice,result" and "P01" in text and "Asha" not in text and "Verma" not in text


def test_pilot_cell_with_a_ticked_box_but_no_finished_practice_match_says_so(monkeypatch, tmp_path, capsys):
    engine = finished_practice_engine(moves=(0, 1))  # one round short of finished
    run_cell(monkeypatch, tmp_path, code_source("export_pilot").replace("SAVE_MY_THROWS = False", "SAVE_MY_THROWS = True", 1), LIVE=True, ENGINE=engine)
    assert not (tmp_path / "data").exists()
    assert "no finished practice match" in capsys.readouterr().out


# --- Part 4 ---

def write_pilot_file(folder, name, players):
    """players: {id: [(choice, result), ...]} -> folder/name, in the pilot's columns."""
    rows = [(player, 1, number, choice, "p", result) for player, throws in players.items() for number, (choice, result) in enumerate(throws, start=1)]
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["id", "game", "period", "choice", "other.choice", "result"]).to_csv(folder / name, index=False)


def test_the_pilot_cell_prints_one_note_when_there_are_no_files(monkeypatch, tmp_path, capsys):
    namespace = run_cell(monkeypatch, tmp_path, code_source("load_pilot"))
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1 and "G3" in lines[0] and "classmates" in lines[0]  # the pilot is gate G3: nothing is invented
    assert "%" not in lines[0] and not re.search(r"\d", lines[0].replace("G3", "").replace("Part 3", ""))  # no number, no percentage: there is nothing to report
    assert namespace["PILOT"] is None


def test_the_pilot_cell_prints_the_counts_the_intervals_and_the_verdict(monkeypatch, tmp_path, capsys):
    folder = tmp_path / "data" / "pilot"
    write_pilot_file(folder, "pilot_2026-10-31.csv", {  # P01 repeats after both losses; P02 repeats after one of two
        "P01": [("r", "lose"), ("r", "tie"), ("p", "lose"), ("p", "win")], "P02": [("s", "lose"), ("p", "lose"), ("p", "tie"), ("p", "tie")]})
    namespace = run_cell(monkeypatch, tmp_path, code_source("load_pilot"))
    out = capsys.readouterr().out
    assert "2 players, 8 rounds" in out and "1 file" in out
    assert "repeat right after a loss: 3 of 4 (75%)" in out and "95% interval 30% to 95%" in out  # Wilson (3, 4) = (30.1%, 95.4%)
    assert "win rate against the bot: 1 of 8 (12%)" in out
    assert "rounds 1 to 8 only" in out.lower()
    assert out.count("inconclusive") == 2  # four rounds after a loss are far too few to say anything, pooled or early
    assert namespace["PILOT"]["players"] == 2 and namespace["PILOT"]["all"]["repeat_after_loss"]["k"] == 3


def test_the_pilot_cell_warns_when_a_match_was_exported_twice(monkeypatch, tmp_path, capsys):
    folder = tmp_path / "data" / "pilot"
    same = {"P01": [("r", "lose"), ("r", "tie"), ("p", "lose")]}
    write_pilot_file(folder, "pilot_2026-10-31.csv", same)
    write_pilot_file(folder, "pilot_2026-10-31_2.csv", same)
    run_cell(monkeypatch, tmp_path, code_source("load_pilot"))
    out = capsys.readouterr().out
    assert "pilot_2026-10-31_2.csv" in out and "P01" in out and "exported twice" in out


def test_part_4_table_and_text_read_the_numbers_of_this_run(built):
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    cells = part(built, 4)
    assert cells, "the notebook has no Part 4"
    text = shown_text(cells)
    brain, eyes, early, live, facts = metrics["brain_table"], metrics["eyes_table"], metrics["brain_early"], metrics["brain_table_live_svm"], metrics["facts"]

    def row(label):
        """[result, luck, always-rock, always-repeat, n] of the table row that starts with this label."""
        line = next(line for line in text.splitlines() if line.startswith(label))
        return line[len(label):].split()

    def pct(share):
        return f"{share:.1%}"

    assert row("Study data: repeat the same move after a loss") == [pct(facts["repeat_after_loss"]), pct(1 / 3), "-", "-", f"{facts['n_after_loss']:,}"]
    best_eyes = max(eyes[name] for name in ["nb", "dt", "logreg", "svm_linear", "svm_rbf", "perceptron"])
    assert row("Eyes: best of the six models, official test pictures") == [pct(best_eyes), pct(eyes["luck"]), "-", "-", "372"]
    assert row("Eyes: live model, test pictures as they are")[0] == pct(metrics["eyes_test_acc"])
    assert row("Eyes: live model, test pictures mirrored")[0] == pct(metrics["eyes_test_acc_mirrored"])
    best_brain = max(brain[name] for name in ["dt", "svm", "logreg", "nb", "perceptron"])
    assert row("Brain: best of the five models, unseen players") == [pct(best_brain), pct(brain["luck"]), pct(brain["always_rock"]), pct(brain["always_repeat"]), f"{facts['n_model_rows']:,}"]
    assert row("Brain: live SVM (20 players), the other 52 players") == [pct(live["accuracy"]), pct(brain["luck"]), "-", pct(live["always_repeat"]), f"{live['n']:,}"]
    for window, label in [("2-10", "Personal brain, rounds 2 to 10"), ("2-20", "Personal brain, rounds 2 to 20")]:
        assert row(label) == [pct(early[window]["personal"]), pct(brain["luck"]), pct(early[window]["always_rock"]), pct(early[window]["always_repeat"]), f"{early[window]['n']:,}"]
    assert len(pictures(cells)) == 1  # the glance figure, drawn from this run's numbers
    glance_cell = next(cell for cell in cells if "glance_figure" in "".join(cell["source"]))
    assert "glance_figure(METRICS)" in "".join(glance_cell["source"]) and pictures([glance_cell])


def test_part4_reads_metrics(built):
    """What the notebook says in Part 4 matches `METRICS` in value (the table above checks every row; this checks the sentences)."""
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    cells = part(built, 4)
    text = shown_text(cells)
    brain, facts = metrics["brain_table"], metrics["facts"]
    best_brain = max(brain[name] for name in ["dt", "svm", "logreg", "nb", "perceptron"])
    gain = (best_brain - brain["always_repeat"]) * 100
    assert f"predicts {best_brain:.1%} of the next moves" in text
    assert f"{gain:.1f} points" in text and f"Always-repeat gets {brain['always_repeat']:.1%}" in text
    assert f"{facts['repeat_after_loss']:.1%} of the time (n = {facts['n_after_loss']:,}" in text
    early = metrics["brain_early"]["2-20"]
    assert f"rounds 2 to 20 the Personal brain gets {early['personal']:.1%}" in text
    assert "computed from the study's public data" in text and "evidence of predictability on unseen players" in text  # D29, word for word


def test_part_4_pilot_section_in_the_build_matches_the_files_there_are(built):
    cells = part(built, 4)
    pilot_cell = next(cell for cell in cells if "load_pilot" in "".join(cell["source"]))
    text = shown_text([pilot_cell])
    files = sorted((ROOT / "data" / "pilot").glob("pilot_*.csv"))
    if files:
        assert f"{len(files)} file" in text and "95% interval" in text
        assert any(verdict in text for verdict in ("inconclusive", "my friends do it too", "less predictable than the study's players"))
    else:
        assert len(text.strip().splitlines()) == 1 and "G3" in text  # no files, one note line, nothing invented
        assert "%" not in text and not re.search(r"\d", text.replace("G3", "").replace("Part 3", ""))


def test_part_4_limits_future_work_and_credits_are_there():
    markdown = "\n".join(cell.source for cell in jupytext.read(SOURCE).cells if cell.cell_type == "markdown")
    part_4 = markdown.split("# Part 4:", 1)[1]
    limits = part_4.split("Limitations", 1)[1].split("Future work", 1)[0]
    for point in ["humans against humans", "72 players", "one lab design", "drawn by a computer", "small pilot", "until the live eyes test"]:
        assert point in limits, point
    credits = part_4.split("Credits", 1)[1]
    for name in ["Wang, Xu & Zhou", "Scientific Reports 4, 5830", "stratEst", "Laurence Moroney", "CC BY 2.0", "MediaPipe Hand Landmarker", "scikit-learn", "OpenCV", "data/LICENSES.md"]:
        assert name in credits, name
    assert "Future work" in part_4


# --- time ---

@pytest.mark.filterwarnings(SVM_WARNING)
def test_quick_start_under_20s(monkeypatch):
    namespace = run_quick_start_cell(monkeypatch, live=False)  # defines quick_start without calling it
    started = time.perf_counter()
    eyes, brains, best = namespace["quick_start"]()
    elapsed = time.perf_counter() - started
    print(f"quick_start took {elapsed:.1f} s")
    assert elapsed <= 20, f"quick_start took {elapsed:.1f} s; the budget (D26) is 20 s"
    assert hasattr(eyes, "predict_proba") and list(brains) == BRAIN_NAMES and best == "dt"  # it really trained both


def test_the_rest_of_parts_0_3_and_4_fit_in_30_seconds(built):
    total = sum(seconds(part(built, number)) for number in (0, 3, 4))
    assert total <= 30, f"Parts 0, 3 and 4 took {total:.1f} s"  # D26: the rest of the fresh Run all


# ---- the build script copies the metrics only after a build that worked ----

def run_build_script(tmp_path, fail_execute):
    """Run tools/build_notebook.sh in a scratch project whose `python` does nothing, or fails the moment it is asked to execute the notebook."""
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "build_notebook.sh").write_text((ROOT / "tools" / "build_notebook.sh").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "metrics.json").write_text('{"fresh": true}', encoding="utf-8")
    (tmp_path / "notebook").mkdir()
    (tmp_path / "notebook" / "metrics_snapshot.json").write_text('{"stale": true}', encoding="utf-8")
    fake = tmp_path / "fake_python"
    fake.write_text('#!/bin/sh\ncase "$*" in *--execute*) [ -n "$FAIL_EXECUTE" ] && exit 1;; esac\nexit 0\n', encoding="utf-8")
    fake.chmod(0o755)
    env = {**os.environ, "PYTHON": str(fake)}
    env.pop("FAIL_EXECUTE", None)
    if fail_execute:
        env["FAIL_EXECUTE"] = "1"
    run = subprocess.run(["bash", str(tmp_path / "tools" / "build_notebook.sh")], env=env, capture_output=True, text=True)
    return run.returncode, (tmp_path / "notebook" / "metrics_snapshot.json").read_text(encoding="utf-8")


def test_the_build_script_copies_the_fresh_metrics_after_a_successful_build(tmp_path):
    code, snapshot = run_build_script(tmp_path, fail_execute=False)
    assert code == 0 and snapshot == '{"fresh": true}'


def test_the_build_script_leaves_the_snapshot_alone_when_the_build_fails(tmp_path):
    code, snapshot = run_build_script(tmp_path, fail_execute=True)
    assert code != 0 and snapshot == '{"stale": true}'  # a failed build must not publish its numbers
