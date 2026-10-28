import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mindreader.plots import BRAIN_MODELS, EYES_MODELS, comparison_figure, glance_figure

BASELINES = {"luck": 1 / 3, "always-rock": 0.356, "always-repeat": 0.44}
CURVE = (np.array([0.0, 0.4, 1.0]), np.array([0.0, 0.7, 1.0]), 0.7)
SVM_MATRIX = [[5, 2, 1], [1, 4, 3], [2, 2, 4]]  # the confusion matrix of the best model (row 1, svm)


def fake_results():
    """Four made-up models. svm (row 1, not row 0) has the top accuracy and its own confusion matrix.
    The last model has no probabilities, so no ROC curve. dt's folds hold one low score that matplotlib would draw as an outlier dot."""
    return pd.DataFrame({
        "model": ["dt", "svm", "logreg", "perceptron"],
        "accuracy": [0.45, 0.46, 0.44, 0.36],
        "precision": [0.40, 0.39, 0.38, 0.30],
        "recall": [0.41, 0.40, 0.39, 0.33],
        "f1": [0.38, 0.37, 0.36, 0.28],
        "cv_scores": [[0.46, 0.46, 0.46, 0.46, 0.40], [0.43, 0.45, 0.46, 0.47, 0.49], [0.42, 0.44, 0.45, 0.46, 0.47], [0.30, 0.33, 0.36, 0.40, 0.42]],
        "roc": [CURVE, CURVE, CURVE, None],
        "cm": [[[9, 0, 0], [0, 9, 0], [0, 0, 9]], SVM_MATRIX, [[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[3, 3, 3], [0, 0, 0], [1, 1, 1]]],
    })


def test_comparison_figure_has_four_panels_and_labels():
    fig = comparison_figure(fake_results(), "Four models", BASELINES, labels=["stay", "up", "down"])
    try:
        fig.canvas.draw()  # draws everything, so a warning raised while drawing fails this test too
        bars, folds, roc, confusion = fig.axes
        assert len(fig.axes) == 4
        assert len(bars.texts) == 16  # a value label on each of 4 models x 4 metrics
        assert len([line for line in bars.lines if line.get_linestyle() == "--"]) == 3
        legend = [text.get_text() for text in bars.get_legend().get_texts()]
        assert all(f"{name} accuracy ({value:.1%})" in legend for name, value in BASELINES.items())  # the dashed lines are accuracy yardsticks
        assert len({patch.get_hatch() for patch in bars.patches}) == 4  # one hatch per metric, so greyscale still reads
        assert len(folds.patches) == 4  # one box per model
        assert [text.get_text() for text in folds.get_xticklabels()] == ["dt", "svm", "logreg", "perceptron"]
        assert len([line for line in folds.lines if line.get_linestyle() == "--"]) == 3
        assert [len(dots.get_offsets()) for dots in folds.collections] == [5, 5, 5, 5]  # every fold is a dot
        assert not [line for line in folds.lines if line.get_marker() == "o"]  # no fold is drawn as an outlier
        assert folds.get_title() == "Cross-validation fold scores (dashed: yardsticks on all rounds)"
        assert len(roc.lines) == 4  # 3 models with probabilities + the chance line
        assert roc.get_title() == "ROC curves (micro-average of the 3 classes)"
        assert "svm" in confusion.get_title()  # the best model is row 1
        assert np.array_equal(np.asarray(confusion.images[0].get_array()), SVM_MATRIX)
        assert [text.get_text() for text in confusion.get_xticklabels()] == ["stay", "up", "down"]
        assert fig.get_suptitle() == "Four models"
        for ax in fig.axes:
            assert ax.get_title() and ax.get_xlabel() and ax.get_ylabel()
    finally:
        plt.close(fig)


def test_comparison_figure_tie_missing_roc_and_no_baselines():
    results = fake_results()
    results["accuracy"] = 0.5  # a four-way tie: the first row is the best model
    results["roc"] = None
    fig = comparison_figure(results, "Tie")
    try:
        bars, folds, roc, confusion = fig.axes
        assert not [line for line in bars.lines if line.get_linestyle() == "--"]
        assert len(roc.lines) == 1  # only the chance line
        assert np.array_equal(np.asarray(confusion.images[0].get_array()), [[9, 0, 0], [0, 9, 0], [0, 0, 9]])
        assert "dt" in confusion.get_title()
    finally:
        plt.close(fig)


def test_comparison_figure_draws_every_baseline_even_past_three():
    fig = comparison_figure(fake_results(), "Four baselines", {**BASELINES, "always-paper": 0.3})
    try:
        assert len([line for line in fig.axes[0].lines if line.get_linestyle() == "--"]) == 4
    finally:
        plt.close(fig)


# ---- the Results at a glance figure (D31) ----

GLANCE = {
    "eyes_table": {"nb": 0.90, "dt": 0.94, "logreg": 0.99, "svm_linear": 0.985, "svm_rbf": 0.98, "perceptron": 0.97, "luck": 1 / 3},
    "brain_table": {"dt": 0.455, "svm": 0.45, "logreg": 0.447, "nb": 0.448, "perceptron": 0.379, "luck": 1 / 3, "always_rock": 0.356, "always_repeat": 0.44},
    "facts": {"a": 1},  # anything else in the metrics is ignored
}


def dashed(ax):
    return {round(float(line.get_ydata()[0]), 6): line for line in ax.lines if line.get_linestyle() == "--"}


def test_glance_figure_puts_both_parts_side_by_side_next_to_their_yardsticks():
    fig = glance_figure(GLANCE)
    try:
        fig.canvas.draw()  # draws everything, so a warning raised while drawing fails this test too
        eyes, brain = fig.axes
        assert len(fig.axes) == 2 and eyes.get_position().x1 < brain.get_position().x0  # eyes on the left, brain on the right
        assert [text.get_text() for text in eyes.get_xticklabels()] == ["nb", "dt", "logreg", "svm_linear", "svm_rbf", "perceptron"]
        assert [text.get_text() for text in brain.get_xticklabels()] == ["dt", "svm", "logreg", "nb", "perceptron"]
        assert [round(patch.get_height(), 3) for patch in eyes.patches] == [90.0, 94.0, 99.0, 98.5, 98.0, 97.0]  # percentages, not fractions
        assert [round(patch.get_height(), 3) for patch in brain.patches] == [45.5, 45.0, 44.7, 44.8, 37.9]
        assert [text.get_text() for text in eyes.texts if text.get_text().endswith("%")][:6] == ["90.0%", "94.0%", "99.0%", "98.5%", "98.0%", "97.0%"]
        assert [text.get_text() for text in brain.texts if text.get_text().endswith("%")][:5] == ["45.5%", "45.0%", "44.7%", "44.8%", "37.9%"]  # a value on every bar
        assert list(dashed(eyes)) == [pytest.approx(100 / 3, abs=1e-5)]  # luck only
        assert sorted(dashed(brain)) == [pytest.approx(100 / 3, abs=1e-5), 44.0]  # luck and always-repeat, not always-rock
        assert {text.get_text() for ax in fig.axes for text in ax.texts} >= {"luck 33.3%", "always-repeat 44.0%"}  # the lines are named on the plot, not by colour
        assert len({patch.get_hatch() for patch in eyes.patches}) == 6 and len({patch.get_hatch() for patch in brain.patches}) == 5  # greyscale: a hatch each
        assert eyes.get_ylim() == brain.get_ylim() and eyes.get_ylim()[0] == 0 and eyes.get_ylim()[1] > 100 * 0.99  # one scale for both, with room for the labels
        assert fig.get_suptitle() and "unseen" in fig.get_suptitle().lower()
        for ax in fig.axes:
            assert ax.get_title() and ax.get_xlabel() and ax.get_ylabel()
    finally:
        plt.close(fig)


def test_glance_figure_draws_the_real_snapshot():
    snapshot = json.loads((Path(__file__).resolve().parent.parent / "notebook" / "metrics_snapshot.json").read_text(encoding="utf-8"))
    fig = glance_figure(snapshot)
    try:
        fig.canvas.draw()
        eyes, brain = fig.axes
        assert [round(patch.get_height(), 4) for patch in eyes.patches] == [round(snapshot["eyes_table"][name] * 100, 4) for name in EYES_MODELS]
        assert [round(patch.get_height(), 4) for patch in brain.patches] == [round(snapshot["brain_table"][name] * 100, 4) for name in BRAIN_MODELS]
    finally:
        plt.close(fig)


@pytest.mark.parametrize("table, key", [("eyes_table", "svm_rbf"), ("eyes_table", "luck"), ("brain_table", "nb"), ("brain_table", "luck"), ("brain_table", "always_repeat")])
def test_glance_figure_names_the_missing_key(table, key):
    metrics = {name: dict(values) for name, values in GLANCE.items()}
    del metrics[table][key]
    open_before = plt.get_fignums()
    with pytest.raises(ValueError, match=rf"{table}.*{key}"):
        glance_figure(metrics)
    assert plt.get_fignums() == open_before  # and no half-drawn figure is left open


@pytest.mark.parametrize("metrics", [{}, {"eyes_table": {}}, {"brain_table": GLANCE["brain_table"]}, {"eyes_table": GLANCE["eyes_table"], "brain_table": None}, None, []])
def test_glance_figure_refuses_metrics_without_the_two_tables(metrics):
    with pytest.raises(ValueError):
        glance_figure(metrics)
