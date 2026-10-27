import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mindreader.plots import comparison_figure

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
