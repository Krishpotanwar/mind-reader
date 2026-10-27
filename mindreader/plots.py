"""Plots shared by the notebook parts: one comparison figure for any set of models (D31)."""
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay

# Each metric gets its own colour AND hatch, so the bars still tell apart in greyscale
METRICS = [("accuracy", "#1b2a49", "//"), ("precision", "#3b7fc4", ".."), ("recall", "#e69f00", "xx"), ("f1", "#f0e442", "\\\\")]
BASELINE_COLOURS = ["#000000", "#d55e00", "#009e73"]
LINE_STYLES = ["-", "-.", (0, (1, 1)), (0, (5, 2, 1, 2))]  # models on the ROC panel differ by line style too
LABEL_BOX = {"facecolor": "white", "edgecolor": "none", "pad": 0.5, "alpha": 0.75}  # so a dashed line does not run through a number


def _bars(ax, results: pd.DataFrame, baselines: dict) -> None:
    """Accuracy, precision, recall and F1 as grouped bars, the baselines as dashed lines."""
    positions = np.arange(len(results))
    width = 0.8 / len(METRICS)
    for k, (metric, colour, hatch) in enumerate(METRICS):
        offset = (k - (len(METRICS) - 1) / 2) * width
        bars = ax.bar(positions + offset, results[metric] * 100, width, label=metric, color=colour, hatch=hatch, edgecolor="black")
        ax.bar_label(bars, fmt="{:.1f}", rotation=90, padding=2, fontsize=8, bbox=LABEL_BOX)
    for (name, value), colour in zip(baselines.items(), cycle(BASELINE_COLOURS)):
        ax.axhline(value * 100, linestyle="--", color=colour, label=f"{name} accuracy ({value:.1%})")  # compare with the accuracy bars only
    ax.set_xticks(positions, list(results["model"]))
    ax.set(title="Accuracy, precision, recall and F1", xlabel="model", ylabel="score (%)", ylim=(0, 115))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, fontsize=8)


def _folds(ax, results: pd.DataFrame, baselines: dict) -> None:
    """A boxplot of each model's cross-validation fold scores, with every fold also drawn as a dot."""
    folds = [np.array(scores) * 100 for scores in results["cv_scores"]]
    ax.boxplot(folds, tick_labels=list(results["model"]), patch_artist=True, showfliers=False,  # a normal fold must not look like an outlier
               medianprops={"color": "black"}, boxprops={"facecolor": "#d9d9d9"})
    for position, scores in enumerate(folds, start=1):
        ax.scatter(position + np.linspace(-0.15, 0.15, len(scores)), scores, color="black", s=14, zorder=3)  # spread sideways so none hides another
    for value, colour in zip(baselines.values(), cycle(BASELINE_COLOURS)):
        ax.axhline(value * 100, linestyle="--", color=colour)  # same lines as the legend on the left
    ax.set(title="Cross-validation fold scores (dashed: yardsticks on all rounds)", xlabel="model", ylabel="accuracy per fold (%)")


def _roc(ax, results: pd.DataFrame) -> None:
    """The ROC curves of the models that have probabilities, and the chance line."""
    curves = [(name, roc) for name, roc in zip(results["model"], results["roc"]) if roc is not None]
    for k, (name, (fpr, tpr, area)) in enumerate(curves):
        ax.plot(fpr, tpr, linestyle=LINE_STYLES[k % len(LINE_STYLES)], label=f"{name} (AUC {area:.2f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", label="chance")
    ax.set(title=f"ROC curves (micro-average of the {len(results.iloc[0]['cm'])} classes)", xlabel="false positive rate", ylabel="true positive rate")
    ax.legend(loc="lower right", fontsize=8)


def _confusion(ax, results: pd.DataFrame, labels) -> None:
    """The confusion matrix of the model with the highest accuracy (the first one on a tie)."""
    best = results.iloc[int(np.argmax(results["accuracy"]))]
    ConfusionMatrixDisplay(np.array(best["cm"]), display_labels=labels).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion matrix of the best model: {best['model']}")


def comparison_figure(results: pd.DataFrame, title: str, baselines: dict[str, float] | None = None, *, labels: list[str] | None = None) -> Figure:
    """Four panels: scores as grouped bars, fold-score boxplots, ROC curves, the best model's confusion matrix.

    `results` has one row per model: model, accuracy, precision, recall, f1, cv_scores (a list),
    roc ((fpr, tpr, auc) or None) and cm (a 2D list). `baselines` maps a name to an accuracy between 0 and 1.
    `labels` names the confusion matrix classes."""
    baselines = baselines or {}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle(title, fontsize=15)
    _bars(axes[0, 0], results, baselines)
    _folds(axes[0, 1], results, baselines)
    _roc(axes[1, 0], results)
    _confusion(axes[1, 1], results, labels)
    return fig
