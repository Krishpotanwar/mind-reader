# %% [markdown]
# # Part 0: Mind-Reader, the bot that reads your next throw
#
# **Mind-Reader** plays rock-paper-scissors against you through your webcam. The bot is called **CHANAKYA**, after the ancient Indian strategist known for reading his opponents.
#
# It reads you in two ways, both with scikit-learn:
# - **The eyes** look at your hand and name the gesture: rock, paper or scissors.
# - **The brain** looks at the last rounds of the game and predicts your next move, so CHANAKYA can play the move that beats it.
#
# This notebook trains both, then plays a live game, and ends with what we learned.

# %% [markdown]
# ## Setup
# On Colab this cell downloads the project and installs MediaPipe, the library that finds the hand. On a laptop it only finds the project folder.
# Both ways, it makes sure the hand model file is there (about 8 MB, checked against its sha256 fingerprint).

# %%
import importlib
import importlib.metadata as metadata
import os
import subprocess
import sys
import warnings

BRANCH = "build/mind-reader"  # becomes "main" once the branch is merged
REPO_URL = "https://github.com/Krishpotanwar/mind-reader.git"

if "google.colab" in sys.modules:
    if not os.path.isdir("/content/mind-reader"):
        clone_cmd = ["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, "/content/mind-reader"]
        if subprocess.run(clone_cmd, capture_output=True).returncode != 0:
            print("Couldn't download the project: check the internet and re-run")
            raise SystemExit(1)
    os.chdir("/content/mind-reader")  # an absolute path, so running this cell twice is safe
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mediapipe==1.1.1"], check=True)
    importlib.invalidate_caches()
    loaded = sys.modules["numpy"].__version__ if "numpy" in sys.modules else None
    if loaded and loaded != metadata.version("numpy"):  # pip changed numpy after this session had already loaded it
        print(f"RESTART NEEDED: numpy {loaded} is already loaded but numpy {metadata.version('numpy')} is installed now")
elif os.path.basename(os.getcwd()) == "notebook":  # on a laptop the notebook runs from its own folder
    os.chdir("..")
sys.path.insert(0, os.getcwd())  # so that `import mindreader` finds the project

# scikit-learn 1.9 warns that SVC(probability=True) goes away in 1.11. The SVM needs it to show a percentage, so only this warning is hidden.
warnings.filterwarnings("ignore", message="The `probability` parameter was deprecated", category=FutureWarning)

from mindreader.eyes import ensure_model

print("hand model ready:", ensure_model("data/raw/hand_landmarker.task"))

# %% [markdown]
# ## Imports and versions

# %%
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display

from mindreader.brain import facts, load_games

METRICS = {}  # the key numbers; Part 2 writes them to build/metrics.json
packages = ("numpy", "pandas", "scikit-learn", "matplotlib", "seaborn", "mediapipe")
print("Python", sys.version.split()[0], "|", " | ".join(f"{name} {metadata.version(name)}" for name in packages))

# %% [markdown]
# ## The hook
# Before any model: do people really play in a pattern? The next cell counts it in the data.

# %%
games = load_games("data/rps_human_games.csv")
hook = facts(games)
display(Markdown(
    f"After losing a round, people **throw the same move again {hook['repeat_after_loss']:.1%} of the time** "
    f"(n = {hook['n_after_loss']:,} rounds). A player picking at random would repeat {1 / 3:.1%} of the time.\n\n"
    f"This is computed from the study's public data: {games['id'].nunique()} players, {games['period'].max()} rounds each. "
    f"People are not random, so a model can learn their habits. That is the brain's job."))

# %% [markdown]
# # Part 1: The Eyes
#
# The eyes look at a picture of a hand and name the gesture: **rock, paper or scissors.**
# MediaPipe's pretrained hand model only turns a picture into 21 points on the hand. We do not train it. Our scikit-learn models learn from those points.
# They learn from the public TensorFlow rock-paper-scissors pictures (hands drawn by a computer), never from our own hands.
#
# The steps are the ones from our labs: check the data, build features, try Find-S, explore with plots, split carefully, train and compare six models, tune them, and try one new example.

# %%
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, auc, classification_report, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_curve)
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_score, learning_curve, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree

from mindreader.eyes import predict_gesture
from mindreader.find_s import find_s
from mindreader.hands import FEATURE_NAMES, features_from_row, finger_states, hand_features, mirror, to_pixels
from mindreader.plots import LINE_STYLES, comparison_figure

IMAGES = {"train": 2520, "test": 372}  # the pictures in the two TensorFlow zips; the CSV only holds the ones where a hand was found
LABELS = ["paper", "rock", "scissors"]  # scikit-learn sorts its classes alphabetically, so every table and plot below uses this order
EYES_COLOURS = {"paper": "#bbbbbb", "rock": "#222222", "scissors": "#777777"}  # three greys, light to dark, so plots still read in greyscale

# %% [markdown]
# ## 1. Look at the data
# `head`, `tail`, `shape` and `info`, the first checks in every lab.
# One row is one picture in which MediaPipe found a hand: the file name, the `split` (TensorFlow's own `train` and `test` pictures), the gesture (`label`), the `width` and `height` of the picture in pixels,
# and then 21 points on the hand, each with an `x`, `y` and `z` (`x0`, `y0`, `z0` up to `x20`, `y20`, `z20`). `x` and `y` are fractions of the width and the height. `z` is MediaPipe's guess at depth, which we never use.
# The table is wide, so `head` and `tail` show only a few columns and `info` prints only its summary.

# %%
hands = pd.read_csv("data/hand_landmarks.csv")
shown = ["file", "split", "label", "width", "height", "x0", "y0"]
display(hands[shown].head())
display(hands[shown].tail())
print("shape:", hands.shape)
hands.info(verbose=False)

# %% [markdown]
# ## 2. Nulls, duplicates and class counts
# The file has no missing values, but it does not hold every picture: MediaPipe did not find a hand in all of them, and those pictures were dropped.
# The duplicate check looks only at the picture size and the 21 points of each hand. The file name is different in every row, so it would hide a repeated picture. The split and the label are left out too, so a hand that shows up in both splits, or under two labels, counts as a duplicate.
# The table shows which gestures MediaPipe missed. For the test pictures this matters most: a picture with no hand can't be read, so it **counts as a wrong answer** in every accuracy below, and the accuracy is always out of all the test pictures.

# %%
print("missing values in the file:", int(hands.isna().sum().sum()))
size_and_landmarks = hands.drop(columns=["file", "split", "label"])  # the file name is different in every row, so it would hide a repeated picture
print(f"duplicate rows (looking only at the {size_and_landmarks.shape[1]} size and landmark columns, not the file name): {int(size_and_landmarks.duplicated().sum())}")
eyes_train = hands[hands["split"] == "train"].reset_index(drop=True)
eyes_test = hands[hands["split"] == "test"].reset_index(drop=True)
per_gesture = {split: n // len(LABELS) for split, n in IMAGES.items()}  # each zip holds the same number of pictures of every gesture
found = pd.crosstab(hands["label"], hands["split"]).reindex(index=LABELS, columns=list(IMAGES))
display(found.rename(columns=lambda split: f"{split} (of {per_gesture[split]})"))
for split, df in [("train", eyes_train), ("test", eyes_test)]:
    print(f"Detection ({split}): {len(df):,} of {IMAGES[split]:,} pictures have a hand ({len(df) / IMAGES[split]:.1%})")
METRICS["eyes_detection_rate"] = len(eyes_test) / IMAGES["test"]  # the test split: the pictures with no hand count as wrong answers later
weakest = (found["train"] / per_gesture["train"]).idxmin()
missed = (per_gesture["test"] - found["test"]).loc[lambda counts: counts > 0]
display(Markdown(
    f"MediaPipe finds the hand least often in **{weakest}** pictures ({found.loc[weakest, 'train']:,} of {per_gesture['train']:,} training pictures). "
    f"The {IMAGES['test'] - len(eyes_test)} test pictures with no hand ({', '.join(f'{count} {name}' for name, count in missed.items())}) count as wrong answers in every accuracy below."))

# %% [markdown]
# **How the file was made.** `tools/make_hand_csv.py` ran MediaPipe once over every picture in the two TensorFlow zips and wrote this file. The pictures themselves are never saved.
# We do not re-run it here: it needs the two zips (a large download) and a hand-finding pass over every picture. To rebuild the file, run this from the project folder:
# `python tools/make_hand_csv.py data/raw/rps.zip data/raw/rps-test-set.zip data/hand_landmarks.csv`

# %% [markdown]
# ## 3. Features
# The model should not care where the hand is in the picture, how big it is or which way it points. `hand_features` (the helper the live game uses too) turns the 21 points into **47 numbers**:
# - **42 numbers for the pose of the hand.** The points in pixels, moved so the wrist is at (0, 0), turned so the middle knuckle is straight above the wrist, and scaled so the wrist-to-knuckle length is 1.
# - **5 numbers for how far each finger reaches** (`ext_thumb` to `ext_pinky`). The distance from the wrist to the fingertip, divided by the distance from the wrist to the finger's middle joint. A curled finger gives a smaller number than a straight one, and turning or resizing the hand does not change it.
#
# Every picture can also be flipped into a **mirrored twin**: the same hand seen in a mirror (every `x` becomes `1 - x`). A mirrored rock, paper or scissors is still the same gesture, so the model never needs to know which hand it is looking at. Step 7 says where the twins are used.
# The cell below checks that flipping the file's `x` columns gives the same 47 numbers as mirroring the hand itself. The features are plain numpy arrays with no column names, the same form the live game uses.

# %%
def feature_matrix(df, flip=False):
    """The 47 features of every picture in df, as a numpy array of shape (pictures, 47). flip=True first mirrors every hand left to right."""
    if flip:
        df = df.copy()
        x_columns = [f"x{point}" for point in range(21)]
        df[x_columns] = 1 - df[x_columns]
    return np.array([features_from_row(row) for _, row in df.iterrows()])


eyes_X_train, eyes_X_train_flipped = feature_matrix(eyes_train), feature_matrix(eyes_train, flip=True)
eyes_X_test, eyes_X_test_flipped = feature_matrix(eyes_test), feature_matrix(eyes_test, flip=True)
eyes_y_train, eyes_y_test = eyes_train["label"].to_numpy(), eyes_test["label"].to_numpy()
print(f"training features {eyes_X_train.shape}, test features {eyes_X_test.shape}: one row per picture, {len(FEATURE_NAMES)} numbers each")

first_five = eyes_train.iloc[:5]
five_hands = first_five.loc[:, "x0":"z20"].to_numpy().reshape(-1, 21, 3)
by_mirroring = [hand_features(mirror(to_pixels(points, width, height))) for points, width, height in zip(five_hands, first_five["width"], first_five["height"])]
assert np.allclose(by_mirroring, eyes_X_train_flipped[:5], atol=1e-9)  # or the mirrored twins would not be mirror images
print("flipping the x columns gives the same numbers as mirroring the hand: checked on", len(first_five), "pictures")

# %%
eyes_ratios = pd.DataFrame(eyes_X_train[:, 42:], columns=FEATURE_NAMES[42:])  # the five finger-reach numbers of the training pictures
display(eyes_ratios.describe())

# %% [markdown]
# ## 4. Find-S (Lab 3)
# Find-S looks for the most specific rule that fits every positive example. Here the positive class is **scissors**, and an example is four facts: is the index, middle, ring and pinky finger **up** or **down**? The thumb is left out, because its joints don't follow the same rule.
# A finger is *up* when its reach number is above 1.15 and *down* when it is below 0.95. A picture with a finger in between is *unclear*.
# - **Clean run:** the unclear pictures are left out.
# - **Noisy run:** every picture is used, with one threshold at 1.0.
#
# Only the pictures that change the rule are printed. `?` means "any value".

# %%
def run_find_s(up, down, leave_out_unclear):
    """Find-S on the training pictures, scissors positive. Prints each picture that changes the rule. Returns the final rule and every picture's finger states."""
    states = [finger_states(ratios, up, down) for ratios in eyes_X_train[:, 42:]]
    kept = [i for i, state in enumerate(states) if not (leave_out_unclear and "unclear" in state)]
    is_scissors = [eyes_y_train[i] == "scissors" for i in kept]
    rule, changes = find_s([states[i] for i in kept], is_scissors)
    print(f"  used {len(kept):,} of {len(states):,} training pictures, {sum(is_scissors):,} of them scissors")
    for index, after in changes:
        print(f"  {eyes_train.loc[kept[index], 'file']} changes the rule to {after}")
    print(f"  final rule (index, middle, ring, pinky): {rule}")
    return rule, states


print("Clean run (up above 1.15, down below 0.95, unclear pictures left out):")
find_s_clean, _ = run_find_s(1.15, 0.95, leave_out_unclear=True)
print("Noisy run (one threshold at 1.0, every picture used):")
find_s_noisy, noisy_states = run_find_s(1.0, 1.0, leave_out_unclear=False)
METRICS["find_s_clean"] = list(find_s_clean)
METRICS["find_s_noisy"] = list(find_s_noisy)

# %%
scissors_ratios = eyes_X_train[eyes_y_train == "scissors", 42:]
ring_or_pinky_up = np.array(["up" in state[2:] for state, label in zip(noisy_states, eyes_y_train) if label == "scissors"])
reach = scissors_ratios[ring_or_pinky_up][:, 3:5].max(axis=1)  # the longer reach of the ring finger and the pinky, in those pictures
display(Markdown(
    f"**Why the noisy run falls apart.** Find-S forgives nothing: one scissors picture that disagrees turns a finger into `?`. "
    f"In {len(reach)} of the {len(scissors_ratios)} scissors pictures the ring finger or the pinky reaches {reach.min():.2f} to {reach.max():.2f}, just over the 1.0 line, so the noisy run calls it *up*. "
    f"The clean run calls those fingers *unclear* and leaves the pictures out. Find-S needs data without noise; real pictures are noisy, so the rest of this part uses models that learn from the whole spread of the numbers."))

# %% [markdown]
# ## 5. Plots
# First how many pictures of each gesture have a hand in them. Then, on the training pictures, how far each finger reaches for each gesture (histograms and boxplots), how the five reach numbers move together (a heatmap) and four of them against each other (a pairplot).
# In the plots by gesture, each gesture has its own grey (paper light, rock dark, scissors in between) and its own marker or hatch, so the plots also read in greyscale.

# %%
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.countplot(data=hands, x="label", hue="split", order=LABELS, hue_order=list(IMAGES), palette=["#555555", "#cccccc"], edgecolor="black", ax=ax)
for container in ax.containers:
    ax.bar_label(container)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1.01, 1))  # beside the bars, not over a number
ax.set(title="Pictures with a hand found, per gesture", xlabel="gesture", ylabel="number of pictures")
plt.show()

# %%
eyes_frame = eyes_ratios.assign(label=eyes_y_train)  # the five reach numbers and the gesture of every training picture
fig, axes = plt.subplots(1, 5, figsize=(18, 3.8), layout="constrained")
for k, (ax, name) in enumerate(zip(axes, FEATURE_NAMES[42:])):
    sns.histplot(data=eyes_frame, x=name, hue="label", hue_order=LABELS, palette=EYES_COLOURS, element="step", legend=(k == 0), ax=ax)
    ax.set(title=name, xlabel="reach number", ylabel="number of pictures")
fig.suptitle("How far each finger reaches, by gesture (training pictures)")
plt.show()

# %%
fig, axes = plt.subplots(1, 5, figsize=(18, 3.8), layout="constrained")
for ax, name in zip(axes, FEATURE_NAMES[42:]):
    boxes = ax.boxplot([eyes_ratios.loc[eyes_y_train == label, name] for label in LABELS], tick_labels=LABELS, patch_artist=True, medianprops={"color": "black"})
    for box, label, hatch in zip(boxes["boxes"], LABELS, ["//", "..", "xx"]):
        box.set(facecolor=EYES_COLOURS[label], hatch=hatch)
    ax.set(title=name, xlabel="gesture", ylabel="reach number")
fig.suptitle("The same reach numbers as boxplots, by gesture (training pictures)")
plt.show()

# %%
fig, ax = plt.subplots(figsize=(6, 4.8))
sns.heatmap(eyes_ratios.corr(), annot=True, fmt=".2f", cmap="Greys", vmin=-1, vmax=1, ax=ax)
ax.set(title="How the five reach numbers move together (correlation)", xlabel="finger", ylabel="finger")
plt.show()

# %%
four_fingers = ["ext_index", "ext_middle", "ext_ring", "ext_pinky"]
pairs = sns.pairplot(eyes_frame, vars=four_fingers, hue="label", hue_order=LABELS, palette=EYES_COLOURS, diag_kind="hist", markers=["o", "s", "^"],
                     height=2, plot_kws={"s": 8, "alpha": 0.5})
pairs.figure.suptitle("Four fingers' reach numbers against each other, by gesture (training pictures)", y=1.02)
plt.show()

# %% [markdown]
# ## 6. Are the test pictures really new?
# The official test is only a fair check if its pictures are not near-copies of training pictures. So we measure it instead of assuming it.
# Every picture is its 47 numbers, scaled so each number counts equally (the scaling the models use too). For each training picture we take the distance to the **nearest other training picture**, and for each test picture the distance to the **nearest training picture**.
# A distance below 1.0 means almost the same hand.

# %%
scaler = StandardScaler().fit(eyes_X_train)
neighbours = NearestNeighbors(n_neighbors=2).fit(scaler.transform(eyes_X_train))
train_gap = neighbours.kneighbors(scaler.transform(eyes_X_train))[0][:, 1]  # column 0 is the picture itself, column 1 the nearest other picture
test_gap = neighbours.kneighbors(scaler.transform(eyes_X_test), n_neighbors=1)[0][:, 0]
print(f"Nearest-neighbour distance (median): training to training {np.median(train_gap):.2f}, test to training {np.median(test_gap):.2f}")
print(f"closer than 1.0: {(train_gap < 1).mean():.0%} of the training pictures to another training picture, {(test_gap < 1).mean():.0%} of the test pictures to a training picture")

fig, ax = plt.subplots(figsize=(6, 4.5))
boxes = ax.boxplot([train_gap, test_gap], tick_labels=["training to training", "test to training"], patch_artist=True, medianprops={"color": "black"})
for box, shade, hatch in zip(boxes["boxes"], ["#cccccc", "#777777"], ["//", ".."]):
    box.set(facecolor=shade, hatch=hatch)
ax.axhline(1.0, linestyle="--", color="black", label="1.0: almost the same hand")
ax.set(title="Distance to the nearest other picture", xlabel="from ... to ...", ylabel="distance (scaled units)")
ax.legend()
plt.show()
display(Markdown(
    f"Half of the training pictures have another training picture within {np.median(train_gap):.2f}, and {(train_gap < 1).mean():.0%} have one closer than 1.0: many training pictures have a near-copy. "
    f"The median test picture is {np.median(test_gap):.2f} from the nearest training picture, and {(test_gap < 1).mean():.0%} are closer than 1.0. "
    f"So the test pictures are not copies of the training pictures, and **the official test is the fair check.** "
    f"A random split *inside* the training pictures is not fair: it puts near-copies on both sides and flatters the score. Part 2 makes the same point about players."))

# %% [markdown]
# ## 7. Splitting the pictures
# Two splits, for two different jobs.
# - **Test: TensorFlow's own test pictures.** No model is fitted or tuned on them. They are used once, to compare the models and pick the live one (step 10), and their mirrored twins are only scored, never trained on. Every accuracy that matters is measured here.
# - **Training, split again the lab way.** `train_test_split(test_size=0.2, stratify=y, random_state=42)` on one row per training picture gives a **fit** part (80%) and a **validation** part (20%).
#
# The mirrored twins of the training pictures are added **after** that split, and only to the fit part. So a twin can never end up on the other side of a split from its original.
# A picture and its twin also share one **group number** (the picture's row in the training table). `GroupKFold` uses it later to keep them together.

# %%
fit_idx, val_idx = train_test_split(np.arange(len(eyes_train)), test_size=0.2, stratify=eyes_y_train, random_state=42)
eyes_X_fit = np.vstack([eyes_X_train[fit_idx], eyes_X_train_flipped[fit_idx]])  # the fit pictures, then their mirrored twins
eyes_y_fit = np.concatenate([eyes_y_train[fit_idx], eyes_y_train[fit_idx]])
eyes_groups_fit = np.concatenate([fit_idx, fit_idx])  # a picture and its twin share one group
eyes_X_val, eyes_y_val = eyes_X_train[val_idx], eyes_y_train[val_idx]  # the validation pictures: no twins
assert not set(val_idx) & set(eyes_groups_fit)  # no validation picture is in a fit group, so none of them has a twin among the fit rows
print(f"fit: {len(fit_idx):,} pictures + {len(fit_idx):,} mirrored twins = {len(eyes_X_fit):,} rows | validation: {len(val_idx):,} pictures | "
      f"test: {len(eyes_X_test):,} pictures with a hand (of {IMAGES['test']})")

# %% [markdown]
# ## 8. Six models
# Each model is a scikit-learn `Pipeline` of two steps, like Lab 8: **scale** the 47 numbers (`StandardScaler`), then **fit a model**.
# - **nb**: Gaussian Naive Bayes (Lab 7). For each gesture it learns the average and the spread of every number. It treats the numbers as unrelated, but some clearly move together (see the heatmap).
# - **dt**: Decision Tree (Lab 4). A flow chart of yes/no questions about the numbers.
# - **logreg**: Logistic Regression (Lab 6). Adds up the weighted numbers into a probability for each gesture.
# - **svm_linear** and **svm_rbf**: Support Vector Machines (Lab 8), with a straight and with a curved boundary between the gestures. They run with `probability=True`, which is slower but lets them give a percentage.
# - **perceptron**: Perceptron (Lab 6). A straight boundary that is nudged only when it gets a picture wrong. It has no probabilities.
#
# All six learn from the same rows: the fit pictures and their mirrored twins. They are fitted on plain numpy arrays, with no column names.

# %%
def eyes_pipeline(clf):
    """Scale the 47 numbers, then fit the model: a Pipeline of two steps, like Lab 8."""
    return Pipeline([("scale", StandardScaler()), ("model", clf)])


EYES_PIPELINES = {  # in this order: when two models tie in the table, the first one wins
    "nb": eyes_pipeline(GaussianNB()),
    "dt": eyes_pipeline(DecisionTreeClassifier(random_state=42)),
    "logreg": eyes_pipeline(LogisticRegression(max_iter=1000)),
    "svm_linear": eyes_pipeline(SVC(kernel="linear", probability=True, random_state=42)),
    "svm_rbf": eyes_pipeline(SVC(kernel="rbf", probability=True, random_state=42)),
    "perceptron": eyes_pipeline(Perceptron(random_state=42)),  # no probabilities, so it comes last
}
for model in EYES_PIPELINES.values():
    model.fit(eyes_X_fit, eyes_y_fit)
print("fitted:", ", ".join(EYES_PIPELINES))

# %% [markdown]
# ## 9. Comparison table and bar chart
# Each model is scored on three sets of pictures: the **fit** rows it learned from (train), the **validation** pictures, and the official **test** pictures.
# The test accuracy is out of **all** the test pictures: the ones with no hand count as wrong, so no model can reach 100%. Precision, recall and F1 are macro averages (worked out for paper, rock and scissors separately, then averaged) over the test pictures that have a hand.
# Two yardsticks sit beside the models. **Luck** guesses at random: one chance in three. The **ceiling** is the best a model can do when it can't read the pictures with no hand.

# %%
def eyes_micro_roc(model, X, y):
    """The ROC curve over the three gestures at once (each one against the rest): false positive rates, true positive rates and the area under the curve."""
    assert list(model.classes_) == LABELS  # predict_proba's columns follow classes_, so they must be in the same order as the columns label_binarize makes
    fpr, tpr, _ = roc_curve(label_binarize(y, classes=LABELS).ravel(), model.predict_proba(X).ravel())
    return fpr, tpr, auc(fpr, tpr)


def score_model(key):
    """One row of the comparison table for the fitted model EYES_PIPELINES[key]."""
    model = EYES_PIPELINES[key]
    predicted = model.predict(eyes_X_test)
    return {
        "model": key,
        "train": accuracy_score(eyes_y_fit, model.predict(eyes_X_fit)),
        "validation": accuracy_score(eyes_y_val, model.predict(eyes_X_val)),
        "accuracy": (predicted == eyes_y_test).sum() / IMAGES["test"],  # the pictures with no hand are wrong (D5)
        "precision": precision_score(eyes_y_test, predicted, labels=LABELS, average="macro", zero_division=0),
        "recall": recall_score(eyes_y_test, predicted, labels=LABELS, average="macro", zero_division=0),
        "f1": f1_score(eyes_y_test, predicted, labels=LABELS, average="macro", zero_division=0),
        "roc": eyes_micro_roc(model, eyes_X_test, eyes_y_test) if hasattr(model, "predict_proba") else None,
        "cm": confusion_matrix(eyes_y_test, predicted, labels=LABELS).tolist(),
    }


eyes_results = pd.DataFrame([score_model(key) for key in EYES_PIPELINES])
EYES_BASELINES = {"luck": 1 / 3, "ceiling": len(eyes_test) / IMAGES["test"]}
eyes_yardsticks = ", ".join(f"{name} {value:.1%}" for name, value in EYES_BASELINES.items())  # printed next to the accuracies below
by_model = eyes_results.set_index("model")
print((pd.concat([by_model[["train", "validation", "accuracy", "precision", "recall", "f1"]], pd.DataFrame({"accuracy": EYES_BASELINES})]) * 100).round(1).to_string(na_rep="-"))
at_ceiling = [key for key, accuracy in by_model["accuracy"].items() if np.isclose(accuracy, EYES_BASELINES["ceiling"])]
display(Markdown(
    f"**Read the three columns together.** On the validation pictures every model scores {by_model['validation'].min():.1%} to {by_model['validation'].max():.1%}. "
    f"That is not all skill: the validation pictures have near-copies among the fit pictures (step 6), so almost any model remembers them. "
    f"The **test** column is the honest one: no model was fitted or tuned on the test pictures, and they are not near-copies of the training pictures (step 6). There, {len(at_ceiling)} of the {len(by_model)} models reach the ceiling ({EYES_BASELINES['ceiling']:.1%}), "
    f"so the test cannot tell those {len(at_ceiling)} apart. Luck would score {EYES_BASELINES['luck']:.1%}."))

# %%
fig, ax = plt.subplots(figsize=(10, 4.8))
positions = np.arange(len(eyes_results))
for k, (column, label, shade, hatch) in enumerate([("train", "train (fit rows)", "#e6e6e6", "//"), ("validation", "validation", "#999999", ".."), ("accuracy", "test (official)", "#333333", "xx")]):
    bars = ax.bar(positions + (k - 1) * 0.27, eyes_results[column] * 100, 0.27, label=label, color=shade, hatch=hatch, edgecolor="black")
    ax.bar_label(bars, fmt="%.1f", rotation=90, padding=2, fontsize=8)
for (name, value), line_style in zip(EYES_BASELINES.items(), ["--", ":"]):
    ax.axhline(value * 100, linestyle=line_style, color="black", label=f"{name} ({value:.1%})")
ax.set_xticks(positions, list(eyes_results["model"]))
ax.set(title="Accuracy of the six models: train, validation and the official test", xlabel="model", ylabel="accuracy (%)", ylim=(0, 118))
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.show()

# %% [markdown]
# ## 10. The model that plays live
# The live game shows a percentage next to its guess, so the live eyes model must be one that gives probabilities (`predict_proba`): every model here except the Perceptron.
# Among those, the one with the best accuracy on the official test pictures is the live eyes model. If several tie, the first one in the table wins. It is used exactly as it was tested above, with no refit.
# This is the one place where the test pictures help to choose something. Nothing is fitted or tuned on them.

# %%
proba_keys = [key for key, model in EYES_PIPELINES.items() if hasattr(model, "predict_proba")]
live_key = by_model.loc[proba_keys, "accuracy"].idxmax()  # the first one on a tie
LIVE_EYES = EYES_PIPELINES[live_key]
assert isinstance(LIVE_EYES, Pipeline) and hasattr(LIVE_EYES, "predict_proba")
assert list(LIVE_EYES.classes_) == LABELS and LIVE_EYES.n_features_in_ == len(FEATURE_NAMES)
assert not hasattr(LIVE_EYES, "feature_names_in_")  # fitted on plain numpy, so the live loop never gets a feature-names warning
print(f"Live eyes model: {live_key} (official test accuracy {by_model.loc[live_key, 'accuracy']:.1%} of {IMAGES['test']})")
tied = [key for key in proba_keys if by_model.loc[key, "accuracy"] == by_model.loc[live_key, "accuracy"]]
print("tied at that score:", ", ".join(tied))
top_models = by_model.index[np.isclose(by_model["accuracy"], by_model["accuracy"].max())]  # every model at the top, with or without probabilities
if len(tied) > 1:
    claim = f"{len(top_models)} of the {len(by_model)} models tie at the top, {len(tied)} of them with probabilities, so the pick cannot flatter the score."
else:
    claim = f"{live_key} is the only model with probabilities at the top, so picking it on these pictures makes its score a little optimistic."
display(Markdown("The official test pictures were used once here, to compare the models and pick the live one. No model was fitted or tuned on them. " + claim))

live_as_is = (LIVE_EYES.predict(eyes_X_test) == eyes_y_test).sum() / IMAGES["test"]
live_mirrored = (LIVE_EYES.predict(eyes_X_test_flipped) == eyes_y_test).sum() / IMAGES["test"]  # the same pictures flipped left to right
print(f"{live_key} on the test pictures, out of {IMAGES['test']}: as they are {live_as_is:.1%}, mirrored {live_mirrored:.1%} (yardsticks: {eyes_yardsticks})")
METRICS["eyes_table"] = {key: float(accuracy) for key, accuracy in by_model["accuracy"].items()} | {"luck": EYES_BASELINES["luck"]}
METRICS["eyes_test_acc"] = float(live_as_is)
METRICS["eyes_test_acc_mirrored"] = float(live_mirrored)

# %% [markdown]
# ## 11. Metrics for the live model
# The **confusion matrix** counts, for each true gesture (rows), what the model said (columns). The **classification report** gives precision, recall and F1 for each gesture, and the number of test pictures it rests on (`support`).
# Both cover the test pictures that have a hand. The pictures with no hand have no answer to put in a matrix: they only lower the accuracy.

# %%
live_predicted = LIVE_EYES.predict(eyes_X_test)
print("model:", live_key)
ConfusionMatrixDisplay.from_predictions(eyes_y_test, live_predicted, labels=LABELS, cmap="Greys")
plt.title(f"Confusion matrix of {live_key}\n{len(eyes_test)} pictures with a hand; the {IMAGES['test'] - len(eyes_test)} without one count as errors in accuracy only")
plt.show()
print(classification_report(eyes_y_test, live_predicted, labels=LABELS, target_names=LABELS, zero_division=0))
display(Markdown(
    f"{live_key} names {(live_predicted == eyes_y_test).sum()} of the {len(eyes_test)} pictures with a hand correctly. "
    f"Add the {IMAGES['test'] - len(eyes_test)} pictures with no hand, which count as wrong, and its accuracy is {live_as_is:.1%} of {IMAGES['test']} (yardsticks: {eyes_yardsticks})."))

# %% [markdown]
# **ROC curves.** For each gesture in turn, the curve for "this gesture against the other two" (that is what `label_binarize` sets up), and the **micro-average** over all three together.
# The dotted line is chance. The more a curve bows towards the top left, the better; the area under it (AUC) is 1.0 for a perfect model and 0.5 for chance.

# %%
gesture_onehot = label_binarize(eyes_y_test, classes=LABELS)  # one 0/1 column per gesture, in LABELS order
live_probabilities = LIVE_EYES.predict_proba(eyes_X_test)  # its columns follow classes_, which is LABELS (checked above)
fig, ax = plt.subplots(figsize=(6, 5.5))
for k, gesture in enumerate(LABELS):
    fpr, tpr, _ = roc_curve(gesture_onehot[:, k], live_probabilities[:, k])
    ax.plot(fpr, tpr, linestyle=LINE_STYLES[k], color="black", label=f"{gesture} vs the rest (AUC {auc(fpr, tpr):.3f})")
    print(f"{gesture} vs the rest: AUC = {auc(fpr, tpr):.3f}")
fpr, tpr, micro_area = eyes_micro_roc(LIVE_EYES, eyes_X_test, eyes_y_test)
ax.plot(fpr, tpr, linestyle=LINE_STYLES[3], color="#777777", linewidth=2.5, label=f"micro-average (AUC {micro_area:.3f})")
ax.plot([0, 1], [0, 1], color="grey", linestyle=":", label="chance")
ax.set(title=f"ROC curves of {live_key}", xlabel="false positive rate", ylabel="true positive rate")
ax.legend(loc="lower right")
plt.show()
print(f"micro-average AUC = {micro_area:.3f}")

# %% [markdown]
# **The Decision Tree, drawn.** The tree of the `dt` model, for display only (it is not the live model).
# Each box shows the question it asks about one of the 47 numbers (after scaling), how many rows reach it, and the gesture most of them are. `plot_tree` stops after 3 levels, so even a big tree fits on the page.

# %%
dt_tree = EYES_PIPELINES["dt"].named_steps["model"]
fig, ax = plt.subplots(figsize=(14, 6))
plot_tree(dt_tree, max_depth=3, feature_names=FEATURE_NAMES, class_names=LABELS, filled=True, fontsize=7, ax=ax)
ax.set_title("The Decision Tree: each box asks about one scaled number")
plt.show()
display(Markdown(f"This tree is {dt_tree.get_depth()} questions deep and has {dt_tree.tree_.node_count} boxes, so "
                 f"{'all of it' if dt_tree.get_depth() <= 3 else 'only its first 3 levels'} is drawn."))

# %% [markdown]
# ## 12. Tuning with GridSearchCV
# `GridSearchCV` tries every setting in a grid, scores each one with 5-fold cross-validation, and keeps the best. Then it refits that best setting on all the fit rows.
# Here it uses `GroupKFold` on the group numbers from step 7, so a picture and its mirrored twin are always in the same fold.
# - **SVM:** the kernel (straight line or curve), `C` (how loosely the boundary may fit the rows; bigger is looser) and, for the curved kernel, `gamma` (how far one row's influence reaches).
# - **Decision Tree:** how deep the tree may grow and how it picks its questions (`criterion`).
#
# Each line shows the best setting, its score as **mean ± std** over the folds, how many settings tie for best, and the tuned model's score on the test pictures. That test score is only printed here: it does not replace a row in the table, and the tuned model is not the live model.

# %%
svm_grid = [
    {"model__kernel": ["linear"], "model__C": [0.1, 1, 10]},
    {"model__kernel": ["rbf"], "model__C": [0.1, 1, 10], "model__gamma": ["scale", 0.01, 0.1]},
]
tree_grid = {"model__max_depth": [3, 5, 8, None], "model__criterion": ["gini", "entropy"]}


def eyes_tune(name, pipeline, grid):
    """GridSearchCV on the fit rows with 5-fold GroupKFold. Prints the best settings, their score as mean ± std over the folds and the tuned model's test accuracy."""
    search = GridSearchCV(pipeline, grid, cv=GroupKFold(n_splits=5), scoring="accuracy").fit(eyes_X_fit, eyes_y_fit, groups=eyes_groups_fit)
    best, ranks = search.best_index_, search.cv_results_["rank_test_score"]
    mean, std = search.cv_results_["mean_test_score"][best], search.cv_results_["std_test_score"][best]
    hits = (search.predict(eyes_X_test) == eyes_y_test).sum()
    print(f"Best parameters ({name}): {search.best_params_}, cross-validation {mean:.4f} ± {std:.4f} (mean ± std over 5 folds), "
          f"{(ranks == 1).sum()} of {len(ranks)} settings tie for best; the tuned model gets {hits} of {IMAGES['test']} test pictures right ({hits / IMAGES['test']:.1%})")
    return search


print(f"yardsticks on the test pictures: {eyes_yardsticks}")
eyes_search_svm = eyes_tune("SVM", eyes_pipeline(SVC(random_state=42)), svm_grid)  # no probability option: the grid only needs predictions, and it is much faster without
eyes_search_tree = eyes_tune("tree", eyes_pipeline(DecisionTreeClassifier(random_state=42)), tree_grid)

# %% [markdown]
# ## 13. Cross-validation: mean ± std
# `cross_val_score` does what the labs did: split the rows into 5 folds, fit on 4 and score on the 5th, five times. Here each model is cross-validated on the fit rows with `GroupKFold`, so twins stay together.
# The mean is the typical score, and a small std means every fold gave about the same score.

# %%
eyes_cv_scores = {}  # model -> its 5 fold scores
for key, model in EYES_PIPELINES.items():
    scores = cross_val_score(model, eyes_X_fit, eyes_y_fit, groups=eyes_groups_fit, cv=GroupKFold(n_splits=5))
    eyes_cv_scores[key] = scores.tolist()
    print(f"Cross-validation {key}: {scores.mean():.4f} ± {scores.std():.4f}")
all_folds = np.array(list(eyes_cv_scores.values()))
lowest_model = by_model["accuracy"].idxmin()  # the model with the lowest test accuracy
display(Markdown(
    f"**Why these scores are so high.** Every model scores {all_folds.mean(axis=1).min():.4f} to {all_folds.mean(axis=1).max():.4f}, even {lowest_model}, "
    f"which gets only {by_model.loc[lowest_model, 'accuracy']:.1%} of the official test pictures right. Two things push the cross-validation scores up. "
    f"The near-copies of a fit picture (step 6) land in other folds, so each fold partly looks like rows the model has already learned from. "
    f"And the held-out folds are training pictures, which sit closer to each other (median distance {np.median(train_gap):.2f}) than the test pictures sit to the training pictures ({np.median(test_gap):.2f}). "
    f"That helps explain why {lowest_model} scores {np.mean(eyes_cv_scores[lowest_model]):.1%} in cross-validation but {by_model.loc[lowest_model, 'accuracy']:.1%} on the test. "
    f"A grid cannot improve on a score that is already at the top. The fair numbers are the official test and the nearest-neighbour check. Luck would score {EYES_BASELINES['luck']:.4f}."))

# %% [markdown]
# ## 14. Learning curve
# Train the live model on a growing share of the fit rows and score it each time on the held-out folds. If the held-out line were still climbing at the end, more pictures would help (these folds hold near-copies, step 13).
# `shuffle=True` mixes the rows first: without it each training set would hold only the pictures as they are, because the twins come after them, and the small sizes would look much worse than they are.

# %%
curve_sizes, curve_train, curve_held_out = learning_curve(
    clone(LIVE_EYES), eyes_X_fit, eyes_y_fit, groups=eyes_groups_fit, cv=GroupKFold(n_splits=5), train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0], shuffle=True, random_state=42)
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for scores, label, marker, line_style, shade in [(curve_train, "rows it trained on", "o", "-", "#222222"), (curve_held_out, "held-out folds", "s", "-.", "#777777")]:
    ax.plot(curve_sizes, scores.mean(axis=1), marker=marker, linestyle=line_style, color=shade, label=label)
    ax.fill_between(curve_sizes, scores.mean(axis=1) - scores.std(axis=1), scores.mean(axis=1) + scores.std(axis=1), color=shade, alpha=0.2)
ax.text(0.02, 0.05, f"luck is {1 / 3:.1%}, far below this chart", transform=ax.transAxes)
ax.set(title=f"Learning curve of {live_key}", xlabel="number of training rows (a picture and its mirrored twin are two rows)", ylabel="accuracy", ylim=(0.9, 1.01))
ax.legend(loc="lower right")
plt.show()
for rows_used, fit_score, held_out in zip(curve_sizes, curve_train.mean(axis=1), curve_held_out.mean(axis=1)):
    print(f"{rows_used:5d} training rows: {fit_score:.4f} on them, {held_out:.4f} on the held-out folds")

# %% [markdown]
# ## 15. One new example
# Take one test picture the model was not trained on: the first scissors picture of the test set. `predict_proba` gives the live eyes model's probability for each gesture.
# Then the same picture goes down the **live path**, `predict_gesture`: it turns the 21 points into the 47 numbers, asks `predict_proba` about the hand **and** about its mirror image, and averages the two answers. That is why a left hand and a right hand get the same answer in the game.
# Last, the live path on every test picture: its **mirror-averaged accuracy**. It is only printed here. The table above holds the model's own answers.

# %%
sample = int(np.flatnonzero(eyes_y_test == "scissors")[0])  # the first scissors picture in the test set
print(f"Predict one new sample: {eyes_test.loc[sample, 'file']} (true: {eyes_y_test[sample]})")
display(pd.DataFrame(LIVE_EYES.predict_proba(eyes_X_test[[sample]]), columns=LABELS, index=[eyes_test.loc[sample, "file"]]).round(3))  # a numpy row, like the live game sends

test_hands = eyes_test.loc[:, "x0":"z20"].to_numpy().reshape(-1, 21, 3)  # the 21 points of every test picture
test_pixels = [to_pixels(points, width, height) for points, width, height in zip(test_hands, eyes_test["width"], eyes_test["height"])]
label, probability = predict_gesture(LIVE_EYES, test_pixels[sample])
print(f"Live path (predict_gesture): {label} ({probability:.0%} sure, averaged with its mirror image)")
live_path_hits = sum(predict_gesture(LIVE_EYES, pixels)[0] == truth for pixels, truth in zip(test_pixels, eyes_y_test))
print(f"Mirror-averaged accuracy (predict_gesture): {live_path_hits / IMAGES['test']:.1%} of {IMAGES['test']}")
print(f"yardsticks: {eyes_yardsticks}")

# %% [markdown]
# ## 16. All six models, side by side
# The results table, then the comparison figure. Everything is scored on the official test pictures.
# - **Top left:** accuracy, precision, recall and F1 as bars. The dashed line is the **accuracy** of luck: compare it with the accuracy bars only. Accuracy is out of all the test pictures, precision, recall and F1 only over the ones with a hand.
# - **Top right:** the 5 cross-validation scores of each model (step 13), one dot per fold. They are all at the top (see step 13 for why), far above the dashed luck line.
# - **Bottom left:** ROC curves, for the models that give probabilities (the Perceptron never does). Each curve is the micro-average over paper, rock and scissors; the curve and AUC of each gesture on its own are in step 11.
# - **Bottom right:** the confusion matrix of the first model with the top accuracy.

# %%
final_results = eyes_results.assign(cv_scores=[eyes_cv_scores[key] for key in eyes_results["model"]])
eyes_summary = final_results.set_index("model")[["accuracy", "precision", "recall", "f1"]].copy()
eyes_summary["cv mean"] = [np.mean(scores) for scores in final_results["cv_scores"]]
eyes_summary["cv std"] = [np.std(scores) for scores in final_results["cv_scores"]]
print((pd.concat([eyes_summary, pd.DataFrame({"accuracy": {"luck": EYES_BASELINES["luck"]}})]) * 100).round(2).to_string(na_rep="-"))
display(Markdown(
    f"**Read this honestly.** {len(at_ceiling)} of the {len(by_model)} models reach the same top test score ({EYES_BASELINES['ceiling']:.1%}), so these pictures cannot say which of them is best. "
    f"Every cross-validation score is at the top too (step 13), so it cannot help either. "
    f"And these are hands drawn by a computer. The real test of the eyes is the live game, with real hands, other people and other light."))
fig = comparison_figure(final_results, "Part 1: the eyes on the official test pictures", baselines={"luck": EYES_BASELINES["luck"]}, labels=LABELS)
top_key = by_model["accuracy"].idxmax()  # the model whose confusion matrix the figure draws: the first one with the top accuracy
tied_note = f" (first of {len(top_models)} tied at the top)" if len(top_models) > 1 else ""
fig.axes[3].set_title(f"Confusion matrix of {top_key}{tied_note}")  # the helper's own title says "best model", but several models tie
plt.show()

# %% [markdown]
# # Part 2: The Brain
#
# The eyes tell the bot which gesture you threw. The brain has a different job: **given the last rounds of a game, predict the next move.**
# It learns from a public rock-paper-scissors study: many players, each playing many rounds. Everything below is computed from the study's public data.
#
# The steps are the ones from our labs: check the data, explore it with plots, build features, split carefully, train and compare models, tune them, and try one new example.

# %%
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, auc, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_curve)
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, cross_val_score, learning_curve
from sklearn.preprocessing import label_binarize

from mindreader.brain import (FEATURES, ModelBrain, PersonalBrain, RandomBrain, add_features, brain_pipeline, decide, history_features,
                              replay_hits)
from mindreader.moves import NAMES, outcome
from mindreader.plots import BASELINE_COLOURS, LINE_STYLES, comparison_figure
from tools.rps_facts import early_window

SHIFT_NAMES = ["stay", "up", "down"]  # the next move relative to the last one: a shift of 0, 1 or 2 (explained in step 5)
SHADES = ["#1b2a49", "#3b7fc4", "#a6cee3"]  # dark to light, so bars still differ in greyscale

# %% [markdown]
# ## 1. Look at the data
# `head`, `tail`, `shape`, `info` and `describe`, the first checks in every lab.
# One row is one round: the player (`id`), the round number (`period`), what they threw (`choice`), what their opponent threw (`other.choice`) and the `result` from the player's side.
# `load_games` also adds `c` and `o`, the same two moves as numbers: rock 0, paper 1, scissors 2.

# %%
games = load_games("data/rps_human_games.csv")
display(games.head())
display(games.tail())
print("shape:", games.shape)
games.info()
display(games.describe())

# %% [markdown]
# ## 2. Nulls, duplicates and class counts
# The file has no missing values. But the brain needs the last two rounds to make a guess, so round 1 (no past at all) and round 2 (only one round of past) of every player cannot be used.
# The class counts show a rock bias: rock is thrown more often than paper or scissors.

# %%
print("missing values in the file:", int(games.isna().sum().sum()))
print("duplicate rows:", int(games.duplicated().sum()))
no_history = games["period"] < 3
print(f"rounds without a 2-round history: {no_history.sum():,} of {len(games):,} (rounds 1 and 2 of each of the {games['id'].nunique()} players)")
display(games["c"].map(NAMES).value_counts().reindex(NAMES.values()).to_frame("throws"))
display(games["result"].value_counts().reindex(["win", "tie", "lose"]).to_frame("rounds"))

# %% [markdown]
# ## 3. Facts the brain can use
# Each fact is printed with `n`, the number of rounds it rests on. The model rows are the rounds that do have a history of 2 or more rounds.
# Two of these are the yardsticks for every brain below: **always-rock** and **always-repeat** (throw your last move again). A brain only counts if it beats them, and it must beat luck too (a random guess, one chance in three).

# %%
FACTS = facts(games)
for label, share, n in [
    ("rock share, all rounds", FACTS["rock_share_all"], len(games)),
    ("first throw was rock", FACTS["first_rock"], FACTS["n_first"]),
    ("repeat right after a loss", FACTS["repeat_after_loss"], FACTS["n_after_loss"]),
    ("always-rock, model rows", FACTS["always_rock_model_rows"], FACTS["n_model_rows"]),
    ("always-repeat, model rows", FACTS["always_repeat_model_rows"], FACTS["n_model_rows"]),
]:
    print(f"{label:27s}{share:6.1%}   n = {n:,}")
print(f"{'luck (random guess)':27s}{1 / 3:6.1%}")

# %% [markdown]
# ## 4. Plots
# First what players throw and how the rounds end, then the habit behind the hook: how often a player repeats their move, depending on how the last round went.

# %%
def labelled_bars(ax, names, values, title, xlabel, ylabel):
    """A bar chart with the count written on each bar."""
    bars = ax.bar(names, values, color=SHADES, edgecolor="black")
    ax.bar_label(bars, labels=[f"{value:,}" for value in values])
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.margins(y=0.12)  # room for the labels


throws = games["c"].map(NAMES).value_counts().reindex(NAMES.values())
results = games["result"].value_counts().reindex(["win", "tie", "lose"])
fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
labelled_bars(left, throws.index, throws.values, "What players threw", "move", "number of rounds")
labelled_bars(right, results.index, results.values, "How the rounds ended (player's side)", "result", "number of rounds")
plt.show()

# %%
previous = games.groupby("id")[["c", "result"]].shift(1)  # the round before each round, from the same player
repeated = games["c"] == previous["c"]
repeat_rate = repeated.groupby(previous["result"]).agg(["mean", "size"]).reindex(["lose", "tie", "win"])

fig, ax = plt.subplots(figsize=(6.5, 4.5))
bars = ax.bar(["after a loss", "after a tie", "after a win"], repeat_rate["mean"] * 100, color=SHADES, edgecolor="black")
ax.bar_label(bars, labels=[f"{share:.1%}\nn = {n:,}" for share, n in zip(repeat_rate["mean"], repeat_rate["size"])])
ax.axhline(100 / 3, linestyle="--", color="black", label=f"luck ({1 / 3:.1%})")
ax.set(title="How often players repeat their move", xlabel="the player's last result", ylabel="repeated the same move (%)", ylim=(0, 65))
ax.legend()
plt.show()

# %% [markdown]
# Heatmaps: for each last result, how often each next move follows each last move. Every row adds up to 100%.
# If players were random, every cell would be about 33%.

# %%
last = games.groupby("id")[["choice", "result"]].shift(1)
words = {"r": "rock", "p": "paper", "s": "scissors"}
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, last_result, title in zip(axes, ["lose", "tie", "win"], ["After a loss", "After a tie", "After a win"]):
    after = last["result"] == last_result
    follows = pd.crosstab(last["choice"][after], games["choice"][after], normalize="index")
    follows = follows.reindex(index=list(words), columns=list(words)).rename(index=words, columns=words)
    sns.heatmap(follows, annot=True, fmt=".0%", cmap="Blues", vmin=0, vmax=0.7, cbar=False, ax=ax)
    ax.set(title=title, xlabel="next move", ylabel="last move")
fig.suptitle("What players throw next, given their last move and last result")
plt.show()

# %% [markdown]
# ## 5. Features
# The brain sees the last two rounds, but not as raw rock, paper and scissors. It gets three small numbers, each coded **relative to the player**:
# - `o_rel1`: how the last round went. It is (opponent move - player move) mod 3, so 1 = the player lost, 2 = the player won, 0 = a tie.
# - `shift1`: how the player moved between the two rounds before: 0 = stayed, 1 = stepped *up* to the move that beats their old move, 2 = stepped *down*.
# - `o_rel2`: how the round before that went, coded like `o_rel1`.
#
# The target `shift` is the same idea for the next move: **stay, up or down**, not rock, paper or scissors.
#
# **Why relative?** Rock-paper-scissors is a circle: turn every move one step and the game is the same. "Stay after a loss" is one habit whether the player lost with rock, paper or scissors, so one relative label learns from all three at once. That is three times the examples for each pattern. To get the real move back, we shift the player's last move.
#
# **Why is `input` dropped?** The study's original file has an `input` column. It only holds the player's previous choice, which we already know from the row before, so it adds nothing. We left it out of the CSV.

# %%
rows = add_features(games)
display(rows.head())
print(f"model rows (rounds with a 2-round history): {len(rows):,}")
counts = rows["shift"].map(dict(enumerate(SHIFT_NAMES))).value_counts().reindex(SHIFT_NAMES)
display(pd.DataFrame({"rows": counts, "share": counts / counts.sum()}).round(3))

# %% [markdown]
# ## 6. Split by player, not by row
# The lab notebooks split rows at random with `train_test_split`. We split by player instead, because the question is whether the brain can predict **new people**.
# Every score below uses `GroupKFold`: all the rounds of one player stay together, on the training side or on the test side, never on both. That is our **evidence of predictability on unseen players**.
# A random split would put rounds of the same person on both sides, which can flatter a model that has learned one person's habits. With only three simple features the effect should be small. The last two lines of the cell measure it with the same Decision Tree.

# %%
X, y, groups = rows[FEATURES], rows["shift"], rows["id"]
for number, (train, test) in enumerate(GroupKFold(n_splits=5).split(X, y, groups), start=1):
    shared = set(groups.iloc[train]) & set(groups.iloc[test])
    print(f"fold {number}: {groups.iloc[train].nunique()} players train, {groups.iloc[test].nunique()} players test, {len(shared)} players on both sides")
random_split = cross_val_score(brain_pipeline("dt"), X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42))
player_split = cross_val_score(brain_pipeline("dt"), X, y, groups=groups, cv=GroupKFold(n_splits=5))
print(f"Decision Tree scored on a random split of the rounds (players on both sides): {random_split.mean():.1%}")
print(f"Decision Tree scored on a split by player (GroupKFold):                       {player_split.mean():.1%}")

# %% [markdown]
# ## 7. Five models, and three yardsticks
# The brain is a scikit-learn `Pipeline` of two steps, like Lab 8: **encode** the three features, then **fit a model**.
# Most models get one-hot columns (a 0/1 column for every value of a feature). Categorical Naive Bayes wants category numbers, so it gets an ordinal encoder.
# - **dt**: Decision Tree (Lab 4). A flow chart of yes/no questions about the features.
# - **svm**: Support Vector Machine with the RBF kernel (Lab 8). Draws a curved boundary between the classes.
# - **logreg**: Logistic Regression (Lab 6). Adds up weighted features into a probability for each class.
# - **nb**: Categorical Naive Bayes. Counts how often each feature value goes with each class. (Lab 7 used the Gaussian kind, for measured numbers; ours are categories.)
# - **perceptron**: Perceptron (Lab 6). A straight-line boundary that is nudged only when it makes a mistake.
#
# Every model is scored the same way, with the 5-fold `GroupKFold` from step 6, so each score is **evidence of predictability on unseen players**.
# `evaluate` makes one pass per model and keeps the prediction for every round, made while that round's player was in the test fold.
# Precision, recall and F1 are macro averages: worked out for stay, up and down separately, then averaged.
#
# The SVM runs here without its `probability` option. It makes the same predictions, but the option makes it much slower. The price: the SVM has no ROC curve below.

# %%
def evaluate(pipeline, X, y, groups):
    """One pass of 5-fold GroupKFold. Returns the accuracy of each fold, and for every round the prediction
    (and the probabilities, if the model has them) made while that round's player was in the test fold."""
    scores = []
    predicted = np.zeros(len(y), dtype=int)
    probabilities = np.zeros((len(y), 3)) if hasattr(pipeline, "predict_proba") else None
    for train, test in GroupKFold(n_splits=5).split(X, y, groups):
        model = clone(pipeline).fit(X.iloc[train], y.iloc[train])
        predicted[test] = model.predict(X.iloc[test])
        scores.append(accuracy_score(y.iloc[test], predicted[test]))
        if probabilities is not None:
            probabilities[test] = model.predict_proba(X.iloc[test])
    return scores, predicted, probabilities


def micro_roc(y, probabilities):
    """The ROC curve over all three classes at once (each class against the rest), or None if the model has no probabilities."""
    if probabilities is None:
        return None
    fpr, tpr, _ = roc_curve(label_binarize(y, classes=[0, 1, 2]).ravel(), probabilities.ravel())
    return fpr, tpr, auc(fpr, tpr)


def result_row(name, scores, predicted, probabilities, y):
    """One row of the comparison table, worked out from the predictions made on players the model had not seen."""
    return {
        "model": name,
        "accuracy": accuracy_score(y, predicted),
        "precision": precision_score(y, predicted, average="macro", zero_division=0),
        "recall": recall_score(y, predicted, average="macro", zero_division=0),
        "f1": f1_score(y, predicted, average="macro", zero_division=0),
        "cv_scores": scores,
        "roc": micro_roc(y, probabilities),
        "cm": confusion_matrix(y, predicted, labels=[0, 1, 2]).tolist(),
    }


# %%
out_of_fold = {}  # model name -> (predictions, probabilities or None), kept for the metrics below
table_rows = []
for name in ["dt", "svm", "logreg", "nb", "perceptron"]:
    pipeline = brain_pipeline(name)
    if name == "svm":
        pipeline.set_params(model__probability=False)  # see the note above
    scores, predicted, probabilities = evaluate(pipeline, X, y, groups)
    out_of_fold[name] = (predicted, probabilities)
    table_rows.append(result_row(name, scores, predicted, probabilities, y))
RESULTS = pd.DataFrame(table_rows)
BASELINES = {"luck": 1 / 3, "always-rock": FACTS["always_rock_model_rows"], "always-repeat": FACTS["always_repeat_model_rows"]}
BEST_BRAIN = RESULTS.loc[RESULTS["accuracy"].idxmax(), "model"]  # the highest accuracy; the first one on a tie

# %% [markdown]
# ## 8. Comparison table and bar chart
# Everything in percent, on the same rounds (the model rows). The three yardsticks have an accuracy only, since they are not trained models:
# **luck** guesses at random, **always-rock** always predicts rock, **always-repeat** always predicts the player's last move again.
#
# Why do dt and svm score the same? The three features can only make 3 x 3 x 3 = 27 different situations. The line under the table checks a guess: that both models just predict, for each situation, the shift that came next most often among the training players. That is a plain **lookup table**, rebuilt in every fold from the training players only.
# If the lookup table gives the same answers, a fancier model has nothing more to find in these features.

# %%
def lookup_table_predictions(situation, y, groups):
    """The plain table: in each fold, predict for every situation the shift that came next most often among the training players."""
    predicted = np.zeros(len(y), dtype=int)
    for train, test in GroupKFold(n_splits=5).split(situation, y, groups):
        counts = pd.crosstab(situation.iloc[train], y.iloc[train])  # rows: situations, columns: stay, up, down
        most_common = counts.idxmax(axis=1)  # a tie goes to the lowest shift
        overall = y.iloc[train].mode()[0]  # for a situation the training players never showed
        predicted[test] = most_common.reindex(situation.iloc[test].to_numpy()).fillna(overall).astype(int).to_numpy()
    return predicted


table = RESULTS.set_index("model")[["accuracy", "precision", "recall", "f1"]]
table = pd.concat([table, pd.DataFrame({"accuracy": BASELINES})])
print((table * 100).round(1).to_string(na_rep="-"))
situation = X["o_rel1"] * 9 + X["shift1"] * 3 + X["o_rel2"]  # one number for each combination of the three features
lookup = lookup_table_predictions(situation, y, groups)
dt_predicted = out_of_fold["dt"][0]
print(f"\ndt and svm made the same prediction on {(dt_predicted == out_of_fold['svm'][0]).mean():.1%} of the rounds; "
      f"the {situation.nunique()}-situation lookup table agrees with dt on {(lookup == dt_predicted).mean():.1%} and scores {(lookup == y).mean():.1%} itself")

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(RESULTS["model"], RESULTS["accuracy"] * 100, color=SHADES[1], edgecolor="black")
ax.bar_label(bars, labels=[f"{accuracy:.1%}" for accuracy in RESULTS["accuracy"]])
for (name, value), colour in zip(BASELINES.items(), BASELINE_COLOURS):
    ax.axhline(value * 100, linestyle="--", color=colour, label=f"{name} ({value:.1%})")
ax.set(title="Accuracy on players the model never saw", xlabel="model", ylabel="accuracy (%)", ylim=(0, 60))
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.show()

# %% [markdown]
# ## 9. Metrics for the best model
# The **confusion matrix** counts, for each true shift (rows), what the model predicted (columns). The **classification report** gives precision, recall and F1 for each shift.
# Then the hook again: **accuracy right after a loss**, on exactly those rounds, next to the yardsticks on the same rounds.

# %%
predicted = out_of_fold[BEST_BRAIN][0]
print("best model:", BEST_BRAIN)
ConfusionMatrixDisplay.from_predictions(y, predicted, display_labels=SHIFT_NAMES, cmap="Blues", colorbar=False)
plt.title(f"Confusion matrix of {BEST_BRAIN}: the next shift")
plt.show()
print(classification_report(y, predicted, target_names=SHIFT_NAMES, zero_division=0))

after_loss = (rows["o_rel1"] == 1).to_numpy()  # o_rel1 = 1: the player lost the last round
shares = {
    BEST_BRAIN: accuracy_score(y[after_loss], predicted[after_loss]),
    "always-repeat": (y[after_loss] == 0).mean(),
    "always-rock": (rows["choice"][after_loss] == "r").mean(),
    "luck": 1 / 3,
}
print(f"accuracy right after a loss (n = {after_loss.sum():,}):")
print(f"  ({after_loss.sum():,} of the {FACTS['n_after_loss']:,} in the hook: the model needs 2 rounds of history)")
for name, share in shares.items():
    print(f"  {name:14s}{share:6.1%}")

# %% [markdown]
# **ROC curves.** For each shift in turn, the curve for "this shift against the other two" (that is what `label_binarize` sets up).
# The dotted line is chance; the more a curve bows towards the top left, the better. This is the best model that has probabilities.

# %%
with_probabilities = RESULTS[RESULTS["roc"].notna()]
roc_model = with_probabilities.loc[with_probabilities["accuracy"].idxmax(), "model"]
shift_onehot = label_binarize(y, classes=[0, 1, 2])
fig, ax = plt.subplots(figsize=(5.5, 5))
for k, shift_name in enumerate(SHIFT_NAMES):
    fpr, tpr, _ = roc_curve(shift_onehot[:, k], out_of_fold[roc_model][1][:, k])
    ax.plot(fpr, tpr, linestyle=LINE_STYLES[k], label=f"{shift_name} vs the rest (AUC {auc(fpr, tpr):.2f})")
ax.plot([0, 1], [0, 1], color="grey", linestyle=":", label="chance")
ax.set(title=f"ROC curves of {roc_model}", xlabel="false positive rate", ylabel="true positive rate")
ax.legend(loc="lower right")
plt.show()

# %% [markdown]
# ## 10. Tuning with GridSearchCV
# `GridSearchCV` tries every setting in a grid, scores each one with the same 5-fold `GroupKFold`, and keeps the best. Then it refits that best setting on all the rounds it was given.
# - **dt**: how deep the tree may grow, and the smallest number of rounds allowed in a leaf.
# - **logreg**: `C`, how loosely the model may fit the training rounds (bigger is looser).
# - **svm**: `C` too. The SVM is slow, so it is tuned and fitted on a small group of players only (the first ones in the file). The printed line says how many players and rounds.
#
# Each line shows the best setting, its accuracy as **mean ± std** over the folds, and how many settings tie for best (a tie means the grid cannot tell them apart). Under them are the three yardsticks **on the same rounds**.
# The best setting is picked and scored on the same folds, so its score is slightly optimistic.
# The SVM line cannot be compared with the others: it is scored on that small group only, in very small folds, so it is noisier, and its yardsticks are different.

# %%
GRIDS = {
    "dt": {"model__max_depth": [3, 4, 5, 6, 8], "model__min_samples_leaf": [1, 20, 50]},
    "logreg": {"model__C": [0.1, 1, 10]},
    "svm": {"model__C": [0.5, 1, 2]},
}


def print_yardsticks(label, subset):
    """Luck, always-rock and always-repeat on the same rounds as the score printed above them."""
    shares = {"luck": 1 / 3, "always-rock": (subset["choice"] == "r").mean(), "always-repeat": (subset["shift"] == 0).mean()}
    print(f"  yardsticks on {label}: " + ", ".join(f"{name} {share:.1%}" for name, share in shares.items()))


def tune(name, pipeline, X, y, groups, note=""):
    """GridSearchCV with the 5-fold GroupKFold. Prints the best settings and their accuracy as mean ± std over the folds."""
    search = GridSearchCV(pipeline, GRIDS[name], cv=GroupKFold(n_splits=5), scoring="accuracy").fit(X, y, groups=groups)
    best, ranks = search.best_index_, search.cv_results_["rank_test_score"]
    mean, std = search.cv_results_["mean_test_score"][best], search.cv_results_["std_test_score"][best]
    print(f"{name}: best {search.best_params_}, accuracy {mean:.1%} ± {std:.1%} over 5 folds; {(ranks == 1).sum()} of {len(ranks)} settings tie for best{note}")
    return search


search_dt = tune("dt", brain_pipeline("dt"), X, y, groups)
search_logreg = tune("logreg", brain_pipeline("logreg"), X, y, groups)
print_yardsticks(f"all {len(rows):,} rounds", rows)

first_players = sorted(rows["id"].unique())[:20]
small = rows[rows["id"].isin(first_players)]
note = f" (tuned on the first {len(first_players)} players, {len(small):,} rounds)"
search_svm = tune("svm", brain_pipeline("svm").set_params(model__probability=False), small[FEATURES], small["shift"], small["id"], note)
print_yardsticks(f"those {len(small):,} rounds", small)
# the live SVM keeps its probability option: the game shows that percentage next to the move
live_svm = brain_pipeline("svm").set_params(model__C=search_svm.best_params_["model__C"]).fit(small[FEATURES], small["shift"])

# %% [markdown]
# **The brains.** These are what the live game plays against. Each model is wrapped in `ModelBrain`: it turns a round history into the three features, asks the model for the likeliest next shift, and turns that into a move.
# `personal` learns only from the current player's own game, and `random` guesses (the luck yardstick). `BEST_BRAIN` is the model with the highest accuracy in step 8.
#
# **Careful with `svm`.** The SVM in the comparison table uses all the players and has no probabilities. The `svm` brain that plays live is a different fit: the first few players only (that is how it stays quick), with the `C` picked in step 10, and with probabilities so the game can show a percentage.
# So the table's SVM score does not describe it. The last lines of the cell score it on the players it never saw, next to always-repeat on those same rounds.

# %%
BRAINS = {
    "dt": ModelBrain(search_dt.best_estimator_, "dt"),
    "logreg": ModelBrain(search_logreg.best_estimator_, "logreg"),
    "svm": ModelBrain(live_svm, "svm"),
    "nb": ModelBrain(brain_pipeline("nb").fit(X, y), "nb"),
    "perceptron": ModelBrain(brain_pipeline("perceptron").fit(X, y), "perceptron"),
    "personal": PersonalBrain(),
    "random": RandomBrain(),
}
print("brains ready:", ", ".join(BRAINS))
print("best brain:", BEST_BRAIN)

other_players = rows[~rows["id"].isin(first_players)]  # the players the live svm never saw
likeliest = live_svm.classes_[live_svm.predict_proba(other_players[FEATURES]).argmax(axis=1)]  # the game plays the likeliest shift from the probabilities
LIVE_SVM = {"accuracy": float((likeliest == other_players["shift"]).mean()), "always_repeat": float((other_players["shift"] == 0).mean()), "n": len(other_players)}
print(f"live svm (fitted on {len(first_players)} players), scored on the other {other_players['id'].nunique()} players ({LIVE_SVM['n']:,} rounds): "
      f"accuracy {LIVE_SVM['accuracy']:.1%}, always-repeat {LIVE_SVM['always_repeat']:.1%}")

# %% [markdown]
# ## 11. Cross-validation: mean ± std
# The five scores of step 7 are cross-validation already (`cross_val_score` in the labs does the same). Here is each model's mean and spread over the folds, each fold being a different group of players the model had not seen.
# A small std means the model does about equally well on every group of players.
# Compare the dt and logreg lines with the tuned lines in step 10: where they match, tuning found nothing better than the default settings.
# Always-repeat is scored on the same test players, fold by fold. A model only really beats it if it is ahead in most folds, not just on average, so the last line counts the folds where dt is ahead.

# %%
for row in RESULTS.itertuples():
    scores = np.array(row.cv_scores)
    print(f"{row.model}: {scores.mean():.1%} ± {scores.std():.1%} over 5 folds")
print_yardsticks(f"the same {len(rows):,} rounds", rows)
repeat_folds = [(y.iloc[test] == 0).mean() for _, test in GroupKFold(n_splits=5).split(X, y, groups)]  # always-repeat on each test fold
dt_folds = RESULTS.set_index("model").loc["dt", "cv_scores"]
print(f"always-repeat: {np.mean(repeat_folds):.1%} ± {np.std(repeat_folds):.1%} over 5 folds (the same test players)")
print(f"dt beats always-repeat in {sum(dt > repeat for dt, repeat in zip(dt_folds, repeat_folds))} of {len(repeat_folds)} folds")

# %% [markdown]
# ## 12. Learning curve: how many players does the brain need?
# Train the Decision Tree on the first few players, then more and more, and score it each time on players it has not seen.
# If the unseen-player line were still climbing at the end, more players would help. The dashed lines are the yardsticks.

# %%
rows_per_player = len(rows) // rows["id"].nunique()
sizes, train_scores, unseen_scores = learning_curve(
    brain_pipeline("dt"), X, y, groups=groups, cv=GroupKFold(n_splits=5), train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0], shuffle=False)
players = sizes / rows_per_player  # the rows are sorted by player, so each training set is the first few whole players
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(players, train_scores.mean(axis=1) * 100, marker="o", linestyle="-", color="#1b2a49", label="players it trained on")
ax.plot(players, unseen_scores.mean(axis=1) * 100, marker="s", linestyle="-.", color="#e69f00", label="unseen players")
for (name, value), colour in zip(BASELINES.items(), BASELINE_COLOURS):
    ax.axhline(value * 100, linestyle="--", color=colour, label=f"{name} ({value:.1%})")
ax.set(title="Learning curve of the Decision Tree", xlabel="number of training players", ylabel="accuracy (%)")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.show()
for count, seen, unseen in zip(players, train_scores.mean(axis=1), unseen_scores.mean(axis=1)):
    print(f"{count:5.1f} training players: {seen:.1%} on them, {unseen:.1%} on unseen players")
gain = 100 * (unseen_scores.mean(axis=1)[-1] - unseen_scores.mean(axis=1)[0])
print(f"on unseen players: {gain:+.1f} points from {players[0]:.1f} to {players[-1]:.1f} training players")

# %% [markdown]
# ## 13. Why the Perceptron is the odd one out
# The Perceptron changes its line only when it gets a round wrong. Here the answers are noisy and overlap: the same situation is followed by different shifts, from different people. So it is always getting some rounds wrong, its line keeps moving, and it **never settles**.
# Where it ends up depends on the order it sees the rows, which the random seed sets. Below, the same Perceptron with several different seeds, next to Logistic Regression. Logistic Regression looks at all the rounds at once and settles on one answer, so it gives the same score every time.

# %%
seed_means = []
for seed in range(5):
    scores = cross_val_score(brain_pipeline("perceptron").set_params(model__random_state=seed), X, y, groups=groups, cv=GroupKFold(n_splits=5))
    seed_means.append(scores.mean())
    print(f"perceptron seed {seed}: {scores.mean():.1%} ± {scores.std():.1%} over 5 folds")
logreg_mean = float(np.mean(RESULTS.set_index("model").loc["logreg", "cv_scores"]))
print(f"logreg (any seed): {logreg_mean:.1%}")

fig, ax = plt.subplots(figsize=(7.5, 4.5))
bars = ax.bar([f"seed {seed}" for seed in range(5)] + ["logreg"], np.array(seed_means + [logreg_mean]) * 100,
              color=[SHADES[2]] * 5 + [SHADES[0]], edgecolor="black")
ax.bar_label(bars, labels=[f"{mean:.1%}" for mean in seed_means + [logreg_mean]])
for (name, value), colour in zip(BASELINES.items(), BASELINE_COLOURS):
    ax.axhline(value * 100, linestyle="--", color=colour, label=f"{name} ({value:.1%})")
ax.set(title=f"Perceptron with {len(seed_means)} seeds, next to Logistic Regression", xlabel="model", ylabel="accuracy (%)", ylim=(0, 60))
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.show()

# %% [markdown]
# ## 14. The early rounds
# The accuracy above is an average over all the rounds of unseen players. A real game starts with no history at all, so the first rounds matter too.
# Two brains are shown, each next to the yardsticks **on the same rounds**:
# - **Personal** learns only from this one player's own game: a small table of counts, one cell per kind of last round and next shift, every cell starting at 1. It needs no training data and works from round 2. Here it is replayed round by round over all the players.
# - The **Decision Tree** is the one from step 7, using its predictions on unseen players. It needs two rounds of history, so it starts at round 3.

# %%
def early_numbers(last_round):
    """The numbers for the first rounds, printed with their n and returned for metrics.json."""
    rock, repeat, personal, n = early_window(games, last_round)
    in_window = (rows["period"] <= last_round).to_numpy()  # the model rows start at round 3
    tree = (out_of_fold["dt"][0][in_window] == y[in_window]).mean()
    print(f"rounds 2-{last_round}, n = {n:,}: luck {1 / 3:.1%} | always-rock {rock:.1%} | always-repeat {repeat:.1%} | personal {personal:.1%}")
    print(f"rounds 3-{last_round}, n = {in_window.sum():,}: luck {1 / 3:.1%} | always-rock {(rows['choice'][in_window] == 'r').mean():.1%} | "
          f"always-repeat {(y[in_window] == 0).mean():.1%} | decision tree {tree:.1%}")
    return {"always_rock": float(rock), "always_repeat": float(repeat), "personal": float(personal), "n": int(n),
            "dt": float(tree), "n_dt": int(in_window.sum())}


EARLY = {f"2-{last_round}": early_numbers(last_round) for last_round in (10, 20)}

# %% [markdown]
# ## 15. How much does the Personal brain gain as the game goes on?
# Linear Regression (Lab 2). Each dot is the share of players the Personal brain guessed right in that round. The straight line is a Linear Regression of "right" (100) or "wrong" (0) on the round number. Its slope is how many points the score rises for every 100 more rounds.
# But that whole slope is not the brain's own gain: players also get more repetitive as a game goes on, so even **always-repeat** scores higher in later rounds. So we fit the same straight line to always-repeat's hits on the same rounds, and subtract.
# The difference is how much more the Personal brain gains than the always-repeat yardstick does. The flat dashed lines are the other two yardsticks over the same rounds.

# %%
def slope_per_100(hits):
    """Fit a straight line to 'right' (100) or 'wrong' (0) against the round number. Returns the line and its slope in points per 100 rounds."""
    line = LinearRegression().fit(hits[["round"]], hits["hit"] * 100)
    return line, float(line.coef_[0]) * 100


hits = replay_hits(games, PersonalBrain(), games["period"].max())
later = games["period"] >= 2  # the same rounds the Personal brain was scored on
last_move = games.groupby("id")["c"].shift(1)  # the player's previous move
repeat_hits = pd.DataFrame({"round": games["period"][later], "hit": (games["c"] == last_move)[later].astype(int)})  # always-repeat is right when the move is the previous one
line, SLOPE_PER_100 = slope_per_100(hits)
repeat_line, REPEAT_SLOPE_PER_100 = slope_per_100(repeat_hits)
print(f"Personal brain: {SLOPE_PER_100:+.1f} points per 100 rounds (n = {len(hits):,} predictions)")
print(f"always-repeat:  {REPEAT_SLOPE_PER_100:+.1f} points per 100 rounds (the same {len(repeat_hits):,} rounds)")
print(f"Personal's own gain over always-repeat: {SLOPE_PER_100 - REPEAT_SLOPE_PER_100:+.1f} points per 100 rounds")

flat_yardsticks = {"luck": 1 / 3, "always-rock": (games["c"] == 0)[later].mean()}
by_round = hits.groupby("round")["hit"].mean() * 100
rounds = by_round.index.to_frame(name="round")
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.scatter(by_round.index, by_round, s=10, color=SHADES[2], edgecolor="black", linewidth=0.3, label=f"Personal brain, {games['id'].nunique()} players")
ax.plot(rounds["round"], line.predict(rounds), color=SHADES[0], label=f"Personal brain, straight line: {SLOPE_PER_100:+.1f} points per 100 rounds")
ax.plot(rounds["round"], repeat_line.predict(rounds), linestyle="--", color=BASELINE_COLOURS[2], label=f"always-repeat, straight line: {REPEAT_SLOPE_PER_100:+.1f} points per 100 rounds")
for (name, value), colour in zip(flat_yardsticks.items(), BASELINE_COLOURS):
    ax.axhline(value * 100, linestyle="--", color=colour, label=f"{name} ({value:.1%})")
ax.set(title="Personal brain and always-repeat both rise as the game goes on", xlabel="round number", ylabel="rounds guessed right (%)")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.show()

# %% [markdown]
# ## 16. One new example
# A player has thrown two rounds so far: round 1, rock against scissors (a win), and round 2, rock against paper (a loss).
# We turn that history into the three features and ask the best brain what comes next. CHANAKYA then plays the move that beats the prediction.

# %%
history = [(0, 2), (0, 1)]  # (player's move, CHANAKYA's move): rock vs scissors, then rock vs paper
for number, (player, bot) in enumerate(history, start=1):
    print(f"round {number}: you threw {NAMES[player]}, CHANAKYA threw {NAMES[bot]}: {outcome(player, bot)}")
display(pd.DataFrame([history_features(history)]))
decision = decide(history, BEST_BRAIN, BRAINS)
how_sure = "" if decision["prob"] is None else f" (probability {decision['prob']:.0%})"
print(f"the {decision['source']} brain expects you to throw {NAMES[decision['predicted_move']]} next{how_sure}")
print(f"CHANAKYA plays {NAMES[decision['bot_move']]}, the move that beats it")

# %% [markdown]
# ## 17. All five models, side by side
# The results table, then the comparison figure. Everything is scored on players the models had not seen.
# - **Top left:** accuracy, precision, recall and F1 as bars. The dashed lines are **accuracy** yardsticks: compare them with the accuracy bars only, not with precision, recall or F1.
# - **Top right:** the accuracy of each cross-validation fold, one dot per fold, with the same dashed yardsticks (on all the rounds). A short box means a steady model.
# - **Bottom left:** ROC curves, for the models that give probabilities (the SVM here has none, and the Perceptron never does). Each curve is the micro-average over stay, up and down; the curve and AUC of each shift on its own are in step 9.
# - **Bottom right:** the confusion matrix of the best model.

# %%
summary = RESULTS.set_index("model")[["accuracy", "precision", "recall", "f1"]].copy()
summary["cv mean"] = [np.mean(scores) for scores in RESULTS["cv_scores"]]
summary["cv std"] = [np.std(scores) for scores in RESULTS["cv_scores"]]
print((pd.concat([summary, pd.DataFrame({"accuracy": BASELINES})]) * 100).round(1).to_string(na_rep="-"))
fig = comparison_figure(RESULTS, "Part 2: five brains on unseen players, next to three yardsticks", BASELINES, labels=SHIFT_NAMES)
plt.show()

# %%
METRICS["brain_table"] = {row.model: float(row.accuracy) for row in RESULTS.itertuples()}
METRICS["brain_table"].update(luck=BASELINES["luck"], always_rock=BASELINES["always-rock"], always_repeat=BASELINES["always-repeat"])
METRICS["brain_early"] = EARLY
METRICS["personal_slope_per_100"] = SLOPE_PER_100
METRICS["repeat_slope_per_100"] = REPEAT_SLOPE_PER_100
METRICS["brain_table_live_svm"] = LIVE_SVM
METRICS["facts"] = FACTS
os.makedirs("build", exist_ok=True)
with open("build/metrics.json", "w") as file:
    json.dump(METRICS, file, indent=2)
