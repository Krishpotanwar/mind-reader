# Mind-Reader viva guide

How to use it. **A** explains the ideas in plain English. **B** walks the notebook cell by cell: the 66 code cells of `notebook/MindReader.py` (the source of the `.ipynb`), numbered in order, markdown cells not counted. **C** lists every helper in `mindreader/`. **D** has likely questions with short answers. **E** has the 5-minute run sheet and the what-if-it-fails playbook.

**The numbers.** Every figure here is printed by the executed notebook or is in `build/metrics.json`. A brain figure always travels with its yardsticks: luck 33.3%, always-rock 35.6%, always-repeat 44.0% (n = 21,456 rounds).
- After losing a round, people threw the same move again 52.7% of the time (n = 7,148), against 33.3% for a random player. 58.3% of first throws were rock (n = 72). Both are computed from the study's public data.
- Eyes, 372 official test pictures: four of six models score 99.2% (the ceiling, because 3 pictures have no hand found), dt 94.4%, nb 89.5%; luck 33.3%. These are computer-drawn hands.
- Brain on unseen players (luck 33.3%, always-rock 35.6%, always-repeat 44.0%): dt 45.5%, svm 45.5%, nb 44.8%, logreg 44.7%, perceptron 37.9%. The best is 1.5 points above always-repeat. That is evidence of predictability on unseen players, nothing more.

**Still to be checked, so claim nothing yet:** the live eyes test with real hands (gate G2), the pilot with classmates (gate G3), and the timed "Run all" on a fresh Colab, whose target is 5:00 or less (gate G4).

## A. Primers

### Find-S (Lab 3)
Find-S learns the most specific rule that fits every positive example. It starts with a rule that matches nothing, copies the first positive example, and for each later positive example turns any attribute that differs into `?` (any value). Negative examples are ignored. Toy example, not from the notebook: positives (red, round) and (red, square) give (red, ?). In the notebook an example is four finger states (index, middle, ring, pinky) and positive means scissors. The clean run ends at (up, up, down, down); the noisy run ends at (up, up, ?, ?). One disagreeing example widens the rule for good, so Find-S needs noise-free data (Cells 10 and 11).

### Decision trees and entropy (Lab 4)
A decision tree is a flow chart of yes/no questions, each about one number. At every box it picks the question that leaves the answers purest. Purity is measured by entropy, minus the sum of p times log2(p) over the classes: 0 when a box holds a single class, highest when the classes are evenly mixed. Information gain is the entropy before a question minus the weighted entropy after it. Gini is the other measure and scikit-learn's default; the grid in Cell 26 tries both. A tree left to grow can memorise, so `max_depth` and `min_samples_leaf` hold it back.

### Naive Bayes (Lab 7)
Bayes' rule scores each class: how common the class is, times how likely each feature value is inside that class. "Naive" means the features are treated as unrelated once the class is known. Gaussian Naive Bayes (the eyes) learns the average and spread of every number for each gesture. Categorical Naive Bayes (the brain) counts how often each feature value goes with each class. The eyes' `nb` is the weakest model (89.5%); the notebook points out that some numbers clearly move together (the heatmap in Cell 15), which "unrelated" ignores.

### The perceptron (Lab 6), and why it never settles (D27)
A perceptron is a straight-line boundary. It reads rows one at a time: right guess, no change; wrong guess, the line is nudged toward the right answer. If a straight line can split the classes perfectly, it stops. In the brain data the same situation is followed by different next moves from different people, so some rows are always wrong, the line keeps moving, and where it ends depends on the row order (the random seed). Cell 50: five seeds give 36.5%, 40.5%, 39.2%, 39.9% and 41.0%, while logistic regression, which weighs all rows at once, gives 44.7% every time. Yardsticks: luck 33.3%, always-rock 35.6%, always-repeat 44.0%.

### Logistic regression (Lab 6)
Despite the name it classifies. It gives each feature a weight, adds them up and turns the total into a probability for each class. It looks at all rows together, so it settles on one answer. Its `C` works like the SVM's (below). In the notebook it is the live eyes model (Cell 22) because it gives `predict_proba` and ties for the top test score.

### SVMs and kernels (Lab 8)
A support vector machine draws the boundary with the widest empty gap on each side; the rows that touch the gap are the support vectors. The linear kernel gives a straight boundary. The RBF kernel gives a curved one by comparing rows through a similarity score (the kernel trick) instead of building new columns. `C` is the price of a wrong training row: a big C bends the boundary to fit more rows (less holding back, more risk of memorising), a small C keeps the gap wide and tolerates a few errors. The notebook words this as "bigger is looser". `gamma` is how far one row's influence reaches. `probability=True` is slower but lets the SVM show a percentage.

### Train/test split and stratify
Keep some rows out of training so the score reflects pictures the model has not seen. Eyes: TensorFlow's own train and test pictures; no model is fitted or tuned on the test pictures. Inside the training pictures, `train_test_split(test_size=0.2, stratify=y, random_state=42)` makes a fit part (80%) and a validation part (20%): 1,925 and 482 pictures. `stratify=y` keeps each gesture's share the same in both parts. `random_state=42` makes the shuffle repeatable.

### GroupKFold and leakage
Leakage is information from the held-out side reaching the training side, so the score flatters the model. Near-copies cause it: a mirrored twin, a very similar picture, other rounds of the same player. `GroupKFold` takes a group number per row and keeps each group wholly on one side of every split. Eyes: a picture and its twin share a group. Brain: all rounds of one player share a group, and Cell 39 prints 0 players on both sides in every fold. On the same Decision Tree a random split scored 46.1% and the split by player 45.5% (luck 33.3%, always-rock 35.6%, always-repeat 44.0%). The leak is small here because there are only three simple features.

### Mirror twins
A mirrored hand is the same hand seen in a mirror (every `x` becomes `1 - x`), and a mirrored rock is still rock. So every fit picture also gets a twin, and the model never needs to know which hand it sees. Twins are added after the split, only to the fit part, and share their picture's group. Live, `predict_gesture` scores the hand and its mirror image and averages the two answers, so a left and a right hand get the same answer.

### Scaling
`StandardScaler` subtracts each column's mean and divides by its spread, so every column has mean 0 and spread 1. Without it, a column with big values dominates models that use distances or gradients (SVM, logistic regression, perceptron). Cell 9 shows the five reach columns already differ in spread (std 0.105 for the thumb, 0.263 for the ring finger). Inside a Pipeline the scaler learns its mean and spread from the training rows only.

### One-hot encoding
A category such as `o_rel1` (0 tie, 1 lost, 2 won) is a label, not a size: 2 is not "twice" 1. One-hot makes one 0/1 column per value, so "lost" becomes 0, 1, 0. `handle_unknown="ignore"` stops a new value from crashing the model. Categorical Naive Bayes wants category numbers instead, so it gets an `OrdinalEncoder`.

### Accuracy, precision, recall and F1
Accuracy is the share of answers that are right. For one class, precision asks "of the rounds I called X, how many were X?", recall asks "of the real X, how many did I find?", and F1 balances the two. A macro average works each class out separately, then averages, so small classes count equally. Example from Cell 44: for `stay`, precision 0.48 and recall 0.75, so the tree finds most stays but is often wrong when it says stay. Accuracy alone can mislead: `stay` is 44.0% of the rows, and that is exactly what always-repeat predicts.

### The confusion matrix
Rows are the true class, columns what the model said. The diagonal is correct; everything else shows which classes get mixed up. Cell 23 (eyes): all 369 test pictures with a hand sit on the diagonal. Cell 44 (brain): the tree says `stay` often and finds few `up` or `down` (recall 0.24 and 0.20).

### ROC and AUC
A model gives probabilities; a threshold turns them into yes/no. Sweep the threshold and plot the true positive rate against the false positive rate: that is the ROC curve. The diagonal is chance (AUC 0.5); a perfect model hugs the top left (AUC 1.0). With three classes each gets a one-vs-rest curve (that is what `label_binarize` sets up), plus a micro-average over all three. Cell 24 prints AUC 1.000 for every gesture on the computer-drawn test pictures.

### GridSearchCV
It tries every combination in a grid of settings, scores each with k-fold cross-validation, keeps the best and refits it. The notebook prints the best setting, its mean ± std, and how many settings tie. The best setting is picked and scored on the same folds, so its score is slightly optimistic. Cell 26 (eyes): 8 of 12 SVM settings tie for best. Cell 46 (brain): 6 of 15 tree settings tie.

### Cross-validation
Split the rows into k folds (5 here), train on four, score on the fifth, repeat until every fold has had its turn, then report mean ± std. A small std means every fold gave about the same score. In the brain each fold is a different group of players, for example `dt: 45.5% ± 1.4% over 5 folds` (Cell 48; luck 33.3%, always-rock 35.6%, always-repeat 44.0%).

### Learning curves
Train on a growing share of the rows and score each time on the training rows and on held-out rows. If the held-out line is still climbing at the end, more data would help. Brain (Cell 49): 44.8% on unseen players with 5.7 training players and 45.5% with 57.0, nearly flat (yardsticks: luck 33.3%, always-rock 35.6%, always-repeat 44.0%).

### MediaPipe as a feature extractor
MediaPipe's pretrained Hand Landmarker (from Google) looks at a picture and returns 21 points on one hand, each with x, y and z. We do not train it. It only turns a picture into numbers, and our scikit-learn models learn from those numbers. `HandReader` wraps it with `num_hands=1`, and the model file (about 8 MB) is checked against a sha256 fingerprint.

### OpenCV's role
OpenCV does the picture plumbing, no learning. `cv2.imdecode` turns a camera JPEG into an image, `cv2.cvtColor` swaps BGR to RGB for MediaPipe, `cv2.line` and `cv2.circle` draw the hand skeleton, and `cv2.imencode` turns the picture back into a JPEG for the page. The one-time script that built `hand_landmarks.csv` also decodes the pictures with OpenCV.

## B. The notebook, cell by cell

Each section gives the source lines of `notebook/MindReader.py`, what the cell does and why, the key lines in backticks, and what the examiner is likely to ask. Cells of 10 lines or fewer are explained line by line; longer cells block by block. Figures come from the executed notebook.

### Cell 1: Setup: project, packages, hand model
Source 18-49. Gets the project and the hand model ready, on Colab or a laptop.
- `if "google.colab" in sys.modules:` clones the repo once (`git clone --depth 1 --branch BRANCH`), moves in, runs `pip install -q mediapipe==1.1.1`, and prints `RESTART NEEDED` if pip changed numpy after it was loaded.
- `elif os.path.basename(os.getcwd()) == "notebook": os.chdir("..")` steps up to the project root on a laptop.
- `sys.path.insert(0, os.getcwd())` lets `import mindreader` work.
- `warnings.filterwarnings("ignore", ...)` hides one scikit-learn 1.9 FutureWarning about `SVC(probability=True)`.
- `ensure_model("data/raw/hand_landmarker.task")` downloads the hand model (about 8 MB) if missing and checks its sha256.

**Ask:** why pin `mediapipe==1.1.1` and check a sha256? Same library every run, and a corrupt or swapped model file is deleted, never used.

### Cell 2: Imports and versions
Source 55-67. Imports the tools and prints versions so a result can be reproduced.
- `import json`: saves the metrics later.
- `import matplotlib.pyplot as plt`, `numpy as np`, `pandas as pd`, `seaborn as sns`: plots, arrays, tables, nicer plots.
- `from IPython.display import Markdown, display`: show formatted text from code.
- `from mindreader.brain import facts, load_games`: our own helpers.
- `METRICS = {}`: collects key numbers; Cell 55 writes them to `build/metrics.json`.
- `packages = (...)`: the libraries to report.
- `print("Python", ...)`: prints Python 3.12.13, scikit-learn 1.9.1, mediapipe 1.1.1 and the rest.

**Ask:** why print versions? Colab and laptops differ, and results can shift between versions.

### Cell 3: The hook: do people repeat after a loss?
Source 74-80. Counts the habit before any model exists.
- `games = load_games("data/rps_human_games.csv")`: reads the 2014 study (72 players, 300 rounds each), sorted, with moves also as numbers 0, 1, 2.
- `hook = facts(games)`: the headline numbers, each with its n.
- `display(Markdown(f"..."))`: prints that after a loss people threw the same move again 52.7% of the time (n = 7,148) against 33.3% for a random player. The 52.7% is computed from the study's public data.

**Ask:** is 52.7% your result? No, it is computed from the study's public data (Wang, Xu & Zhou, 2014). It is the habit the brain tries to learn.

### Cell 4: Results at a glance (saved snapshot)
Source 89-96. Draws the one-picture summary before the details.
- `from mindreader.plots import glance_figure`: the shared plot helper.
- `if os.path.exists("notebook/metrics_snapshot.json"):` Part 2 has not run yet at this point, so it uses numbers saved by the last full build.
- `fig = glance_figure(json.load(snapshot_file))`: two bar panels, eyes next to luck, brain next to luck and always-repeat.
- `plt.show()`: shows it. `else: print("No saved results yet...")` is the fallback.

**Ask:** where do the bars come from if Part 2 has not run? From a JSON saved by the previous build. Cell 64 redraws the same picture from this run's own numbers.

### Cell 5: Part 1 imports and constants
Source 108-127. Brings in everything Part 1 uses.
- scikit-learn imports: the models (`GaussianNB`, `DecisionTreeClassifier`, `LogisticRegression`, `Perceptron`, `SVC`), `Pipeline`, `StandardScaler`, the metrics, and `GridSearchCV`, `GroupKFold`, `learning_curve`.
- our helpers: `find_s`, `hand_features`, `predict_gesture`, `comparison_figure`.
- `IMAGES = {"train": 2520, "test": 372}`: the picture counts in the two zips. The CSV only holds pictures where a hand was found.
- `LABELS = ["paper", "rock", "scissors"]`: alphabetical, the order scikit-learn uses.
- `EYES_COLOURS`: three greys that still read in greyscale.

**Ask:** why alphabetical labels? `predict_proba` columns follow `classes_`, so tables and plots must use the same order.

### Cell 6: Look at the eyes data
Source 137-142. The first checks of every lab.
- `hands = pd.read_csv("data/hand_landmarks.csv")`: one row per picture in which MediaPipe found a hand.
- `shown = [...]`: a few columns, because the table is 68 wide.
- `display(hands[shown].head())` and `.tail()`: first and last five rows.
- `print("shape:", hands.shape)`: (2776, 68).
- `hands.info(verbose=False)`: only the summary: 2776 entries, 63 float columns.

**Ask:** what is one row? A picture: file name, split, gesture, width, height, then 21 points with x, y, z. The pictures themselves are never stored.

### Cell 7: Nulls, duplicates, class counts
Source 151-166. Checks the table is clean and counts pictures per gesture.
- `hands.isna().sum().sum()`: 0 missing values.
- `hands.drop(columns=["file", "split", "label"])` then `.duplicated().sum()`: 0 duplicates over the 65 size and landmark columns. File names are always different and would hide copies.
- `eyes_train`, `eyes_test`: TensorFlow's own `split` column separates the pictures.
- `pd.crosstab(hands["label"], hands["split"])`: pictures with a hand per gesture, for example rock 733 of 840 in training.
- Detection: 2,407 of 2,520 training and 369 of 372 test pictures have a hand.

**Ask:** why can no model reach 100%? The 3 test pictures with no hand (all rock) count as wrong, so the ceiling is 99.2% of 372.

### Cell 8: 47 features and mirrored twins
Source 183-201. Turns every picture into 47 numbers and makes mirrored copies.
- `def feature_matrix(df, flip=False)`: runs `features_from_row` on every row. With `flip=True` it first sets every `x` to `1 - x`.
- Four matrices (training, training flipped, test, test flipped) and the labels. Printed shapes: (2407, 47) and (369, 47).
- `by_mirroring = [hand_features(mirror(to_pixels(...))) ...]` rebuilds five flipped hands the other way.
- `assert np.allclose(by_mirroring, eyes_X_train_flipped[:5], atol=1e-9)`: both routes agree, so twins are true mirror images.

**Ask:** what are the 47 numbers? 42 hand-shape coordinates (wrist at the origin, hand turned upright, scaled) plus 5 finger-reach ratios.

### Cell 9: Finger reach summary
Source 204-205. Two lines.
- `eyes_ratios = pd.DataFrame(eyes_X_train[:, 42:], columns=FEATURE_NAMES[42:])`: keeps columns 42 to 46, the five reach numbers (thumb to pinky), as a named table.
- `display(eyes_ratios.describe())`: count, mean, spread, min, quartiles, max. A median below 1 means a curled finger (ring 0.97, pinky 0.96); above 1 means a straight one (index and middle medians are above 1.3).

**Ask:** what is a reach number? The wrist-to-fingertip distance divided by the wrist-to-middle-joint distance. Turning or resizing the hand does not change it.

### Cell 10: Find-S, clean run and noisy run
Source 217-235. Lab 3 on the finger states, with scissors as the positive class.
- `run_find_s(up, down, leave_out_unclear)`: `finger_states` turns each picture's four reach numbers into up, down or unclear; optionally drops unclear ones; `find_s` (hand-written) returns the rule; only pictures that change the rule are printed.
- Clean run `(1.15, 0.95, True)`: 1,657 of 2,407 pictures used, final rule (up, up, down, down).
- Noisy run `(1.0, 1.0, False)`: all 2,407 used, rule falls to (up, up, ?, ?).
- Both rules are saved in `METRICS`.

**Ask:** why does the noisy run collapse? One scissors picture that disagrees turns a finger into `?`, and Find-S never narrows again.

### Cell 11: Why the noisy run falls apart
Source 238-244. Finds the pictures that broke the rule.
- `scissors_ratios = eyes_X_train[eyes_y_train == "scissors", 42:]`: reach numbers of scissors pictures only.
- `ring_or_pinky_up = np.array(["up" in state[2:] ...])`: flags scissors where the noisy run called the ring or pinky finger up.
- `reach = ...[:, 3:5].max(axis=1)`: the larger of the two reaches.
- `display(Markdown(...))`: 108 of 840 scissors pictures reach 1.00 to 1.13, just over the 1.0 line.

**Ask:** so is Find-S useless? No. It is exact on noise-free data, but real pictures are noisy, so the rest of Part 1 uses models that learn from the whole spread.

### Cell 12: Plot: pictures with a hand, per gesture
Source 252-258. A bar chart.
- `fig, ax = plt.subplots(figsize=(7, 4.5))`: the canvas.
- `sns.countplot(data=hands, x="label", hue="split", ...)`: bars per gesture, split into train and test, with two greys.
- `ax.bar_label(container)`: writes the counts on the bars.
- `sns.move_legend(...)`: legend beside the bars.
- `ax.set(title=...)` and `plt.show()`: labels and display.

**Ask:** what does it show? MediaPipe misses rock pictures most often (733 of 840 training pictures found), and each test gesture has about the same number of pictures.

### Cell 13: Plot: histograms of finger reach
Source 261-267. Five histograms, one per finger, coloured by gesture.
- `eyes_frame = eyes_ratios.assign(label=eyes_y_train)`: adds the gesture to the reach table.
- `fig, axes = plt.subplots(1, 5, ...)`: five panels.
- `sns.histplot(..., hue="label", element="step", legend=(k == 0))`: step outlines so overlaps stay visible; one legend only.
- `ax.set(...)`, `fig.suptitle(...)`, `plt.show()`: titles and display.

**Ask:** how do you read it? Where two gestures' outlines overlap, that finger cannot tell them apart, so the model needs several fingers together.

### Cell 14: Plot: boxplots of finger reach
Source 270-277. The same five numbers as boxplots.
- `ax.boxplot([...for label in LABELS], tick_labels=LABELS, patch_artist=True, ...)`: one box per gesture. The box holds the middle half of the pictures and the black line is the median.
- `box.set(facecolor=..., hatch=hatch)`: a grey and a hatch per gesture so it reads in greyscale.
- `ax.set(...)`, `fig.suptitle(...)`, `plt.show()`.

**Ask:** why a boxplot and a histogram? The boxplot shows the median and spread at a glance; the histogram shows the shape.

### Cell 15: Plot: correlation heatmap
Source 280-283. Four lines.
- `fig, ax = plt.subplots(figsize=(6, 4.8))`: the canvas.
- `sns.heatmap(eyes_ratios.corr(), annot=True, fmt=".2f", cmap="Greys", vmin=-1, vmax=1, ax=ax)`: how the five reach numbers move together, from -1 to 1.
- `ax.set(...)`: titles. `plt.show()`: display.

**Ask:** why does it matter? Gaussian Naive Bayes treats the numbers as unrelated; strong correlations break that idea, a likely reason `nb` is the weakest eyes model (89.5%).

### Cell 16: Plot: pairplot of four fingers
Source 286-290. Five lines.
- `four_fingers = [...]`: index, middle, ring, pinky.
- `pairs = sns.pairplot(eyes_frame, vars=four_fingers, hue="label", diag_kind="hist", markers=["o", "s", "^"], height=2, ...)`: every finger against every other, histograms on the diagonal, a marker per gesture.
- `pairs.figure.suptitle(...)`: the title.
- `plt.show()`: display.

**Ask:** what does it show? Which pairs of fingers separate the gestures. Scissors (index and middle up, ring and pinky down) is a pattern across fingers, which is why Find-S looks at all four.

### Cell 17: Are the test pictures really new?
Source 299-318. Measures how close pictures are, instead of assuming.
- `scaler = StandardScaler().fit(eyes_X_train)` and `NearestNeighbors(n_neighbors=2)`: finds nearest pictures in scaled 47-number space.
- `train_gap`: distance to the nearest other training picture (column 1, as column 0 is itself). `test_gap`: distance to the nearest training picture.
- Printed: median 0.79 training to training, 3.95 test to training. Closer than 1.0: 66% of training pictures, 0% of test pictures.
- A boxplot with a dashed line at 1.0.

**Ask:** what is the point? Training pictures have near-copies, test pictures do not. So a random split inside training would leak, and the official test is the fair check.

### Cell 18: Splitting the pictures
Source 330-337. Eight lines.
- `fit_idx, val_idx = train_test_split(np.arange(len(eyes_train)), test_size=0.2, stratify=eyes_y_train, random_state=42)`: 80/20 split of row numbers, same gesture shares, repeatable.
- `eyes_X_fit = np.vstack([...[fit_idx], ...flipped[fit_idx]])`: fit pictures, then their twins.
- `eyes_y_fit`: labels, repeated for the twins.
- `eyes_groups_fit = np.concatenate([fit_idx, fit_idx])`: a picture and its twin share a group.
- `eyes_X_val, eyes_y_val`: validation pictures, no twins.
- `assert not set(val_idx) & set(eyes_groups_fit)`: no validation picture has a twin among the fit rows.
- Printed: 1,925 + 1,925 twins = 3,850 rows, validation 482, test 369.

**Ask:** why add twins after the split? So no twin lands across the split from its original.

### Cell 19: Six models in Pipelines
Source 351-366. Builds and fits the six eyes models.
- `eyes_pipeline(clf)`: `Pipeline([("scale", StandardScaler()), ("model", clf)])`, scale then fit, like Lab 8.
- `EYES_PIPELINES`: `nb` (GaussianNB), `dt`, `logreg`, `svm_linear`, `svm_rbf` (both `probability=True`), `perceptron`. Order matters: the first wins a tie.
- `for model in EYES_PIPELINES.values(): model.fit(eyes_X_fit, eyes_y_fit)`: all six learn from the same 3,850 rows, as plain numpy arrays.

**Ask:** why a Pipeline? The scaler learns its mean and spread from training rows only, and the same two steps run at predict time, so nothing leaks and nothing is forgotten.

### Cell 20: Comparison table
Source 375-409. Scores each model on three sets of pictures.
- `eyes_micro_roc(model, X, y)`: ROC over the three gestures at once; asserts `classes_` matches `LABELS`.
- `score_model(key)`: train accuracy on the fit rows, validation accuracy, and test accuracy as `(predicted == eyes_y_test).sum() / IMAGES["test"]`, out of all 372. Macro precision, recall, F1 only over pictures with a hand.
- `EYES_BASELINES = {"luck": 1 / 3, "ceiling": ...}`.
- Printed test accuracy: nb 89.5, dt 94.4, logreg, svm_linear, svm_rbf and perceptron 99.2; luck 33.3, ceiling 99.2.

**Ask:** why is validation about 100% but nb only 89.5% on test? Validation pictures have near-copies among the fit pictures; the test pictures do not (Cell 17).

### Cell 21: Bar chart: train, validation, official test
Source 412-422. Draws Cell 20's table.
- `for k, (column, label, shade, hatch) in enumerate([...])`: three bars per model, shifted side by side.
- `ax.bar_label(bars, fmt="%.1f", rotation=90, ...)`: writes the values.
- `ax.axhline(value * 100, linestyle=line_style, ...)`: luck dashed, ceiling dotted.
- `ax.set_xticks(...)`, `ax.set(... ylim=(0, 118))`, `ax.legend(...)`, `plt.show()`: labels, room for numbers, legend.

**Ask:** which bar counts? The test bar (hatched `xx`). Train and validation are flattered by near-copies.

### Cell 22: The model that plays live
Source 431-452. Picks the live eyes model and checks it.
- `proba_keys`: models with `predict_proba` (all but the Perceptron). `live_key = by_model.loc[proba_keys, "accuracy"].idxmax()`: best test accuracy, first on a tie, so `logreg`.
- Asserts: it is a Pipeline with `predict_proba`, `classes_ == LABELS`, 47 features, and no `feature_names_in_`.
- Printed: logreg 99.2% of 372; tied: logreg, svm_linear, svm_rbf.
- `live_as_is`, `live_mirrored`: 99.2% and 99.2%, saved in `METRICS`.

**Ask:** you chose on the test set; is that leakage? The test pictures were used once, to choose among a three-way tie, and nothing was fitted or tuned on them, so the choice cannot inflate the 99.2%.

### Cell 23: Confusion matrix and report for the live model
Source 460-468. Nine lines.
- `live_predicted = LIVE_EYES.predict(eyes_X_test)`: answers for the 369 pictures with a hand.
- `print("model:", live_key)`: logreg.
- `ConfusionMatrixDisplay.from_predictions(eyes_y_test, live_predicted, labels=LABELS, cmap="Greys")`: the matrix; `plt.title(...)`, `plt.show()`.
- `print(classification_report(...))`: precision, recall, F1 1.00 each; support 124, 121, 124.
- `display(Markdown(...))`: 369 of 369 correct; add the 3 no-hand pictures and it is 99.2% of 372.

**Ask:** what is `support`? The number of true pictures of that gesture.

### Cell 24: ROC curves for the live eyes model
Source 475-488. One-vs-rest ROC curves.
- `gesture_onehot = label_binarize(eyes_y_test, classes=LABELS)`: one 0/1 column per gesture.
- `live_probabilities = LIVE_EYES.predict_proba(eyes_X_test)`: the percentages, columns in `LABELS` order.
- The loop uses `roc_curve(gesture_onehot[:, k], live_probabilities[:, k])` and `auc(fpr, tpr)` per gesture. Each prints AUC 1.000.
- `eyes_micro_roc(...)`: the micro-average over all three, AUC 1.000. The dotted diagonal is chance.

**Ask:** is AUC 1.000 suspicious? It says the probabilities rank perfectly on these easy computer-drawn test pictures. It is not proof for real hands.

### Cell 25: The Decision Tree, drawn
Source 495-501. Seven lines.
- `dt_tree = EYES_PIPELINES["dt"].named_steps["model"]`: pulls the tree out of its Pipeline.
- `fig, ax = plt.subplots(figsize=(14, 6))`: a wide canvas.
- `plot_tree(dt_tree, max_depth=3, feature_names=FEATURE_NAMES, class_names=LABELS, filled=True, fontsize=7, ax=ax)`: each box shows its question, its rows and the majority gesture.
- `ax.set_title(...)` and `plt.show()`: title and display.
- `dt_tree.get_depth()` and `.tree_.node_count`: 2 questions deep, 7 boxes, so all of it is drawn.

**Ask:** what picks the questions? The default `gini`; Cell 26 also tries `entropy`.

### Cell 26: Tuning the eyes models with GridSearchCV
Source 513-533. Lab tuning, with twins kept together.
- `svm_grid`: linear with `C` in 0.1, 1, 10; rbf with `C` and `gamma` in scale, 0.01, 0.1: 12 settings. `tree_grid`: `max_depth` 3, 5, 8, None times `criterion` gini, entropy: 8 settings.
- `GridSearchCV(pipeline, grid, cv=GroupKFold(n_splits=5), scoring="accuracy").fit(eyes_X_fit, eyes_y_fit, groups=eyes_groups_fit)`.
- Printed SVM: rbf, `C` 0.1, CV 1.0000 ± 0.0000, 8 of 12 tie, 369 of 372 on test (99.2%). Tree: gini, depth 3, CV 0.9990 ± 0.0021, 8 of 8 tie, 351 of 372 (94.4%).

**Ask:** did tuning help? No. CV is already at the top and many settings tie. The tuned model is only printed; it does not replace a table row.

### Cell 27: Cross-validation for the eyes models
Source 541-554. Lab cross-validation, mean ± std.
- `cross_val_score(model, eyes_X_fit, eyes_y_fit, groups=eyes_groups_fit, cv=GroupKFold(n_splits=5))` for each of the six pipelines.
- Printed: every model 0.9990 to 1.0000, for example nb 1.0000 ± 0.0000 and dt 0.9990 ± 0.0021.
- The markdown explains why: near-copies of a fit picture land in other folds, and held-out folds sit closer to the fit rows (median distance 0.79) than test pictures do (3.95). So nb is 100.0% in CV but 89.5% on test.

**Ask:** why is CV so high? Near-copies flatter it. The fair numbers are the official test and the nearest-neighbour check.

### Cell 28: Learning curve for the eyes
Source 562-573. Does more data help?
- `learning_curve(clone(LIVE_EYES), eyes_X_fit, eyes_y_fit, groups=eyes_groups_fit, cv=GroupKFold(n_splits=5), train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0], shuffle=True, random_state=42)`: train on growing shares, score on held-out folds. `shuffle=True` because twins come after the originals; without it small sizes would hold only originals.
- The plot draws mean ± std bands and the note "luck is 33.3%".
- Printed: 0.9984 held-out at 308 rows, 0.9995 from 770 rows up.

**Ask:** would more pictures help? The line is flat, but those folds hold near-copies, so I would not lean on it.

### Cell 29: One new example, and the live path
Source 582-592. Ten lines.
- `sample = int(np.flatnonzero(eyes_y_test == "scissors")[0])`: the first scissors test picture.
- `print(f"Predict one new sample: ...")`: its file and the true gesture.
- `display(pd.DataFrame(LIVE_EYES.predict_proba(eyes_X_test[[sample]]), ...).round(3))`: paper 0.0, rock 0.0, scissors 1.0.
- `test_hands` and `test_pixels`: the 21 points of every test picture, as pixels.
- `predict_gesture(LIVE_EYES, test_pixels[sample])` then its `print`: the live path, mirror-averaged: scissors, 100% sure.
- `live_path_hits = sum(...)` and the next `print`: live-path accuracy 99.2% of 372.
- Last `print`: yardsticks luck 33.3%, ceiling 99.2%.

**Ask:** why also score the mirror image? So a left and a right hand get the same answer in the game.

### Cell 30: All six eyes models, side by side
Source 603-616. The graded comparison figure (D31).
- `final_results` and `eyes_summary`: accuracy, precision, recall, F1, CV mean and std, printed with a luck row (33.3).
- The markdown: 4 of 6 models share the top test score (99.2%), so these pictures cannot say which is best, and the hands are computer-drawn.
- `comparison_figure(final_results, "...", baselines={"luck": ...}, labels=LABELS)`: four panels: grouped bars, fold boxplots, ROC curves, confusion matrix of the best model.
- `fig.axes[3].set_title(...)` fixes the title because several models tie.

**Ask:** which model is best? The test cannot tell four apart. I use logreg because it gives probabilities and ties for the top.

### Cell 31: Part 2 imports and constants
Source 627-641. Brings in what the brain needs.
- `LinearRegression` (Lab 2, for the slope in Cell 52), `KFold` (the random-split check in Cell 39), and the metric and tuning tools again.
- `from mindreader.brain import (FEATURES, ModelBrain, PersonalBrain, RandomBrain, add_features, brain_pipeline, decide, history_features, replay_hits)`: the brain code.
- `from mindreader.moves import NAMES, outcome` and `from tools.rps_facts import early_window`.
- `SHIFT_NAMES = ["stay", "up", "down"]`: the target classes. `SHADES`: three blues that differ in greyscale.

**Ask:** what is a shift? The next move relative to the last: 0 stay, 1 step up to the move that beats the old one, 2 step down.

### Cell 32: Look at the brain data
Source 650-655. Six lines.
- `games = load_games("data/rps_human_games.csv")`: 21,600 rows, one per round.
- `display(games.head())` and `display(games.tail())`: first and last five rounds.
- `print("shape:", games.shape)`: (21600, 8).
- `games.info()`: 21,600 non-null in all 8 columns.
- `display(games.describe())`: count, mean, spread of the numeric columns.

**Ask:** what is a row? A round: player `id`, `period` (round number), `choice`, `other.choice`, `result` from the player's side, plus `c` and `o`, the moves as 0, 1, 2.

### Cell 33: Nulls, duplicates, class counts
Source 663-668. Six lines.
- `print("missing values in the file:", ...)`: 0.
- `print("duplicate rows:", ...)`: 0.
- `no_history = games["period"] < 3` and its `print`: 144 of 21,600 rounds (rounds 1 and 2 of 72 players) cannot be used, because the brain needs two rounds of past.
- `display(games["c"].map(NAMES).value_counts()...)`: rock 7702, paper 6961, scissors 6937, a rock bias.
- `display(games["result"].value_counts()...)`: win 7170, tie 7260, lose 7170.

**Ask:** why do wins equal losses? The players played each other, so one player's win is another's loss.

### Cell 34: Facts the brain can use
Source 676-685. Ten lines; each fact is printed with its n.
- `FACTS = facts(games)`: the dictionary of headline numbers.
- The `for` loop prints: rock share of all rounds 35.7% (n = 21,600); first throw rock 58.3% (n = 72); repeat after a loss 52.7% (n = 7,148); always-rock on model rows 35.6% and always-repeat 44.0% (n = 21,456).
- The last `print` gives luck, 33.3%.
- The 58.3% and the 52.7% are computed from the study's public data.

**Ask:** what are the yardsticks? Always-rock, always-repeat and luck. A brain only counts if it beats them.

### Cell 35: Plots: what players threw, how rounds ended
Source 692-705. Two bar charts.
- `labelled_bars(ax, names, values, title, xlabel, ylabel)`: a bar chart with the count written on every bar.
- `throws` and `results`: `value_counts().reindex(...)`, in a fixed order.
- `plt.subplots(1, 2, figsize=(11, 4))`: two panels, one `labelled_bars` call each.
- `plt.show()`: display.

**Ask:** what do the counts show? Rock is thrown most (7702 against 6961 and 6937), wins and losses are equal (7170 each) and ties are 7260.

### Cell 36: Plot: how often players repeat their move
Source 708-718. The 52.7% bar chart.
- `previous = games.groupby("id")[["c", "result"]].shift(1)`: each player's round before.
- `repeated = games["c"] == previous["c"]`: threw the same move again.
- `repeat_rate = repeated.groupby(previous["result"]).agg(["mean", "size"]).reindex(["lose", "tie", "win"])`: the repeat share after a loss, a tie, a win.
- `plt.subplots`, `ax.bar(...)`, `ax.bar_label(...)`: bars with share and n.
- `ax.axhline(100 / 3, ...)`: the dashed luck line.
- `ax.set(... ylim=(0, 65))`, `ax.legend()`, `plt.show()`.

**Ask:** which bar is the hook? "After a loss", 52.7% (n = 7,148), computed from the study's public data, against the 33.3% luck line. The other bars, as printed on the chart: after a tie 43.3%, after a win 36.0%.

### Cell 37: Plot: last move to next move, by last result
Source 725-735. Three heatmaps.
- `last = games.groupby("id")[["choice", "result"]].shift(1)`: the previous round's move and result.
- The loop runs over `["lose", "tie", "win"]`; `after = last["result"] == last_result` picks those rounds.
- `pd.crosstab(last["choice"][after], games["choice"][after], normalize="index")`: for each last move, the share of each next move; every row adds to 100%.
- `sns.heatmap(follows, annot=True, fmt=".0%", cmap="Blues", vmin=0, vmax=0.7, cbar=False, ax=ax)`: one panel each.

**Ask:** how do you read it? If players were random every cell would be about 33%; cells far from 33% are habits.

### Cell 38: Features for the brain
Source 751-755. Five lines.
- `rows = add_features(games)`: one row per round from round 3, with `o_rel1`, `shift1`, `o_rel2` and the target `shift`.
- `display(rows.head())`: the first rows.
- `print(f"model rows ...")`: 21,456 rows.
- `counts = rows["shift"].map(dict(enumerate(SHIFT_NAMES))).value_counts().reindex(SHIFT_NAMES)`: stay 9437, up 6043, down 5976.
- `display(pd.DataFrame({"rows": counts, "share": counts / counts.sum()}).round(3))`: shares 0.440, 0.282, 0.279.

**Ask:** why relative moves? Rock-paper-scissors is a circle, so one habit ("stay after a loss") learns from rock, paper and scissors at once. Note stay is 0.440, which is what always-repeat predicts.

### Cell 39: Split by player, not by row
Source 764-771. Eight lines.
- `X, y, groups = rows[FEATURES], rows["shift"], rows["id"]`: three features, the target, the player as the group.
- The `for` loop over `GroupKFold(n_splits=5).split(X, y, groups)` and its `print`: 57 or 58 players train, 14 or 15 test, 0 on both sides.
- `random_split = cross_val_score(brain_pipeline("dt"), X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42))`: 46.1%, players on both sides.
- `player_split = cross_val_score(..., groups=groups, cv=GroupKFold(n_splits=5))`: 45.5% (luck 33.3%, always-rock 35.6%, always-repeat 44.0%).

**Ask:** why split by player? The question is whether the brain predicts new people. Random rounds put one person on both sides, which can flatter a model.

### Cell 40: Helpers: evaluate, micro_roc, result_row
Source 790-824. Three functions, no output.
- `evaluate(pipeline, X, y, groups)`: for each of 5 folds, `clone(pipeline).fit(X.iloc[train], y.iloc[train])`, predict the held-out players, and keep every round's prediction (and probabilities if the model has them) from the fold where its player was held out.
- `micro_roc(y, probabilities)`: ROC over the three classes at once, or `None` without probabilities.
- `result_row(...)`: one table row: accuracy, macro precision, recall, F1, fold scores, ROC, confusion matrix.

**Ask:** why `clone`? Each fold needs a fresh unfitted copy, or one fold's learning would leak into the next.

### Cell 41: Fit and score the five brains
Source 828-839. Where the numbers come from.
- `for name in ["dt", "svm", "logreg", "nb", "perceptron"]:` builds `brain_pipeline(name)`. For `svm`, `set_params(model__probability=False)`: same predictions, much faster, but no ROC.
- `evaluate(pipeline, X, y, groups)`: predictions on players the model had not seen; kept in `out_of_fold` and `table_rows`.
- `RESULTS = pd.DataFrame(table_rows)`.
- `BASELINES = {"luck": 1/3, "always-rock": ..., "always-repeat": ...}`: the three yardsticks.
- `BEST_BRAIN`: highest accuracy, first on a tie: `dt`.

**Ask:** why does every score mean unseen players? Each prediction was made while that round's player sat in the test fold.

### Cell 42: Comparison table, and why dt and svm tie
Source 850-868. The brain table, plus a check.
- `lookup_table_predictions`: in each fold, a crosstab of situation by next shift on the training players; each situation predicts its most common shift; an unseen situation gets the overall mode.
- `situation = X["o_rel1"] * 9 + X["shift1"] * 3 + X["o_rel2"]`: one number per combination, 27 in all.
- Printed (luck 33.3, always-rock 35.6, always-repeat 44.0): dt 45.5, svm 45.5, nb 44.8, logreg 44.7, perceptron 37.9.
- dt and svm agree on 100.0% of rounds; the lookup table agrees with dt on 100.0% and scores 45.5% itself.

**Ask:** why identical? Three simple features make only 27 situations, both models learn the same table, so a fancier model finds nothing more.

### Cell 43: Bar chart of the brain table
Source 871-878. Eight lines.
- `fig, ax = plt.subplots(figsize=(8, 4.5))`: the canvas.
- `bars = ax.bar(RESULTS["model"], RESULTS["accuracy"] * 100, ...)`: one bar per model.
- `ax.bar_label(bars, labels=[...])`: the percentages.
- The `for ... zip(BASELINES.items(), BASELINE_COLOURS)` loop with `ax.axhline(...)`: dashed lines for luck 33.3%, always-rock 35.6%, always-repeat 44.0%.
- `ax.set(... ylim=(0, 60))`, `ax.legend(...)`, `plt.show()`.

**Ask:** what is the message? The tree and SVM (45.5%) sit just above the always-repeat line (44.0%); the Perceptron (37.9%) is below it.

### Cell 44: Metrics for the best brain
Source 886-903. Confusion matrix, report, and the hook again.
- `predicted = out_of_fold[BEST_BRAIN][0]` and `ConfusionMatrixDisplay.from_predictions(y, predicted, display_labels=SHIFT_NAMES, ...)`: stay, up, down.
- `classification_report`: stay precision 0.48, recall 0.75; up 0.38 and 0.24; down 0.40 and 0.20. The tree mostly says stay.
- `after_loss = (rows["o_rel1"] == 1)`: rounds right after the player lost.
- Printed on 7,130 such rounds: dt 54.3%, always-repeat 52.7% (computed from the study's public data), always-rock 35.8%, luck 33.3%.

**Ask:** is 54.3% much better than 52.7%? No. It is slightly above always-repeat, which is the whole story.

### Cell 45: ROC curves for the best brain with probabilities
Source 910-920. One-vs-rest ROC for stay, up and down.
- `with_probabilities = RESULTS[RESULTS["roc"].notna()]` and `roc_model = ...idxmax()`: the most accurate model that has probabilities (the SVM here has none), which is the tree.
- `shift_onehot = label_binarize(y, classes=[0, 1, 2])`: one 0/1 column per shift.
- The loop uses `roc_curve(shift_onehot[:, k], out_of_fold[roc_model][1][:, k])`: the probabilities each round got while its player was held out. Each legend line shows its AUC.
- The dotted diagonal is chance.

**Ask:** why is the SVM missing? It ran with `probability=False` for speed, and a ROC needs probabilities.

### Cell 46: Tuning the brains with GridSearchCV
Source 934-966. Tuning on the same 5-fold `GroupKFold`.
- `GRIDS`: dt `max_depth` (3, 4, 5, 6, 8) by `min_samples_leaf` (1, 20, 50); logreg `C`; svm `C`.
- `tune(...)` prints the best setting, mean ± std and the ties. On all 21,456 rounds (luck 33.3%, always-rock 35.6%, always-repeat 44.0%): dt depth 6, leaf 1: 45.5% ± 1.4%, 6 of 15 tie; logreg `C` 0.1: 44.7% ± 1.7%, 3 of 3 tie.
- The SVM is slow, so `small` is the first 20 players (5,960 rounds): `C` 0.5, 47.3% ± 5.2%; there luck is 33.3%, always-rock 35.0%, always-repeat 46.0%. `live_svm` is refitted on `small` with probabilities for the game.

**Ask:** why is the SVM's 47.3% not comparable? Different rows, noisier folds, and a different always-repeat (46.0%).

### Cell 47: The brains the game plays against
Source 976-992. Wraps the models as brains.
- `BRAINS = {...}`: `dt` and `logreg` use the grids' best estimators, `svm` is `live_svm`, `nb` and `perceptron` are fitted on all rounds, plus `PersonalBrain()` and `RandomBrain()`. `ModelBrain` wraps a model so it takes a history and returns a move.
- `likeliest = live_svm.classes_[live_svm.predict_proba(...).argmax(axis=1)]`: scores the live SVM on the 52 players it never saw.
- Printed: brains ready: dt, logreg, svm, nb, perceptron, personal, random; best brain dt; live SVM 43.9% against always-repeat 43.2% on 15,496 rounds (luck 33.3%; no always-rock figure is printed for those rows).

**Ask:** is the live SVM better than always-repeat? Barely: 43.9% to 43.2%.

### Cell 48: Cross-validation: mean ± std, fold by fold
Source 1002-1009. Eight lines.
- `for row in RESULTS.itertuples():` and its `print`: dt 45.5% ± 1.4%, svm 45.5% ± 1.4%, logreg 44.7% ± 1.7%, nb 44.8% ± 1.7%, perceptron 37.9% ± 3.7%.
- `print_yardsticks(...)`: luck 33.3%, always-rock 35.6%, always-repeat 44.0%.
- `repeat_folds = [(y.iloc[test] == 0).mean() ...]`: always-repeat on each test fold.
- `dt_folds = ...`: dt's fold scores.
- The next `print`: always-repeat 44.0% ± 3.3%. The last: `dt beats always-repeat in 3 of 5 folds`.

**Ask:** does the brain reliably beat always-repeat? No. It is ahead in 3 of 5 folds.

### Cell 49: Learning curve: how many players does the brain need?
Source 1017-1032. Train on more and more players.
- `rows_per_player = len(rows) // rows["id"].nunique()`: rounds per player.
- `learning_curve(brain_pipeline("dt"), X, y, groups=groups, cv=GroupKFold(n_splits=5), train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0], shuffle=False)`: `shuffle=False` because rows are sorted by player, so each training set is whole players. `players = sizes / rows_per_player`.
- Plot with yardsticks dashed. Printed: 44.8% on unseen players at 5.7 training players, 45.5% at 57.0 (+0.7 points). Luck 33.3%, always-rock 35.6%, always-repeat 44.0%.

**Ask:** would more players help? The line is nearly flat, so probably not much.

### Cell 50: Why the Perceptron is the odd one out
Source 1040-1056. The D27 story, with evidence.
- `for seed in range(5):` runs `cross_val_score(brain_pipeline("perceptron").set_params(model__random_state=seed), ...)`. Printed: 36.5%, 40.5%, 39.2%, 39.9%, 41.0%.
- `logreg_mean`: logistic regression gives 44.7% whatever the seed.
- The bar chart puts the five seeds next to logreg, with luck 33.3%, always-rock 35.6% and always-repeat 44.0% dashed.

**Ask:** why does the Perceptron change with the seed? It updates only on its mistakes. With noisy, overlapping labels it is always wrong on some rows, so its line never settles and the final spot depends on row order.

### Cell 51: The early rounds
Source 1066-1078. A real game starts with no history.
- `early_numbers(last_round)`: `early_window` scores always-rock, always-repeat and the Personal brain on rounds 2 to `last_round`; the tree's out-of-fold predictions start at round 3 (it needs two rounds).
- `EARLY = {f"2-{last_round}": ... for last_round in (10, 20)}`.
- Printed as luck | always-rock | always-repeat | brain: rounds 2-10 33.3 | 34.9 | 39.2 | personal 40.0 (n = 648); rounds 3-10 33.3 | 33.7 | 39.4 | tree 39.6; rounds 2-20 33.3 | 37.0 | 41.5 | personal 42.4; rounds 3-20 33.3 | 36.6 | 41.7 | tree 42.1.

**Ask:** how good is the bot in an 8-round match? Barely above always-repeat, so a single match says little (the notebook's words).

### Cell 52: How fast does the Personal brain improve? (Linear Regression)
Source 1087-1114. Lab 2 applied.
- `slope_per_100(hits)`: `LinearRegression().fit(hits[["round"]], hits["hit"] * 100)`: a straight line of right (100) or wrong (0) against the round number; the slope is points per 100 rounds.
- `replay_hits(games, PersonalBrain(), ...)`: replays every player round by round. `repeat_hits`: the same for always-repeat.
- Printed: Personal +3.2 points per 100 rounds (n = 21,528); always-repeat +1.3; Personal's own gain +1.8.
- The scatter shows both lines with luck and always-rock flat.

**Ask:** why subtract the always-repeat slope? Players get more repetitive as the game goes on, so the plain slope overstates what the brain learns.

### Cell 53: One new example (brain)
Source 1122-1129. Eight lines.
- `history = [(0, 2), (0, 1)]`: (player move, CHANAKYA move): rock vs scissors, then rock vs paper.
- The `for` loop with `print`: round 1 a win, round 2 a loss.
- `display(pd.DataFrame([history_features(history)]))`: `o_rel1` 1 (lost), `shift1` 0 (stayed), `o_rel2` 2 (won).
- `decision = decide(history, BEST_BRAIN, BRAINS)`: the best brain's call.
- `how_sure = ...`: adds the probability when there is one.
- First `print`: the dt brain expects rock next (probability 56%). Last `print`: CHANAKYA plays paper, the move that beats it.

**Ask:** how is the counter chosen? Predicted move = last move + shift; the bot plays `beats(m) = (m + 1) % 3`.

### Cell 54: All five brains, side by side
Source 1140-1145. The graded comparison figure (D31).
- `summary = RESULTS.set_index("model")[[...]].copy()`: accuracy, precision, recall, F1.
- `summary["cv mean"]`, `summary["cv std"]`: fold mean and spread.
- `print((pd.concat([summary, pd.DataFrame({"accuracy": BASELINES})]) * 100).round(1)...)`: the table with luck 33.3, always-rock 35.6, always-repeat 44.0.
- `fig = comparison_figure(RESULTS, "Part 2: ...", BASELINES, labels=SHIFT_NAMES)`: bars, fold boxplots, ROC curves, the best model's confusion matrix.
- `plt.show()`.

**Ask:** what does the figure say? Five brains next to luck 33.3%, always-rock 35.6% and always-repeat 44.0%; dt and svm (45.5%) are barely above always-repeat.

### Cell 55: Save the numbers
Source 1148-1157. Ten lines.
- `METRICS["brain_table"] = {row.model: float(row.accuracy) ...}`: accuracy per brain.
- `METRICS["brain_table"].update(luck=..., always_rock=..., always_repeat=...)`: the yardsticks.
- `METRICS["brain_early"] = EARLY`: the early-round numbers.
- `METRICS["personal_slope_per_100"]`, `METRICS["repeat_slope_per_100"]`: the two slopes.
- `METRICS["brain_table_live_svm"] = LIVE_SVM`, `METRICS["facts"] = FACTS`.
- `os.makedirs("build", exist_ok=True)`, then `json.dump(METRICS, file, indent=2)` into `build/metrics.json`.

**Ask:** why a JSON file? Part 4 and the snapshot read it, so no number is typed twice.

### Cell 56: The LIVE switch
Source 1181. One line.
- `LIVE = os.environ.get("MR_SKIP_LIVE") != "1"`: `os.environ.get` reads an environment variable (None if unset), so `LIVE` is True on Colab and on a laptop. The build script sets `MR_SKIP_LIVE=1`, so every cell that starts `if LIVE:` does nothing in the saved copy.

**Ask:** why? The saved notebook is built without a camera, and a check rejects any live cell with outputs or any `image/jpeg` output, so no camera picture can be saved.

### Cell 57: Quick start: train again what the game needs
Source 1194-1239. Needed after a runtime restart.
- `quick_start()` retrains the brains with default settings (no tuning; the SVM on the first 20 players) plus Personal and Random, then the eyes: logistic regression on the same fit part and mirrored twins as Part 1. It returns `eyes, brains, "dt"`.
- `missing = [name for name in ("LIVE_EYES", "BRAINS", "BEST_BRAIN") if name not in globals()]`: only trains what is missing, then prints "Quick start done" or "Parts 1 and 2 have run".

**Ask:** why retrain instead of saving the models? A pickled scikit-learn model breaks when Colab updates scikit-learn.

### Cell 58: The game: stage mode
Source 1248-1257. Nine lines, live only.
- `if LIVE:` skips everything in the saved copy.
- `from mindreader.bridge import register, show_widget`, `HandReader`, `GameEngine`: the pieces.
- `if "READER" not in globals(): READER = HandReader(ensure_model(...)).__enter__()`: opens the hand reader once.
- `ENGINE = GameEngine(BRAINS)`: a fresh game session.
- `register(ENGINE, READER, LIVE_EYES)`: the Colab callback `mindreader.call` now reaches the game in Python.
- `show_widget("stage", brain=BEST_BRAIN)`: shows the page; the camera lives in the browser, everything that reads and decides runs in Python.

**Ask:** can the bot see my throw first? No. Python locks its move before the countdown; the page only gets a round number.

### Cell 59: Practice mode
Source 1265-1266. Two lines.
- `if LIVE:` skips it in the saved copy.
- `show_widget("practice", brain=BEST_BRAIN)`: the same game over 20 rounds instead of 8, Space starts each round, and a dropdown lets you try the other brains, starting on `dt`.

**Ask:** what is it for? Rehearsal, trying other brains, and the pilot. It needs Cell 58 to have run, because `show_widget` uses the engine that `register` stored.

### Cell 60: The eyes test: real hands
Source 1274-1275. Two lines.
- `if LIVE:` skips it in the saved copy.
- `show_widget("eyes_test")`: asks for ROCK, PAPER or SCISSORS 30 times (10 each, shuffled); Python says whether each throw was accepted and correct. Pass marks: at least 85% accepted and at least 95% of the accepted correct, over 3 or more people, both hands, 2 kinds of light.

**Ask:** has it passed? It has not been run yet. Until gate G2 is run in Colab with real hands, the 99.2% says nothing about the webcam.

### Cell 61: The pilot: save practice throws
Source 1288-1296. Eight lines, live only.
- `SAVE_MY_THROWS = False  #@param {type:"boolean"}`: a Colab checkbox, unticked by default.
- `if LIVE:` and `from mindreader.pilot import export_pilot`.
- `saved_file = export_pilot(ENGINE, consent=SAVE_MY_THROWS)`: writes `data/pilot/pilot_<date>.csv`, moves only, anonymous ids, never overwriting a file.
- `try:` runs the export and `print(f"Saved {saved_file}: ...")` tells you which file to send.
- `except ValueError as problem: print("Not saved:", problem)`: the box is unticked, or no practice match has finished.

**Ask:** privacy? Nothing is saved unless the box is ticked; no pictures, no names.

### Cell 62: Shut down
Source 1303-1305. Three lines.
- `if LIVE and "READER" in globals():` only if the reader was opened.
- `READER.__exit__(None, None, None)`: closes MediaPipe's hand reader, which must be closed or it raises when Python exits.
- `del READER`: forgets it, so Cell 58 opens a fresh one.

**Ask:** what happens to an open game page? It says it cannot see your hand, and offers the keys, until Cell 58 is run again.

### Cell 63: Part 4: the numbers in one table
Source 1319-1344. Eight rows, each with its yardsticks and n.
- `from mindreader.plots import BRAIN_MODELS, EYES_MODELS`: the model lists.
- `percent(share)`: "-" when a yardstick does not apply, else one decimal.
- `eyes_scores, brain_scores = METRICS["eyes_table"], METRICS["brain_table"]` and the next lines read this run's numbers; `summary_rows` builds the rows.
- Printed: study repeat-after-loss 52.7% (computed from the study's public data); eyes 99.2% three times (luck 33.3%); brain 45.5% (luck 33.3%, always-rock 35.6%, always-repeat 44.0%, n = 21,456); live SVM 43.9% (luck 33.3%, always-repeat 43.2%, always-rock a dash); Personal rounds 2-10 40.0% (always-rock 34.9%, always-repeat 39.2%) and 2-20 42.4% (37.0%, 41.5%), luck 33.3% in both.

**Ask:** which row is the honest headline? The brain row: 45.5% next to always-repeat 44.0%.

### Cell 64: Results at a glance, from this run
Source 1351-1354. Three lines.
- `from mindreader.plots import glance_figure`: the helper.
- `fig = glance_figure(METRICS)`: the Cell 4 picture, now from this run's numbers. It needs the `eyes_table` and `brain_table` that Parts 1 and 2 stored in `METRICS`; if one is missing it raises a ValueError that names it.
- `plt.show()`.

**Ask:** how does it differ from Cell 4? Cell 4 reads a saved file; this one reads the numbers computed above.

### Cell 65: The pilot: do our friends do it too?
Source 1362-1384. Pilot in, verdict out.
- `PILOT_ROWS = load_pilot()`; `PILOT = summarize(PILOT_ROWS) if len(PILOT_ROWS) else None`: every `data/pilot/pilot_*.csv`.
- Counts the rounds after a loss (n) and the repeats (k), with a 95% Wilson interval, for all rounds and for rounds 1 to 8.
- `verdict(k, n)`: "my friends do it too" if the whole interval is above 1/3; "less predictable than the study's players" only if it is below 52.7% (computed from the study's public data) with at least 60 rounds; else "inconclusive".
- Printed now: "No pilot files yet: the pilot is gate G3 ..."

**Ask:** what did your friends show? Nothing yet. The pilot has not been collected (G3), so I make no claim about them.

### Cell 66: What we learned
Source 1390-1408. Builds the closing bullets from this run's numbers.
- `best_eyes`, `best_brain`, and `gain = (best_brain - brain_scores["always_repeat"]) * 100`: 1.5 points.
- `pilot_line`: with no pilot files it says the notebook makes no claim about our friends.
- `display(Markdown(...))`: 52.7% repeat after a loss (n = 7,148, computed from the study's public data); eyes 99.2% of 372 (luck 33.3%), computer-drawn hands; brain 45.5% against always-repeat 44.0%, always-rock 35.6%, luck 33.3%; Personal 42.4% in rounds 2-20 (luck 33.3%, always-repeat 41.5%, always-rock 37.0%).

**Ask:** what can you not claim? That the live eyes work on real hands (G2), that friends behave like the study (G3), or that the bot beats you. The brain is evidence of predictability on unseen players.

## C. Module functions

Every top-level function and class under `mindreader/*.py` (86 rows). Methods are explained in their class's row.

| function | file | what it does |
|---|---|---|
| `code` | moves.py | Turns the letter 'r', 'p' or 's' into 0, 1, 2; any other letter raises KeyError. |
| `letter` | moves.py | The reverse: 0, 1, 2 back to 'r', 'p', 's'. |
| `beats` | moves.py | The move that beats m: (m + 1) % 3, so paper beats rock. |
| `outcome` | moves.py | 'win', 'lose' or 'tie' from the player's point of view. |
| `shift` | moves.py | How far the player moved: (now - prev) % 3, so 0 stay, 1 up, 2 down. |
| `apply_shift` | moves.py | The move you get by shifting prev by s; the inverse of `shift`. |
| `rel` | moves.py | The opponent's move relative to the player's: 0 tie, 1 the player lost, 2 the player won. |
| `load_games` | brain.py | Reads the study CSV sorted by player and round and adds `c` and `o`, the two moves as 0, 1, 2. |
| `add_features` | brain.py | One row per round from round 3 with `o_rel1`, `shift1`, `o_rel2` and the target `shift`. |
| `facts` | brain.py | The study's headline numbers (repeat after a loss, first throw rock, rock share, the two yardsticks), each with its n. |
| `history_features` | brain.py | The three features for the next round from a list of (player, bot) moves; None if fewer than 2 rounds. |
| `brain_pipeline` | brain.py | An unfitted Pipeline (encoder, then model) for 'nb', 'dt', 'logreg', 'perceptron' or 'svm'; any other name raises ValueError. |
| `PersonalBrain` | brain.py | Learns one player from this game alone: a 3x3 table of counts starting at 1; predicts the most counted shift after the last kind of round. |
| `ModelBrain` | brain.py | Wraps a fitted pipeline: turns a history into features and returns a move and a probability, taking the likeliest shift from `predict_proba` when the model has it. |
| `RandomBrain` | brain.py | Guesses a move at random from a seeded generator: the luck brain, with no percentage. |
| `decide` | brain.py | The bot's move: round 1 assumes rock (0.583) and plays paper, round 2 uses Personal, later rounds the chosen brain; always plays the move that beats the prediction. |
| `replay_hits` | brain.py | Replays rounds 2 to N of every player with the brain seeing only earlier rounds; one row per prediction, hit 1 or 0. |
| `replay_accuracy` | brain.py | The mean of `replay_hits`: the share of rounds where the predicted move was the move thrown. |
| `find_s` | find_s.py | Find-S: copies the first positive example, turns every attribute that later differs into '?'; returns the final rule and a log of changes. |
| `matches` | find_s.py | True if every attribute equals the rule or the rule says '?'; a rule of None fits nothing. |
| `to_pixels` | hands.py | Turns normalised (21, 3) landmarks into (21, 2) pixel x, y so a 4:3 frame does not squash the hand. |
| `canonical` | hands.py | Moves the wrist to the origin, turns wrist-to-middle-knuckle to point straight up and scales it to length 1. |
| `mirror` | hands.py | The hand as seen in a mirror: every x negated; returns a new array. |
| `extension_ratios` | hands.py | Per finger, tip-to-wrist distance divided by middle-joint-to-wrist distance; unchanged by turning, resizing or mirroring. |
| `hand_features` | hands.py | The 47 features: 42 canonical coordinates, then the 5 ratios. |
| `finger_states` | hands.py | 'up', 'down' or 'unclear' for the index, middle, ring and pinky from their ratios and two thresholds. |
| `features_from_row` | hands.py | The 47 features for one row of `hand_landmarks.csv`. |
| `ensure_model` | eyes.py | Downloads the hand model if missing and checks its sha256; a wrong file is deleted and ValueError raised. |
| `HandReader` | eyes.py | Context manager around MediaPipe's HandLandmarker (one hand); `landmarks(rgb)` returns the 21 points or None, and closing it avoids an error at exit. |
| `jpeg_size` | eyes.py | Reads width and height from a JPEG's header without decoding pixels; refuses cut-off or odd headers. |
| `decode_jpeg` | eyes.py | A data URL or base64 JPEG into a BGR image; checks base64, the JPEG start bytes and the size limit before decoding. |
| `encode_jpeg` | eyes.py | A BGR image into a JPEG data URL the browser can show. |
| `predict_gesture` | eyes.py | Scores the hand and its mirror image in one `predict_proba` call and averages; returns (label, probability). |
| `Vote` | eyes.py | A NamedTuple for one frame's vote: frame number, label, probability, landmarks. |
| `read_throw` | eyes.py | Votes over a burst of frames; accepted if at least 3 frames agree and their mean probability is at least 0.7; returns label, prob, agree, best frame, landmarks. |
| `draw_skeleton` | eyes.py | A copy of the frame with green bones and red joints drawn on it (OpenCV); frames are never flipped. |
| `Round` | game.py | A NamedTuple for one counted round: player move, bot move, predicted move, probability, source brain. |
| `Match` | game.py | A dataclass for one match: name, mode, brain, total rounds, counted rounds, locked round, decision, last round, rejects; has `done` and `history()`. |
| `is_move` | game.py | True only for the ints 0, 1 and 2 (not True, not 1.0, not a string). |
| `standing` | game.py | Score, health, done and winner; a win takes one health point from the other side, a tie takes none. |
| `call_made` | game.py | The mind-read call in round i: 'opener' in round 1, 'repeat_after_loss' if the player lost the round before and the bot predicted a stay, else None. |
| `moment` | game.py | The call from `call_made` if it came true (the player threw the predicted move), else None. |
| `reveal` | game.py | What the screen learns about a counted round: both moves, result, prediction, probability, moment, score, health, done, winner. |
| `moment_counts` | game.py | k of n: calls that came true over calls made. |
| `repeat_rates` | game.py | After a loss, a tie and a win: how often the player threw the same move again, as k of n. |
| `habits` | game.py | The player's habits as counts: repeats by last result, favourite move, and how often they stay, step up or step down. |
| `unpredictability` | game.py | 1 minus correct bot predictions over predictions in rounds 2 to N, with the counts. |
| `replay_table` | game.py | Each brain's hits on rounds 2 to N of this match, using only earlier rounds; labelled "illustrative", not a ranking. |
| `GameEngine` | game.py | Keeps matches in memory with `start`, `lock`, `throw`, `correct`, `report`, `leaderboard`, `finished_matches`, `export_log`; the bot's move is fixed at lock and never passed to `throw`. |
| `check_frame` | bridge.py | One camera frame to a decoded BGR image, or ValueError: must be a string, at most 250,000 characters, a good JPEG, longest side at most 1280 pixels. |
| `validate_frames` | bridge.py | A burst must be a list of 3 or 5 frames; decodes each and names the bad one. |
| `clean_name` | bridge.py | Removes control, format, surrogate, private-use, unassigned and line-separator characters from a name and trims it to 20 characters. |
| `fail` | bridge.py | Builds the error reply with ok set to False and the message. |
| `text_field` | bridge.py | payload[key] as text of at most 64 characters, or a ValueError naming the field. |
| `move_field` | bridge.py | payload["move"] as the int 0, 1 or 2; True and 1.0 are refused. |
| `read_burst` | bridge.py | Calls `read_throw`; any failure becomes one plain ValueError and the traceback goes to `last_error`, never to the page. |
| `seen_reply` | bridge.py | What the page shows about a burst: label, probability, frames agreeing, and the clearest frame with the skeleton drawn (None if no hand). |
| `do_start` | bridge.py | Handler: cleans the name, refuses the eyes-test mode, starts a match and returns its id, name and length. |
| `do_lock` | bridge.py | Handler: asks the engine to lock the next round; returns only the round id and period. |
| `do_throw` | bridge.py | Handler: counts a throw from either a frame burst or a key press (exactly one of them); never receives the bot's move. |
| `do_correct` | bridge.py | Handler: changes the last counted throw (D18) through the engine. |
| `do_report` | bridge.py | Handler: the match report for one match. |
| `do_leaderboard` | bridge.py | Handler: the ranking of finished matches this session (names live in memory only). |
| `do_preview` | bridge.py | Handler: the gesture in one frame for the live "I see" label; one frame with any probability is enough. |
| `do_eyes_target` | bridge.py | Handler: one eyes-test throw; says whether the burst was accepted and whether it matched the gesture asked for. |
| `handle` | bridge.py | The dispatcher behind every widget call; checks the action and payload and never raises: it returns ok True with the answer, or ok False with an error. |
| `register` | bridge.py | Registers the Colab callback "mindreader.call" (inner function `call`) so the page reaches `handle`; Colab is imported inside so the module also loads on a laptop. |
| `js_string` | bridge.py | A value as JSON for inside a script tag, with "<" escaped so a closing script tag cannot end it early. |
| `widget_page` | bridge.py | Builds the page HTML from widget.html, widget.js and the settings (mode, brains, default brain). |
| `show_widget` | bridge.py | Displays the game ("stage", "practice") or the eyes test in the cell's output; `register` must have run first. |
| `wilson` | pilot.py | The 95% Wilson interval for k hits in n tries; None when n is 0. |
| `verdict` | pilot.py | D30: "my friends do it too" if the whole interval is above 1/3; "less predictable than the study's players" if below 52.7% (computed from the study's public data) with at least 60 rounds; else "inconclusive". |
| `export_pilot` | pilot.py | With consent exactly True, writes the finished practice matches to `data/pilot/pilot_<date>.csv` (ids P01, P02, ...), never overwriting a file. |
| `load_pilot` | pilot.py | Reads every pilot_*.csv into one table with a `file` column; refuses a file with the wrong columns. |
| `_count` | pilot.py | Small helper: k, n and the Wilson low and high ends as a dict. |
| `count_rounds` | pilot.py | Repeats right after a loss (k of n) and the human win rate against the bot, optionally for the first rounds only. |
| `summarize` | pilot.py | Players and rounds counted, plus the two counts for all rounds and for rounds 1 to 8 (D29). |
| `repeated_players` | pilot.py | Finds a player whose throws exactly match one in an earlier file: probably one match exported twice. |
| `_bars` | plots.py | Grouped bars for accuracy, precision, recall and F1, with the baselines as dashed lines. |
| `_folds` | plots.py | A boxplot of each model's cross-validation fold scores, every fold also drawn as a dot. |
| `_roc` | plots.py | ROC curves of the models that have probabilities, plus the chance line. |
| `_confusion` | plots.py | The confusion matrix of the model with the highest accuracy (first on a tie). |
| `comparison_figure` | plots.py | The shared 2x2 comparison figure (D31): bars, fold boxplots, ROC curves, the best model's confusion matrix. |
| `_scores` | plots.py | metrics[table] with every model and yardstick present, or a ValueError saying what is missing. |
| `_glance_panel` | plots.py | One panel of the glance figure: a bar per model in percent, yardsticks as named dashed lines. |
| `glance_figure` | plots.py | "Results at a glance": the eyes panel and the brain panel side by side, each next to its yardsticks. |

## D. Likely questions

Short answers. Brain figures always come with luck (33.3%), always-rock (35.6%) and always-repeat (44.0%).

**The project**

**Q1.** What is Mind-Reader in one sentence?
**A.** A webcam rock-paper-scissors bot called CHANAKYA: the eyes (MediaPipe hand points plus a scikit-learn model) name your gesture, and the brain (scikit-learn models over the last rounds) predicts your next move so the bot plays the move that beats it.

**Q2.** Which lab algorithms did you use, and where?
**A.** Find-S (Lab 3) for the scissors rule; Decision Tree (Lab 4); Logistic Regression and Perceptron (Lab 6); Naive Bayes (Lab 7); SVM (Lab 8), the last four in both parts; and Linear Regression (Lab 2) for the Personal brain's learning slope.

**Q3.** Is this supervised or unsupervised, classification or regression?
**A.** Supervised classification: the eyes pick one of three gestures, the brain one of three shifts (stay, up, down). The only regression is the straight-line slope in Cell 52.

**Q4.** What did you train, and what was already trained?
**A.** MediaPipe's hand model is pretrained by Google and only turns a picture into 21 points. Everything that decides, the six eyes models and five brains, we train with scikit-learn.

**Q5.** Where does the data come from?
**A.** Eyes: Laurence Moroney's Rock Paper Scissors pictures (computer-drawn hands, CC BY 2.0), turned into hand points by MediaPipe. Brain: Wang, Xu & Zhou (2014), 72 players with 300 rounds each, from the R package stratEst (GPL-3).

**Q6.** Why is the bot called CHANAKYA?
**A.** After the ancient Indian strategist known for reading his opponents (the notebook's opening cell says so).

**Honesty about the numbers**

**Q7.** Always repeat already gets 44%, so why is the brain worth anything?
**A.** Honest answer: the brain's offline accuracy is about 45.5%, only about 1.5 points above always-repeat (44.0%; always-rock 35.6%, luck 33.3%). The claim is evidence of predictability on unseen players, not that the bot wins.

**Q8.** Can your bot actually beat people?
**A.** I do not claim that. The study measured humans against humans, so the numbers say how predictable those players were, not how CHANAKYA does against you. A stage match is 8 rounds, a small sample, and a single match says little.

**Q9.** Where does 52.7% come from, and is it your result?
**A.** It is computed from the study's public data: after losing a round, players threw the same move again 52.7% of the time (n = 7,148), against 33.3% for a random player. It is the habit the brain learns, not our model's score.

**Q10.** Why is the brain's 45.5% so much lower than 52.7%?
**A.** They measure different things. 52.7% (computed from the study's public data) is the repeat rate in rounds right after a loss; 45.5% is accuracy over all rounds of unseen players. Right after a loss the tree scores 54.3% against always-repeat 52.7%, always-rock 35.8% and luck 33.3%.

**Q11.** Does the brain beat always-repeat reliably?
**A.** No. The tree is ahead in 3 of 5 folds: 45.5% ± 1.4% against always-repeat 44.0% ± 3.3% (luck 33.3%, always-rock 35.6%).

**Q12.** Why do the Decision Tree and the SVM score exactly the same?
**A.** Three simple features make only 27 possible situations. Both models end up predicting the most common next shift for each situation, and a plain lookup table does the same (45.5%, luck 33.3%, always-rock 35.6%, always-repeat 44.0%), so a fancier model finds nothing more.

**Q13.** Why is the Perceptron the worst brain?
**A.** It updates only on its mistakes and the labels are noisy and overlapping, so it never settles. Five seeds gave 36.5% to 41.0%, against 44.7% for logistic regression (luck 33.3%, always-rock 35.6%, always-repeat 44.0%).

**Q14.** The live SVM is fitted on only 20 players. Is that fair?
**A.** It is a different fit, kept small so it stays quick, and the notebook says so. On the 52 players it never saw it scores 43.9% against always-repeat 43.2% (luck 33.3%; no always-rock figure is printed for those rows). The default brain is the tree.

**Q15.** How good is the bot in the first rounds of a match?
**A.** Only just above the yardsticks. Personal brain, rounds 2-10: 40.0% against always-repeat 39.2%, always-rock 34.9%, luck 33.3%. Rounds 2-20: 42.4% against always-repeat 41.5%, always-rock 37.0%, luck 33.3%.

**Q16.** What is a "mind-read moment"?
**A.** A call the bot makes that comes true: the round-1 opener (assuming rock) or a predicted repeat after the player lost. The match report shows k of n so everyone sees how small the sample is.

**The eyes**

**Q17.** Why 47 features?
**A.** 42 are the 21 hand points as x, y after moving the wrist to the origin, turning the hand upright and scaling it; 5 are how far each finger reaches. Together they ignore where the hand is, how big it looks and which way it points.

**Q18.** What is a finger reach number?
**A.** The wrist-to-fingertip distance divided by the wrist-to-middle-joint distance. A curled finger gives a smaller number, and turning or resizing the hand does not change it.

**Q19.** Why mirrored twins, and why add them after the split?
**A.** A mirrored rock is still rock, so twins teach the model that left and right hands are the same. They go in after the split, only on the fit part, so a twin never sits across a split from its original.

**Q20.** How do a left and a right hand get the same answer live?
**A.** `predict_gesture` scores the hand and its mirror image in one call and averages the two answers.

**Q21.** Why does the eyes model stop at 99.2% and not 100%?
**A.** 3 of the 372 test pictures (all rock) have no hand found, and they count as wrong. That is the ceiling: 99.2% of 372.

**Q22.** Four models tie at 99.2%. How did you pick the live one?
**A.** Among models with `predict_proba` (all but the Perceptron) the best test accuracy wins, and the first listed wins a tie: logreg. Three models tie, so the pick cannot flatter the score.

**Q23.** Did you tune or choose on the test set?
**A.** Nothing was fitted or tuned on it. The test pictures were used once, to compare models and pick among the tie.

**Q24.** Cross-validation says 100% but nb scores 89.5% on the test. Why?
**A.** Near-copies of a fit picture land in other folds, and the held-out folds are training pictures that sit closer together than test pictures do (median distance 0.79 against 3.95). So CV is flattered; the official test is the honest number.

**Q25.** Does 99.2% mean it works on my hand?
**A.** Not known yet. The 99.2% is on computer-drawn hands. The real check is the live eyes test (Cell 60), which has not been run in Colab with real hands: that is gate G2. If it misses its pass marks, the plan is to record real webcam hands and retrain.

**Q26.** What are the pass marks of the live eyes test?
**A.** At least 85% of throws accepted and at least 95% of the accepted throws correct, over 3 or more people, both hands and 2 kinds of light.

**Q27.** Why does a throw need 3 of 5 frames and 70%?
**A.** A burst of 5 frames guards against one bad frame. The throw counts only if at least 3 frames agree and their mean probability is at least 0.7; otherwise the screen says "Show me your hand" and the round is tried again.

**Q28.** What if the camera fails during the demo?
**A.** Press K and play with keys 1, 2 and 3. After 3 rejected throws in a row the round also takes a key press. The full playbook is in E.

**Q29.** Why does Find-S fail on the noisy run, and what does it teach?
**A.** One scissors picture that disagrees turns a finger into `?`: 108 of 840 scissors pictures have the ring or pinky reach just over the 1.0 line. Find-S needs noise-free data, so real data needs the statistical learners.

**Q30.** Why is the thumb left out of Find-S?
**A.** Its joints do not follow the same up/down rule as the other four fingers.

**ML concepts**

**Q31.** What are entropy and information gain?
**A.** Entropy measures how mixed a box is: 0 for one class, highest for an even mix. Information gain is the entropy before a question minus the weighted entropy after it; the tree asks the question with the best gain.

**Q32.** Gini or entropy?
**A.** Both measure impurity. Gini is scikit-learn's default and the eyes tree uses it; Cell 26 tries both and finds ties.

**Q33.** Why is Naive Bayes "naive"?
**A.** It assumes the features are unrelated once the class is known. The eyes' numbers clearly move together (Cell 15), a likely reason `nb` is the weakest eyes model at 89.5%.

**Q34.** Logistic regression classifies, so why "regression"?
**A.** It fits a weighted sum of the features, like regression, then turns the sum into class probabilities.

**Q35.** What do a kernel, `C` and `gamma` do in an SVM?
**A.** A kernel (RBF) lets the boundary curve instead of being a straight line. `C` is the price of a wrong training row, and `gamma` is how far one row's influence reaches.

**Q36.** Why scale the features?
**A.** So no column dominates just because its numbers are bigger. SVM, logistic regression and the Perceptron are sensitive to that. The scaler sits inside the Pipeline and learns from training rows only.

**Q37.** Why one-hot encoding for the brain's features?
**A.** `o_rel1` is a category (tie, lost, won), not a size. One-hot gives each value its own 0/1 column; Categorical Naive Bayes instead wants category numbers.

**Q38.** What are accuracy, precision, recall and F1, and why macro?
**A.** Accuracy is the share right; precision is how many calls of a class were right; recall is how many real members of a class were found; F1 balances the two. Macro averages the classes equally, so small classes count.

**Q39.** How do you read your confusion matrices?
**A.** Rows are true, columns predicted. Eyes: all 369 pictures with a hand are on the diagonal. Brain: the tree says "stay" a lot (recall 0.75) and finds few "up" (0.24) or "down" (0.20).

**Q40.** What are ROC and AUC, and is AUC 1.000 suspicious?
**A.** ROC plots true against false positive rate as the threshold moves; AUC is the area under it (1.0 perfect, 0.5 chance). 1.000 only says the probabilities rank perfectly on these easy computer-drawn test pictures, not that real hands will.

**Q41.** What does GridSearchCV do, and why is its best score optimistic?
**A.** It tries every setting in a grid, scores each with cross-validation and keeps the best. The best is picked and scored on the same folds, so its score is slightly high.

**Q42.** Why GroupKFold instead of plain cross-validation?
**A.** To keep near-copies on one side: a picture with its twin in the eyes, all of a player's rounds in the brain. Otherwise one person could sit on both sides and flatter the model.

**Q43.** How big is the leakage in the brain?
**A.** Small. The same tree scored 46.1% on a random split of the rounds and 45.5% split by player (luck 33.3%, always-rock 35.6%, always-repeat 44.0%). With only three simple features it cannot memorise a person much.

**Q44.** Why `stratify` and `random_state=42`?
**A.** `stratify=y` keeps each gesture's share the same in the fit and validation parts; `random_state=42` makes the shuffle repeatable.

**Q45.** What is the difference between validation and test?
**A.** Validation is carved from the training pictures and has near-copies in the fit part, so it flatters. Test is TensorFlow's own test set, not near the training pictures and never fitted or tuned on.

**Q46.** Why a Pipeline?
**A.** Scaling or encoding plus the model are one object, so in cross-validation the scaler learns only from the training folds and the same steps run when predicting live.

**Q47.** What does a learning curve tell you?
**A.** Whether more data would help. The brain's unseen-player line is nearly flat, 44.8% to 45.5% from 5.7 to 57.0 training players (luck 33.3%, always-rock 35.6%, always-repeat 44.0%), so more players would probably not add much.

**Q48.** Are you overfitting?
**A.** In the eyes, train and validation are near 100% because of near-copies, which is why the test column is the one to trust. The brain shows little gap: at the end of the learning curve the tree scores 46.5% on its training players and 45.5% on unseen ones (luck 33.3%, always-rock 35.6%, always-repeat 44.0%).

**Q49.** What is a baseline (yardstick), and why three?
**A.** A trivial rule a model must beat to count: luck (a random guess), always-rock, always-repeat. A brain that cannot beat always-repeat adds nothing.

**Q50.** Why relative moves (stay, up, down) and not rock, paper, scissors?
**A.** Rock-paper-scissors is a circle, so "stay after a loss" is one habit whatever the move was. One relative label learns from all three moves at once: three times the examples.

**Q51.** What is the Personal brain, and what is Laplace smoothing in it?
**A.** A 3x3 table of counts, one cell per kind of last round and next shift, learned from this player's own game. Every cell starts at 1 (Laplace smoothing), so no shift has zero chance before it is seen. It works from round 2.

**Q52.** Why Linear Regression, and why subtract the always-repeat slope?
**A.** The line gives "points gained per 100 rounds": Personal +3.2, always-repeat +1.3, so Personal's own gain is +1.8. Players repeat more as a game goes on, so the plain slope would overstate what the brain learns.

**Q53.** What does MediaPipe do, and is it machine learning?
**A.** It is a pretrained neural-network hand detector from Google that returns 21 points per hand. We use it only as a feature extractor and do not train it.

**Q54.** What does OpenCV do here?
**A.** The picture plumbing: decode the camera's JPEG, convert BGR to RGB, draw the skeleton, encode the picture back. No learning.

**Q55.** Why depth 6 for the brain's tree, and why five folds?
**A.** The grid picked depth 6 with leaf size 1 (45.5% ± 1.4%; luck 33.3%, always-rock 35.6%, always-repeat 44.0%), but 6 of 15 settings tie, so the grid cannot tell them apart. Five folds is the labs' usual choice; each brain fold holds 14 or 15 players.

**The live game, privacy and the pilot**

**Q56.** Can the bot cheat by seeing my throw first?
**A.** No. It locks its move before the countdown, Python keeps it, and the page only gets a round number. `throw` is never given the bot's move, and the move comes back only with the counted throw.

**Q57.** How do the browser camera and Python talk?
**A.** At "shoot" the browser takes 5 frames and sends them in one call to a Colab callback (`mindreader.call`). Python checks them (count, size, JPEG), reads the hand, votes and replies.

**Q58.** What about privacy?
**A.** A consent prompt comes before the camera starts, frames are read in memory and never saved, names live in session memory, and the pilot file holds throws only and is written only if the box is ticked.

**Q59.** Why retrain at quick start instead of saving the models?
**A.** A pickled scikit-learn model breaks when Colab updates scikit-learn, so Cell 57 retrains whatever is missing.

**Q60.** Has the pilot been done? What did your friends show?
**A.** Not yet. No pilot files exist, and Cell 65 prints "No pilot files yet" (gate G3). I make no claim about my friends.

**Q61.** How will you judge the pilot?
**A.** By a rule fixed in advance (D30): count the rounds after a loss (n) and the repeats (k), take the 95% Wilson interval, and say "my friends do it too" only if the whole interval is above 1/3, "less predictable than the study's players" only if it is below 52.7% (computed from the study's public data) with at least 60 rounds, otherwise "inconclusive".

**Q62.** Why the Wilson interval?
**A.** It stays between 0 and 1 and still works for the few rounds a pilot has. It treats rounds as independent, which one player's rounds are not, so the real uncertainty is a little wider.

**Q63.** How long does Run all take?
**A.** The target is 5:00 or less on a fresh Colab, installs included, but it has not been timed there yet (gate G4). Do not quote a time.

**Q64.** What are the limitations?
**A.** The study was humans against humans, 72 players in one lab design; the eyes learned from computer-drawn hands; the pilot is small and not yet collected; a stage match is 8 rounds, a small sample.

**Q65.** What would you do next?
**A.** Record real webcam hands and train on them if the live eyes test misses its pass marks; run a bigger pilot; build a brain that mixes the study's habits with the player's own; try longer histories, tested on unseen players.

**Q66.** What do you say if the bot loses on stage?
**A.** "One match is luck": 8 rounds is a small sample. Show the match replay table and the study numbers (52.7% repeat after a loss, n = 7,148, computed from the study's public data), and the pilot only if it has numbers.

## E. The 5-minute run sheet and the what-if-it-fails playbook

### The 5-minute run sheet
From `docs/superpowers/specs/2026-10-27-mind-reader-design.md`; the "On screen" column is added here. The notebook is run in full before presenting, and the talk scrolls through the saved outputs. The words for each slot are in `PRESENTATION_SCRIPT.md`.

| Time | What happens | On screen |
|---|---|---|
| 0:00 | Hook: the 53% line | Cell 3 |
| 0:30 | Part 1 highlights: dataset, Find-S, comparison chart, confusion matrix | Cells 6, 10, 30, 23 |
| 1:30 | Part 2 highlights: the 53% chart, comparison against the luck line, the unseen-player test | Cells 36, 39, 54 |
| 2:30 | A classmate takes the seat and types a name (about 10 s) | Cell 58 |
| 2:40 | 8-round stage match (about 65 s) | Cell 58 |
| 3:45 | Match report: mind-read moments, then the pilot and study evidence (about 30 s) | Cell 58 report, Cell 65 |
| 4:15 | Spare time for a replay or a slow round | Cells 51, 52 |
| 4:45 | One-line conclusion | Cell 66 |

Three cautions. The "53%" is 52.7%, computed from the study's public data. The 3:45 slot has no pilot numbers yet (gate G3), so say that and lean on the study. The timings are the design's targets: the stage match and the shoot-to-reveal time have not been measured on a fresh Colab (gate G4).

### What if it fails?

| If | Do |
|---|---|
| The bot loses on stage | Say "one match is luck" (8 rounds is a small sample). Show the match replay table (labelled illustrative) and the study numbers: 52.7% repeat after a loss, n = 7,148, computed from the study's public data. Mention the pilot only if gate G3 has produced numbers. Equal wins is a draw. |
| The page says "Camera blocked: allow it in the address bar, or press K to play with keys" | Allow the camera in the address bar, or press K and throw with keys 1, 2, 3. To go back to the camera, press K again and click "I agree, start camera" once more. |
| The page says "Couldn't see your hand: press 1, 2 or 3 for this round" (3 rejected throws in a row) | Press 1, 2 or 3 for rock, paper, scissors. |
| The eyes misread a throw | After the reveal, and before the next round locks, press 1, 2 or 3 (or the "I actually threw" buttons) to correct it. |
| The page says "Python isn't answering: re-run the game cell" | Re-run Cell 58. It starts a fresh game session (earlier matches and names are forgotten). If the runtime restarted, run Cell 1 first; Cell 57 retrains what is missing. |
| The page says "Something went wrong in this page: press Start over" | Press Start over. |
| Cell 1 prints "RESTART NEEDED" | pip changed numpy after it was loaded: restart the Colab session and run from Cell 1. |
| Cell 1 prints "Couldn't download the project: check the internet and re-run" | Check the network or the hotspot and run Cell 1 again. |
| No speech | The captions stay. M mutes the voice. |
| Replies are slow (the plan wants the median shoot-to-reveal at 1.5 s or less, to be measured at gate G4) | The plan's fallback: 3 frames at quality 0.6; if it is still slow, play the stage match in keyboard mode. |
| A cell misbehaves live | Show the saved outputs instead of re-running it: the notebook is saved with outputs (also as `notebook/MindReader.html`). |
| A question you cannot answer | Say what the notebook prints and point to the cell. Never state a number the notebook has not printed. |
| You think of the local OpenCV window (Approach B) | It is not built: there is no `play_local.py` in the repo. Do not count on it. |
