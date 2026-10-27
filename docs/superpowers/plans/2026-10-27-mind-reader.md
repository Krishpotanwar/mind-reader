# Mind-Reader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Colab notebook, `MindReader.ipynb`, that trains and compares the lab algorithms on two public datasets (hand gestures, and real human rock-paper-scissors moves), then runs a live webcam rock-paper-scissors game against a bot that predicts the player's next move.

**Architecture:**
- Testable logic lives in a small package, `mindreader/`, with pytest tests:
  - move coding
  - the brain (features and brains)
  - Find-S
  - hand geometry
  - the eyes (MediaPipe + OpenCV)
  - the game engine
  - the Colab bridge
- The notebook's source is a jupytext percent file, `notebook/MindReader.py`. It imports the package and shows the lab-style pipelines inline. A build script turns it into an executed `MindReader.ipynb` and an HTML export.
- The live game is an HTML/JS widget in a Colab output cell. It calls one Python callback through `google.colab.kernel.invokeFunction`.

**Tech Stack:**
- Python 3.12, scikit-learn, pandas, numpy, matplotlib, seaborn
- OpenCV (opencv-contrib-python 5.0, installed as a MediaPipe dependency)
- MediaPipe 1.1.1, Tasks API (`HandLandmarker`)
- jupytext, nbconvert, pytest
- Playwright, for the widget smoke test
- vanilla JS

**Spec:** `docs/superpowers/specs/2026-10-27-mind-reader-design.md` (APPROVED 2026-10-27). The Decisions section below settles the spec's "Reviewer Concerns". Where the two differ, Decisions win.

## Global Constraints

**Repo and environment**
- Repo root: `/media/psf/project/ML Project/mind-reader`. Every path below is relative to it.
- Krish pushes. Agents never push.
- The venv is `.venv` (Python 3.12), built with uv:
  - Binary: `UV=$HOME/.cache/nidhinetra-vm/bin/uv`. If it's missing, install uv to `~/.local/bin`.
  - Environment: `UV_CACHE_DIR=$HOME/.cache/mind-reader-uv/cache` and `UV_PYTHON_INSTALL_DIR=$HOME/.cache/mind-reader-uv/python`.
  - Never install into `~/.cache/nidhinetra-vm/venv`.
- Every Python command runs with `OPENBLAS_CORETYPE=ARMV8`; numpy crashes on this VM without it.
- Tests: `OPENBLAS_CORETYPE=ARMV8 .venv/bin/python -m pytest -q`.

**Pinned dependencies**
- `mediapipe==1.1.1`.
- Hand model: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`, sha256 `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`, 7,819,105 bytes.
- MediaPipe 1.1.1 has no `mp.solutions`. Use `mediapipe.tasks.python.vision.HandLandmarker`, and always open it in a `with` block; otherwise its `__del__` raises a TypeError when Python exits.
- Measured on this VM: hands detected in 369 of 372 TensorFlow test images, at about 10 ms per image.

**Move coding** (used everywhere)
- Rock `r` = 0, paper `p` = 1, scissors `s` = 2.
- The move that beats m is `(m + 1) % 3`.
- A shift is `(now - previous) % 3`: 0 = stay, 1 = up (to the move that beats your last), 2 = down.
- `result` is always from the player's point of view.

**Notebook**
- `notebook/MindReader.py` (jupytext percent) is the source of truth. Never hand-edit `MindReader.ipynb`; rebuild it with `tools/build_notebook.sh`.
- Training sections follow the lab steps listed in the spec's Constraints.
- Every brain accuracy is shown next to luck (33.3%), always-rock and always-repeat, computed on the same rows.
- The notebook writes its key numbers to `build/metrics.json` (gitignored). Tests and Part 4 read that file.

**Code style:** plain code Krish can explain. Functions of 30 lines or fewer, descriptive names, no metaprogramming, and comments only where a line's purpose isn't obvious.

**Privacy:** no camera pixels are ever written to disk or committed. Names live in session memory only. A consent prompt comes before the camera starts.

**Commits:** each task ends with a commit of only its own files: `git add <files>`, then `git commit`. Before committing:
- Run `node /media/psf/project/.gitnexus/run.cjs detect-changes --scope all --repo .` from the repo root. Re-run it if the result says `partial` or `truncated`.
- Before editing an existing function: `node /media/psf/project/.gitnexus/run.cjs impact <symbol> --direction upstream --repo .`

**Worker prompts:** every worker prompt says to run in ponytail ultra mode: laziest working solution and terse output, never dropping validation, security, accessibility or required behaviour. It also names the skills to use:
- superpowers:test-driven-development for module code
- superpowers:systematic-debugging on any failure
- superpowers:verification-before-completion before reporting done
- superpowers:brainstorming only if the brief leaves a real choice open. Then stop and report the options instead of guessing.

## Review Focus

1. **No hand, half a hand, or two hands in the burst.** The throw is rejected with "Show me your hand", nothing crashes, and the round isn't counted. After 3 rejections in a row, the round takes a 1/2/3 key throw. Tests: Task 4 `test_read_throw_rejects_*`; Task 7 `test_three_rejections_enable_keys`.
2. **Left hand vs right hand, and the mirrored preview.** Both give the same predicted gesture. Tests: Task 3 `test_extension_ratios_mirror_invariant`; Task 4 `test_predict_gesture_uses_mirror_average`.
3. **A correction arrives while stage mode is auto-advancing.** The next lock is computed from the corrected history, and a correction sent after the next lock is refused. Tests: Task 7 `test_correction_recomputes_next_lock` and `test_correction_refused_after_next_lock`.
4. **Malformed input from the browser:** wrong frame count, an oversized or non-JPEG frame, a stale round id, an unknown brain, HTML in a name. Each returns a clear error dict, no exception escapes, and names render as text. Tests: Task 4 `test_validate_frames_*`; Task 8 `test_handle_*` and the Playwright name test.
5. **A recycled Colab runtime, or re-run cells.** The quick-start cell restores the live game in 20 s or less, and registering the callback twice does no harm. Tests: Task 8 `test_register_is_idempotent`; Task 9 `test_quick_start_under_20s`.

---

## Decisions (these settle the spec's Reviewer Concerns)

**D1 to D4: the eyes**
- **D1, mirroring:** no handedness columns, and no pixel flip of webcam frames. The preview is mirrored with CSS only (`transform: scaleX(-1)`), and frames are sent unflipped.
- **D2, mirror average:** the eyes predict by averaging `predict_proba` over the hand features and the features of the mirrored hand. Official test-set accuracy is reported both as-is and mirrored; the target is 95% or more for both.
- **D3, live model must give probabilities:** the live eyes model is the best Part 1 model that has `predict_proba` (Logistic Regression, SVM with `probability=True`, Gaussian NB or Decision Tree).
- **D4, burst vote:** for each frame, predict a (label, probability) pair. A frame with no hand counts as disagreeing.
  - The winning label is the most frequent one among detected frames.
  - The throw is accepted if at least `need` frames (default 3 of 5) agree and their mean probability is at least `min_prob` (default 0.7).
  - Tune `need` and `min_prob` only on Krish's spike data, never on the people in the pass test.

**D5 to D9: data and evidence**
- **D5, test-set denominator:** test images where no hand is detected count as errors, so the denominator is all 372. The detection rate is printed too.
- **D6, leakage check:** soften "different hand models". Part 1 adds a nearest-neighbour check comparing train→test distances with train→train distances.
- **D7, numbers on row sets:**
  - 35.7% rock = all 21,600 rows.
  - 35.6% always-rock = the 21,456 model rows (history of 2 or more rounds).
  - 44.0% always-repeat = the same model rows.
  - Early windows, always-rock / always-repeat / Personal:
    - rounds 2–10: 34.9% / 39.2% / 40.0%
    - rounds 2–20: 37.0% / 41.5% / 42.4%
  - Print every one of these with its n.
- **D8, `input` column:** dropped. It equals the player's previous choice on 100% of rows.
- **D9, licences:** repo code is MIT.
  - `data/LICENSES.md` credits the TensorFlow images (Laurence Moroney, CC BY 2.0).
  - It also credits the WXZ2014 data: the paper (Wang, Xu & Zhou 2014, Sci Rep 4:5830), its distribution in CRAN stratEst 1.1.8 (GPL-3), and a GPL-3 notice for the converted CSV.

**D10: the brain**
- Target: the relative shift.
- Features: `o_rel1`, `shift1` and `o_rel2`, where:
  - `o_rel` = (opponent − player) % 3
  - 1 = the player lost, 2 = the player won, 0 = a tie
- Encoding: one-hot inside a ColumnTransformer for the Decision Tree, Logistic Regression, Perceptron and SVM; OrdinalEncoder for CategoricalNB.
- Measured, 5-fold GroupKFold: Decision Tree 45.5, SVM 45.5, Logistic Regression 44.7, Naive Bayes 44.4, Perceptron 35.2.

**D11 and D12: the Personal brain**
- **D11, context:** the opponent's relative move only (`o_rel1`), giving a 3 × 3 table of next-shift counts with Laplace smoothing (every cell starts at 1).
  - It is rebuilt from the history on every prediction, so it is stateless and corrections need no special handling.
  - Ties break stay > up > down (numpy argmax order).
  - Replay accuracy: 40.0% (rounds 2–10), 42.4% (rounds 2–20).
- **D12, cold start:**
  - Round 1: the bot assumes rock (58.3% of first throws) and plays paper.
  - Round 2: Personal brain.
  - Round 3 onwards: the selected brain.

**D13 and D14: what the screen shows**
- **D13, reveal percentage:** shown only when the brain has one.
  - Personal: its count share.
  - Model brains: `predict_proba`, except the Perceptron and Random, which show none.
  - Round 1: "58% of people open with rock".
- **D14, mind-read moment:**
  - (a) Round 1, the bot assumed rock and the player threw rock; or
  - (b) the previous result was a loss, the predicted shift was stay, and the player stayed.
  - n = the number of calls of type (a) or (b) that were made; k = the number that came true.
  - The spoken "Called it" line and the report use the same rule.

**D15 to D17: match rules and the report**
- **D15, match rules:** a health bar counts wins, and ties do no damage. A match is a fixed length: 8 rounds in stage mode, 20 in practice. More wins takes the match, equal wins is a draw, and there's no early finish.
- **D16, report and leaderboard:**
  - Unpredictability = 1 − (correct bot predictions ÷ predictions made in rounds 2..N), shown as k/n with a "small sample" note.
  - The brain-replay table is labelled "illustrative".
- **D17, pacing and keys:**
  - Stage mode auto-advances; P pauses.
  - Practice mode advances on Space.
  - K toggles keyboard mode; 1, 2 and 3 throw rock, paper and scissors in keyboard mode, or correct the last throw after a reveal; M mutes.

**D18 to D22: the browser boundary**
- **D18, corrections:** accepted only for the last completed round, and only before the next lock. Stage mode locks the next round only after a 1 s pause, so corrections made during the reveal or the pause go through.
- **D19, the throw handler:** it receives `(match_id, round_id, frames)` and never the bot's move. The engine's `throw` receives `(match_id, round_id, player_move)`.
- **D20, input validation:**
  - Exactly 3 or 5 frames.
  - Each frame at most 250,000 characters of base64 JPEG, decodable, with a longest side of 1280 px or less.
  - The round id must be the active one, and the brain id must be on an allow-list.
  - Names: stripped of control characters and at most 20 characters, and the widget renders them with `textContent`.
- **D21, failure messages (exact copy):**
  - Camera blocked: "Camera blocked: allow it in the address bar, or press K to play with keys"
  - No reply: "Python isn't answering: re-run the game cell"
  - No speech: captions only
  - 3 rejections in a row: "Couldn't see your hand: press 1, 2 or 3 for this round"
- **D22, consent overlay (exact copy):** "This game sends camera frames to this Colab session on Google's servers only to read your hand. Nothing is saved." Button: "I agree, start camera".

**D23 to D27: logs, checks and timing**
- **D23, pilot logs:**
  - Columns exactly `id, game, period, choice, other.choice, result`, with `id` an anonymous `P01`… and `other.choice` the bot's move.
  - Files go in `data/pilot/` only after a consent checkbox. Throws only, no images.
- **D24, notebook clean check:** `tools/check_notebook_clean.py` fails if a cell tagged `live` has outputs, or if any output contains `image/jpeg`.
- **D25, live models are retrained, not pickled:** the quick-start cell retrains the two live models (scikit-learn pickles break across Colab versions).
- **D26, time budgets:** fresh Colab "Run all" ≤ 5:00, broken down as setup ≤ 90 s, Part 1 ≤ 60 s, Part 2 ≤ 90 s and the rest ≤ 30 s.
  - The brain SVM is tuned and fitted on the first 20 players (5,960 rows).
  - Grids:
    - Decision Tree: `max_depth` ∈ {3, 4, 5, 6, 8}, `min_samples_leaf` ∈ {1, 20, 50}
    - Logistic Regression: `C` ∈ {0.1, 1, 10}
    - SVM: `C` ∈ {0.5, 1, 2}
  - Report the numbers the notebook actually produces.
- **D27, the Perceptron story:** it updates only on its mistakes and never settles on noisy, overlapping labels. The notebook shows its accuracy over 5 seeds next to Logistic Regression's.

**D28 to D30: pass marks and wording**
- **D28, pass marks:** mock viva 8 of 10. Live eyes test: accuracy ≥ 95% on accepted throws and ≥ 85% of throws accepted, across 3 or more people, both hands and 2 lighting conditions. A miss triggers Task 1b.
- **D29, wording:**
  - Say "computed from the study's public data" and "evidence of predictability on unseen players".
  - Report the pilot with a Wilson 95% interval, and report rounds 1–8 separately.
  - Pilot players must not have heard the pitch.
- **D30, the Assignment's second line** ("my friends were less predictable") is allowed only if the interval's upper bound is below 52.7% and n is at least 60. Otherwise say "inconclusive". Keep one opponent type per pooled count.

**D31 and D32: what the professor marks (added 2026-10-27, from today's presentations by other students)**
- **D31, graphical comparison of at least 4 algorithms.** The professor mainly looked at results compared graphically across at least 4 algorithms. Every comparison figure is produced by one shared helper, `mindreader/plots.py`:
  - `comparison_figure(results: pd.DataFrame, title: str, baselines: dict[str, float] | None = None) -> matplotlib.figure.Figure` draws a 2×2 grid:
    - accuracy, precision, recall and F1 as grouped bars, with the baselines (luck, always-rock, always-repeat) drawn as dashed lines
    - a boxplot of the cross-validation fold scores for each model
    - ROC curves overlaid, for the models with `predict_proba`
    - the best model's confusion matrix
  - `results` has one row per model and the columns `model, accuracy, precision, recall, f1, cv_scores` (a list), `roc` (a tuple of fpr, tpr and auc, or None) and `cm` (a 2D list).
  - Parts 1 and 2 each end with a comparison figure.
  - Part 0 shows a "Results at a glance" figure: both parts' accuracy bars side by side, read from `build/metrics.json` on a rebuild, with a placeholder note on the first run.
  - Every figure has a title, axis labels, value labels on the bars, and colours that stay readable in greyscale.
- **D32, word-for-word presentation script.** `PRESENTATION_SCRIPT.md` gives the exact words for the 5-minute run sheet: about 650 words at roughly 140 words per minute. Each paragraph names the notebook cell or figure on screen. It ends with the 3 most likely classmate questions and their 20-second answers.

## Schedule and cut line (presentation window 2026-11-03 to 2026-11-10)

| Dates | Work |
|---|---|
| Oct 27–28 | Tasks 1–4, then hand the spike to Krish (G1) |
| Oct 29 | Tasks 5–7 |
| Oct 30 | Task 8, then G2 (Krish's Colab checkpoint and the 3-person eyes test) |
| Oct 31 | Task 9, then G3 (pilot) |
| Nov 1 | Tasks 10–11; mock viva |
| Nov 2 | G4 (final test and venue dry run) |
| Nov 3 onwards | Buffer |

- **Must** (minimum viable demo): Tasks 1–8, 11, and G4.
- **Should:** Task 9.
- **Could:** the leaderboard, Task 1b, and the pilot size.
- If late, cut in this order: leaderboard → pilot size → practice logging → Task 1b → spoken taunts (captions stay).

## File map

**`mindreader/`**

| File | Responsibility |
|---|---|
| `__init__.py` | (empty) |
| `moves.py` | Move coding, outcome, shift and rel helpers |
| `brain.py` | Load games, features, facts, pipelines, Personal/Model/Random brains, cold-start `decide` |
| `find_s.py` | Find-S with a change log (Lab 3) |
| `hands.py` | Landmark geometry: pixels, canonical pose, mirror, extension ratios, features, finger states |
| `eyes.py` | `ensure_model`, `HandReader`, JPEG decode/encode, mirror-averaged prediction, burst vote, skeleton drawing |
| `game.py` | `GameEngine`: matches, lock, throw, corrections, scoring, moments, report, leaderboard, export |
| `bridge.py` | Input validation, the `handle` dispatcher, Colab callback registration, `show_widget` |
| `widget/widget.html` | Face-off layout, consent overlay, report panel |
| `widget/widget.js` | Camera, countdown, speech plus captions, burst capture, preview, keys, failure states |

**`notebook/` and the built notebook**

| File | Responsibility |
|---|---|
| `notebook/MindReader.py` | Notebook source (jupytext percent) |
| `MindReader.ipynb` | Built and executed |
| `MindReader.html` | Built export |

**`tools/`**

| File | Responsibility |
|---|---|
| `rda_to_csv.py`, `rps_check.py`, `personal_check.py`, `detect_check.py` | Already copied in (exploration scripts; adapt them) |
| `make_hand_csv.py` | Hand images to landmarks CSV |
| `rps_facts.py` | Prints the facts in D7 |
| `build_notebook.sh` | Builds the notebook and the HTML export |
| `check_notebook_clean.py` | The D24 check |

**Other files**

| File | Responsibility |
|---|---|
| `data/` | `rps_human_games.csv`, `hand_landmarks.csv`, `LICENSES.md`, `pilot/` |
| `spike/spike.py` → `spike/Spike.ipynb` | Throwaway Colab spike |
| `tests/` | One test file per module, plus `test_data_files.py`, `test_notebook_build.py`, `test_widget_smoke.py`, `widget_harness.html` |
| `docs/mockups/` | `game-layout.html` (option B is the Face-off), `notebook-results.html` |
| Repo root | `README.md`, `LICENSE`, `VIVA_GUIDE.md`, `requirements.txt`, `requirements-dev.txt`, `.gitignore` |

---

### Task 1: Repo, environment, data CSVs, HandReader

**Worker:** sonnet-max · **Review:** Opus

**Files:**
- Create:
  - `.gitignore` (`.venv/`, `__pycache__/`, `.pytest_cache/`, `.gitnexus/`, `*.task`, `data/raw/`, `build/`, `.ipynb_checkpoints/`)
  - `LICENSE` (MIT, "Krish Potanwar")
  - `requirements.txt`: `mediapipe==1.1.1`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`
  - `requirements-dev.txt`: `-r requirements.txt`, `pytest`, `jupytext`, `nbconvert`, `nbclient`, `ipykernel`, `playwright`
  - `mindreader/__init__.py`, `mindreader/eyes.py` (only `MODEL_URL`, `MODEL_SHA256`, `ensure_model` and `HandReader` here)
  - `tools/make_hand_csv.py`
  - `data/rps_human_games.csv`, `data/hand_landmarks.csv`, `data/LICENSES.md` (D9)
  - `tests/test_data_files.py`
- Modify: `tools/rda_to_csv.py`, so it drops the `input` column (D8).

**Interfaces:**
- Produces: `ensure_model(path: str = "hand_landmarker.task") -> str`. It downloads the model if missing, checks the sha256, and raises `ValueError("hand model checksum mismatch")` on a mismatch.
- Produces: `HandReader(model_path: str)`, a context manager with `num_hands=1`.
  - `.landmarks(rgb: np.ndarray) -> np.ndarray | None` returns a (21, 3) array of normalised x, y, z, or None when no hand is found.
- Produces two CSV schemas:
  - `rps_human_games.csv` = `id,game,period,choice,other.choice,result` (21,600 rows)
  - `hand_landmarks.csv` = `file,split,label,width,height,x0,y0,z0,…,x20,y20,z20` (one row per image with a hand; split ∈ {train, test}; label ∈ {rock, paper, scissors})

- [ ] **Step 1:** `git init`. Write `.gitignore`, `LICENSE` and both requirements files. Create the venv and install: `$UV venv --python 3.12 .venv && $UV pip install --python .venv/bin/python -r requirements-dev.txt`.
- [ ] **Step 2:** Write the failing tests in `tests/test_data_files.py`:

```python
def test_games_csv_shape_and_columns():
    g = pd.read_csv("data/rps_human_games.csv")
    assert list(g.columns) == ["id", "game", "period", "choice", "other.choice", "result"]
    assert len(g) == 21600 and g.id.nunique() == 72 and set(g.choice) == {"r", "p", "s"}

def test_games_results_follow_rules():
    g = pd.read_csv("data/rps_human_games.csv")
    beats = {("r", "s"), ("p", "r"), ("s", "p")}
    expected = ["tie" if a == b else ("win" if (a, b) in beats else "lose")
                for a, b in zip(g.choice, g["other.choice"])]
    assert (g.result == pd.Series(expected)).all()

def test_hand_csv_detection_and_schema():
    h = pd.read_csv("data/hand_landmarks.csv")
    assert {"file", "split", "label", "width", "height", "x0", "z20"} <= set(h.columns)
    assert (h.split == "test").sum() >= 365 and (h.split == "train").sum() >= 2400
    assert set(h.label) == {"rock", "paper", "scissors"}
```

- [ ] **Step 3:** Run `pytest tests/test_data_files.py -q`. Expected: 3 failures (files missing).
- [ ] **Step 4:** Build `data/rps_human_games.csv`:
  - `curl -sSL --retry 5 -o data/raw/stratEst.tar.gz https://cran.r-project.org/src/contrib/stratEst_1.1.8.tar.gz`
  - Extract `stratEst/data/data.WXZ2014.rda`, then run `python tools/rda_to_csv.py <rda> data/rps_human_games.csv`.
- [ ] **Step 5:** Write `ensure_model` and `HandReader` (logic as in `tools/detect_check.py`, with a `with` block). Then write `tools/make_hand_csv.py <rps.zip> <rps-test-set.zip> <out.csv>`:
  - Use `curl --retry 5 -C -` for the 200 MB download into `data/raw/`; the VM's network drops connections.
  - It prints the detection rate per split and class.
- [ ] **Step 6:** Write `data/LICENSES.md` (D9). Index the repo: `node /media/psf/project/.gitnexus/run.cjs analyze --index-only .`
- [ ] **Step 7:** Run `pytest -q`. Expected: all pass. Commit with `feat: data, environment and hand reader`, including the copied `tools/*` and `docs/`.

### Task 2: Moves and brain

**Worker:** sonnet-max · **Review:** Opus

**Files:**
- Create: `mindreader/moves.py`, `mindreader/brain.py`, `tools/rps_facts.py`, `tests/test_moves.py`, `tests/test_brain.py`

**Interfaces:**
- Produces in `moves.py`:
  - `code(m: str) -> int`
  - `letter(i: int) -> str`
  - `beats(m: int) -> int`
  - `outcome(player: int, opponent: int) -> str` (returns "win", "lose" or "tie")
  - `shift(prev: int, now: int) -> int`
  - `apply_shift(prev: int, s: int) -> int`
  - `rel(opponent: int, player: int) -> int`
  - `NAMES = {0: "rock", 1: "paper", 2: "scissors"}`
- Produces in `brain.py`:
  - `FEATURES = ["o_rel1", "shift1", "o_rel2"]`
  - `load_games(path) -> pd.DataFrame`: adds integer columns `c` and `o`, sorted by id and period.
  - `add_features(games) -> pd.DataFrame`: rows for period ≥ 3 only, with columns `id, period, choice, o_rel1, shift1, o_rel2, shift`.
  - `facts(games) -> dict`, with keys `repeat_after_loss`, `n_after_loss`, `first_rock`, `n_first`, `rock_share_all`, `always_rock_model_rows`, `always_repeat_model_rows`, `n_model_rows`.
  - `history_features(history: list[tuple[int, int]]) -> dict | None`: history is a list of (player_move, bot_move) pairs; returns None if it has fewer than 2 rounds.
  - `brain_pipeline(name: str) -> Pipeline` for names in {"nb", "dt", "logreg", "perceptron", "svm"}, using the D10 encoders. Defaults: dt `max_depth=6`; logreg `max_iter=1000`; svm `probability=True`; `random_state=42` wherever it applies.
  - `PersonalBrain().predict(history) -> tuple[int, float]` (D11)
  - `ModelBrain(pipeline, name).predict(history) -> tuple[int, float | None]`
  - `RandomBrain(seed=0).predict(history) -> tuple[int, None]`
  - `decide(history, brain_name: str, brains: dict) -> dict`, with keys `bot_move`, `predicted_move`, `prob`, `source` (D12, D13)

- [ ] **Step 1:** Write the failing tests:
  - `test_moves.py`:
    - `beats` cycles 0→1→2→0.
    - For all 9 pairs, `rel(o, p) == 1` exactly when `outcome(p, o) == "lose"`, `2` when "win", and `0` when "tie".
    - `shift` / `apply_shift` round-trip.
  - `test_brain.py`:
    - `test_facts_match_study_data`: `n_after_loss == 7148`, `round(repeat_after_loss, 3) == 0.527`, `n_first == 72`, `round(first_rock, 3) == 0.583`, `round(rock_share_all, 3) == 0.357`, `n_model_rows == 21456`, `round(always_rock_model_rows, 3) == 0.356`, `round(always_repeat_model_rows, 3) == 0.440`.
    - `test_add_features_rows_and_columns`: 21,456 rows and no NaN.
    - `test_history_features_matches_add_features`: for player 1, periods 3, 10, 150 and 300.
    - `test_personal_brain_replay_matches_measured`: over all 72 players, predict each round t from rounds < t. Rounds 2–10 average `0.400 ± 0.005`; rounds 2–20 average `0.424 ± 0.005`.
    - `test_personal_tie_break_prefers_stay`: `PersonalBrain().predict([(0, 1)]) == (0, 1/3)`.
    - `test_decide_cold_start`:
      - `[]` gives `bot_move == 1`, `predicted_move == 0`, `prob == 0.583`, `source == "opener"`.
      - One round gives `source == "personal"`.
      - Two rounds give `source == "dt"`.
    - `test_model_brain_proba`: None for the perceptron, a float for dt.
    - `test_random_brain_seeded`.
- [ ] **Step 2:** Run them. Expected: FAIL (module missing).
- [ ] **Step 3:** Implement `moves.py`, then `brain.py`, with the signatures above.
- [ ] **Step 4:** Run `pytest tests/test_moves.py tests/test_brain.py -q`. Expected: PASS.
- [ ] **Step 5:** Write `tools/rps_facts.py`, which prints every D7 figure with its n. Run it and check that the printed values match D7.
- [ ] **Step 6:** Commit with `feat: move coding and brains`.

### Task 3: Find-S and hand geometry

**Worker:** sonnet-max · **Review:** Opus

**Files:**
- Create: `mindreader/find_s.py`, `mindreader/hands.py`, `tests/test_find_s.py`, `tests/test_hands.py`

**Interfaces:**
- Produces in `find_s.py`:
  - `find_s(examples: list[tuple[str, ...]], positive: list[bool]) -> tuple[tuple[str, ...] | None, list[tuple[int, tuple[str, ...]]]]`, returning the final hypothesis and a list of (example index, hypothesis after that change).
  - `matches(hypothesis, example) -> bool`
- Produces in `hands.py`:
  - Constants: `WRIST = 0`, `MIDDLE_MCP = 9`, `TIPS = (4, 8, 12, 16, 20)`, `JOINTS = (2, 6, 10, 14, 18)`.
  - `to_pixels(landmarks, width, height) -> np.ndarray` (21, 2).
  - `canonical(points) -> np.ndarray`: wrist at the origin, wrist→point 9 rotated to point straight up (0, -1), scaled to unit length.
  - `mirror(points) -> np.ndarray`: negates x.
  - `extension_ratios(points) -> np.ndarray` (5,): |tip − wrist| ÷ |joint − wrist|.
  - `hand_features(points) -> np.ndarray` (47,): the canonical points flattened (42), plus the 5 ratios.
  - `FEATURE_NAMES` (47 names).
  - `finger_states(ratios, up=1.15, down=0.95) -> tuple[str, ...]`: for index, middle, ring and pinky, each "up", "down" or "unclear".
  - `features_from_row(row) -> np.ndarray`, for a `hand_landmarks.csv` row.

- [ ] **Step 1:** Write the failing tests:
  - `test_find_s_enjoysport`: Lab 3's 4 EnjoySport rows. The final hypothesis is `("Sunny", "Warm", "?", "Strong", "?", "?")` and the changes happen at indices [0, 1, 3].
  - `test_find_s_scissors_concept`: gives `("up", "up", "down", "down")`.
  - `test_find_s_noise_collapses`: one noisy positive turns that attribute into "?".
  - `test_canonical_invariant_to_translation_rotation_scale` (`atol=1e-9`).
  - `test_canonical_puts_middle_knuckle_up`: point 9 ≈ (0, -1).
  - `test_extension_ratios_mirror_invariant`.
  - `test_features_shape_47`.
  - `test_to_pixels_removes_aspect_skew`: the same hand normalised for a 640×480 frame and for a 480×640 frame gives equal canonical features.
  - `test_finger_states_thresholds`: ratios [x, 1.2, 1.2, 0.9, 0.9] give `("up", "up", "down", "down")`, and 1.0 gives "unclear".
  - `test_features_from_csv_rows_finite`: the first 50 CSV rows.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement. The canonical rotation angle is `atan2(v_x, -v_y)`, where `v` = point 9 − wrist; rotate every point by `-angle`.
- [ ] **Step 4:** Run them. Expected: PASS.
- [ ] **Step 5:** Commit with `feat: Find-S and hand geometry`.

### Task 4: Eyes, validation, and the Colab spike

**Worker:** sonnet-max · **Review:** Opus

**Files:**
- Modify: `mindreader/eyes.py`
- Create:
  - `mindreader/bridge.py` (validation only: `validate_frames`, `clean_name`, plus the constants `FRAME_COUNTS = (3, 5)`, `MAX_FRAME_CHARS = 250_000`, `MAX_SIDE = 1280`, `MAX_NAME = 20`)
  - `tests/test_eyes.py`, `tests/test_bridge_validation.py`
  - `spike/spike.py` → `spike/Spike.ipynb`

**Interfaces:**
- Consumes: `HandReader` (Task 1); `hand_features`, `mirror`, `to_pixels` and `finger_states` (Task 3).
- Produces in `eyes.py`:
  - `decode_jpeg(data_url: str) -> np.ndarray` (BGR; raises `ValueError` on undecodable input)
  - `encode_jpeg(bgr, quality=80) -> str` (a data URL)
  - `predict_gesture(model, points) -> tuple[str, float]` (D2)
  - `read_throw(reader, model, frames_bgr, need=3, min_prob=0.7) -> dict`, with keys `label` (str or None), `prob`, `agree`, `best` and `landmarks` (D4)
  - `draw_skeleton(bgr, landmarks) -> np.ndarray`
  - `CONNECTIONS`: the 21-point hand connections, written out explicitly
- Produces in `bridge.py`:
  - `validate_frames(frames: list) -> list[np.ndarray]` (D20; raises `ValueError` with a message)
  - `clean_name(name: str) -> str`

- [ ] **Step 1:** Write the failing tests:
  - `test_decode_jpeg_roundtrip`, `test_decode_jpeg_rejects_garbage`.
  - `test_validate_frames_count` (4 frames fail), `test_validate_frames_size`, `test_validate_frames_dimensions` (2000×2000 fails).
  - `test_clean_name_strips_and_truncates`.
  - `test_predict_gesture_uses_mirror_average`: a stub model returns [1, 0, 0] for the hand and [0, 1, 0] for its mirror, so the result is `("rock", 0.5)`.
  - `test_read_throw_accepts_3_of_5`: 4 hands and 1 None; labels rock/rock/rock/paper with probabilities .9/.8/.7/.9, so the result is rock, `agree == 3`, `prob == 0.8`.
  - `test_read_throw_rejects_low_confidence`: 3 agree at a mean of 0.6, so `label` is None.
  - `test_read_throw_rejects_no_hand`.
  - `test_draw_skeleton_draws`.
  - `test_live_path_matches_csv` (skipped if `data/raw` is missing): send 20 test images through JPEG encode → decode → landmarks → features, both as-is and flipped.
    - As-is: labels match the CSV pipeline for at least 19 of 20, and the ratios are within 0.02.
    - Flipped: compare the ratios and labels only.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement `eyes.py` and the validation in `bridge.py`.
- [ ] **Step 4:** Run them. Expected: PASS.
- [ ] **Step 5:** Write `spike/spike.py` (jupytext). Throwaway code; the label comes from a `finger_states` heuristic. Cells:
  1. `pip install mediapipe==1.1.1`. Print the numpy, opencv and mediapipe versions before and after, and print "RESTART NEEDED" if numpy was already imported at a different version.
  2. `ensure_model()`.
  3. Camera with consent (D22) and a CSS-mirrored preview. "Run 20 bursts": 5 JPEG frames (640×480, quality 0.8), 80 ms apart, sent in one `invokeFunction` call, with a 4 fps single-flight preview running. Print the median and p90 round-trip in ms.
  4. Real-hand test: rock, paper and scissors 10 times each with each hand. Print a detection-percentage table, and offer a download of the landmark rows (no images).
  5. Speech: say "Rock, paper, scissors, shoot!" while a big SHOOT flashes. Krish notes whether the voice and the flash line up.
  6. Print a summary block for Krish to paste back.

  Build `spike/Spike.ipynb` with `jupytext --to ipynb`.
- [ ] **Step 6:** Commit with `feat: eyes, input validation and Colab spike`.

**HUMAN GATE G1 (Krish):** open `spike/Spike.ipynb` in Colab on the demo laptop and browser, run every cell, and paste back the summary block and the landmark CSV.
- **Approach B** is built if the camera or `invokeFunction` fails, or the median exceeds 1.5 s. If the median is between 1.5 and 2.5 s, first try 3 frames at quality 0.6.
- `need` and `min_prob` are tuned on Krish's CSV.
- The capture offset is set from the speech check.

### Task 5: Notebook build, Part 0 and Part 2 (the brain)

**Worker:** sonnet-max · **Review:** Opus

**Files:**
- Create: `notebook/MindReader.py`, `tools/build_notebook.sh`, `tools/check_notebook_clean.py`, `tests/test_notebook_build.py`, `mindreader/plots.py` (D31), `tests/test_plots.py`
- Extra step before Step 1: write `tests/test_plots.py::test_comparison_figure_has_four_panels_and_labels`. Given 4 fake models, it checks for 4 axes, value labels on the bars, the dashed baseline lines and non-empty titles. Then implement `comparison_figure` per D31.
- Part 2 ends with `comparison_figure(...)` over at least 4 models, with the luck, always-rock and always-repeat baselines.

**Interfaces:**
- Consumes: `brain.*` and `moves.*` (Task 2).
- Produces:
  - `build/metrics.json` keys: `brain_table` (model → accuracy), `brain_early` (window → accuracy per brain and baseline), `personal_slope_per_100`, `facts`.
  - Notebook variables used later: `BRAINS` (dict, name → `ModelBrain`/`PersonalBrain`/`RandomBrain`), `BEST_BRAIN` (str).
  - `tools/build_notebook.sh` runs `jupytext --to ipynb`, then `jupyter nbconvert --execute --inplace` (timeout 600 s), then `check_notebook_clean.py`, then `nbconvert --to html`.
  - Outside Colab, the env var `MR_SKIP_LIVE=1` makes live cells print a placeholder.

- [ ] **Step 1:** Write the failing test `test_notebook_builds_parts_0_2`: run `tools/build_notebook.sh`, then assert that no cell output has `output_type == "error"`, that metrics.json has `brain_table["dt"] >= 0.44`, that `facts.repeat_after_loss` rounds to 0.527, and that the Part 2 execution time is ≤ 90 s (from cell timing metadata). Also add `test_check_notebook_clean_flags_live_outputs` against a 2-cell fixture.
- [ ] **Step 2:** Run it. Expected: FAIL.
- [ ] **Step 3:** Write Part 0:
  - Title, the 53% hook (computed from data with `facts`), and the CHANAKYA intro: the bot is named after the ancient Indian strategist known for reading his opponents.
  - Setup cell: in Colab, `git clone --depth 1 https://github.com/Krishpotanwar/mind-reader.git` if absent, and `%cd`. If that fails, print "Couldn't download the project: check the internet and re-run". Then `pip install -q mediapipe==1.1.1` and `ensure_model()`.
  - Imports, and a version print.
- [ ] **Step 4:** Write Part 2 with the lab steps:
  - Data checks: `head`, `tail`, `shape`, `info`, `describe`; nulls (round 1 has no history); duplicates; class counts.
  - Facts with their n (D7).
  - Plots: choice and result distributions; the repeat-rate-by-last-result bar chart with the 33% line; transition heatmaps split by last result.
  - Features, with a markdown explanation of the relative coding and why `input` is dropped.
  - GroupKFold explanation (why a random split leaks).
  - Models via `brain_pipeline`, plus baselines (luck, always-rock, always-repeat). Comparison DataFrame and bar chart.
  - Metrics: confusion matrix, classification report, "accuracy right after a loss" (with n), ROC (label_binarize).
  - GridSearchCV per D26, cross-validation mean ± std, and a learning curve over the number of training players.
  - Perceptron over 5 seeds next to Logistic Regression (D27).
  - Early-round table (D7) with Personal-brain replay.
  - LinearRegression of Personal accuracy against round number, printing "+X points per 100 rounds".
  - Predict one new sample: "lost with rock against paper".
  - Write `metrics.json`.
- [ ] **Step 5:** Run `bash tools/build_notebook.sh && pytest tests/test_notebook_build.py -q`. Expected: PASS.
- [ ] **Step 6:** Commit with `feat: notebook part 2 (brain)`. Include the built `MindReader.ipynb` and `.html`.

### Task 6: Notebook Part 1 (the eyes)

**Worker:** sonnet-max · **Review:** Opus

**Files:**
- Modify: `notebook/MindReader.py`, `tests/test_notebook_build.py`

**Interfaces:**
- Consumes: `hands.*`, `find_s.*`, `predict_gesture`.
- Produces:
  - Notebook variable `LIVE_EYES`: a fitted Pipeline with `predict_proba` (D3).
  - metrics.json keys: `eyes_table`, `eyes_test_acc`, `eyes_test_acc_mirrored`, `eyes_detection_rate`, `find_s_clean`, `find_s_noisy`.

- [ ] **Step 1:** Add the failing test `test_part1_metrics`: `eyes_test_acc >= 0.95` and `eyes_test_acc_mirrored >= 0.95`, both over a denominator of 372 (D5); `find_s_clean == ["up", "up", "down", "down"]`; Part 1 time ≤ 60 s.
- [ ] **Step 2:** Run it. Expected: FAIL.
- [ ] **Step 3:** Write Part 1 with the lab steps:
  - The data checks, as in Part 2, plus the detection rate per split.
  - Features via `features_from_row`.
  - Find-S: the clean run and the noisy run from the spec's Part 1, printing only the changes.
  - Plots: countplot; ratio histograms and boxplots by gesture; correlation heatmap; pairplot of 4 ratios.
  - Nearest-neighbour leakage check (D6).
  - Official split. Inside training, `train_test_split(stratify)` on one row per image, then mirrored twins with group ids, and GroupKFold for CV, GridSearch and the learning curve.
  - Models in Pipelines with StandardScaler: GaussianNB, Decision Tree, Perceptron, Logistic Regression, SVM (linear and RBF, with `probability=True`).
  - Comparison table and bar chart; confusion matrix; classification report; ROC; `plot_tree`.
  - GridSearchCV and `cross_val_score`.
  - Mirrored test accuracy.
  - Predict one new sample with `predict_proba`.
  - Choose `LIVE_EYES`.
  - End with `comparison_figure(...)` (D31) over at least 4 models, on the official test set.
- [ ] **Step 4:** Build and run the tests. Expected: PASS.
- [ ] **Step 5:** Commit with `feat: notebook part 1 (eyes)`.

### Task 7: Game engine

**Worker:** sonnet-max · **Review:** Opus

**Files:**
- Create: `mindreader/game.py`, `tests/test_game.py`

**Interfaces:**
- Consumes: `decide`, the brains and `history_features` (Task 2); `outcome` and `shift` (moves).
- Produces `GameEngine(brains: dict, stage_rounds=8, practice_rounds=20)`, with these methods (each returns a dict):

  | Method | Returns |
  |---|---|
  | `.start(name, mode, brain)` | `{"match_id"}` (mode ∈ {stage, practice, eyes_test}) |
  | `.lock(match_id)` | `{"round_id", "period"}` |
  | `.throw(match_id, round_id, player_move)` | rejected: `{"accepted": False, "rejects", "keys_enabled"}`; accepted: the 13 keys listed below |
  | `.correct(match_id, round_id, player_move)` | `{"ok", "score"}` or `{"ok": False, "error"}` |
  | `.report(match_id)` | the D14–D16 contents |
  | `.leaderboard()` | a list |
  | `.export_log(match_id, player_id)` | a DataFrame (D23) |

  - `player_move` is an int, or None for a rejected throw (D19).
  - An accepted throw returns `{"accepted": True, "bot_move", "player_move", "result", "predicted", "prob", "source", "moment", "score", "health", "period", "done", "winner"}`.

- [ ] **Step 1:** Write the failing tests:
  - Lock and throw:
    - `test_lock_hides_bot_move`: the keys are exactly {round_id, period}.
    - `test_throw_signature_has_no_bot_move`: the parameters are (self, match_id, round_id, player_move).
    - `test_locked_move_unchanged_after_throw`.
    - `test_stale_round_id_rejected`.
    - `test_cold_start_sources`: round 1 is the opener (paper), round 2 is personal, round 3 is the selected brain.
  - Scoring and rejections:
    - `test_scoring_health_and_draw`: 3 wins, 3 losses and 2 ties end in a draw.
    - `test_three_rejections_enable_keys`.
  - Corrections:
    - `test_correction_recomputes_next_lock`.
    - `test_correction_refused_after_next_lock`.
  - Report and export:
    - `test_moment_rule` (D14), with k/n in the report.
    - `test_report_habits_k_of_n`.
    - `test_unpredictability_score` (D16).
    - `test_export_log_columns_and_anonymous_id`.
  - Parity:
    - `test_parity_replay_rounds_3_300`: player 72 through the engine with the same fitted `ModelBrain` object gives predictions identical to `brain.predict` on the `add_features` rows.
    - `test_parity_rounds_1_2_rules`.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement. A match state holds its history list, its rounds, the locked round id and its rejects counter.
- [ ] **Step 4:** Run them. Expected: PASS.
- [ ] **Step 5:** Commit with `feat: game engine`.

### Task 8: Live widget, bridge handlers, Part 3

**Worker:** sonnet-max · **Review:** Opus

**Files:**
- Modify: `mindreader/bridge.py`, `notebook/MindReader.py`
- Create: `mindreader/widget/widget.html`, `mindreader/widget/widget.js`, `tests/test_bridge.py`, `tests/widget_harness.html`, `tests/test_widget_smoke.py`

**Interfaces:**
- Consumes: `GameEngine` (Task 7); `read_throw`, `draw_skeleton` and `encode_jpeg` (Task 4); `LIVE_EYES` and `BRAINS` (notebook).
- Produces in `bridge.py`:
  - `handle(engine, reader, eyes_model, action: str, payload: dict) -> dict`, with actions start, lock, throw, correct, report, preview, eyes_target. It always returns `{"ok": bool, …}` and never raises.
  - `register(engine, reader, eyes_model) -> None`: registers the callback `"mindreader.call"`. Idempotent; it imports `google.colab` inside the function.
  - `show_widget(mode: str = "stage") -> None`
  - The JS calls `google.colab.kernel.invokeFunction("mindreader.call", [action, payload], {})` with a 5 s timeout.

- [ ] **Step 1:** Write the failing tests:
  - `test_handle_rejects_unknown_action`, `test_handle_rejects_unknown_brain`, `test_handle_rejects_bad_frames`, `test_handle_rejects_stale_round`.
  - `test_handle_throw_end_to_end_with_stub_reader`.
  - `test_register_is_idempotent`: inject a fake `google.colab.output` into `sys.modules`.
  - Playwright, opening `widget_harness.html` with a stubbed `invokeFunction` and Chromium's `--use-fake-device-for-media-stream`:
    - `test_widget_consent_then_round`: Space starts; the captions show the countdown; the reveal shows the bot's move text.
    - `test_widget_name_rendered_as_text`: the name `<img src=x onerror=alert(1)>` appears literally.
    - `test_widget_camera_denied_keyboard_mode`: the harness denies the camera; the D21 message is shown and the 1/2/3 keys play a round.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement the widget:
  - Layout: the Face-off from `docs/mockups/game-layout.html`, option B.
  - The consent overlay (D22), a CSS-mirrored preview, captions for every spoken line, and the keys in D17.
  - The failure messages in D21.
  - The burst starts 150 ms after "SHOOT" appears; set this from the spike's measured offset.
  - The preview runs at 4 fps, single-flight, and pauses during the countdown and the capture.
  - Eyes-test mode prompts "Show ROCK" (and so on): 10 per gesture, shuffled, recording accepted and correct counts.
  - All text is set with `textContent`.
- [ ] **Step 4:** Implement `handle`, `register` and `show_widget`. Add Part 3 cells tagged `live`:
  - A quick-start cell that retrains `LIVE_EYES` and `BEST_BRAIN` when they're missing (D25).
  - `register(...)` and `show_widget("stage")`.
  - A practice-mode cell and an eyes-test cell.
- [ ] **Step 5:** Run `pytest -q` and the build. Expected: PASS, and `check_notebook_clean` passes.
- [ ] **Step 6:** Commit with `feat: live widget and Part 3`.

**HUMAN GATE G2 (Krish):**
- Run the notebook in Colab and play one practice match.
- Run the eyes test with at least 3 people, both hands and 2 lighting conditions. Paste back the totals.
- A miss on D28 triggers Task 1b.

### Task 9: Match report screen, pilot logging, Part 4

**Worker:** sonnet-max · **Review:** Opus

**Files:**
- Modify: `mindreader/widget/*`, `notebook/MindReader.py`, `tests/test_widget_smoke.py`, `tests/test_notebook_build.py`

**Interfaces:**
- Consumes: `.report()`, `.leaderboard()` and `.export_log()` (Task 7); `metrics.json`.

- [ ] **Step 1:** Write the failing tests:
  - `test_report_panel_shows_k_of_n_and_small_sample_note` (Playwright).
  - `test_pilot_cell_requires_consent`: without the checkbox, no file is written.
  - `test_quick_start_under_20s`: time the quick-start function locally; it must be ≤ 20 s.
  - `test_part4_reads_metrics`.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement:
  - The report panel: score, winner or draw, mind-read moments k/n, habits k/n with the "small sample" note, the illustrative brain-replay table, and the leaderboard.
  - The spoken taunts: a fixed list of lines of 6 words or fewer, plus the D14 "Called it" lines.
  - A pilot export cell: consent checkbox, anonymous ids, writing `data/pilot/pilot_<date>.csv`.
  - A pilot summary cell: pooled k/n with a Wilson interval, and rounds 1–8 separately (D29).
  - Part 4: a summary table from `metrics.json`, what we learned, limitations, future work and credits.
  - The Part 0 "Results at a glance" figure (D31): both parts' accuracy bars side by side, read from `metrics.json`.
- [ ] **Step 4:** Run the tests and the build. Expected: PASS.
- [ ] **Step 5:** Commit with `feat: match report, pilot logging, part 4`.

**HUMAN GATE G3 (Krish):** 3 classmates who haven't heard the pitch each play a 20-round practice match. Export the logs with consent, then send the files back for committing.

### Task 10: README, credits, licences, export

**Worker:** haiku (effort max) · **Review:** Opus

**Files:**
- Create: `README.md`
- Modify: `data/LICENSES.md`, `tools/build_notebook.sh` (only if the HTML export step is missing)

- [ ] **Step 1:** Write `README.md`:
  - An "Open in Colab" badge for `https://colab.research.google.com/github/Krishpotanwar/mind-reader/blob/main/MindReader.ipynb`.
  - A 3-line pitch (the 53% hook).
  - How to run in Colab and locally (the uv commands from Global Constraints).
  - Repo layout, credits (D9), licences, and links to `docs/mockups/`.
- [ ] **Step 2:** Run `bash tools/build_notebook.sh` and confirm that `MindReader.html` exists and opens.
- [ ] **Step 3:** Commit with `docs: readme, credits and html export`.

### Task 11: VIVA_GUIDE.md and PRESENTATION_SCRIPT.md (D32)

**Worker:** sonnet-max · **Review:** Opus. Then a mock viva in chat with Opus: 10 questions, pass mark 8 (D28).

**Files:**
- Create: `VIVA_GUIDE.md`

- [ ] **Step 1:** Write the guide:
  - Every notebook code cell and every `mindreader/*.py` function, quoted and explained line by line in plain English. The JS is explained block by block, with the key lines quoted.
  - Primers: Find-S, Decision Trees and entropy, Naive Bayes, the Perceptron (and D27), Logistic Regression, SVMs and kernels, the train/test split, stratify, GroupKFold and leakage, mirror twins, scaling, one-hot encoding, accuracy/precision/recall/F1, the confusion matrix, ROC/AUC, GridSearchCV, cross-validation, learning curves, MediaPipe as a feature extractor, and OpenCV's role.
  - 50+ likely questions with short answers, including the honest "always repeat already gets 44%" question.
  - The 5-minute run sheet (spec), and a what-if-it-fails playbook.
- [ ] **Step 2:** Verify that every code cell in `notebook/MindReader.py` appears in the guide. The two numbers below must be equal:
  - code cells: `grep -cE '^# %%( tags=.*)?$' notebook/MindReader.py`
  - guide sections: `grep -c '^### Cell ' VIVA_GUIDE.md`
- [ ] **Step 3:** Write `PRESENTATION_SCRIPT.md` (D32). Check the length with `wc -w PRESENTATION_SCRIPT.md`: 600–700 words, excluding the Q&A section.
- [ ] **Step 4:** Commit with `docs: viva guide and presentation script`.

### Task 12: Final test (HUMAN GATE G4) and fixes

**Worker:** Krish runs it; fixes go to sonnet-max · **Review:** Opus

- [ ] **Step 1 (Krish):** on a fresh Colab runtime, "Run all" must finish in 5:00 or less. Play an 8-round stage match and run the eyes test. Report the median shoot-to-reveal time from the widget's debug line.
- [ ] **Step 2 (Krish):** a dry run on the venue network or the hotspot Krish will bring. On a latency miss, use 3 frames at quality 0.6; if it still misses, run the stage match in keyboard mode.
- [ ] **Step 3:** Fix what broke (each fix is a small TDD task), rebuild, and commit with `fix: final test findings`.

### Task B (only if G1 triggers Approach B): local OpenCV runner

**Worker:** sonnet-max · **Review:** Opus

- **Files:** create `play_local.py` and `requirements-local.txt`.
- **What it does:** `cv2.VideoCapture(0)` and a `cv2.imshow` window, reusing `GameEngine`, `read_throw` and the brains. Keys as in D17.
- **Test:** `test_play_local_smoke`, which replays a recorded frame list through the loop function without a camera.
- **Open question:** whether a local window is acceptable is Krish's question for the professor.

### Task 1b (only if the G2 eyes test misses D28): our webcam data

**Worker:** sonnet-max · **Review:** Opus

- **Recording:** a recording cell (consent, anonymous person ids) saves landmark rows only to `data/recordings/`, at least 3 people × 50 samples per gesture.
- **Part 1b:** trains on CGI plus the recordings and tests on held-out people (GroupKFold by person). The criterion becomes ≥ 95% on held-out people.
- **Test:** `test_part1b_metrics`.
