# Mind-Reader

A webcam rock-paper-scissors bot called CHANAKYA that reads your hand (MediaPipe) and predicts your next move (scikit-learn).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Krishpotanwar/mind-reader/blob/build/mind-reader/notebook/MindReader.ipynb)

Until the build branch is merged into `main`, the badge opens the build branch. After the merge, change `build/mind-reader` to `main` in the badge link and in `BRANCH` in `notebook/MindReader.py`.

## The idea

- In a 2014 study of 72 players, people repeated their last move 52.7% of the time right after losing (computed from the study's public data).
- Mind-Reader turns that habit into a predictor. Its offline brain scores about 45.5% against 44.0% for always repeating the last move, which is evidence of predictability on unseen players.
- That is a small edge over a simple rule. It is not proof that the bot beats people.

## Run it in Colab

1. Click the badge above.
2. In Colab, choose **Runtime**, then **Run all**.

The setup cell downloads the project and installs MediaPipe. The live game needs a webcam, and it asks for your consent before the camera starts. Camera pixels are never saved to disk.

The five-minute target is not measured yet. It will be timed on a fresh Colab runtime.

## Run it locally

The project uses a `.venv` with Python 3.12, built with uv (the commands are from Task 1 of the plan):

```
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt
```

`requirements.txt` holds the runtime packages (MediaPipe is pinned to `mediapipe==1.1.1`). `requirements-dev.txt` adds the test and notebook tools.

Run the tests:

```
OPENBLAS_CORETYPE=ARMV8 .venv/bin/python -m pytest -q
```

Python commands in this project set `OPENBLAS_CORETYPE=ARMV8`, because numpy crashes without it on the build VM.

The notebook also runs from its own folder. `notebook/MindReader.py` is the source of truth. Edit that file, not the `.ipynb`, then rebuild the notebook with `tools/build_notebook.sh`.

## Repo layout

```
mindreader/   the code: brain, eyes, game engine, bridge, hands, widget
notebook/     MindReader.py (source), MindReader.ipynb and MindReader.html (built)
tools/        build_notebook.sh, the data converter and the check scripts
tests/        pytest tests
data/         the CSV data files and LICENSES.md
docs/         mockups, the design spec and the implementation plan
spike/        the early Colab spike
```

## Credits

- Rock Paper Scissors images: Laurence Moroney, CC BY 2.0. The hand landmark data is derived from these images.
- Game data (WXZ2014): Wang, Xu & Zhou (2014), Sci Rep 4:5830.
- Distribution of the game data: CRAN package stratEst 1.1.8, GPL-3.
- The converted `rps_human_games.csv` carries a GPL-3 notice.
- Hand landmarks: computed with Google's MediaPipe HandLandmarker model, which the code downloads and checks by sha256.

## Licences

- Project code: MIT. See [LICENSE](LICENSE).
- Data: each file keeps the licence of its source. `rps_human_games.csv` is under GPL-3. `hand_landmarks.csv` is under CC BY 2.0. Full details are in [data/LICENSES.md](data/LICENSES.md).

## Mockups, spec and plan

- [docs/mockups/game-layout.html](docs/mockups/game-layout.html): game layout mockup.
- [docs/mockups/notebook-results.html](docs/mockups/notebook-results.html): notebook results mockup.
- [Design spec](docs/superpowers/specs/2026-10-27-mind-reader-design.md)
- [Implementation plan](docs/superpowers/plans/2026-10-27-mind-reader.md)
