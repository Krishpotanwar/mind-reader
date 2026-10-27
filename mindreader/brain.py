"""The bot's brain: features from the round history, the five lab models, the Personal and Random brains, and the cold-start rules.

A history is a list of (player_move, bot_move) pairs, oldest first. Moves are 0 rock, 1 paper, 2 scissors.
"""
import random

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from mindreader.moves import apply_shift, beats, code, rel, shift

FEATURES = ["o_rel1", "shift1", "o_rel2"]
OPENER_PROB = 0.583  # 58.3% of the study's first throws were rock


def load_games(path: str) -> pd.DataFrame:
    """Read the study CSV, sorted by player and round, and add c and o: the player's and the opponent's move as 0/1/2."""
    games = pd.read_csv(path).sort_values(["id", "period"]).reset_index(drop=True)
    games["c"] = games["choice"].map(code)  # code() raises KeyError on any letter that is not r, p or s
    games["o"] = games["other.choice"].map(code)
    return games


def add_features(games: pd.DataFrame) -> pd.DataFrame:
    """One row per round from period 3 on: how the last two rounds went, and the shift that came next."""
    last = games.groupby("id")[["c", "o"]].shift(1)  # the previous round's moves
    before = games.groupby("id")[["c", "o"]].shift(2)  # and the round before that
    rows = games[["id", "period", "choice"]].copy()
    rows["o_rel1"] = rel(last["o"], last["c"])
    rows["shift1"] = shift(before["c"], last["c"])
    rows["o_rel2"] = rel(before["o"], before["c"])
    rows["shift"] = shift(last["c"], games["c"])
    return rows[games["period"] >= 3].astype({"o_rel1": int, "shift1": int, "o_rel2": int, "shift": int})


def facts(games: pd.DataFrame) -> dict:
    """The study's headline numbers, each next to the number of rows (n) it rests on."""
    model_rows = add_features(games)  # the rounds that have a history of 2 or more rounds
    previous = games.groupby("id")[["c", "result"]].shift(1)
    after_loss = previous["result"] == "lose"
    repeated = games["c"] == previous["c"]
    first_throws = games[games["period"] == 1]
    return {
        "repeat_after_loss": float(repeated[after_loss].mean()),
        "n_after_loss": int(after_loss.sum()),
        "first_rock": float((first_throws["choice"] == "r").mean()),
        "n_first": len(first_throws),
        "rock_share_all": float((games["choice"] == "r").mean()),
        "always_rock_model_rows": float((model_rows["choice"] == "r").mean()),
        "always_repeat_model_rows": float((model_rows["shift"] == 0).mean()),
        "n_model_rows": len(model_rows),
    }


def history_features(history: list[tuple[int, int]]) -> dict | None:
    """The features for predicting the next round, or None if there are fewer than 2 rounds to learn from."""
    if len(history) < 2:
        return None
    (player2, bot2), (player1, bot1) = history[-2], history[-1]  # 2 rounds ago, 1 round ago
    return {"o_rel1": rel(bot1, player1), "shift1": shift(player2, player1), "o_rel2": rel(bot2, player2)}


def brain_pipeline(name: str) -> Pipeline:
    """An unfitted pipeline (encode the features, then the model) for 'nb', 'dt', 'logreg', 'perceptron' or 'svm'."""
    models = {
        "nb": CategoricalNB(),
        "dt": DecisionTreeClassifier(max_depth=6, random_state=42),
        "logreg": LogisticRegression(max_iter=1000, random_state=42),
        "perceptron": Perceptron(random_state=42),
        "svm": SVC(probability=True, random_state=42),
    }
    if name not in models:
        raise ValueError(f"unknown brain {name!r}, expected one of {list(models)}")
    # Naive Bayes wants each feature as a category number; the others take one-hot columns
    encoder = OrdinalEncoder() if name == "nb" else OneHotEncoder(handle_unknown="ignore")
    return Pipeline([("encode", ColumnTransformer([("features", encoder, FEATURES)])), ("model", models[name])])


class PersonalBrain:
    """Learns one player from this game alone: after each kind of result, which shift came next most often?"""

    def predict(self, history: list[tuple[int, int]]) -> tuple[int, float]:
        counts = np.ones((3, 3))  # row: how a round went (rel), column: the shift after it. Every cell starts at 1.
        for (player, bot), (next_player, _) in zip(history, history[1:]):
            counts[rel(bot, player), shift(player, next_player)] += 1
        last_player, last_bot = history[-1]
        row = counts[rel(last_bot, last_player)]
        best = int(np.argmax(row))  # a tie goes to the lowest shift: stay, then up, then down
        return apply_shift(last_player, best), float(row[best] / row.sum())


class ModelBrain:
    """A fitted pipeline that predicts the player's next move from the round history."""

    def __init__(self, pipeline: Pipeline, name: str):
        self.pipeline = pipeline
        self.name = name

    def predict(self, history: list[tuple[int, int]]) -> tuple[int, float | None]:
        features = pd.DataFrame([history_features(history)], columns=FEATURES)
        if hasattr(self.pipeline, "predict_proba"):
            # Take the likeliest shift from the probabilities, not .predict(): SVC's two can disagree
            # and the screen shows this percentage next to the move.
            probs = self.pipeline.predict_proba(features)[0]
            best = int(np.argmax(probs))
            s, prob = int(self.pipeline.classes_[best]), float(probs[best])
        else:
            s, prob = int(self.pipeline.predict(features)[0]), None
        return apply_shift(history[-1][0], s), prob


class RandomBrain:
    """Guesses at random: the 'luck' baseline brain. It has no percentage to show."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def predict(self, history: list[tuple[int, int]]) -> tuple[int, None]:
        return self.rng.randrange(3), None


def decide(history: list[tuple[int, int]], brain_name: str, brains: dict) -> dict:
    """Pick the bot's move for the next round: round 1 assume rock, round 2 Personal, then the chosen brain."""
    if not history:
        predicted, prob, source = 0, OPENER_PROB, "opener"
    elif len(history) == 1:
        predicted, prob = PersonalBrain().predict(history)
        source = "personal"
    else:
        predicted, prob = brains[brain_name].predict(history)
        source = brain_name
    return {"bot_move": beats(predicted), "predicted_move": predicted, "prob": prob, "source": source}


def replay_accuracy(games: pd.DataFrame, brain, last_round: int) -> float:
    """Replay every player's rounds 2..last_round, the brain seeing only the rounds before each one.
    Returns the share of rounds where the move it predicted was the move thrown."""
    hits = []
    for _, player in games.groupby("id"):
        history = list(zip(player["c"].tolist(), player["o"].tolist()))
        for round_number in range(2, last_round + 1):
            predicted, _ = brain.predict(history[: round_number - 1])
            hits.append(predicted == history[round_number - 1][0])
    return float(np.mean(hits))
