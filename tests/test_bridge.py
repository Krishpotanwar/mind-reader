"""Tests for the live side of the bridge: what `handle` answers to each call from the page, the Colab registration, and the page builder.

The hand reader and the eyes model are stand-ins: every frame holds the same made-up hand and the model always names the same gesture.
The bot's first move is always paper (round 1 assumes the player opens with rock), so a rock throw in round 1 loses.
"""
import json
import os
import re
import sys
import types
import zipfile

import cv2
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mindreader import bridge, eyes
from mindreader.game import GameEngine
from mindreader.hands import features_from_row

HAND = np.random.default_rng(0).uniform(0.1, 0.9, size=(21, 3))  # any 21 points that are not degenerate
FRAME = eyes.encode_jpeg(np.zeros((48, 64, 3), np.uint8))
SECRET = "/home/secret/hand_landmarker.task"


class Stay:
    """A stand-in brain that always predicts the player's last move again."""

    def predict(self, history):
        return history[-1][0], 0.5


class FakeReader:
    """Stands in for HandReader: every frame holds the same hand, or no hand."""

    def __init__(self, hand=HAND):
        self.hand = hand

    def landmarks(self, rgb):
        return self.hand


class BrokenReader:
    """A hand reader that fails the way MediaPipe could, with a file path in its message."""

    def landmarks(self, rgb):
        raise RuntimeError(f"cannot open {SECRET}")


class FakeModel:
    """Stands in for the eyes model: always the same gesture with the same probability."""
    classes_ = ["paper", "rock", "scissors"]  # alphabetical, like scikit-learn's

    def __init__(self, label="rock", prob=0.9):
        self.label, self.prob = label, prob

    def predict_proba(self, rows):
        row = np.full(3, (1 - self.prob) / 2)
        row[self.classes_.index(self.label)] = self.prob
        return np.tile(row, (len(rows), 1))


class ScriptedModel(FakeModel):
    """An eyes model that names rock every time, with the next probability of the list on every call (one call per frame)."""

    def __init__(self, probs):
        super().__init__("rock")
        self.probs = list(probs)

    def predict_proba(self, rows):
        self.prob = self.probs.pop(0)
        return super().predict_proba(rows)


class Game:
    """One engine, one reader and one model, with shortcuts for the calls the page makes."""

    def __init__(self, label="rock", reader=None, prob=0.9, brains=None):
        self.engine = GameEngine(brains or {"stay": Stay()})
        self.reader, self.model = reader or FakeReader(), FakeModel(label, prob)

    def call(self, action, payload):
        return bridge.handle(self.engine, self.reader, self.model, action, payload)

    def start(self, mode="stage", name="Asha"):
        return self.call("start", {"name": name, "mode": mode, "brain": "stay"})["match_id"]

    def lock(self, match):
        return self.call("lock", {"match_id": match})["round_id"]

    def throw(self, match, round_id, **how):
        return self.call("throw", {"match_id": match, "round_id": round_id, **how})

    def round(self, match, **how):
        """Lock a round and throw in it. Returns the throw's reply."""
        return self.throw(match, self.lock(match), **how)


def as_page_would_see_it(reply):
    """The reply after a trip through JSON, which is how it reaches the browser."""
    return json.loads(json.dumps(reply))


# ---- calls the page must never be able to break ----

@pytest.mark.parametrize("action", ["nope", "", None, 5, ["start"], {"start": 1}])
def test_handle_rejects_unknown_action(action):
    reply = Game().call(action, {})
    assert reply == {"ok": False, "error": "unknown action"}


@pytest.mark.parametrize("action", bridge.ACTIONS)
@pytest.mark.parametrize("payload", [None, [], "text", 5])
def test_handle_rejects_a_payload_that_is_not_an_object(action, payload):
    reply = Game().call(action, payload)
    assert reply["ok"] is False and "object" in reply["error"]


IDS = [5, None, ["a"], {"a": 1}, True, "x" * (bridge.MAX_ID + 1)]  # not text, or too long
MOVES_THAT_ARE_NOT_MOVES = [True, 1.0, 3, -1, "1", None, [1]]
MOVE_MESSAGE = r"move must be 0 \(rock\), 1 \(paper\) or 2 \(scissors\)"  # the bridge's own check, not the engine's "the move must be ..."


@pytest.mark.parametrize("bad", IDS, ids=repr)
@pytest.mark.parametrize("field", ["match_id", "round_id"])
def test_handle_rejects_an_id_that_is_not_short_text(field, bad):
    game = Game()
    match = game.start()
    round_id = game.lock(match)
    reply = game.call("throw", {"match_id": match, "round_id": round_id, "move": 0, field: bad})  # a real match and round, one bad id
    assert reply["ok"] is False
    assert re.fullmatch(f"{field} must be text of at most {bridge.MAX_ID} characters", reply["error"])
    assert game.lock(match) == round_id  # nothing was counted


@pytest.mark.parametrize("bad", MOVES_THAT_ARE_NOT_MOVES, ids=repr)
@pytest.mark.parametrize("action", ["throw", "correct"])
def test_handle_rejects_a_move_that_is_not_0_1_or_2(action, bad):
    game = Game()
    match = game.start()
    round_id = game.lock(match)
    reply = game.call(action, {"match_id": match, "round_id": round_id, "move": bad})
    assert reply["ok"] is False and re.fullmatch(MOVE_MESSAGE, reply["error"])
    assert game.lock(match) == round_id  # nothing was counted


@pytest.mark.parametrize("how", [{}, {"frames": [FRAME] * 3, "move": 1}], ids=["neither", "both"])
def test_handle_wants_either_frames_or_a_move(how):
    game = Game()
    match = game.start()
    round_id = game.lock(match)
    assert game.throw(match, round_id, **how) == {"ok": False, "error": "send either frames or a move"}
    assert game.lock(match) == round_id


@pytest.mark.parametrize("action, payload, message", [
    ("start", {"mode": "stage", "brain": "stay"}, "a name must be text"),
    ("start", {"name": 5, "mode": "stage", "brain": "stay"}, "a name must be text"),
    ("start", {"name": "Asha", "mode": 5, "brain": "stay"}, f"mode must be text of at most {bridge.MAX_ID} characters"),
    ("start", {"name": "Asha", "mode": "stage", "brain": ["stay"]}, f"brain must be text of at most {bridge.MAX_ID} characters"),
    ("lock", {}, f"match_id must be text of at most {bridge.MAX_ID} characters"),
    ("report", {"match_id": 7}, f"match_id must be text of at most {bridge.MAX_ID} characters"),
    ("preview", {}, "preview frame is not a string"),
    ("eyes_target", {"frames": [FRAME] * 5}, r"target must be one of \['rock', 'paper', 'scissors'\]"),
    ("eyes_target", {"frames": [FRAME] * 5, "target": ["rock"]}, r"target must be one of \['rock', 'paper', 'scissors'\]"),
    ("eyes_target", {"frames": [FRAME] * 5, "target": "lizard"}, r"target must be one of \['rock', 'paper', 'scissors'\]"),
    ("eyes_target", {"target": "rock"}, "send a list of 3 or 5 frames"),
])
def test_handle_names_the_field_that_is_missing_or_wrong(action, payload, message):
    reply = Game().call(action, payload)
    assert reply["ok"] is False and re.fullmatch(message, reply["error"])


def test_handle_rejects_unknown_brain():
    reply = Game().call("start", {"name": "Asha", "mode": "stage", "brain": "nope"})
    assert reply["ok"] is False and "brain" in reply["error"]


def test_handle_rejects_unknown_mode():
    reply = Game().call("start", {"name": "Asha", "mode": "boss fight", "brain": "stay"})
    assert reply["ok"] is False and "mode" in reply["error"]


def test_handle_refuses_to_start_an_eyes_test_match():
    game = Game()
    reply = game.call("start", {"name": "Asha", "mode": "eyes_test", "brain": "stay"})  # the page asks eyes_target for the eyes test
    assert reply["ok"] is False and "eyes test" in reply["error"] and "eyes_target" in reply["error"]
    assert game.call("start", {"name": "Asha", "mode": "practice", "brain": "stay"})["ok"] is True  # the other modes still start


@pytest.mark.parametrize("frames", [
    [FRAME] * 4, [FRAME] * 6, [], "not a list", None, tuple([FRAME] * 3),
    [FRAME, FRAME, "not a jpeg"], [FRAME, FRAME, 42],
    [FRAME, FRAME, "data:image/png;base64,iVBORw0KGgo="],
    [FRAME, FRAME, "A" * (bridge.MAX_FRAME_CHARS + 1)],
], ids=["four", "six", "none", "string", "None", "tuple", "garbage", "number", "png", "too long"])
def test_handle_rejects_bad_frames(frames):
    game = Game()
    match = game.start()
    round_id = game.lock(match)
    reply = game.throw(match, round_id, frames=frames)
    assert reply["ok"] is False and "frame" in reply["error"]
    assert game.lock(match) == round_id  # nothing was counted: the same round is still waiting for its throw
    game.reader = FakeReader(hand=None)
    assert game.throw(match, round_id, frames=[FRAME] * 5)["rejects"] == 1  # and the bad call was not a rejected throw
    game.reader = FakeReader()
    assert game.throw(match, round_id, frames=[FRAME] * 5)["accepted"] is True


def test_handle_rejects_stale_round():
    game = Game()
    match = game.start()
    round_id = game.lock(match)
    stale = game.throw(match, "0" * 16, frames=[FRAME] * 5)
    assert stale["ok"] is False and "not the active one" in stale["error"]
    unknown_match = game.call("lock", {"match_id": "nope"})
    assert unknown_match["ok"] is False and "unknown match" in unknown_match["error"]
    assert game.throw(match, round_id, frames=[FRAME] * 5)["ok"] is True  # the real round is untouched
    again = game.throw(match, round_id, frames=[FRAME] * 5)  # that round is now counted: it cannot be thrown twice
    assert again["ok"] is False and "not the active one" in again["error"]


def test_handle_never_raises_and_never_leaks_a_path(monkeypatch):
    monkeypatch.setattr(bridge, "last_error", "")
    game = Game(reader=BrokenReader())
    match = game.start()
    round_id = game.lock(match)
    reply = game.throw(match, round_id, frames=[FRAME] * 5)
    assert reply == {"ok": False, "error": bridge.CANT_READ}
    assert SECRET not in json.dumps(reply)
    assert "RuntimeError" in bridge.last_error and SECRET in bridge.last_error  # the traceback stays with the notebook author
    assert game.lock(match) == round_id  # the round was not counted: it is the same round, still waiting
    game.reader = FakeReader(hand=None)
    assert game.throw(match, round_id, frames=[FRAME] * 5)["rejects"] == 1  # a failure of the reader is not a rejected throw

    def boom(*args):
        raise KeyError(SECRET)

    monkeypatch.setattr(game.engine, "lock", boom)
    broken = game.call("lock", {"match_id": match})
    assert broken == {"ok": False, "error": bridge.TRY_AGAIN}
    assert SECRET not in json.dumps(broken) and "Traceback" not in json.dumps(broken)
    assert "KeyError" in bridge.last_error and SECRET in bridge.last_error  # this failure's traceback replaced the last one


def test_handle_keeps_the_throw_when_the_skeleton_cannot_be_drawn(monkeypatch):
    monkeypatch.setattr(bridge, "last_error", "")

    def broken_drawing(*args):
        raise RuntimeError(f"cannot draw with {SECRET}")

    monkeypatch.setattr(bridge, "draw_skeleton", broken_drawing)
    game = Game()
    reply = game.round(game.start(), frames=[FRAME] * 5)
    assert reply["ok"] is True and reply["accepted"] is True and reply["player_move"] == 0  # the throw still counts
    assert reply["seen"]["skeleton"] is None and reply["seen"]["label"] == "rock"
    assert "RuntimeError" in bridge.last_error and SECRET in bridge.last_error  # the failure is kept for the notebook author
    assert SECRET not in json.dumps(reply)  # and never sent to the page


def test_handle_passes_on_at_most_200_characters_of_an_error(monkeypatch):
    game = Game()

    def long_error(*args):
        raise ValueError("x" * 1000)

    monkeypatch.setattr(game.engine, "lock", long_error)
    assert game.call("lock", {"match_id": "m"}) == {"ok": False, "error": "x" * 200}


# ---- the happy paths ----

def test_handle_throw_end_to_end_with_stub_reader():
    game = Game(label="rock")
    started = game.call("start", {"name": "  Asha\x00\n ", "mode": "stage", "brain": "stay"})
    assert started["ok"] is True and started["name"] == "Asha" and started["total"] == 8  # the cleaned name comes back
    locked = game.call("lock", {"match_id": started["match_id"]})
    assert set(locked) == {"ok", "round_id", "period"} and locked["period"] == 1  # the bot's move is not in it (D19)
    reply = as_page_would_see_it(game.throw(started["match_id"], locked["round_id"], frames=[FRAME] * 5))
    assert reply["ok"] is True and reply["accepted"] is True
    assert reply["player_move"] == 0 and reply["bot_move"] == 1 and reply["result"] == "lose"  # rock against the opener's paper
    assert reply["source"] == "opener" and reply["predicted"] == 0 and reply["prob"] == pytest.approx(0.583)
    assert reply["score"] == {"player": 0, "bot": 1, "ties": 0} and reply["period"] == 1 and reply["done"] is False
    seen = reply["seen"]
    assert seen["label"] == "rock" and seen["prob"] == pytest.approx(0.9) and seen["agree"] == 5
    assert seen["skeleton"].startswith("data:image/jpeg;base64,")
    assert eyes.decode_jpeg(seen["skeleton"]).shape == (48, 64, 3)  # the clearest frame, with the hand drawn on it
    assert game.call("lock", {"match_id": started["match_id"]})["period"] == 2


@pytest.mark.parametrize("label, move", [("rock", 0), ("paper", 1), ("scissors", 2)])
def test_handle_maps_the_gesture_to_a_move(label, move):
    game = Game(label=label)
    assert game.round(game.start(), frames=[FRAME] * 3)["player_move"] == move  # a burst of 3 works too


def test_handle_unclear_hand_is_a_rejected_throw_and_three_enable_the_keys():
    for game in (Game(reader=FakeReader(hand=None)), Game(prob=0.5)):  # no hand at all, and a hand the model is unsure of
        match = game.start()
        round_id = game.lock(match)
        rejects = [game.throw(match, round_id, frames=[FRAME] * 5) for _ in range(3)]
        assert [r["accepted"] for r in rejects] == [False] * 3 and [r["ok"] for r in rejects] == [True] * 3
        assert [r["rejects"] for r in rejects] == [1, 2, 3] and [r["keys_enabled"] for r in rejects] == [False, False, True]
        assert game.lock(match) == round_id  # still the same round, with the bot's move still locked
        kept = game.throw(match, round_id, move=2)
        assert kept["accepted"] is True and kept["player_move"] == 2 and kept["seen"] is None  # a key press has no camera reading


def test_handle_correct_rewrites_the_last_round():
    game = Game(label="rock")
    match = game.start()
    round_id = game.lock(match)
    assert game.throw(match, round_id, frames=[FRAME] * 5)["result"] == "lose"  # rock against paper
    fixed = game.call("correct", {"match_id": match, "round_id": round_id, "move": 2})  # "I actually threw scissors": a win
    assert fixed["ok"] is True and fixed["player_move"] == 2 and fixed["result"] == "win" and fixed["score"]["player"] == 1
    refused = game.call("correct", {"match_id": match, "round_id": "0" * 16, "move": 2})
    assert refused["ok"] is False and "last round" in refused["error"]


def test_handle_correct_is_refused_once_the_next_round_is_locked():
    game = Game(label="rock")
    match = game.start()
    round_id = game.lock(match)
    game.throw(match, round_id, frames=[FRAME] * 5)
    game.lock(match)
    late = game.call("correct", {"match_id": match, "round_id": round_id, "move": 2})
    assert late["ok"] is False and "too late" in late["error"]


def test_handle_plays_a_whole_stage_match_and_reports_it():
    game = Game(label="rock")  # the player throws rock every round; the Stay brain predicts it, so the bot plays paper every round
    match = game.start(mode="stage")
    replies = [game.round(match, frames=[FRAME] * 5) for _ in range(8)]
    assert [r["period"] for r in replies] == list(range(1, 9)) and [r["done"] for r in replies] == [False] * 7 + [True]
    assert replies[-1]["winner"] == "bot" and replies[-1]["score"] == {"player": 0, "bot": 8, "ties": 0}
    assert replies[-1]["health"] == {"player": 0, "bot": 8}
    over = game.call("lock", {"match_id": match})
    assert over["ok"] is False and "over" in over["error"]  # there is no ninth round
    report = as_page_would_see_it(game.call("report", {"match_id": match}))
    assert report["ok"] is True and report["rounds"] == report["total"] == 8 and report["done"] is True and report["winner"] == "bot"
    assert (report["name"], report["mode"], report["brain"]) == ("Asha", "stage", "stay")
    assert report["moments"] == {"k": 8, "n": 8}  # the opener and seven repeats after a loss, all called
    assert report["habits"]["repeat_after"]["lose"] == {"k": 7, "n": 7}
    assert report["unpredictability"] == {"score": 0.0, "k": 7, "n": 7} and report["small_sample"] is True
    assert set(report) >= {"habits", "replay", "score", "health"} and report["replay"]["label"] == "illustrative"


def test_handle_leaderboard_lists_the_finished_matches_most_unpredictable_first():
    game = Game()
    assert game.call("leaderboard", {}) == {"ok": True, "rows": []}  # nothing has finished yet: an empty list, not an error
    for name, moves in [("Asha", [0] * 8), ("Bo", [0, 1, 2, 0, 1, 2, 0, 1])]:  # Stay reads every repeat; Bo never repeats
        match = game.start(name=name)
        for move in moves:
            game.round(match, move=move)
    game.round(game.start(name="Cy"), move=0)  # one round of eight: not on the board yet
    reply = as_page_would_see_it(game.call("leaderboard", {}))
    assert reply == {"ok": True, "rows": [
        {"name": "Bo", "mode": "stage", "score": 1.0, "k": 0, "n": 7},
        {"name": "Asha", "mode": "stage", "score": 0.0, "k": 7, "n": 7}]}
    assert as_page_would_see_it(game.call("leaderboard", {"match_id": 5, "mode": None, "junk": [1]})) == reply  # extra fields are ignored
    assert game.engine.leaderboard() == reply["rows"]  # the page gets exactly what the engine ranks, nothing added


def test_handle_leaderboard_never_raises(monkeypatch):
    monkeypatch.setattr(bridge, "last_error", "")
    game = Game()

    def boom():
        raise KeyError(SECRET)

    monkeypatch.setattr(game.engine, "leaderboard", boom)
    broken = game.call("leaderboard", {})
    assert broken == {"ok": False, "error": bridge.TRY_AGAIN} and SECRET not in json.dumps(broken)
    assert "KeyError" in bridge.last_error


def test_handle_preview_accepts_one_unsure_frame():
    seen = Game(label="paper", prob=0.5).call("preview", {"frame": FRAME})  # one frame, and only 50% sure: the throw rules would refuse it
    assert seen == {"ok": True, "label": "paper", "prob": pytest.approx(0.5)}


@pytest.mark.parametrize("name", ["", "   ", "\x00\x01\n\t", "\u202e\u200b"])
def test_handle_calls_a_player_whose_name_is_empty_once_cleaned_player(name):
    reply = Game().call("start", {"name": name, "mode": "stage", "brain": "stay"})
    assert reply["ok"] is True and reply["name"] == "Player"


def test_the_skeleton_is_drawn_on_the_best_frame():
    frames = [eyes.encode_jpeg(np.zeros((48, width, 3), np.uint8)) for width in (64, 66, 68, 70, 72)]  # the width tells the frames apart
    game = Game()
    game.model = ScriptedModel([0.80, 0.85, 0.95, 0.90, 0.80])  # the third frame is the surest
    seen = game.round(game.start(), frames=frames)["seen"]
    assert seen["agree"] == 5 and seen["prob"] == pytest.approx(0.86)
    assert eyes.decode_jpeg(seen["skeleton"]).shape == (48, 68, 3)


def test_a_rejected_throw_still_says_what_was_seen():
    game = Game(reader=FakeReader(hand=None))
    nothing = game.round(game.start(), frames=[FRAME] * 5)
    assert nothing["accepted"] is False and nothing["seen"] == {"label": None, "prob": 0.0, "agree": 0, "skeleton": None}
    unsure = Game(prob=0.5)
    seen = unsure.round(unsure.start(), frames=[FRAME] * 5)["seen"]  # a hand, but not sure enough to count
    assert seen["label"] is None and seen["prob"] == pytest.approx(0.5) and seen["agree"] == 5
    assert seen["skeleton"].startswith("data:image/jpeg;base64,")


def test_handle_preview_names_the_gesture_in_one_frame():
    seen = Game(label="scissors").call("preview", {"frame": FRAME})
    assert seen["ok"] is True and seen["label"] == "scissors" and seen["prob"] == pytest.approx(0.9)
    none = Game(reader=FakeReader(hand=None)).call("preview", {"frame": FRAME})
    assert none == {"ok": True, "label": None, "prob": None}
    bad = Game().call("preview", {"frame": "not a jpeg"})
    assert bad["ok"] is False


def test_handle_eyes_target_says_whether_it_was_accepted_and_correct():
    game = Game(label="rock")
    right = game.call("eyes_target", {"target": "rock", "frames": [FRAME] * 5})
    assert right["ok"] and right["accepted"] is True and right["correct"] is True and right["seen"]["label"] == "rock"
    wrong = game.call("eyes_target", {"target": "paper", "frames": [FRAME] * 5})
    assert wrong["ok"] and wrong["accepted"] is True and wrong["correct"] is False
    unsure = Game(label="rock", prob=0.5).call("eyes_target", {"target": "rock", "frames": [FRAME] * 5})
    assert unsure["ok"] and unsure["accepted"] is False and unsure["correct"] is False
    bad = game.call("eyes_target", {"target": "lizard", "frames": [FRAME] * 5})
    assert bad["ok"] is False and "target" in bad["error"]


# ---- the real hand reader and a real eyes model, on three of the computer-drawn test hands ----

MODEL, TEST_ZIP = "data/raw/hand_landmarker.task", "data/raw/rps-test-set.zip"
needs_data = pytest.mark.skipif(not (os.path.exists(MODEL) and os.path.exists(TEST_ZIP)), reason="needs data/raw/")


@pytest.fixture(scope="module")
def real_eyes():
    """The real HandReader and a Logistic Regression trained as the notebook's quick start does (training pictures plus mirrored twins)."""
    train = pd.read_csv("data/hand_landmarks.csv").query("split == 'train'")
    twins = train.copy()
    x_columns = [f"x{point}" for point in range(21)]
    twins[x_columns] = 1 - twins[x_columns]
    both = pd.concat([train, twins])
    features = np.array([features_from_row(row) for _, row in both.iterrows()])
    model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]).fit(features, both["label"].to_numpy())
    with eyes.HandReader(MODEL) as reader:
        yield reader, model


@needs_data
@pytest.mark.parametrize("gesture, move", [("rock", 0), ("paper", 1), ("scissors", 2)])
def test_handle_throw_with_the_real_reader_and_model(real_eyes, gesture, move):
    reader, model = real_eyes
    with zipfile.ZipFile(TEST_ZIP) as archive:
        picture = cv2.imdecode(np.frombuffer(archive.read(f"rps-test-set/{gesture}/test{gesture}01-00.png"), np.uint8), cv2.IMREAD_COLOR)
    frames = [eyes.encode_jpeg(picture)] * 5  # the same JPEG path the browser's frames take
    engine = GameEngine({"stay": Stay()})
    match = bridge.handle(engine, reader, model, "start", {"name": "Asha", "mode": "stage", "brain": "stay"})["match_id"]
    round_id = bridge.handle(engine, reader, model, "lock", {"match_id": match})["round_id"]
    reply = bridge.handle(engine, reader, model, "throw", {"match_id": match, "round_id": round_id, "frames": frames})
    assert reply["ok"] and reply["accepted"] is True and reply["player_move"] == move
    assert reply["seen"]["label"] == gesture and reply["seen"]["prob"] >= 0.7 and reply["seen"]["agree"] == 5
    assert reply["seen"]["skeleton"].startswith("data:image/jpeg;base64,")
    preview = bridge.handle(engine, reader, model, "preview", {"frame": frames[0]})
    assert preview["label"] == gesture and 0.7 <= preview["prob"] <= 1
    nothing = bridge.handle(engine, reader, model, "preview", {"frame": eyes.encode_jpeg(np.full((240, 320, 3), 128, np.uint8))})
    assert nothing == {"ok": True, "label": None, "prob": None}  # a grey wall has no hand


# ---- the Colab side ----

@pytest.fixture
def fake_colab(monkeypatch):
    """A stand-in for google.colab.output that remembers the callbacks registered with it."""
    registered = {}
    colab = types.ModuleType("google.colab")
    colab.output = types.SimpleNamespace(register_callback=lambda name, function: registered.__setitem__(name, function))
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setattr(bridge, "_registered", {})
    return registered


def test_register_is_idempotent(fake_colab):
    first, second = Game(), Game()
    bridge.register(first.engine, first.reader, first.model)
    bridge.register(second.engine, second.reader, second.model)
    assert list(fake_colab) == ["mindreader.call"]  # one callback under one name, not two
    answer = fake_colab["mindreader.call"]("start", {"name": "Asha", "mode": "stage", "brain": "stay"})
    reply = answer.data  # an IPython JSON object: Colab hands it to the page as data['application/json']
    assert reply["ok"] is True
    assert second.engine.report(reply["match_id"])["name"] == "Asha"  # the later registration is the one that answers
    with pytest.raises(ValueError):
        first.engine.report(reply["match_id"])
    assert fake_colab["mindreader.call"]("nope", {}).data == {"ok": False, "error": "unknown action"}  # and the callback never raises


def test_registering_twice_replaces_the_engine_the_widget_is_built_for(fake_colab, monkeypatch):
    shown = []
    monkeypatch.setattr("IPython.display.display", shown.append)
    first, second = Game(brains={"alpha": Stay()}), Game(brains={"beta": Stay(), "gamma": Stay()})
    bridge.register(first.engine, first.reader, first.model)
    bridge.register(second.engine, second.reader, second.model)
    bridge.show_widget("stage")
    assert '"brains": ["beta", "gamma"], "defaultBrain": "beta"' in shown[0].data and "alpha" not in shown[0].data


def test_show_widget_displays_the_page_for_the_registered_engine(fake_colab, monkeypatch):
    shown = []
    monkeypatch.setattr("IPython.display.display", shown.append)
    with pytest.raises(ValueError, match="register"):
        bridge.show_widget("stage")
    game = Game()
    bridge.register(game.engine, game.reader, game.model)
    bridge.show_widget("practice")
    bridge.show_widget("eyes_test", brain="stay")
    assert len(shown) == 2 and all(type(page).__name__ == "HTML" for page in shown)
    assert '"mode": "practice"' in shown[0].data and '"brains": ["stay"]' in shown[0].data
    with pytest.raises(ValueError, match="brain"):
        bridge.show_widget("stage", brain="nope")
    with pytest.raises(ValueError, match="mode"):
        bridge.show_widget("boss fight")


def test_widget_page_is_one_self_contained_page_with_safe_settings():
    page = bridge.widget_page("stage", ["dt", "</script><b>x"], "dt")
    assert "__CONFIG__" not in page and "__WIDGET_JS__" not in page  # both placeholders were filled
    assert page.count("<script") == 1 and page.count("</script>") == 1  # a brain name cannot close the script early
    script = page.split("<script>", 1)[1]
    assert "\\u003c/script>\\u003cb>x" in script and "<!--" not in script  # inside the script, nothing can start a comment either
    assert bridge.js_string({"a": "</script><!--"}) == '{"a": "\\u003c/script>\\u003c!--"}'
    with pytest.raises(ValueError, match="default"):
        bridge.widget_page("stage", ["dt"], "svm")
    with pytest.raises(ValueError, match="mode"):
        bridge.widget_page("boss fight", ["dt"], "dt")


def test_widget_page_settings_hold_the_mode_the_brains_and_the_default():
    page = bridge.widget_page("practice", ["dt", "svm"], "svm")
    assert 'const CONFIG = {"mode": "practice", "brains": ["dt", "svm"], "defaultBrain": "svm"};' in page


def test_a_brain_named_like_a_placeholder_does_not_get_the_script():
    script = (bridge.WIDGET_DIR / "widget.js").read_text(encoding="utf-8")
    page = bridge.widget_page("stage", ["dt", "__WIDGET_JS__"], "dt")
    assert page.count(script) == 1 and '"brains": ["dt", "__WIDGET_JS__"]' in page


def test_widget_files_hold_no_text_that_would_end_the_script_early():
    script = (bridge.WIDGET_DIR / "widget.js").read_text(encoding="utf-8")
    assert "</script" not in script.lower() and "<!--" not in script and "__CONFIG__" not in script
