"""Tests for the game engine.

The hand-worked games use Stay, a stand-in brain that always predicts "the same move again". Rounds 1 and 2 never ask the
selected brain (the opener assumes rock, Personal predicts a stay), so in every game below the bot plays the move that
beats the player's previous move (paper in round 1, as if the player's last move was rock). A repeat then loses, one step
up (rock to paper to scissors) ties and one step down wins. Moves: 0 rock, 1 paper, 2 scissors.
"""
import inspect
import json
from collections import namedtuple

import pandas as pd
import pytest

from mindreader import bridge, game
from mindreader.brain import (FEATURES, ModelBrain, PersonalBrain, RandomBrain, add_features, brain_pipeline, decide,
                              load_games)
from mindreader.game import GameEngine

Game = namedtuple("Game", "player bot results score health winner")
GAMES = {  # worked out by hand: what the bot throws, how each round ends, and where the match stands after round 8
    "bot wins": Game([0, 0, 0, 1, 2, 2, 1, 0], [1, 1, 1, 1, 2, 0, 0, 2],
                     ["lose", "lose", "lose", "tie", "tie", "lose", "win", "win"],
                     {"player": 2, "bot": 4, "ties": 2}, {"player": 4, "bot": 6}, "bot"),
    "draw": Game([2, 2, 0, 2, 2, 0, 2, 2], [1, 0, 0, 1, 0, 0, 1, 0],
                 ["win", "lose", "tie", "win", "lose", "tie", "win", "lose"],
                 {"player": 3, "bot": 3, "ties": 2}, {"player": 5, "bot": 5}, "draw"),
    "player wins": Game([2, 1, 0, 0, 1, 0, 1, 2], [1, 0, 2, 1, 1, 2, 1, 2],
                        ["win", "win", "win", "lose", "tie", "win", "tie", "tie"],
                        {"player": 4, "bot": 1, "ties": 3}, {"player": 7, "bot": 4}, "player"),
}
REVEAL_KEYS = {"bot_move", "player_move", "result", "predicted", "prob", "source", "moment", "score", "health",
               "period", "done", "winner"}  # an accepted throw adds "accepted"; a correction adds "ok"
ALWAYS_UP = [1, 2, 0, 1, 2, 0, 1, 2]  # each throw one step up from the last: against Stay every round ties


class Stay:
    """A stand-in brain that always predicts the player's last move again."""

    def predict(self, history):
        return history[-1][0], 0.5


class Up:
    """A stand-in brain that always predicts one step up from the player's last move."""

    def predict(self, history):
        return (history[-1][0] + 1) % 3, None


class Always:
    """A stand-in brain that always predicts the same move."""

    def __init__(self, move):
        self.move = move

    def predict(self, history):
        return self.move, None


class Spy:
    """A stand-in brain that predicts scissors and writes down every history it is asked about."""

    def __init__(self):
        self.asked = []

    def predict(self, history):
        self.asked.append(list(history))
        return 2, None


def started(brains=None, mode="stage", brain="stay", name="Asha", **lengths):
    """A new engine with one match started. Returns (engine, match_id)."""
    engine = GameEngine(brains or {"stay": Stay()}, **lengths)
    return engine, engine.start(name, mode, brain)["match_id"]


def play(engine, match, moves):
    """Lock, then throw each move in turn. Returns the replies to the throws."""
    replies = []
    for move in moves:
        round_id = engine.lock(match)["round_id"]
        replies.append(engine.throw(match, round_id, move))
    return replies


@pytest.fixture(scope="module")
def games():
    return load_games("data/rps_human_games.csv")


@pytest.fixture(scope="module")
def rows(games):
    return add_features(games)


@pytest.fixture(scope="module")
def study_brains(rows):
    """Personal and Random, plus a decision tree and a perceptron fitted on the study data (as in test_brain)."""
    def fitted(name):
        return ModelBrain(brain_pipeline(name).fit(rows[FEATURES], rows["shift"]), name)
    return {"personal": PersonalBrain(), "random": RandomBrain(), "dt": fitted("dt"), "perceptron": fitted("perceptron")}


# ---- lock and throw ----

def test_start_returns_only_the_match_id():
    engine = GameEngine({"stay": Stay()})
    assert set(engine.start("Asha", "stage", "stay")) == {"match_id"}


def test_lock_hides_bot_move():
    engine, match = started()
    reply = engine.lock(match)
    assert set(reply) == {"round_id", "period"}
    assert reply["period"] == 1


def test_throw_signature_has_no_bot_move():
    assert list(inspect.signature(GameEngine.throw).parameters) == ["self", "match_id", "round_id", "player_move"]


@pytest.mark.parametrize("move", [0, 1, 2])
def test_locked_move_unchanged_after_throw(move):
    spy = Spy()
    engine, match = started({"spy": spy}, brain="spy")
    play(engine, match, [0, 1])  # rounds 1 and 2 do not ask the brain
    round_id = engine.lock(match)["round_id"]  # round 3: the brain is asked once, here
    asked = list(spy.asked)
    assert len(asked) == 1 and len(asked[0]) == 2
    assert engine.throw(match, round_id, None)["accepted"] is False  # a throw that was not seen changes nothing
    assert engine.lock(match)["round_id"] == round_id  # nor does asking for the same round again
    assert spy.asked == asked
    reply = engine.throw(match, round_id, move)
    assert spy.asked == asked  # the throw did not ask the brain again
    assert (reply["predicted"], reply["bot_move"]) == (2, 0)  # scissors predicted whatever was thrown, so rock


def test_lock_is_idempotent_until_the_throw_is_counted():
    engine, match = started()
    first = engine.lock(match)
    assert engine.lock(match) == first
    engine.throw(match, first["round_id"], 0)
    second = engine.lock(match)
    assert second["period"] == 2 and second["round_id"] != first["round_id"]


def test_stale_round_id_rejected():
    engine, match = started()
    with pytest.raises(ValueError, match="not the active one"):  # nothing is locked yet
        engine.throw(match, "made-up", 0)
    first = engine.lock(match)["round_id"]
    engine.throw(match, first, 0)
    with pytest.raises(ValueError, match="not the active one"):  # already counted
        engine.throw(match, first, 1)
    second = engine.lock(match)["round_id"]
    for stale in (first, "made-up", None, 5, ["x"]):
        with pytest.raises(ValueError, match="not the active one"):
            engine.throw(match, stale, 1)
    assert engine.report(match)["rounds"] == 1  # none of that counted
    assert engine.throw(match, second, 1)["accepted"] is True  # and the real round is still open


def test_throw_reveal_has_the_listed_keys():
    engine, match = started()
    reply = play(engine, match, [0])[0]
    assert set(reply) == REVEAL_KEYS | {"accepted"} and reply["accepted"] is True


def test_cold_start_sources(study_brains):
    engine, match = started(study_brains, brain="dt")
    first, second, third = play(engine, match, [0, 1, 2])
    assert (first["source"], first["bot_move"], first["predicted"], first["prob"]) == ("opener", 1, 0, 0.583)
    assert (second["source"], second["predicted"], second["prob"]) == ("personal", 0, 1 / 3)  # a stay: rock again
    expected = decide([(0, 1), (1, 1)], "dt", study_brains)
    assert third["source"] == "dt"
    assert (third["bot_move"], third["predicted"], third["prob"]) == (
        expected["bot_move"], expected["predicted_move"], expected["prob"])


# ---- scoring, rejections, match length ----

def test_scoring_health_and_draw():
    draw = GAMES["draw"]
    engine, match = started()
    replies = play(engine, match, draw.player)
    for number, reply in enumerate(replies, start=1):
        so_far = draw.results[:number]
        assert reply["period"] == number
        assert (reply["bot_move"], reply["result"]) == (draw.bot[number - 1], draw.results[number - 1])
        assert reply["score"] == {"player": so_far.count("win"), "bot": so_far.count("lose"), "ties": so_far.count("tie")}
        assert reply["health"] == {"player": 8 - so_far.count("lose"), "bot": 8 - so_far.count("win")}  # ties do no damage
        assert reply["done"] is (number == 8)
        assert reply["winner"] == ("draw" if number == 8 else None)
    assert replies[-1]["score"] == draw.score and replies[-1]["health"] == draw.health  # 3 wins, 3 losses, 2 ties


@pytest.mark.parametrize("name", ["bot wins", "player wins"])
def test_more_wins_takes_the_match(name):
    expected = GAMES[name]
    engine, match = started()
    replies = play(engine, match, expected.player)
    assert [r["bot_move"] for r in replies] == expected.bot and [r["result"] for r in replies] == expected.results
    last = replies[-1]
    assert (last["score"], last["health"], last["winner"], last["done"]) == (
        expected.score, expected.health, expected.winner, True)


def test_a_finished_match_takes_no_more_rounds():
    engine, match = started(stage_rounds=2)
    replies = play(engine, match, [0, 1])
    assert replies[-1]["done"] is True
    with pytest.raises(ValueError, match="match is over"):
        engine.lock(match)
    with pytest.raises(ValueError, match="match is over"):
        engine.throw(match, "anything", 0)
    assert engine.report(match)["done"] is True  # but the report still works


def test_matches_have_the_planned_lengths():
    engine = GameEngine({"stay": Stay()})  # 8 rounds in stage mode, 20 in practice, and an eyes test never ends
    for mode, length in (("stage", 8), ("practice", 20)):
        match = engine.start("Asha", mode, "stay")["match_id"]
        assert not any(r["done"] for r in play(engine, match, [0] * (length - 1)))
        assert play(engine, match, [0])[0]["done"] is True
    match = engine.start("Asha", "eyes_test", "stay")["match_id"]
    last = play(engine, match, [0, 1, 2] * 14)[-1]  # 42 throws
    assert (last["done"], last["winner"], last["health"]) == (False, None, None)


def test_constructor_lengths_are_used():
    engine = GameEngine({"stay": Stay()}, stage_rounds=2, practice_rounds=3)
    stage = engine.start("Asha", "stage", "stay")["match_id"]
    practice = engine.start("Asha", "practice", "stay")["match_id"]
    assert [r["done"] for r in play(engine, stage, [0, 0])] == [False, True]
    assert [r["done"] for r in play(engine, practice, [0, 0, 0])] == [False, False, True]


def test_three_rejections_enable_keys():
    engine, match = started()
    round_id = engine.lock(match)["round_id"]
    for number in (1, 2, 3):
        assert engine.throw(match, round_id, None) == {"accepted": False, "rejects": number, "keys_enabled": number >= 3}
    assert engine.throw(match, round_id, None) == {"accepted": False, "rejects": 4, "keys_enabled": True}
    assert engine.report(match)["rounds"] == 0  # a rejected throw is not counted
    reply = engine.throw(match, round_id, 1)  # the round now takes a key throw, and it is still the same round
    assert reply["accepted"] is True and reply["period"] == 1
    next_round = engine.lock(match)["round_id"]  # and the count starts again
    assert engine.throw(match, next_round, None) == {"accepted": False, "rejects": 1, "keys_enabled": False}


def test_an_accepted_throw_resets_the_rejection_count():
    engine, match = started()
    round_id = engine.lock(match)["round_id"]
    engine.throw(match, round_id, None)
    engine.throw(match, round_id, None)
    engine.throw(match, round_id, 0)  # an accepted throw resets the count
    next_round = engine.lock(match)["round_id"]
    assert engine.throw(match, next_round, None)["keys_enabled"] is False
    assert engine.throw(match, next_round, None)["keys_enabled"] is False


# ---- corrections ----

def test_correction_recomputes_next_lock():
    engine, match = started()
    first = engine.lock(match)["round_id"]
    assert engine.throw(match, first, 0)["bot_move"] == 1  # round 1: paper, against the opener's rock
    fixed = engine.correct(match, first, 2)  # the camera misread: it was scissors
    assert set(fixed) == REVEAL_KEYS | {"ok"}  # the brief's {"ok", "score"}, plus the rest of the round for the screen
    assert fixed["ok"] is True and fixed["result"] == "win"  # scissors beat paper
    assert fixed["score"] == {"player": 1, "bot": 0, "ties": 0}
    reply = engine.throw(match, engine.lock(match)["round_id"], 1)
    # Round 2 predicts a stay. From the corrected history that is scissors, so the bot plays rock;
    # from the uncorrected rock it would have played paper.
    assert (reply["predicted"], reply["bot_move"]) == (2, 0)


def test_correction_refused_after_next_lock():
    engine, match = started()
    first = engine.lock(match)["round_id"]
    engine.throw(match, first, 0)
    second = engine.lock(match)["round_id"]
    refused = engine.correct(match, first, 2)
    assert refused["ok"] is False and set(refused) == {"ok", "error"} and refused["error"]
    reply = engine.throw(match, second, 1)
    assert (reply["predicted"], reply["bot_move"]) == (0, 1)  # still built from the uncorrected rock
    assert engine.correct(match, first, 2)["ok"] is False  # round 2 is counted now, so round 1 is not the last any more
    assert engine.export_log(match, "P01")["choice"].tolist() == ["r", "p"]  # nothing was overwritten
    assert engine.correct(match, second, 2)["ok"] is True  # the last round still can be
    assert engine.export_log(match, "P01")["choice"].tolist() == ["r", "s"]


def test_correction_refused_when_nothing_can_be_corrected():
    engine, match = started()
    assert engine.correct(match, "made-up", 0)["ok"] is False  # no round counted yet
    first = engine.lock(match)["round_id"]
    assert engine.correct(match, first, 0)["ok"] is False  # locked, not thrown
    engine.throw(match, first, 0)
    for wrong in ("made-up", None, 5):
        assert engine.correct(match, wrong, 1)["ok"] is False
    assert engine.report(match)["habits"]["favourite"] == {"move": 0, "k": 1, "n": 1}  # still the original throw


def test_correction_after_the_last_round_can_change_the_winner():
    draw = GAMES["draw"]
    engine, match = started()
    play(engine, match, draw.player[:-1])
    last_round = engine.lock(match)["round_id"]
    assert engine.throw(match, last_round, draw.player[-1])["winner"] == "draw"
    # The finished match has no next lock, so its last round stays open to a correction. Round 8 was scissors against
    # rock (a loss); if it was really paper, that is a win: 4 wins to 2 losses.
    fixed = engine.correct(match, last_round, 1)
    assert fixed["ok"] is True and fixed["result"] == "win"
    assert (fixed["score"], fixed["health"], fixed["done"], fixed["winner"]) == (
        {"player": 4, "bot": 2, "ties": 2}, {"player": 6, "bot": 4}, True, "player")
    assert engine.report(match)["winner"] == "player"


def test_correction_moves_the_moment():
    engine, match = started()
    round_id = engine.lock(match)["round_id"]
    assert engine.throw(match, round_id, 1)["moment"] is None  # the opener assumed rock; paper is a miss
    assert engine.report(match)["moments"] == {"k": 0, "n": 1}
    fixed = engine.correct(match, round_id, 0)
    assert fixed["moment"] == "opener"
    assert engine.report(match)["moments"] == {"k": 1, "n": 1}


# ---- report ----

def test_moment_rule():
    # Calls are made in rounds 1 (the opener), 2, 3, 4 and 7: the player lost the round before and the bot predicted
    # a stay. They come true in rounds 1, 2 and 3 (rock, then a stay, then a stay). Rounds 5, 6 and 8 follow a tie
    # or a win: no call.
    engine, match = started()
    replies = play(engine, match, GAMES["bot wins"].player)
    assert [r["moment"] for r in replies] == [
        "opener", "repeat_after_loss", "repeat_after_loss", None, None, None, None, None]
    assert engine.report(match)["moments"] == {"k": 3, "n": 5}


def test_moment_opener_counts_even_when_missed():
    engine, match = started()
    assert play(engine, match, [1])[0]["moment"] is None  # the bot assumed rock; the player threw paper
    assert engine.report(match)["moments"] == {"k": 0, "n": 1}


def test_moment_needs_a_predicted_stay_after_a_loss():
    engine, match = started({"up": Up()}, brain="up")
    replies = play(engine, match, [0, 0, 0])  # rock loses to paper twice, so round 3 follows a loss...
    assert [r["moment"] for r in replies] == ["opener", "repeat_after_loss", None]
    assert replies[2]["predicted"] == 1  # ...but this brain predicted a step up, not a stay: no call
    assert engine.report(match)["moments"] == {"k": 2, "n": 2}


def test_report_habits_k_of_n():
    engine, match = started()
    play(engine, match, GAMES["bot wins"].player)
    report = engine.report(match)
    assert report["habits"] == {
        "repeat_after": {"lose": {"k": 2, "n": 4}, "tie": {"k": 1, "n": 2}, "win": {"k": 0, "n": 1}},
        "favourite": {"move": 0, "k": 4, "n": 8},
        "cycle": {"stay": 3, "up": 2, "down": 2, "n": 7},
    }
    assert report["small_sample"] is True  # 8 rounds


def test_favourite_move_ties_go_to_the_lowest_move():
    engine, match = started()
    play(engine, match, [2, 1])  # one paper, one scissors, no rock
    assert engine.report(match)["habits"]["favourite"] == {"move": 1, "k": 1, "n": 2}


@pytest.mark.parametrize("moves, steps", [
    ([0, 1, 2, 0], {"stay": 0, "up": 3, "down": 0, "n": 3}),  # rock to paper to scissors to rock: up, up, up
    ([0, 2, 1, 0], {"stay": 0, "up": 0, "down": 3, "n": 3}),
    ([1, 1, 1], {"stay": 2, "up": 0, "down": 0, "n": 2}),
])
def test_cycle_counts_the_direction_of_each_step(moves, steps):
    engine, match = started()
    play(engine, match, moves)
    assert engine.report(match)["habits"]["cycle"] == steps


def test_report_contents():
    engine, match = started(name="Asha")
    play(engine, match, GAMES["bot wins"].player)
    report = engine.report(match)
    assert (report["name"], report["mode"], report["brain"]) == ("Asha", "stage", "stay")
    assert (report["rounds"], report["total"], report["done"], report["winner"]) == (8, 8, True, "bot")
    assert report["score"] == GAMES["bot wins"].score and report["health"] == GAMES["bot wins"].health


def test_report_of_a_match_with_no_rounds_is_empty_not_an_error():
    engine, match = started()
    report = engine.report(match)
    assert report["rounds"] == 0 and report["winner"] is None
    assert report["moments"] == {"k": 0, "n": 0}
    assert report["unpredictability"] == {"score": None, "k": 0, "n": 0}
    assert report["habits"]["favourite"] == {"move": None, "k": 0, "n": 0}
    assert [row["n"] for row in report["replay"]["rows"]] == [0]


def test_unpredictability_score():
    # Stay is right in rounds 2, 4, 5 and 8: 4 of the 7 predictions (rounds 2..8). So 1 - 4/7.
    engine, match = started({"stay": Stay(), "rock": Always(0), "random": RandomBrain(3)})
    play(engine, match, [0, 0, 1, 1, 1, 2, 0, 0])
    report = engine.report(match)
    assert report["unpredictability"] == {"score": pytest.approx(3 / 7), "k": 4, "n": 7}
    replay = report["replay"]
    assert replay["label"] == "illustrative"  # D16: not a ranking
    assert [row["brain"] for row in replay["rows"]] == ["stay", "rock", "random"]  # every brain, in the engine's order
    assert replay["rows"][0] == {"brain": "stay", "k": 4, "n": 7}  # the brain that played reads the same 4 of 7
    # Round 2 is always Personal (a hit); then rock is right in rounds 7 and 8 only
    assert replay["rows"][1] == {"brain": "rock", "k": 3, "n": 7}
    assert replay["rows"][2]["n"] == 7 and 0 <= replay["rows"][2]["k"] <= 7


def test_report_is_repeatable_and_leaves_the_live_random_brain_alone():
    engine, match = started({"random": RandomBrain(5)}, brain="random")
    play(engine, match, [0, 1, 2])
    assert engine.report(match) == engine.report(match)
    live = play(engine, match, [0])[0]
    reference = RandomBrain(5)
    reference.predict([])  # the draw round 3 used
    round_4 = reference.predict([])[0]  # and the one round 4 should use, had the report not touched the generator
    assert live["predicted"] == round_4


def test_leaderboard_ranks_finished_matches_by_unpredictability():
    engine = GameEngine({"stay": Stay()})

    def finished(name, mode, moves):
        play(engine, engine.start(name, mode, "stay")["match_id"], moves)

    finished("Asha", "stage", [0, 0, 1, 1, 1, 2, 0, 0])  # Stay read 4 of 7
    finished("Bo", "stage", ALWAYS_UP)  # read 0 of 7
    finished("Chen", "practice", ([1, 2, 0] * 7)[:20])  # read 0 of 19: as unpredictable as Bo, over more rounds
    play(engine, engine.start("Dev", "stage", "stay")["match_id"], [0, 1, 2])  # not finished
    play(engine, engine.start("Eve", "eyes_test", "stay")["match_id"], [0, 1, 2] * 3)  # not a game
    board = engine.leaderboard()
    assert [(row["name"], row["mode"], row["k"], row["n"]) for row in board] == [
        ("Chen", "practice", 0, 19), ("Bo", "stage", 0, 7), ("Asha", "stage", 4, 7)]
    assert [row["score"] for row in board] == [1.0, 1.0, pytest.approx(3 / 7)]
    assert set(board[0]) == {"name", "mode", "score", "k", "n"}


def test_leaderboard_is_empty_before_any_match_finishes():
    engine, match = started()
    play(engine, match, [0, 1])
    assert engine.leaderboard() == []


# ---- export ----

def test_export_log_columns_and_anonymous_id():
    engine, match = started(name="Asha Verma")
    play(engine, match, GAMES["bot wins"].player)
    log = engine.export_log(match, "P01")
    assert list(log.columns) == ["id", "game", "period", "choice", "other.choice", "result"]  # the study's, D23
    assert log["id"].tolist() == ["P01"] * 8 and log["game"].tolist() == [1] * 8
    assert log["period"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert log["choice"].tolist() == ["r", "r", "r", "p", "s", "s", "p", "r"]
    assert log["other.choice"].tolist() == ["p", "p", "p", "p", "s", "r", "r", "s"]  # the bot's throws
    assert log["result"].tolist() == GAMES["bot wins"].results  # from the player's point of view
    assert "Asha" not in log.to_string() and "Verma" not in log.to_string()  # the name never reaches the log


def test_export_log_of_an_unfinished_match_has_the_rounds_so_far():
    engine, match = started()
    play(engine, match, [0, 1])
    log = engine.export_log(match, "P12")
    assert isinstance(log, pd.DataFrame) and len(log) == 2 and log["id"].tolist() == ["P12", "P12"]
    engine.lock(match)  # a locked round is not a round yet
    assert len(engine.export_log(match, "P12")) == 2


@pytest.mark.parametrize("bad", ["Asha", "p01", "P1", "P", "P0a", "P01\n", "", None, 1, ["P01"]])
def test_export_log_refuses_ids_that_are_not_anonymous(bad):
    engine, match = started()
    play(engine, match, [0])
    with pytest.raises(ValueError, match="anonymous"):
        engine.export_log(match, bad)


def test_export_log_refuses_an_eyes_test():
    engine, match = started(mode="eyes_test")
    play(engine, match, [0, 1, 2])
    with pytest.raises(ValueError, match="eyes test"):
        engine.export_log(match, "P01")


# ---- parity with the offline brain ----

def test_parity_replay_rounds_3_300(games, study_brains):
    brain = study_brains["dt"]
    engine = GameEngine(study_brains, practice_rounds=300)
    match = engine.start("P72", "practice", "dt")["match_id"]
    player = games[games["id"] == 72].reset_index(drop=True)  # 300 real throws
    replies = play(engine, match, [int(move) for move in player["c"]])
    assert len(replies) == 300 and replies[-1]["done"] is True
    # The same 300 rounds as the offline code sees them: the player's throws against what the engine's bot threw.
    seen = player[["id", "period", "choice", "c"]].assign(o=[r["bot_move"] for r in replies])
    history = list(zip(seen["c"].tolist(), seen["o"].tolist()))
    for period in range(3, 301):  # the engine against ModelBrain.predict, on the history of rounds 1..period-1
        move, prob = brain.predict(history[:period - 1])
        assert (replies[period - 1]["predicted"], replies[period - 1]["prob"]) == (move, prob)
        assert replies[period - 1]["bot_move"] == (move + 1) % 3 and replies[period - 1]["source"] == "dt"
    # And against the offline route: add_features rows through the same fitted pipeline, all at once.
    offline = add_features(seen)
    probs = brain.pipeline.predict_proba(offline[FEATURES])
    shifts = brain.pipeline.classes_[probs.argmax(axis=1)]
    last_move = seen["c"].shift(1)[offline.index]
    assert [replies[p - 1]["predicted"] for p in offline["period"]] == ((last_move + shifts) % 3).tolist()
    assert [replies[p - 1]["prob"] for p in offline["period"]] == pytest.approx(probs.max(axis=1).tolist())
    assert len(offline) == 298  # rounds 3..300
    # The report replays the match through every brain; the brain that played reads exactly what it read live.
    report = engine.report(match)
    assert {row["brain"]: row["k"] for row in report["replay"]["rows"]}["dt"] == report["unpredictability"]["k"]
    assert report["small_sample"] is False and report["unpredictability"]["n"] == 299


@pytest.mark.parametrize("brain", ["dt", "perceptron", "random", "personal"])
@pytest.mark.parametrize("first", [0, 1, 2])
def test_parity_rounds_1_2_rules(study_brains, brain, first):
    engine, match = started(study_brains, brain=brain)
    one, two = play(engine, match, [first, 0])
    # Round 1: the opener assumes rock (58.3% of first throws) and plays paper, whichever brain was picked.
    assert (one["source"], one["predicted"], one["prob"], one["bot_move"]) == ("opener", 0, 0.583, 1)
    # Round 2: Personal, from the one round there is. Nothing to learn from yet, so the tie-break says stay.
    expected_move, expected_prob = PersonalBrain().predict([(first, 1)])
    assert expected_move == first
    assert (two["source"], two["predicted"], two["prob"]) == ("personal", expected_move, expected_prob)
    assert two["bot_move"] == (expected_move + 1) % 3


# ---- validation ----

@pytest.mark.parametrize("name, mode, brain", [
    (None, "stage", "stay"), (5, "stage", "stay"), ("x" * (game.MAX_NAME + 1), "stage", "stay"),
    ("Asha", "arcade", "stay"), ("Asha", None, "stay"), ("Asha", ["stage"], "stay"), ("Asha", "STAGE", "stay"),
    ("Asha", "stage", "knn"), ("Asha", "stage", "opener"), ("Asha", "stage", ""), ("Asha", "stage", None),
    ("Asha", "stage", ["stay"]), ("Asha", "stage", {"stay": 1}),
])
def test_start_refuses_a_bad_name_mode_or_brain(name, mode, brain):
    engine = GameEngine({"stay": Stay()})
    with pytest.raises(ValueError):
        engine.start(name, mode, brain)


def test_start_accepts_exactly_the_brains_it_was_given():
    brains = {"stay": Stay(), "up": Up()}
    engine = GameEngine(brains)
    for name in brains:
        assert engine.start("Asha", "practice", name)["match_id"]
    brains["sneaky"] = Always(0)  # a brain added to the caller's dict later is not on the engine's list
    with pytest.raises(ValueError, match="brain"):
        engine.start("Asha", "practice", "sneaky")


def test_name_limit_matches_the_bridge():
    assert game.MAX_NAME == bridge.MAX_NAME
    GameEngine({"stay": Stay()}).start("x" * bridge.MAX_NAME, "stage", "stay")


UNKNOWN_MATCHES = ["nope", "", None, 5, ["x"], {"a": 1}]


@pytest.mark.parametrize("match", UNKNOWN_MATCHES)
def test_an_unknown_match_is_a_value_error_everywhere(match):
    engine, _ = started()
    calls = [lambda: engine.lock(match), lambda: engine.throw(match, "r", 0), lambda: engine.correct(match, "r", 0),
             lambda: engine.report(match), lambda: engine.export_log(match, "P01")]
    for call in calls:
        with pytest.raises(ValueError, match="unknown match"):
            call()


@pytest.mark.parametrize("move", [3, -1, 10 ** 9, "1", "rock", 1.0, True, False, [1], {"m": 1}, b"1"])
def test_a_bad_move_is_refused_and_not_counted(move):
    engine, match = started()
    round_id = engine.lock(match)["round_id"]
    with pytest.raises(ValueError, match="move"):
        engine.throw(match, round_id, move)
    assert engine.throw(match, round_id, None)["rejects"] == 1  # a bad move is not a rejected throw either
    engine.throw(match, round_id, 0)
    with pytest.raises(ValueError, match="move"):
        engine.correct(match, round_id, move)
    assert engine.correct(match, round_id, 1)["ok"] is True


def test_a_correction_needs_a_real_move():
    engine, match = started()
    round_id = engine.lock(match)["round_id"]
    engine.throw(match, round_id, 0)
    with pytest.raises(ValueError, match="move"):
        engine.correct(match, round_id, None)  # None means 'not seen' for a throw, and means nothing for a correction


@pytest.mark.parametrize("brains, kwargs", [
    ({}, {}), (None, {}), ([Stay()], {}),
    ({"stay": Stay()}, {"stage_rounds": 0}), ({"stay": Stay()}, {"practice_rounds": -1}),
    ({"stay": Stay()}, {"stage_rounds": "8"}), ({"stay": Stay()}, {"practice_rounds": 2.5}),
    ({"stay": Stay()}, {"stage_rounds": True}),
])
def test_the_constructor_refuses_nonsense(brains, kwargs):
    with pytest.raises(ValueError):
        GameEngine(brains, **kwargs)


def test_replies_are_plain_json(study_brains):
    engine, match = started(study_brains, brain="dt", mode="practice")
    replies = play(engine, match, [0, 1, 2, 2, 1, 0])
    rejected = engine.throw(match, engine.lock(match)["round_id"], None)
    json.dumps([replies, rejected, engine.report(match), engine.leaderboard()])  # a numpy number would raise here
