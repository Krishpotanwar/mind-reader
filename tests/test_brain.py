import itertools

import numpy as np
import pandas as pd
import pytest

from mindreader.brain import (FEATURES, ModelBrain, PersonalBrain, RandomBrain, add_features, brain_pipeline, decide,
                              facts, history_features, load_games, replay_accuracy)

# scikit-learn 1.9 warns that SVC(probability=True) goes away in 1.11. The brief (D13: the SVM shows a percentage) needs it, so only the SVM case may warn.
SVM_WARNING = pytest.mark.filterwarnings("ignore:The `probability` parameter was deprecated")


@pytest.fixture(scope="module")
def games():
    return load_games("data/rps_human_games.csv")


@pytest.fixture(scope="module")
def rows(games):
    return add_features(games)


@pytest.fixture(scope="module")
def brains(rows):
    """Personal and Random, plus a decision tree and a perceptron fitted on the study data."""
    def fitted(name):
        return ModelBrain(brain_pipeline(name).fit(rows[FEATURES], rows["shift"]), name)
    return {"personal": PersonalBrain(), "random": RandomBrain(), "dt": fitted("dt"), "perceptron": fitted("perceptron")}


@pytest.fixture(scope="module")
def tie_up_else_down():
    """A tree fitted on made-up rows: after a tie the player goes up (shift 1), otherwise down (shift 2).
    No row says 'stay', so the model's classes_ are [1, 2] and column 0 of predict_proba means shift 1."""
    made_up = pd.DataFrame(list(itertools.product(range(3), repeat=3)), columns=FEATURES)
    made_up["shift"] = np.where(made_up["o_rel1"] == 0, 1, 2)
    return ModelBrain(brain_pipeline("dt").fit(made_up[FEATURES], made_up["shift"]), "dt")


def test_facts_match_study_data(games):
    f = facts(games)
    assert f["n_after_loss"] == 7148
    assert round(f["repeat_after_loss"], 3) == 0.527
    assert f["n_first"] == 72
    assert round(f["first_rock"], 3) == 0.583
    assert round(f["rock_share_all"], 3) == 0.357
    assert f["n_model_rows"] == 21456
    assert round(f["always_rock_model_rows"], 3) == 0.356
    assert round(f["always_repeat_model_rows"], 3) == 0.440


def test_add_features_rows_and_columns(rows):
    assert list(rows.columns) == ["id", "period", "choice", "o_rel1", "shift1", "o_rel2", "shift"]
    assert len(rows) == 21456 and rows["period"].min() == 3
    assert not rows.isna().any().any()


def test_add_features_hand_checked_row(rows):
    # Player 1, period 5: s vs p (win). Period 6: r vs r (tie). Period 7: s.
    row = rows[(rows["id"] == 1) & (rows["period"] == 7)].iloc[0]
    assert (row["choice"], row["o_rel1"], row["shift1"], row["o_rel2"], row["shift"]) == ("s", 0, 1, 2, 2)


def test_history_features_hand_checked_row():
    # Same rounds as above, as (player, bot) pairs: scissors vs paper, then rock vs rock.
    assert history_features([(2, 1), (0, 0)]) == {"o_rel1": 0, "shift1": 1, "o_rel2": 2}


def test_history_features_needs_two_rounds():
    assert history_features([]) is None
    assert history_features([(0, 1)]) is None


@pytest.mark.parametrize("period", [3, 10, 150, 300])
def test_history_features_matches_add_features(games, rows, period):
    player = games[games["id"] == 1]
    history = list(zip(player["c"].tolist(), player["o"].tolist()))[: period - 1]  # rounds 1..period-1
    row = rows[(rows["id"] == 1) & (rows["period"] == period)].iloc[0]
    assert history_features(history) == {name: row[name] for name in FEATURES}


def test_personal_brain_replay_matches_measured(games):
    assert replay_accuracy(games, PersonalBrain(), 10) == pytest.approx(0.400, abs=0.005)
    assert replay_accuracy(games, PersonalBrain(), 20) == pytest.approx(0.424, abs=0.005)


def test_personal_tie_break_prefers_stay():
    assert PersonalBrain().predict([(0, 1)]) == (0, 1 / 3)


def test_personal_brain_counts_shifts_per_kind_of_result():
    # After a tie: stay once. After a loss: up once, down once. The last round is a loss, so the counts to use
    # are stay 1, up 2, down 2 (everything starts at 1): up wins the tie, with share 2 of 5.
    history = [(0, 0), (0, 1), (1, 2), (0, 1)]
    assert PersonalBrain().predict(history) == (1, pytest.approx(0.4))


def test_decide_cold_start(brains):
    opener = decide([], "dt", brains)
    assert opener == {"bot_move": 1, "predicted_move": 0, "prob": 0.583, "source": "opener"}
    assert decide([(0, 1)], "dt", brains)["source"] == "personal"
    assert decide([(0, 1), (1, 2)], "dt", brains)["source"] == "dt"


def test_decide_plays_the_move_that_beats_the_prediction(tie_up_else_down):
    got = decide([(0, 0), (2, 2)], "dt", {"dt": tie_up_else_down})
    assert (got["predicted_move"], got["bot_move"], got["source"]) == (0, 1, "dt")  # rock predicted, so paper


def test_model_brain_proba(brains):
    history = [(0, 1), (1, 2), (2, 0)]
    move, prob = brains["dt"].predict(history)
    assert move in (0, 1, 2) and isinstance(prob, float) and 0 < prob <= 1
    move, prob = brains["perceptron"].predict(history)
    assert move in (0, 1, 2) and prob is None


def test_model_brain_maps_probability_columns_to_shifts(tie_up_else_down):
    # The player's last move is scissors (2): up wraps round to rock (0), down is paper (1).
    assert tie_up_else_down.predict([(0, 0), (2, 2)]) == (0, pytest.approx(1.0))  # last round a tie
    assert tie_up_else_down.predict([(0, 0), (2, 0)]) == (1, pytest.approx(1.0))  # last round a loss


class DisagreeingPipeline:
    """Stands in for an SVM pipeline whose predict() disagrees with its own predict_proba()."""
    classes_ = np.array([0, 1, 2])

    def predict(self, features):
        return np.array([0])  # stay

    def predict_proba(self, features):
        return np.array([[0.2, 0.5, 0.3]])  # but up is likeliest


def test_model_brain_shows_the_probability_of_the_move_it_plays():
    # Last move scissors: up wraps to rock. The move must come with the 0.5 that backs it, not with predict()'s stay.
    assert ModelBrain(DisagreeingPipeline(), "svm").predict([(0, 0), (2, 2)]) == (0, 0.5)


def test_random_brain_seeded():
    first, same, other = RandomBrain(seed=7), RandomBrain(seed=7), RandomBrain(seed=8)
    moves = [first.predict([])[0] for _ in range(30)]
    assert moves == [same.predict([])[0] for _ in range(30)]
    assert moves != [other.predict([])[0] for _ in range(30)]
    assert set(moves) <= {0, 1, 2} and first.predict([])[1] is None


@pytest.mark.parametrize("name", ["nb", "dt", "logreg", "perceptron", pytest.param("svm", marks=SVM_WARNING)])
def test_every_pipeline_fits_and_predicts_a_shift(rows, name):
    sample = rows.iloc[::14]  # about 1,500 rows from every player: enough to prove it runs, and keeps the SVM quick
    pipeline = brain_pipeline(name).fit(sample[FEATURES], sample["shift"])
    assert set(pipeline.predict(sample[FEATURES])) <= {0, 1, 2}
    # Naive Bayes sees the 3 features as category numbers; the others see 3 features x 3 one-hot columns
    assert pipeline["model"].n_features_in_ == (3 if name == "nb" else 9)


def test_only_the_perceptron_has_no_probabilities():
    expected = {"nb": True, "dt": True, "logreg": True, "perceptron": False, "svm": True}
    assert {name: hasattr(brain_pipeline(name), "predict_proba") for name in expected} == expected


def test_brain_pipeline_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown brain"):
        brain_pipeline("knn")
