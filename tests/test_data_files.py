import pandas as pd


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
