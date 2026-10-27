"""Print the study facts behind plan D7, each with the number of rows (n) it rests on.

Usage (from the repo root):
    python tools/rps_facts.py [data/rps_human_games.csv]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `python tools/rps_facts.py` can import mindreader
from mindreader.brain import PersonalBrain, facts, load_games, replay_accuracy


def early_window(games, last_round):
    """Accuracy of always-rock, always-repeat and the Personal brain on rounds 2..last_round, and how many predictions that is."""
    previous = games.groupby("id")["c"].shift(1)
    window = games["period"].between(2, last_round)
    always_rock = (games["choice"] == "r")[window].mean()
    always_repeat = (games["c"] == previous)[window].mean()
    return always_rock, always_repeat, replay_accuracy(games, PersonalBrain(), last_round), int(window.sum())


def main(path):
    games = load_games(path)
    f = facts(games)
    shares = [
        ("rock share, all rounds", f["rock_share_all"], len(games)),
        ("first throw was rock", f["first_rock"], f["n_first"]),
        ("repeat right after a loss", f["repeat_after_loss"], f["n_after_loss"]),
        ("always-rock, model rows", f["always_rock_model_rows"], f["n_model_rows"]),
        ("always-repeat, model rows", f["always_repeat_model_rows"], f["n_model_rows"]),
    ]
    for label, share, n in shares:
        print(f"{label:27s}{share:6.1%}   n = {n:,}")
    print("\nearly rounds:              always-rock / always-repeat / Personal")
    for last_round in (10, 20):
        rock, repeat, personal, n = early_window(games, last_round)
        label = f"rounds 2-{last_round}"
        print(f"{label:27s}{rock:6.1%} /{repeat:6.1%} /{personal:6.1%}   n = {n:,}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "rps_human_games.csv")
