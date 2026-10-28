"""Tests for the pilot: the consent gate and the file it writes, the Wilson interval, the two counts and the D30 verdict.

Matches are short (3 or 4 rounds) and played against Stay, a stand-in brain, so every expected number below is worked out by hand.
"""
import json
from datetime import date

import pandas as pd
import pytest

from mindreader import pilot
from mindreader.game import LOG_COLUMNS, GameEngine
from test_game import Stay, play

DAY = date(2026, 10, 31)
HEADER = "id,game,period,choice,other.choice,result"


def engine_with(*matches, practice_rounds=3):
    """An engine with the matches [(name, mode, moves)] started in that order and played. Returns (engine, ids)."""
    engine = GameEngine({"stay": Stay()}, stage_rounds=2, practice_rounds=practice_rounds)
    ids = [engine.start(name, mode, "stay")["match_id"] for name, mode, _ in matches]
    for match_id, (_, _, moves) in zip(ids, matches):
        play(engine, match_id, moves)
    return engine, ids


def written(folder):
    return sorted(path.name for path in folder.iterdir())


# ---- the Wilson interval ----

@pytest.mark.parametrize("k, n, low, high", [  # worked by hand from the closed formula, z = 1.96
    (0, 10, 0.0, 0.2775), (5, 10, 0.2366, 0.7634), (10, 10, 0.7225, 1.0), (53, 100, 0.4329, 0.6249), (1, 3, 0.0615, 0.7923),
])
def test_wilson_matches_the_hand_computed_interval(k, n, low, high):
    got_low, got_high = pilot.wilson(k, n)
    assert got_low == pytest.approx(low, abs=5e-4) and got_high == pytest.approx(high, abs=5e-4)


def test_wilson_is_symmetric_and_stays_inside_zero_and_one():
    for n in (1, 7, 60):
        for k in range(n + 1):
            low, high = pilot.wilson(k, n)
            assert 0 <= low <= k / n <= high <= 1
            mirror_low, mirror_high = pilot.wilson(n - k, n)
            assert (1 - high, 1 - low) == pytest.approx((mirror_low, mirror_high))


def test_wilson_gets_narrower_with_more_rounds():
    narrow, wide = pilot.wilson(50, 100), pilot.wilson(5, 10)
    assert narrow[1] - narrow[0] < wide[1] - wide[0]


def test_wilson_with_no_rounds_has_no_interval_and_no_division_by_zero():
    assert pilot.wilson(0, 0) is None


@pytest.mark.parametrize("k, n", [(3, 2), (-1, 5), (1, 0), (0, -2)])
def test_wilson_refuses_counts_that_cannot_be(k, n):
    with pytest.raises(ValueError, match="between 0 and n"):
        pilot.wilson(k, n)


# ---- the verdict (D30) ----

@pytest.mark.parametrize("k, n, expected", [
    (30, 60, "my friends do it too"),  # 50%: the interval (37.7%, 62.3%) is above 1/3
    (22, 60, "less predictable than the study's players"),  # 36.7%: (25.6%, 49.3%) reaches below 1/3 and stays under 52.7%
    (8, 30, "inconclusive"),  # 26.7%: (14.2%, 44.4%) is under 52.7%, but 30 rounds are too few
    (15, 30, "inconclusive"),  # 50%: (33.2%, 66.8%) is wide: not above 1/3, not below 52.7%
    (0, 0, "inconclusive"),  # nobody has lost a round yet: nothing to say
])
def test_verdict_follows_d30(k, n, expected):
    assert pilot.verdict(k, n) == expected


def test_verdict_edges_60_rounds_and_the_two_bounds():
    assert pilot.verdict(22, 60) == "less predictable than the study's players"
    assert pilot.verdict(22, 59) == "inconclusive"  # the 60-round floor is inclusive, so 59 is one short
    assert pilot.verdict(25, 60) == "inconclusive"  # 41.7%: (30.1%, 54.3%) reaches past 52.7%
    low, high = pilot.wilson(86, 200)
    assert low > 1 / 3 and high < pilot.STUDY_REPEAT and pilot.verdict(86, 200) == "my friends do it too"  # both bounds hold: "do it too" comes first
    assert pilot.wilson(33, 100)[0] < 1 / 3 and pilot.verdict(33, 100) == "less predictable than the study's players"


# ---- the export: consent first, then a plain file of throws ----

@pytest.mark.parametrize("consent", [False, None, 0, 1, "yes", "True", [True], ()], ids=repr)
def test_pilot_cell_requires_consent(tmp_path, consent):
    engine, _ = engine_with(("Asha", "practice", [0, 1, 2]))
    folder = tmp_path / "data" / "pilot"
    with pytest.raises(ValueError, match="box is ticked"):
        pilot.export_pilot(engine, consent, folder, DAY)
    assert not (tmp_path / "data").exists()  # not the file and not even the folder


def test_nothing_is_written_when_no_practice_match_has_finished(tmp_path):
    folder = tmp_path / "data" / "pilot"
    engine, _ = engine_with(("Asha", "practice", [0, 1]), ("Bo", "stage", [0, 1]), ("Cy", "eyes_test", [0, 1, 2] * 3), ("Di", "practice", []))
    with pytest.raises(ValueError, match="no finished practice match"):  # a stage match, an eyes test and a short practice match do not count
        pilot.export_pilot(engine, True, folder, DAY)
    assert not (tmp_path / "data").exists()
    with pytest.raises(ValueError, match="no finished practice match"):
        pilot.export_pilot(GameEngine({"stay": Stay()}), True, folder, DAY)  # a session with no match at all
    assert not (tmp_path / "data").exists()


def test_the_export_writes_the_finished_practice_matches_with_anonymous_ids(tmp_path):
    engine, (_, _, _, _, _) = engine_with(
        ("Asha Verma", "practice", [0, 0, 0]), ("Bo Chen", "practice", [1, 2, 0]), ("Stage Sam", "stage", [0, 0]),
        ("Cy", "practice", [2, 2]), ("Di", "practice", [2, 1, 0]))
    folder = tmp_path / "data" / "pilot"
    path = pilot.export_pilot(engine, True, folder, DAY)
    assert path == folder / "pilot_2026-10-31.csv" and written(folder) == ["pilot_2026-10-31.csv"]
    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == HEADER  # the columns, exactly (D23)
    log = pd.read_csv(path)
    assert list(log.columns) == LOG_COLUMNS == HEADER.split(",")
    assert log["id"].tolist() == ["P01"] * 3 + ["P02"] * 3 + ["P03"] * 3  # Asha, Bo and Di: Cy has one round to go and Sam played a stage match
    assert log["period"].tolist() == [1, 2, 3] * 3 and log["game"].tolist() == [1] * 9
    assert log["choice"].tolist() == list("rrr") + list("psr") + list("spr")  # the players' own throws, in the order the matches were started
    assert log["other.choice"].tolist() == list("ppp") + list("psr") + list("prs")  # the bot's: paper, then the move that beats the last throw
    for name in ("Asha", "Verma", "Bo", "Chen", "Sam", "Cy", "Di", "data:", "jpeg"):
        assert name not in text.replace("other.choice", "")  # no names, no pictures: throws only
    assert text == pd.concat([engine.export_log(match, f"P{number:02d}") for number, match in enumerate(engine.finished_matches("practice"), 1)]).to_csv(index=False)


def test_ids_follow_the_order_the_matches_started_not_the_order_they_finished(tmp_path):
    engine, (first, second, third) = engine_with(("A", "practice", [0, 0, 0]), ("B", "practice", [1, 1, 1]), ("C", "practice", [2, 2, 2]))
    fresh = GameEngine({"stay": Stay()}, practice_rounds=3)  # the same three matches, thrown in the opposite order
    ids = [fresh.start(name, "practice", "stay")["match_id"] for name in "ABC"]
    for match, move in reversed(list(zip(ids, [0, 1, 2]))):
        play(fresh, match, [move] * 3)
    log = pd.read_csv(pilot.export_pilot(fresh, True, tmp_path / "pilot", DAY))
    assert [log[log["id"] == f"P0{number}"]["choice"].iloc[0] for number in (1, 2, 3)] == ["r", "p", "s"]  # A is P01 although it finished last


def test_an_export_never_overwrites_an_earlier_one(tmp_path):
    engine, _ = engine_with(("Asha", "practice", [0, 1, 2]))
    folder = tmp_path / "pilot"
    first = pilot.export_pilot(engine, True, folder, DAY)
    original = first.read_bytes()
    engine, _ = engine_with(("Bo", "practice", [2, 2, 2]))  # another session the same day: its players start from P01 again
    second, third = pilot.export_pilot(engine, True, folder, DAY), pilot.export_pilot(engine, True, folder, DAY)
    assert [path.name for path in (first, second, third)] == ["pilot_2026-10-31.csv", "pilot_2026-10-31_2.csv", "pilot_2026-10-31_3.csv"]
    assert first.read_bytes() == original  # the first file is untouched
    assert second.read_bytes() == third.read_bytes() != original
    assert written(folder) == ["pilot_2026-10-31.csv", "pilot_2026-10-31_2.csv", "pilot_2026-10-31_3.csv"]
    other_day = pilot.export_pilot(engine, True, folder, date(2026, 11, 1))
    assert other_day.name == "pilot_2026-11-01.csv"


def test_the_export_names_the_file_after_today_by_default(tmp_path):
    engine, _ = engine_with(("Asha", "practice", [0, 1, 2]))
    assert pilot.export_pilot(engine, True, tmp_path / "pilot").name == f"pilot_{date.today().isoformat()}.csv"


def test_the_default_folder_is_data_pilot_in_the_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    engine, _ = engine_with(("Asha", "practice", [0, 1, 2]))
    assert pilot.export_pilot(engine, True, day=DAY).as_posix() == "data/pilot/pilot_2026-10-31.csv"


# ---- reading the files back ----

def rounds(file, player, moves, bot, results, first_period=1):
    """The rows of one player: moves, the bot's moves and the results are strings like "rrp", "ppp" and "lose,tie,win".
    The counts read only `choice`, `period` and `result`; the bot's moves are filled in to look real and are not checked."""
    return [{"file": file, "id": player, "game": 1, "period": first_period + i, "choice": move, "other.choice": other, "result": result}
            for i, (move, other, result) in enumerate(zip(moves, bot, results.split(",")))]


def table(*players):
    return pd.DataFrame([row for player in players for row in player])


def test_load_pilot_reads_every_file_and_tells_them_apart(tmp_path):
    engine, _ = engine_with(("Asha", "practice", [0, 1, 2]), ("Bo", "practice", [2, 2, 2]))
    pilot.export_pilot(engine, True, tmp_path, DAY)
    pilot.export_pilot(engine, True, tmp_path, date(2026, 11, 1))
    (tmp_path / "notes.txt").write_text("not a pilot file")
    rows = pilot.load_pilot(tmp_path)
    assert rows["file"].unique().tolist() == ["pilot_2026-10-31.csv", "pilot_2026-11-01.csv"]  # the older file first
    assert len(rows) == 12 and rows.groupby(["file", "id"]).ngroups == 4  # P01 and P02 in each: four players, not two
    assert list(rows.columns) == LOG_COLUMNS + ["file"]


def test_load_pilot_with_no_files_is_an_empty_table_and_a_wrong_file_is_refused(tmp_path):
    empty = pilot.load_pilot(tmp_path / "nowhere")
    assert empty.empty and list(empty.columns) == LOG_COLUMNS + ["file"]
    (tmp_path / "pilot_2026-10-31.csv").write_text("id,choice\nP01,r\n")
    with pytest.raises(ValueError, match="pilot_2026-10-31.csv is not a pilot file"):
        pilot.load_pilot(tmp_path)


def test_the_counts_for_one_player_worked_by_hand():
    summary = pilot.summarize(table(rounds("a.csv", "P01", "rrpps", "ppsrs", "lose,tie,lose,win,tie")))
    # after a loss: round 1 -> round 2 stays r (a repeat); round 3 -> round 4 stays p (a repeat): 2 of 2
    assert summary["all"]["repeat_after_loss"]["k"] == 2 and summary["all"]["repeat_after_loss"]["n"] == 2
    assert summary["all"]["win_rate"]["k"] == 1 and summary["all"]["win_rate"]["n"] == 5  # one win in five rounds
    assert (summary["players"], summary["rounds"]) == (1, 5)
    low, high = pilot.wilson(2, 2)
    assert summary["all"]["repeat_after_loss"]["low"] == pytest.approx(low) and summary["all"]["repeat_after_loss"]["high"] == pytest.approx(high)


def test_a_loss_followed_by_a_change_is_not_a_repeat_and_ties_and_wins_do_not_start_a_count():
    summary = pilot.summarize(table(rounds("a.csv", "P01", "rpspr", "ppsss", "lose,tie,win,lose,win")))
    assert (summary["all"]["repeat_after_loss"]["k"], summary["all"]["repeat_after_loss"]["n"]) == (0, 2)  # r then p, s then p: both changed
    assert (summary["all"]["win_rate"]["k"], summary["all"]["win_rate"]["n"]) == (2, 5)


def test_the_last_round_after_a_loss_has_no_next_round_to_count():
    summary = pilot.summarize(table(rounds("a.csv", "P01", "rrr", "ppp", "lose,lose,lose")))
    assert (summary["all"]["repeat_after_loss"]["k"], summary["all"]["repeat_after_loss"]["n"]) == (2, 2)  # three losses leave two rounds that follow one


def test_a_player_is_a_file_and_an_id_not_the_id_alone():
    # a.csv's P01 ends on a loss with r; b.csv's P01 begins with r. If the two were one player, a spurious "repeat after a loss" would appear
    first = rounds("a.csv", "P01", "psr", "ssp", "win,win,lose", first_period=1)
    second = rounds("b.csv", "P01", "rpp", "sss", "tie,lose,lose", first_period=1)
    summary = pilot.summarize(table(first, second))
    assert summary["players"] == 2 and summary["rounds"] == 6
    assert (summary["all"]["repeat_after_loss"]["k"], summary["all"]["repeat_after_loss"]["n"]) == (1, 1)  # only b's round 2 (loss) -> round 3 (p again)
    assert (summary["all"]["win_rate"]["k"], summary["all"]["win_rate"]["n"]) == (2, 6)


def test_rounds_1_to_8_are_counted_on_their_own():
    moves = "rrppppppppsppp"  # 14 rounds: r, r, then eight p, then s, then p, p, p
    results = ["tie"] * 14
    results[0], results[7], results[10] = "lose", "lose", "lose"  # losses in rounds 1, 8 and 11
    summary = pilot.summarize(table(rounds("a.csv", "P01", moves, "s" * 14, ",".join(results))))
    assert (summary["all"]["repeat_after_loss"]["k"], summary["all"]["repeat_after_loss"]["n"]) == (2, 3)  # r->r repeats, p->p repeats, round 11 s->p changes
    # rounds 1-8: the loss in round 1 and its follower (round 2) count; the loss in round 8 is followed by round 9, outside the window
    assert (summary["early"]["repeat_after_loss"]["k"], summary["early"]["repeat_after_loss"]["n"]) == (1, 1)
    results[8] = "win"
    summary = pilot.summarize(table(rounds("a.csv", "P01", moves, "s" * 14, ",".join(results))))
    assert (summary["early"]["win_rate"]["k"], summary["early"]["win_rate"]["n"]) == (0, 8)  # round 9's win is outside rounds 1 to 8
    assert (summary["all"]["win_rate"]["k"], summary["all"]["win_rate"]["n"]) == (1, 14)


def test_a_missing_round_breaks_the_chain():
    rows = table(rounds("a.csv", "P01", "rr", "pp", "lose,lose", first_period=1) + rounds("a.csv", "P01", "r", "p", "lose", first_period=4))
    summary = pilot.summarize(rows)  # rounds 1, 2 and 4: round 2 and round 4 are not neighbours
    assert (summary["all"]["repeat_after_loss"]["k"], summary["all"]["repeat_after_loss"]["n"]) == (1, 1)


def test_pooled_counts_add_up_the_players_of_every_file():
    one = rounds("a.csv", "P01", "rr", "pp", "lose,tie") + rounds("a.csv", "P02", "rp", "pp", "lose,win")
    two = rounds("b.csv", "P01", "ps", "ss", "win,lose") + rounds("b.csv", "P02", "ss", "pp", "lose,lose")
    summary = pilot.summarize(table(one, two))
    assert summary["players"] == 4 and summary["rounds"] == 8
    assert (summary["all"]["repeat_after_loss"]["k"], summary["all"]["repeat_after_loss"]["n"]) == (2, 3)  # a.P01 repeats, a.P02 changes, b.P02 repeats
    assert (summary["all"]["win_rate"]["k"], summary["all"]["win_rate"]["n"]) == (2, 8)


def test_summarize_an_empty_pilot_has_no_intervals_and_does_not_fail():
    summary = pilot.summarize(pilot.load_pilot("nowhere"))
    assert (summary["players"], summary["rounds"]) == (0, 0)
    for window in ("all", "early"):
        for name in ("repeat_after_loss", "win_rate"):
            assert summary[window][name] == {"k": 0, "n": 0, "low": None, "high": None}


def test_a_match_exported_twice_is_pointed_out_not_silently_counted_twice(tmp_path):
    engine, _ = engine_with(("Asha", "practice", [0, 1, 2]))
    pilot.export_pilot(engine, True, tmp_path, DAY)
    assert pilot.repeated_players(pilot.load_pilot(tmp_path)) == []
    play(engine, engine.start("Bo", "practice", "stay")["match_id"], [2, 2, 1])  # a second player, then the cell is run again
    pilot.export_pilot(engine, True, tmp_path, DAY)
    assert pilot.repeated_players(pilot.load_pilot(tmp_path)) == [("pilot_2026-10-31_2.csv", "P01")]  # Asha's throws are in both files
    same_file = table(rounds("a.csv", "P01", "rrr", "ppp", "lose,lose,lose") + rounds("a.csv", "P02", "rrr", "ppp", "lose,lose,lose"))
    assert pilot.repeated_players(same_file) == []  # two players in one file with the same throws are two players


def test_pilot_files_load_by_date_then_copy_number(tmp_path):
    names = ["pilot_2026-10-28_10.csv", "pilot_2026-10-28_2.csv", "pilot_2026-10-29.csv", "pilot_2026-10-28.csv"]
    for name in names:
        (tmp_path / name).write_text("id,game,period,choice,other.choice,result\nP01,1,1,rock,paper,lose\n", encoding="utf-8")
    loaded = pilot.load_pilot(tmp_path)["file"].drop_duplicates().tolist()
    assert loaded == ["pilot_2026-10-28.csv", "pilot_2026-10-28_2.csv", "pilot_2026-10-28_10.csv", "pilot_2026-10-29.csv"]  # "_10" after "_2"


def test_the_summary_is_plain_json():
    json.dumps(pilot.summarize(table(rounds("a.csv", "P01", "rrsps", "pppps", "lose,lose,lose,lose,tie"))))  # a numpy number would raise here
