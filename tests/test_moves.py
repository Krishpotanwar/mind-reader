import pytest

from mindreader.moves import apply_shift, beats, code, letter, outcome, rel, shift

# (player, opponent, result for the player). Rock 0, paper 1, scissors 2: paper covers rock, scissors cut paper, rock blunts scissors.
RULES = [
    (0, 0, "tie"), (0, 1, "lose"), (0, 2, "win"),
    (1, 0, "win"), (1, 1, "tie"), (1, 2, "lose"),
    (2, 0, "lose"), (2, 1, "win"), (2, 2, "tie"),
]


def test_code_and_letter_are_inverses():
    assert [code(x) for x in "rps"] == [0, 1, 2]
    assert [letter(i) for i in range(3)] == ["r", "p", "s"]


def test_unknown_moves_fail_instead_of_wrapping_around():
    with pytest.raises(KeyError):
        code("x")
    for bad in (-1, 3):
        with pytest.raises(KeyError):
            letter(bad)  # a plain string index would turn -1 into 's'


def test_beats_cycles_rock_paper_scissors():
    assert [beats(m) for m in (0, 1, 2)] == [1, 2, 0]


@pytest.mark.parametrize("player, opponent, expected", RULES)
def test_outcome_follows_the_rules(player, opponent, expected):
    assert outcome(player, opponent) == expected


def test_rel_is_one_exactly_when_the_player_lost():
    from_outcome = {"tie": 0, "lose": 1, "win": 2}
    for player in range(3):
        for opponent in range(3):
            assert rel(opponent, player) == from_outcome[outcome(player, opponent)], (player, opponent)


def test_shift_counts_steps_up_the_beats_cycle():
    assert shift(0, 0) == 0  # stay
    assert shift(0, 1) == 1  # rock to paper: up, to the move that beats rock
    assert shift(1, 0) == 2  # paper to rock: down
    assert apply_shift(2, 1) == 0  # up from scissors wraps round to rock


def test_apply_shift_undoes_shift():
    for prev in range(3):
        for now in range(3):
            assert apply_shift(prev, shift(prev, now)) == now
