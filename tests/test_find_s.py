import pytest

from mindreader.find_s import find_s, matches

# Mitchell's EnjoySport table (Lab 3): Sky, AirTemp, Humidity, Wind, Water, Forecast.
ENJOYSPORT = [
    ("Sunny", "Warm", "Normal", "Strong", "Warm", "Same"),
    ("Sunny", "Warm", "High", "Strong", "Warm", "Same"),
    ("Rainy", "Cold", "High", "Strong", "Warm", "Change"),
    ("Sunny", "Warm", "High", "Strong", "Cool", "Change"),
]
ENJOYED = [True, True, False, True]

# Finger states for index, middle, ring, pinky.
SCISSORS = ("up", "up", "down", "down")
ROCK = ("down", "down", "down", "down")
PAPER = ("up", "up", "up", "up")


def test_find_s_enjoysport():
    hypothesis, changes = find_s(ENJOYSPORT, ENJOYED)
    assert hypothesis == ("Sunny", "Warm", "?", "Strong", "?", "?")
    assert changes == [  # Mitchell's S1, S2 and S4; the negative example 2 changes nothing
        (0, ("Sunny", "Warm", "Normal", "Strong", "Warm", "Same")),
        (1, ("Sunny", "Warm", "?", "Strong", "Warm", "Same")),
        (3, ("Sunny", "Warm", "?", "Strong", "?", "?")),
    ]


def test_find_s_scissors_concept():
    hypothesis, changes = find_s([ROCK, SCISSORS, PAPER, SCISSORS], [False, True, False, True])
    assert hypothesis == ("up", "up", "down", "down")
    assert changes == [(1, SCISSORS)]  # the second scissors hand fits already, so it is not logged


def test_find_s_noise_collapses():
    noisy = ("up", "up", "up", "down")  # a scissors hand whose ring finger was misread as up
    hypothesis, changes = find_s([SCISSORS, SCISSORS, noisy], [True, True, True])
    assert hypothesis == ("up", "up", "?", "down")
    assert changes == [(0, SCISSORS), (2, ("up", "up", "?", "down"))]


def test_find_s_without_a_positive_example_has_no_hypothesis():
    assert find_s([ROCK, PAPER], [False, False]) == (None, [])
    assert find_s([], []) == (None, [])


def test_matches_follows_the_hypothesis():
    hypothesis = ("Sunny", "Warm", "?", "Strong", "?", "?")
    assert [matches(hypothesis, row) for row in ENJOYSPORT] == ENJOYED
    assert not matches(None, ENJOYSPORT[0])  # no positive example seen yet, so nothing fits
    with pytest.raises(ValueError):
        matches(hypothesis, ("Sunny", "Warm"))


@pytest.mark.parametrize("examples, positive", [
    ([SCISSORS, ROCK], [True]),  # a label is missing
    ([SCISSORS], [True, False]),  # an example is missing
    ([SCISSORS, ("up", "up")], [True, True]),  # the second example has fewer attributes
])
def test_find_s_rejects_input_that_does_not_line_up(examples, positive):
    with pytest.raises(ValueError):
        find_s(examples, positive)
