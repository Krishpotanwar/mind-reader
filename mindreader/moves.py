"""Move coding used everywhere: rock 0, paper 1, scissors 2. The move that beats m is (m + 1) % 3.

beats, shift, apply_shift and rel are plain arithmetic mod 3, so they also work on whole pandas columns at once.
"""

NAMES = {0: "rock", 1: "paper", 2: "scissors"}
_CODES = {"r": 0, "p": 1, "s": 2}
_LETTERS = {0: "r", 1: "p", 2: "s"}


def code(m: str) -> int:
    """'r' -> 0, 'p' -> 1, 's' -> 2. Anything else raises KeyError."""
    return _CODES[m]


def letter(i: int) -> str:
    """0 -> 'r', 1 -> 'p', 2 -> 's'. Anything else raises KeyError."""
    return _LETTERS[i]


def beats(m: int) -> int:
    """The move that beats m."""
    return (m + 1) % 3


def outcome(player: int, opponent: int) -> str:
    """'win', 'lose' or 'tie', from the player's point of view."""
    if player == opponent:
        return "tie"
    return "lose" if opponent == beats(player) else "win"


def shift(prev: int, now: int) -> int:
    """How far the player moved: 0 = stayed, 1 = up (to the move that beats the last one), 2 = down."""
    return (now - prev) % 3


def apply_shift(prev: int, s: int) -> int:
    """The move you get by shifting prev by s. The inverse of shift."""
    return (prev + s) % 3


def rel(opponent: int, player: int) -> int:
    """The opponent's move relative to the player's: 0 = tie, 1 = the player lost, 2 = the player won."""
    return (opponent - player) % 3
