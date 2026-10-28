"""The game engine: matches, the locked bot move, throws, corrections, scoring, report, leaderboard and pilot log.

Everything lives in memory. The engine never reads or writes a file and never prints; names and logs are gone when the
session ends. Moves are 0 rock, 1 paper, 2 scissors, and `result` is always from the player's point of view.

The bot's move is fixed when a round is locked and only read when a throw is counted: `throw` is never given the bot's
move and never asks a brain for one. A malformed call raises ValueError with a short message; a correction that comes
too late is refused with {"ok": False, "error": ...} instead. Not thread-safe: calls must come one at a time.
"""
import re
import secrets
from dataclasses import dataclass, field
from typing import NamedTuple

import pandas as pd

from mindreader.brain import RandomBrain, decide
from mindreader.moves import letter, outcome, shift

MODES = ("stage", "practice", "eyes_test")
KEYS_AFTER = 3  # rejected throws in a row before the round takes a 1/2/3 key throw (D21)
MAX_NAME = 20  # the same limit as bridge.MAX_NAME: the bridge cleans names, the engine only refuses a bad one
MAX_MATCHES = 500  # matches one engine keeps: a session starts a few dozen, a page stuck in a loop would start thousands
SMALL_SAMPLE = 30  # a report of fewer counted rounds than this carries the "small sample" flag
LOG_COLUMNS = ["id", "game", "period", "choice", "other.choice", "result"]  # the study's columns (D23)
MOVE_ERROR = "the move must be 0 (rock), 1 (paper) or 2 (scissors)"


class Round(NamedTuple):
    """One counted round: what each side threw, and what the bot predicted when it locked."""
    player: int
    bot: int
    predicted: int
    prob: float | None
    source: str


@dataclass
class Match:
    name: str
    mode: str
    brain: str
    total: int | None  # rounds in the match; None for an eyes test, which has no end
    rounds: list = field(default_factory=list)  # counted rounds, oldest first; the brains' history is built from these
    locked_id: str | None = None  # the round waiting for its throw
    decision: dict | None = None  # what `decide` returned when that round was locked
    last_id: str | None = None  # the last counted round: the only one a correction may touch
    rejects: int = 0  # rejected throws in a row

    @property
    def done(self) -> bool:
        return self.total is not None and len(self.rounds) >= self.total

    def history(self) -> list[tuple[int, int]]:
        return [(r.player, r.bot) for r in self.rounds]


def is_move(move) -> bool:
    """True for the ints 0, 1 and 2 only: not True, not 1.0, not a string."""
    return type(move) is int and 0 <= move <= 2


def standing(match: Match) -> dict:
    """Score, health, done and winner (D15): a win takes one point of health from the other side, a tie takes none."""
    results = [outcome(r.player, r.bot) for r in match.rounds]
    score = {"player": results.count("win"), "bot": results.count("lose"), "ties": results.count("tie")}
    health = None
    if match.total is not None:
        health = {"player": match.total - score["bot"], "bot": match.total - score["player"]}
    winner = None
    if match.done:
        winner = "draw" if score["player"] == score["bot"] else "player" if score["player"] > score["bot"] else "bot"
    return {"score": score, "health": health, "done": match.done, "winner": winner}


def call_made(rounds: list, i: int) -> str | None:
    """The mind-read call the bot made in round i (0-based), or None (D14).
    'opener': round 1, where the bot always assumes rock. 'repeat_after_loss': the player lost the round before
    and the bot predicted a stay."""
    if i == 0:
        return "opener"
    before = rounds[i - 1]
    if outcome(before.player, before.bot) == "lose" and shift(before.player, rounds[i].predicted) == 0:
        return "repeat_after_loss"
    return None


def moment(rounds: list, i: int) -> str | None:
    """The call made in round i if it came true, else None. A call is a prediction, so it came true when the player
    threw the predicted move: rock for the opener, the same move again for a repeat.
    The spoken 'Called it' line and the report both use this."""
    kind = call_made(rounds, i)
    return kind if kind and rounds[i].player == rounds[i].predicted else None


def reveal(match: Match, i: int) -> dict:
    """What the screen learns about counted round i (0-based) once it is counted: the keys of an accepted throw."""
    r, st = match.rounds[i], standing(match)
    return {
        "bot_move": r.bot, "player_move": r.player, "result": outcome(r.player, r.bot),
        "predicted": r.predicted, "prob": r.prob, "source": r.source,
        "moment": moment(match.rounds, i), "score": st["score"], "health": st["health"],
        "period": i + 1, "done": st["done"], "winner": st["winner"],
    }


def moment_counts(rounds: list) -> dict:
    """D14: n = the calls that were made, k = the calls that came true."""
    made = [i for i in range(len(rounds)) if call_made(rounds, i)]
    return {"k": sum(moment(rounds, i) is not None for i in made), "n": len(made)}


def repeat_rates(rounds: list) -> dict:
    """After a loss, a tie and a win: how often the player threw the same move again (k of n)."""
    counts = {"lose": [0, 0], "tie": [0, 0], "win": [0, 0]}  # [repeats, rounds that followed]
    for before, now in zip(rounds, rounds[1:]):
        seen = counts[outcome(before.player, before.bot)]
        seen[0] += now.player == before.player
        seen[1] += 1
    return {result: {"k": k, "n": n} for result, (k, n) in counts.items()}


def habits(rounds: list) -> dict:
    """The player's habits as counts: repeats by last result, favourite move, and which way they step (stay, up, down)."""
    thrown = [r.player for r in rounds]
    steps = [shift(before, now) for before, now in zip(thrown, thrown[1:])]
    favourite = max(range(3), key=thrown.count) if thrown else None  # a tie goes to the lowest move
    return {
        "repeat_after": repeat_rates(rounds),
        "favourite": {"move": favourite, "k": thrown.count(favourite), "n": len(thrown)},
        "cycle": {"stay": steps.count(0), "up": steps.count(1), "down": steps.count(2), "n": len(steps)},
    }


def unpredictability(rounds: list) -> dict:
    """D16: 1 - (correct bot predictions / predictions made in rounds 2..N), with the counts behind it."""
    made = rounds[1:]  # round 1 is the opener's assumption, not a prediction
    hits = sum(r.predicted == r.player for r in made)
    return {"score": 1 - hits / len(made) if made else None, "k": hits, "n": len(made)}


def replay_table(rounds: list, brains: dict) -> dict:
    """Each brain's predictions for rounds 2..N of this match, every one made from the rounds before it (D16).
    Round 2 is always Personal, as in the game. Illustrative, not a ranking."""
    history = [(r.player, r.bot) for r in rounds]
    rows = []
    for name, brain in brains.items():
        # Random is replayed from a fresh seed: a report is repeatable and never uses up the live brain's draws
        replayed = {**brains, name: RandomBrain(0)} if isinstance(brain, RandomBrain) else brains
        hits = sum(decide(history[:t], name, replayed)["predicted_move"] == history[t][0] for t in range(1, len(history)))
        rows.append({"brain": name, "k": hits, "n": max(len(history) - 1, 0)})
    return {"label": "illustrative", "rows": rows}


class GameEngine:
    """Matches against the bot. Ask for a lock, then a throw; the bot's move comes back only with the counted throw."""

    def __init__(self, brains: dict, stage_rounds: int = 8, practice_rounds: int = 20):
        if not isinstance(brains, dict) or not brains:
            raise ValueError("brains must be a dict with at least one brain")
        for length in (stage_rounds, practice_rounds):
            if type(length) is not int or length < 1:
                raise ValueError("a match needs a whole number of rounds, at least 1")
        self.brains = dict(brains)  # the allow-list: a brain added to the caller's dict later cannot be started
        self._lengths = {"stage": stage_rounds, "practice": practice_rounds, "eyes_test": None}
        self._matches = {}  # ponytail: kept for the whole session, no eviction; start() refuses after MAX_MATCHES

    def start(self, name: str, mode: str, brain: str) -> dict:
        """Open a match and return {"match_id"}. The name only labels it; the brain must be one of the engine's."""
        if not isinstance(name, str) or len(name) > MAX_NAME:
            raise ValueError(f"the name must be text of at most {MAX_NAME} characters")
        if mode not in MODES:
            raise ValueError(f"the mode must be one of {list(MODES)}")
        if not isinstance(brain, str) or brain not in self.brains:
            raise ValueError(f"the brain must be one of {list(self.brains)}")
        if len(self._matches) >= MAX_MATCHES:
            raise ValueError("too many matches in this session: run the game cell again")
        match_id = secrets.token_hex(8)
        self._matches[match_id] = Match(name, mode, brain, self._lengths[mode])
        return {"match_id": match_id}

    def lock(self, match_id: str) -> dict:
        """Fix the bot's move for the next round and return only {"round_id", "period"}. Asking again before the
        throw is counted returns the same round: the move is decided once."""
        match = self._match(match_id)
        if match.done:
            raise ValueError("the match is over")
        if match.locked_id is None:
            match.decision = decide(match.history(), match.brain, self.brains)
            match.locked_id = secrets.token_hex(8)
        return {"round_id": match.locked_id, "period": len(match.rounds) + 1}

    def throw(self, match_id: str, round_id: str, player_move: int | None) -> dict:
        """Count the player's throw for the locked round, or reject it (player_move None: no hand was seen)."""
        match = self._match(match_id)
        if match.done:
            raise ValueError("the match is over")
        if match.locked_id is None or round_id != match.locked_id:
            raise ValueError("that round is not the active one")
        if player_move is not None and not is_move(player_move):
            raise ValueError(MOVE_ERROR)
        if player_move is None:  # the round stays locked, with the same bot move and the same id
            match.rejects += 1
            return {"accepted": False, "rejects": match.rejects, "keys_enabled": match.rejects >= KEYS_AFTER}
        d = match.decision
        match.rounds.append(Round(player_move, d["bot_move"], d["predicted_move"], d["prob"], d["source"]))
        match.last_id, match.locked_id, match.decision, match.rejects = match.locked_id, None, None, 0
        return {"accepted": True, **reveal(match, len(match.rounds) - 1)}

    def correct(self, match_id: str, round_id: str, player_move: int) -> dict:
        """Change what the player threw in the last counted round, before the next round is locked (D18).
        Returns {"ok": True, ...the round's reveal} or {"ok": False, "error"}.
        The next lock is built from the new history."""
        match = self._match(match_id)
        if not is_move(player_move):
            raise ValueError(MOVE_ERROR)
        if match.locked_id is not None:
            return {"ok": False, "error": "too late: the next round is already locked"}
        if match.last_id is None or round_id != match.last_id:
            return {"ok": False, "error": "only the last round can be corrected"}
        match.rounds[-1] = match.rounds[-1]._replace(player=player_move)
        return {"ok": True, **reveal(match, len(match.rounds) - 1)}

    def report(self, match_id: str) -> dict:
        """The match report (D14 to D16), worked out from the counted rounds so far."""
        match = self._match(match_id)
        return {
            "name": match.name, "mode": match.mode, "brain": match.brain,
            "rounds": len(match.rounds), "total": match.total, **standing(match),
            "moments": moment_counts(match.rounds), "habits": habits(match.rounds),
            "unpredictability": unpredictability(match.rounds), "small_sample": len(match.rounds) < SMALL_SAMPLE,
            "replay": replay_table(match.rounds, self.brains),
        }

    def leaderboard(self) -> list:
        """Finished matches, most unpredictable first; a tie goes to the match with more rounds. An eyes test has no
        end, so it never finishes and never ranks."""
        rows = []
        for match in self._matches.values():
            u = unpredictability(match.rounds)
            if match.done and u["n"] > 0:
                rows.append({"name": match.name, "mode": match.mode, **u})
        return sorted(rows, key=lambda row: (-row["score"], -row["n"]))  # sorted is stable: earlier matches stay first

    def export_log(self, match_id: str, player_id: str) -> pd.DataFrame:
        """The counted rounds in the study's columns, for the pilot (D23). The id must be anonymous, like P01."""
        match = self._match(match_id)
        if match.mode == "eyes_test":
            raise ValueError("an eyes test has no game log to export")
        if not isinstance(player_id, str) or not re.fullmatch(r"P[0-9]{2,}", player_id):
            raise ValueError("the player id must be anonymous, like P01")
        rows = [(player_id, 1, number, letter(r.player), letter(r.bot), outcome(r.player, r.bot))
                for number, r in enumerate(match.rounds, start=1)]
        return pd.DataFrame(rows, columns=LOG_COLUMNS)

    def _match(self, match_id: str) -> Match:
        if not isinstance(match_id, str) or match_id not in self._matches:  # a list from the browser would not hash
            raise ValueError("unknown match: start a new game")
        return self._matches[match_id]
