"""The pilot (D23, D29, D30): save practice matches with consent, then count what the classmates did.

Nothing is written unless the box was ticked. A file holds throws only: an anonymous id (P01, P02, ...), the round number,
both moves and the result. No names and no pictures. The opponent is always the bot, so a pooled count never mixes opponents.

Every export numbers its players from P01, so a player is the pair (file, id), never the id alone. The counts use the
Wilson 95% interval, which stays between 0 and 1 and still works for the few rounds a pilot has. It treats every round as
independent, which one player's rounds are not, so the real uncertainty is a little wider.
"""
import math
from datetime import date
from pathlib import Path

import pandas as pd

from mindreader.game import LOG_COLUMNS

PILOT_DIR = "data/pilot"
Z = 1.96  # the normal multiplier for a 95% interval
STUDY_REPEAT = 0.527  # the study's players threw the same move again after a loss 52.7% of the time
MIN_ROUNDS = 60  # D30: "less predictable" needs at least this many rounds that followed a loss
EARLY = 8  # D29: the first 8 rounds are reported on their own, like a stage match
PLAYER = ["file", "id", "game"]  # who a row belongs to
DO_IT_TOO, LESS_PREDICTABLE, INCONCLUSIVE = "my friends do it too", "less predictable than the study's players", "inconclusive"


def wilson(k: int, n: int) -> tuple[float, float] | None:
    """The 95% Wilson score interval for k hits in n tries, or None when n is 0 (there is nothing to measure)."""
    if not 0 <= k <= n:
        raise ValueError("k must be between 0 and n")
    if n == 0:
        return None
    p, shrink = k / n, 1 + Z * Z / n
    middle = (p + Z * Z / (2 * n)) / shrink
    spread = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / shrink
    return (0.0 if k == 0 else middle - spread), (1.0 if k == n else middle + spread)  # the ends are exactly 0 and 1; rounding must not move them


def verdict(k: int, n: int) -> str:
    """D30: what k repeats in n rounds that followed a loss let us say. 'Do it too' needs the whole interval above luck
    (1/3). 'Less predictable' needs the whole interval below the study's 52.7% and at least 60 rounds. Else 'inconclusive'."""
    interval = wilson(k, n)
    if interval is None:
        return INCONCLUSIVE
    if interval[0] > 1 / 3:
        return DO_IT_TOO
    if interval[1] < STUDY_REPEAT and n >= MIN_ROUNDS:
        return LESS_PREDICTABLE
    return INCONCLUSIVE


def export_pilot(engine, consent, folder: str = PILOT_DIR, day: date | None = None) -> Path:
    """Write every finished practice match of this session to <folder>/pilot_<date>.csv and return the path.
    The players are P01, P02, ... in the order their matches were started. `consent` must be exactly True. Nothing is
    written, not even the folder, if it is not, or if no practice match has finished. A file is never overwritten:
    the second one on a day ends in _2, the third in _3."""
    if consent is not True:
        raise ValueError("nothing is saved unless the box is ticked")
    matches = engine.finished_matches("practice")
    if not matches:
        raise ValueError("no finished practice match to save: play a whole practice match first")
    logs = [engine.export_log(match, f"P{number:02d}") for number, match in enumerate(matches, start=1)]
    text = pd.concat(logs).to_csv(index=False)
    Path(folder).mkdir(parents=True, exist_ok=True)
    stem = f"pilot_{(day or date.today()).isoformat()}"
    number = 1
    while True:
        path = Path(folder) / (f"{stem}.csv" if number == 1 else f"{stem}_{number}.csv")
        try:
            with open(path, "x", encoding="utf-8", newline="") as file:  # "x" refuses a file that is already there
                file.write(text)
            return path
        except FileExistsError:
            number += 1


def load_pilot(folder: str = PILOT_DIR) -> pd.DataFrame:
    """The rows of every pilot_*.csv in the folder as one table, with a `file` column saying which file each came from."""
    tables = []
    for path in sorted(Path(folder).glob("pilot_*.csv"), key=lambda path: (path.stem.split("_")[1], len(path.name), path.name)):  # by date, then copy number: "_10" after "_2"
        table = pd.read_csv(path)
        if list(table.columns) != LOG_COLUMNS:
            raise ValueError(f"{path.name} is not a pilot file: its columns must be {', '.join(LOG_COLUMNS)}")
        tables.append(table.assign(file=path.name))
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=[*LOG_COLUMNS, "file"])


def _count(k: int, n: int) -> dict:
    interval = wilson(k, n)
    return {"k": k, "n": n, "low": interval and interval[0], "high": interval and interval[1]}


def count_rounds(rows: pd.DataFrame, last_round: int | None = None) -> dict:
    """The two pilot counts, over the first `last_round` rounds of every player (all of them if None):
    repeats right after a loss (the Assignment's k of n: of the rounds that followed a loss, how many threw the same move
    again) and the human win rate against the bot. With `last_round`, the loss and the round after it must both be in it."""
    if last_round is not None:
        rows = rows[rows["period"] <= last_round]
    rows = rows.sort_values([*PLAYER, "period"])
    before = rows.groupby(PLAYER)[["period", "choice", "result"]].shift(1)  # the same player's round before this one
    after_loss = (before["period"] == rows["period"] - 1) & (before["result"] == "lose")  # a missing round breaks the chain
    repeated = after_loss & (rows["choice"] == before["choice"])
    return {"repeat_after_loss": _count(int(repeated.sum()), int(after_loss.sum())),
            "win_rate": _count(int((rows["result"] == "win").sum()), len(rows))}


def summarize(rows: pd.DataFrame) -> dict:
    """The players and rounds counted, and the two counts for all rounds and for rounds 1 to 8 on their own (D29)."""
    players = rows.groupby(PLAYER).ngroups if len(rows) else 0
    return {"players": players, "rounds": len(rows), "all": count_rounds(rows), "early": count_rounds(rows, EARLY)}


def repeated_players(rows: pd.DataFrame) -> list[tuple[str, str]]:
    """The (file, id) of every player whose throws are exactly those of a player in an earlier file: most likely one match
    exported twice, which would be counted twice. Two players in the same file with the same throws are two players."""
    first_seen, repeats = {}, []
    for (file, player, _), log in rows.groupby(PLAYER):  # files in date order, then ids
        throws = tuple(log.sort_values("period")[["period", "choice", "other.choice", "result"]].itertuples(index=False, name=None))
        if first_seen.setdefault(throws, file) != file:
            repeats.append((file, player))
    return repeats
