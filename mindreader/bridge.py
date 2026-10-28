"""The Colab side of the live game.

Checks on what the browser sends (the burst of camera frames, the player's name), the `handle` dispatcher behind every call
the widget makes, the Colab callback that routes the calls to it, and `show_widget`, which puts the widget in a cell.

Everything the browser sends is untrusted. `handle` checks types and ranges, works the answer out first, and only then lets
the engine count anything. It never raises: a problem comes back as {"ok": False, "error": <short plain English>}, and no
traceback or file path ever reaches the page. Camera frames live in memory only.
"""
import json
import traceback
import unicodedata
from pathlib import Path

import numpy as np

from mindreader.eyes import decode_jpeg, draw_skeleton, encode_jpeg, read_throw
from mindreader.game import MODES
from mindreader.moves import NAMES

FRAME_COUNTS = (3, 5)  # a burst is this many frames
MAX_FRAME_CHARS = 250_000  # longest data URL for one frame, prefix included
MAX_SIDE = 1280  # longest side of a frame, in pixels
MAX_NAME = 20  # longest player name, in characters
MAX_ID = 64  # longest match id, round id, mode or brain name the browser may send, in characters


def check_frame(frame, label: str) -> np.ndarray:
    """One frame of a burst -> its decoded BGR image, or a ValueError naming `label` and the problem."""
    if not isinstance(frame, str):
        raise ValueError(f"{label} is not a string")
    if len(frame) > MAX_FRAME_CHARS:  # judged before decoding, so a huge string is never handed to the decoder
        raise ValueError(f"{label} is {len(frame)} characters long; the limit is {MAX_FRAME_CHARS}")
    try:
        return decode_jpeg(frame, MAX_SIDE)  # the size is read from the header, so a huge frame never gets its pixel buffer
    except ValueError as problem:
        raise ValueError(f"{label}: {problem}") from None


def validate_frames(frames: list) -> list[np.ndarray]:
    """The decoded BGR frames of a burst, in order, or a ValueError saying what is wrong and with which frame (D20)."""
    if not isinstance(frames, list) or len(frames) not in FRAME_COUNTS:
        counts = " or ".join(str(count) for count in FRAME_COUNTS)
        raise ValueError(f"send a list of {counts} frames")
    return [check_frame(frame, f"frame {number} of {len(frames)}") for number, frame in enumerate(frames, start=1)]


def clean_name(name: str) -> str:
    """A player's name with control, format, surrogate, private-use, unassigned and line-separator characters removed,
    trimmed to MAX_NAME. Only the first 10 * MAX_NAME characters are looked at."""
    if not isinstance(name, str):
        raise ValueError("a name must be text")
    kept = []
    for char in name[:10 * MAX_NAME]:  # cut first, so a name of millions of characters costs nothing to scan
        category = unicodedata.category(char)
        if not category.startswith("C") and category not in ("Zl", "Zp"):  # C... is non-text; Zl, Zp split lines in a log
            kept.append(char)
    return "".join(kept).strip()[:MAX_NAME].strip()


# ---- the live game: what the widget asks and how Python answers ----

CALLBACK = "mindreader.call"  # the name the widget's JavaScript calls
WIDGET_DIR = Path(__file__).parent / "widget"
GESTURE_MOVES = {name: move for move, name in NAMES.items()}  # "rock" -> 0, "paper" -> 1, "scissors" -> 2
DEFAULT_NAME = "Player"  # for a name that is empty once cleaned
TRY_AGAIN = "Something went wrong on my side: try that again"
CANT_READ = "I couldn't read that picture: try that again"
NO_EYES_MATCH = "the eyes test has no match: the page asks with eyes_target"

last_error = ""  # the traceback of the latest unexpected failure, for the notebook author (print bridge.last_error); never sent to the page
_registered = {}  # what show_widget needs from the latest register() call


def fail(message: str) -> dict:
    return {"ok": False, "error": message}


def text_field(payload: dict, key: str) -> str:
    """payload[key] as short text (an id, a mode, a brain name), or a ValueError that names the field."""
    value = payload.get(key)
    if type(value) is not str or len(value) > MAX_ID:
        raise ValueError(f"{key} must be text of at most {MAX_ID} characters")
    return value


def move_field(payload: dict) -> int:
    """payload["move"] as 0, 1 or 2. True and 1.0 are not moves."""
    move = payload.get("move")
    if type(move) is not int or not 0 <= move <= 2:
        raise ValueError("move must be 0 (rock), 1 (paper) or 2 (scissors)")
    return move


def read_burst(reader, eyes_model, frames: list, **limits) -> dict:
    """eyes.read_throw, with any failure of the hand reader (MediaPipe, OpenCV, a degenerate hand) turned into one plain ValueError.
    The traceback goes to `last_error`, never to the page."""
    global last_error
    try:
        return read_throw(reader, eyes_model, frames, **limits)
    except Exception:
        last_error = traceback.format_exc()
        raise ValueError(CANT_READ) from None


def seen_reply(frames: list, seen: dict) -> dict:
    """What the page shows about a read burst: the gesture, how sure, how many frames agreed, and the clearest frame with the
    hand's skeleton drawn on it (None when there is no hand, or if drawing fails: the throw counts without its picture,
    and the traceback goes to `last_error`)."""
    global last_error
    skeleton = None
    if seen["best"] is not None:
        try:
            skeleton = encode_jpeg(draw_skeleton(frames[seen["best"]], seen["landmarks"]), quality=70)
        except Exception:
            last_error = traceback.format_exc()
    return {"label": seen["label"], "prob": seen["prob"], "agree": seen["agree"], "skeleton": skeleton}


def do_start(engine, reader, eyes_model, payload: dict) -> dict:
    name = clean_name(payload.get("name")) or DEFAULT_NAME  # the page shows this cleaned name, from the reply
    mode = text_field(payload, "mode")
    if mode == "eyes_test":  # an engine match for it would never end; the page asks eyes_target instead
        raise ValueError(NO_EYES_MATCH)
    started = engine.start(name, mode, text_field(payload, "brain"))
    return {**started, "name": name, "total": engine.report(started["match_id"])["total"]}


def do_lock(engine, reader, eyes_model, payload: dict) -> dict:
    return engine.lock(text_field(payload, "match_id"))


def do_throw(engine, reader, eyes_model, payload: dict) -> dict:
    """Count a throw: from a burst of camera frames, or from a key press (a "move"). The bot's move is never an input (D19)."""
    match_id, round_id = text_field(payload, "match_id"), text_field(payload, "round_id")
    if ("frames" in payload) == ("move" in payload):
        raise ValueError("send either frames or a move")
    if "move" in payload:
        return {**engine.throw(match_id, round_id, move_field(payload)), "seen": None}
    frames = validate_frames(payload["frames"])
    seen = read_burst(reader, eyes_model, frames)
    picture = seen_reply(frames, seen)  # all the work is done before the engine counts anything
    return {**engine.throw(match_id, round_id, GESTURE_MOVES.get(seen["label"])), "seen": picture}  # no clear gesture: a rejected throw


def do_correct(engine, reader, eyes_model, payload: dict) -> dict:
    return engine.correct(text_field(payload, "match_id"), text_field(payload, "round_id"), move_field(payload))


def do_report(engine, reader, eyes_model, payload: dict) -> dict:
    return engine.report(text_field(payload, "match_id"))


def do_leaderboard(engine, reader, eyes_model, payload: dict) -> dict:
    """The ranking of every finished match this session (names are session memory only). It takes no arguments."""
    return {"rows": engine.leaderboard()}


def do_preview(engine, reader, eyes_model, payload: dict) -> dict:
    """The gesture in one frame, for the live "I see" label. One frame is enough here, with any probability."""
    frame = check_frame(payload.get("frame"), "preview frame")
    seen = read_burst(reader, eyes_model, [frame], need=1, min_prob=0.0)
    return {"label": seen["label"], "prob": seen["prob"] if seen["label"] else None}


def do_eyes_target(engine, reader, eyes_model, payload: dict) -> dict:
    """One throw of the eyes test: was the burst accepted, and did it show the gesture the page asked for?"""
    target = payload.get("target")
    if type(target) is not str or target not in GESTURE_MOVES:
        raise ValueError(f"target must be one of {list(GESTURE_MOVES)}")
    frames = validate_frames(payload.get("frames"))
    seen = read_burst(reader, eyes_model, frames)
    return {"accepted": seen["label"] is not None, "correct": seen["label"] == target, "seen": seen_reply(frames, seen)}


HANDLERS = {"start": do_start, "lock": do_lock, "throw": do_throw, "correct": do_correct, "report": do_report,
            "leaderboard": do_leaderboard, "preview": do_preview, "eyes_target": do_eyes_target}
ACTIONS = tuple(HANDLERS)


def handle(engine, reader, eyes_model, action: str, payload: dict) -> dict:
    """Answer one call from the widget with {"ok": True, ...} or {"ok": False, "error": ...}. It never raises.

    A ValueError (from the checks above or from the engine) has a plain English message that is passed on. Anything else is
    an unexpected failure: the page gets a fixed message and the traceback is kept in `last_error`."""
    global last_error
    if type(action) is not str or action not in HANDLERS:
        return fail("unknown action")
    if not isinstance(payload, dict):
        return fail("the payload must be an object")
    try:
        return {"ok": True, **HANDLERS[action](engine, reader, eyes_model, payload)}  # an engine reply with "ok": False wins
    except ValueError as problem:
        return fail(str(problem)[:200])  # a long message is cut, not trusted
    except Exception:
        last_error = traceback.format_exc()
        return fail(TRY_AGAIN)


# ---- Colab and the notebook ----

def register(engine, reader, eyes_model) -> None:
    """Make the Colab callback "mindreader.call" answer the widget with handle(...). Calling it again replaces the old engine,
    reader and model, so re-running the cell is safe."""
    from google.colab import output  # only exists on Colab, so it is imported here: this module also loads on a laptop
    from IPython.display import JSON

    def call(action, payload):
        return JSON(handle(engine, reader, eyes_model, action, payload))  # Colab gives the page data['application/json']

    output.register_callback(CALLBACK, call)
    _registered["engine"] = engine


def js_string(value) -> str:
    """A value as JSON for inside a <script>. No "<" is left, so no </script> or <!-- can end the script early."""
    return json.dumps(value).replace("<", "\\u003c")


def widget_page(mode: str, brains: list, default_brain: str) -> str:
    """The widget as one HTML string: widget.html with widget.js and the settings (mode, brains, default brain) filled in."""
    if mode not in MODES:
        raise ValueError(f"the mode must be one of {list(MODES)}")
    if default_brain not in brains:
        raise ValueError(f"the default brain must be one of {list(brains)}")
    page = (WIDGET_DIR / "widget.html").read_text(encoding="utf-8")
    script = (WIDGET_DIR / "widget.js").read_text(encoding="utf-8")
    config = {"mode": mode, "brains": list(brains), "defaultBrain": default_brain}
    # widget.js is ours and holds no placeholder; a brain name could, so the settings go in last
    return page.replace("__WIDGET_JS__", script).replace("__CONFIG__", js_string(config))


def show_widget(mode: str = "stage", *, brain: str | None = None) -> None:
    """Show the game ("stage" or "practice") or the eyes test ("eyes_test") in this cell's output. register(...) must have run first.
    `brain` is the bot's brain in the dropdown at the start; it is the engine's first brain unless given."""
    from IPython.display import HTML, display

    engine = _registered.get("engine")
    if engine is None:
        raise ValueError("run register(...) first: the widget needs the game engine")
    brains = list(engine.brains)
    display(HTML(widget_page(mode, brains, brain or brains[0])))
