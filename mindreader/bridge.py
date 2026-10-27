"""Checks on what the browser sends: the burst of camera frames and the player's name."""
import unicodedata

import numpy as np

from mindreader.eyes import decode_jpeg

FRAME_COUNTS = (3, 5)  # a burst is this many frames
MAX_FRAME_CHARS = 250_000  # longest data URL for one frame, prefix included
MAX_SIDE = 1280  # longest side of a frame, in pixels
MAX_NAME = 20  # longest player name, in characters


def check_frame(frame, label: str) -> np.ndarray:
    """One frame of a burst -> its decoded BGR image, or a ValueError naming `label` and the problem."""
    if not isinstance(frame, str):
        raise ValueError(f"{label} is not a string")
    if len(frame) > MAX_FRAME_CHARS:  # judged before decoding, so a huge string is never handed to the decoder
        raise ValueError(f"{label} is {len(frame)} characters long; the limit is {MAX_FRAME_CHARS}")
    try:
        bgr = decode_jpeg(frame)
    except ValueError as problem:
        raise ValueError(f"{label}: {problem}") from None
    height, width = bgr.shape[:2]
    if max(height, width) > MAX_SIDE:
        raise ValueError(f"{label} is {width} x {height} pixels; the longest side may be {MAX_SIDE}")
    return bgr


def validate_frames(frames: list) -> list[np.ndarray]:
    """The decoded BGR frames of a burst, in order, or a ValueError saying what is wrong and with which frame (D20)."""
    if not isinstance(frames, list) or len(frames) not in FRAME_COUNTS:
        counts = " or ".join(str(count) for count in FRAME_COUNTS)
        raise ValueError(f"send a list of {counts} frames")
    return [check_frame(frame, f"frame {number} of {len(frames)}") for number, frame in enumerate(frames, start=1)]


def clean_name(name: str) -> str:
    """A player's name with control, format, surrogate, private-use and unassigned characters removed, trimmed to MAX_NAME."""
    if not isinstance(name, str):
        raise ValueError("a name must be text")
    kept = "".join(char for char in name if not unicodedata.category(char).startswith("C"))  # C... = every kind of non-text
    return kept.strip()[:MAX_NAME].strip()
