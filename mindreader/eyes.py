"""The bot's eyes: find one hand in an RGB image and return its 21 landmark points, then name the gesture."""
import base64
import hashlib
import os
import urllib.request
from collections import Counter
from typing import NamedTuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from mindreader.hands import hand_features, mirror, to_pixels

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"


def ensure_model(path: str = "hand_landmarker.task") -> str:
    """Download the hand model if it is missing, then check its sha256. A wrong file is deleted and ValueError raised."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, path + ".part")  # a dropped connection leaves only the .part file
        os.replace(path + ".part", path)
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != MODEL_SHA256:
        os.remove(path)  # so the next run downloads a fresh copy
        raise ValueError("hand model checksum mismatch")
    return path


class HandReader:
    """Use in a `with` block: MediaPipe's HandLandmarker must be closed, or it raises when Python exits."""

    def __init__(self, model_path: str):
        self._options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path), num_hands=1)

    def __enter__(self):
        self._landmarker = vision.HandLandmarker.create_from_options(self._options)
        return self

    def __exit__(self, *exc_info):
        self._landmarker.close()

    def landmarks(self, rgb: np.ndarray) -> np.ndarray | None:
        """Return a (21, 3) array of normalised x, y, z for the hand in the image, or None if no hand is found."""
        rgb = np.ascontiguousarray(rgb)  # MediaPipe reads raw memory, so a flipped or sliced view would be misread
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"expected a uint8 image of shape (height, width, 3), got {rgb.dtype} {rgb.shape}")
        result = self._landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.hand_landmarks:
            return None
        return np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]])


JPEG_MARKER = b"\xff\xd8\xff"  # every JPEG file starts with these three bytes
CONNECTIONS = (  # the bones: pairs of landmark points. 0 is the wrist, then four points per finger from the palm out
    (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # index finger
    (5, 9), (9, 10), (10, 11), (11, 12),  # middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),  # ring finger
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
)


def jpeg_size(raw: bytes) -> tuple[int, int]:
    """(width, height) from a JPEG's frame header, read without decoding any pixels. ValueError if the header is cut off or odd.

    Only the segments that real files put before the frame header (tables, comments, application data) are hopped over.
    Anything else is refused, so this reader and the decoder cannot disagree about where the frame header is."""
    i = 2  # past the FF D8 start marker
    while i + 9 <= len(raw) and raw[i] == 0xFF:  # a segment is FF, a marker byte, a 2-byte length that counts itself, then data
        marker, length = raw[i + 1], int.from_bytes(raw[i + 2:i + 4], "big")
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):  # the frame header: precision, height, width
            return int.from_bytes(raw[i + 7:i + 9], "big"), int.from_bytes(raw[i + 5:i + 7], "big")
        if not (marker in (0xC4, 0xCC, 0xDB, 0xDD, 0xFE) or 0xE0 <= marker <= 0xEF):  # tables, comment, application data
            break
        i += 2 + length  # always forward, so the loop ends
    raise ValueError("the JPEG header is cut off or not one this game reads")


def decode_jpeg(data_url: str, max_side: int | None = None) -> np.ndarray:
    """A JPEG as a data URL (or bare base64) -> a BGR image. Raises ValueError for anything that is not a good JPEG.

    With max_side, a JPEG whose header declares a longer side is refused before any pixels are allocated."""
    if not isinstance(data_url, str):
        raise ValueError("a JPEG must be given as a string")
    if data_url.startswith("data:"):
        data_url = data_url.partition(",")[2]  # keep what follows "data:image/jpeg;base64,"
    try:
        raw = base64.b64decode(data_url, validate=True)
    except ValueError:  # binascii.Error is a ValueError
        raise ValueError("not valid base64") from None
    if not raw.startswith(JPEG_MARKER):  # so only JPEG bytes ever reach a JPEG parser
        raise ValueError("not a JPEG")
    if max_side is not None:
        width, height = jpeg_size(raw)
        if max(width, height) > max_side:
            raise ValueError(f"{width} x {height} pixels; the longest side may be {max_side}")
    try:
        bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        bgr = None
    if bgr is None:
        raise ValueError("the JPEG could not be decoded")
    return bgr


def encode_jpeg(bgr: np.ndarray, quality: int = 80) -> str:
    """A BGR image -> a JPEG data URL, the form a browser <img> accepts."""
    ok, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("the image could not be encoded as a JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode("ascii")


def predict_gesture(model, points: np.ndarray) -> tuple[str, float]:
    """The model's (label, probability) for one hand, averaged with its answer for the mirrored hand (D2).

    `points` are the (21, 2) pixel points from to_pixels. One predict_proba call scores both: row 0 the hand, row 1 its mirror."""
    both = np.vstack([hand_features(points), hand_features(mirror(points))])
    average = model.predict_proba(both).mean(axis=0)
    best = int(np.argmax(average))  # a tie goes to the first class
    return str(model.classes_[best]), float(average[best])  # classes_ is sorted by scikit-learn, not rock, paper, scissors


class Vote(NamedTuple):
    frame: int  # position in the burst
    label: str
    prob: float
    landmarks: np.ndarray


def read_throw(reader, model, frames_bgr: list, need: int = 3, min_prob: float = 0.7) -> dict:
    """Name the gesture in a burst of BGR frames by vote (D4).

    Each frame with a hand votes for a label. The most frequent label wins (a tie goes to the one seen first). The throw
    is accepted, so `label` is not None, if at least `need` frames agree and their mean probability is `min_prob` or more.
    Even when it is not accepted, prob, agree, best and landmarks describe the winner: `best` is the index in frames_bgr
    of its most confident frame and `landmarks` that frame's (21, 3) points. With no hand at all they are 0.0, 0, None, None."""
    votes = []
    for frame, bgr in enumerate(frames_bgr):
        landmarks = reader.landmarks(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if landmarks is not None:  # a frame without a hand casts no vote
            height, width = bgr.shape[:2]
            label, prob = predict_gesture(model, to_pixels(landmarks, width, height))
            votes.append(Vote(frame, label, prob, landmarks))
    if not votes:
        return {"label": None, "prob": 0.0, "agree": 0, "best": None, "landmarks": None}
    winner = Counter(vote.label for vote in votes).most_common(1)[0][0]
    winning = [vote for vote in votes if vote.label == winner]
    prob = sum(vote.prob for vote in winning) / len(winning)
    best = max(winning, key=lambda vote: vote.prob)  # a tie goes to the earlier frame
    accepted = len(winning) >= need and prob >= min_prob
    return {"label": winner if accepted else None, "prob": prob, "agree": len(winning),
            "best": best.frame, "landmarks": best.landmarks}


def draw_skeleton(bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """A copy of the frame with the hand's bones and joints drawn on it. Frames are never flipped (D1), nor are the points."""
    height, width = bgr.shape[:2]
    points = to_pixels(landmarks, width, height).round().astype(int).tolist()
    drawn = bgr.copy()
    for a, b in CONNECTIONS:
        cv2.line(drawn, tuple(points[a]), tuple(points[b]), (0, 255, 0), 2)  # green bones
    for x, y in points:
        cv2.circle(drawn, (x, y), 4, (0, 0, 255), -1)  # red joints
    return drawn
