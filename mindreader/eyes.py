"""The bot's eyes: find one hand in an RGB image and return its 21 landmark points."""
import hashlib
import os
import urllib.request

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

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
