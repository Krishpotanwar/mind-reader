"""Hand geometry: turn the 21 MediaPipe landmarks into 47 numbers.

The numbers do not depend on where the hand is, how big it is or which way it points."""
import numpy as np

WRIST = 0
MIDDLE_MCP = 9  # the knuckle at the base of the middle finger
TIPS = (4, 8, 12, 16, 20)  # thumb, index, middle, ring, pinky
JOINTS = (2, 6, 10, 14, 18)  # the middle joint of each of those fingers

FEATURE_NAMES = (
    [f"c{axis}{point}" for point in range(21) for axis in "xy"]  # cx0, cy0, cx1, cy1, ... like the flattened points
    + [f"ext_{finger}" for finger in ("thumb", "index", "middle", "ring", "pinky")]
)


def to_pixels(landmarks: np.ndarray, width: int, height: int) -> np.ndarray:
    """(21, 3) normalised landmarks -> (21, 2) pixel x, y. Without this a 4:3 frame would squash the hand."""
    return landmarks[:, :2] * (width, height)


def canonical(points: np.ndarray) -> np.ndarray:
    """Wrist at the origin, wrist-to-point-9 turned to point straight up (0, -1) and scaled to length 1."""
    shifted = points - points[WRIST]
    vx, vy = shifted[MIDDLE_MCP]
    length = np.hypot(vx, vy)
    if length == 0:
        raise ValueError("the wrist and the middle knuckle are at the same point")
    turn = -np.arctan2(vx, -vy)  # minus the knuckle's angle away from straight up
    rotation = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
    return shifted @ rotation.T / length


def mirror(points: np.ndarray) -> np.ndarray:
    """The hand as seen in a mirror: every x negated. Returns a new array."""
    return points * (-1, 1)


def extension_ratios(points: np.ndarray) -> np.ndarray:
    """Per finger, |tip - wrist| divided by |middle joint - wrist|. Rotation, scale and mirroring do not change it."""
    reach = np.linalg.norm(points[list(TIPS)] - points[WRIST], axis=1)
    base = np.linalg.norm(points[list(JOINTS)] - points[WRIST], axis=1)
    if (base == 0).any():
        raise ValueError("a middle joint is at the same point as the wrist")
    return reach / base


def hand_features(points: np.ndarray) -> np.ndarray:
    """The 47 features: the canonical points flattened as x0, y0, x1, y1, ... (42), then the 5 ratios."""
    return np.concatenate([canonical(points).ravel(), extension_ratios(points)])


def finger_states(ratios, up: float = 1.15, down: float = 0.95) -> tuple[str, ...]:
    """'up', 'down' or 'unclear' for the index, middle, ring and pinky. The thumb's joints don't follow the same rule."""
    states = []
    for ratio in ratios[1:5]:
        if ratio > up:
            states.append("up")
        elif ratio < down:
            states.append("down")
        else:
            states.append("unclear")
    return tuple(states)


def features_from_row(row) -> np.ndarray:
    """The 47 features for one row of data/hand_landmarks.csv (a pandas Series)."""
    normalised = np.array([[row[f"x{point}"], row[f"y{point}"]] for point in range(21)])
    return hand_features(to_pixels(normalised, row["width"], row["height"]))
