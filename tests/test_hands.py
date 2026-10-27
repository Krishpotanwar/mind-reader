import numpy as np
import pandas as pd
import pytest

from mindreader.hands import (FEATURE_NAMES, MIDDLE_MCP, canonical, extension_ratios, features_from_row,
                              finger_states, hand_features, mirror, to_pixels)

# One hand worked out on paper. Its wrist is at pixel (10, 20) and its middle knuckle (point 9) sits at (+3, +4),
# 5 pixels away, so every number below comes from a 3-4-5 triangle. A point not listed sits on the wrist.
WRIST_PIXEL = (10, 20)
WORKED = {  # point -> offset from the wrist, in pixels
    2: (0, 5), 4: (0, 10), 5: (4, -3), 6: (6, 8), 8: (9, 12), 9: (3, 4),
    10: (8, 6), 12: (4, 3), 14: (0, -4), 16: (0, -1), 18: (5, 0), 20: (15, 0),
}
# The same points after canonical. Turning (3, 4) onto (0, -1) and dividing by 5 maps an offset (x, y)
# to (-0.16x + 0.12y, -0.12x - 0.16y).
WORKED_CANONICAL = {
    2: (0.6, -0.8), 4: (1.2, -1.6), 5: (-1, 0), 6: (0, -2), 8: (0, -3), 9: (0, -1),
    10: (-0.56, -1.92), 12: (-0.28, -0.96), 14: (-0.48, 0.64), 16: (-0.12, 0.16), 18: (-0.8, -0.6), 20: (-2.4, -1.8),
}
WORKED_RATIOS = [2.0, 1.5, 0.5, 0.25, 3.0]  # thumb, index, middle, ring, pinky: tip distance / joint distance


def hand_from(offsets, wrist=(0, 0)):
    """A (21, 2) hand: each listed point at its offset from the wrist, every other point on the wrist."""
    hand = np.tile(wrist, (21, 1)).astype(float)
    for point, offset in offsets.items():
        hand[point] += offset
    return hand


def worked_pixels():
    return hand_from(WORKED, WRIST_PIXEL)


def worked_features():
    return np.concatenate([hand_from(WORKED_CANONICAL).ravel(), WORKED_RATIOS])


def random_hand():
    """21 arbitrary pixel points, fixed by the seed, with no symmetry for a bug to hide behind."""
    return np.random.default_rng(0).uniform(0, 400, size=(21, 2))


def transform(points, degrees, scale, shift):
    """The same hand held differently: rotated, scaled, then moved."""
    a = np.radians(degrees)
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return scale * (points @ rotation.T) + shift


@pytest.mark.parametrize("degrees, scale, shift", [(37, 2.5, (100, -40)), (180, 0.1, (0, 0)), (-90, 1, (-500, 300)), (359.9, 3, (7, 7))])
def test_canonical_invariant_to_translation_rotation_scale(degrees, scale, shift):
    hand = random_hand()
    np.testing.assert_allclose(canonical(transform(hand, degrees, scale, shift)), canonical(hand), atol=1e-9)


@pytest.mark.parametrize("knuckle", [(0, -5), (5, 0), (0, 5), (-5, 0), (3, 4), (-3, 4), (3, -4), (-3, -4)])
def test_canonical_puts_middle_knuckle_up(knuckle):
    hand = hand_from({MIDDLE_MCP: knuckle}, wrist=(50, 60))
    np.testing.assert_allclose(canonical(hand)[MIDDLE_MCP], (0, -1), atol=1e-9)


def test_canonical_worked_example():
    # Pins which way the other points swing (index knuckle on the left), not just where point 9 lands.
    np.testing.assert_allclose(canonical(worked_pixels()), hand_from(WORKED_CANONICAL), atol=1e-12)


def test_mirror_negates_x_and_leaves_the_input_alone():
    points = np.array([[1.0, 2.0], [-3.0, 4.0]])
    np.testing.assert_array_equal(mirror(points), [[-1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(points, [[1.0, 2.0], [-3.0, 4.0]])


def test_canonical_commutes_with_mirror():
    mirrored_literal = hand_from(WORKED_CANONICAL) * (-1, 1)
    np.testing.assert_allclose(canonical(mirror(worked_pixels())), mirrored_literal, atol=1e-12)


def test_extension_ratios_worked_example():
    pixels = worked_pixels()
    np.testing.assert_allclose(extension_ratios(pixels), WORKED_RATIOS)
    np.testing.assert_allclose(extension_ratios(transform(pixels, 37, 2.5, (100, -40))), WORKED_RATIOS)


def test_extension_ratios_mirror_invariant():
    np.testing.assert_allclose(extension_ratios(mirror(worked_pixels())), WORKED_RATIOS)


def test_features_shape_47():
    features = hand_features(random_hand())
    assert features.shape == (47,)
    assert len(FEATURE_NAMES) == 47 and len(set(FEATURE_NAMES)) == 47
    assert FEATURE_NAMES[:4] == ["cx0", "cy0", "cx1", "cy1"]
    assert FEATURE_NAMES[-5:] == ["ext_thumb", "ext_index", "ext_middle", "ext_ring", "ext_pinky"]


def test_hand_features_layout_and_names():
    pixels = worked_pixels()
    before = pixels.copy()
    features = hand_features(pixels)
    np.testing.assert_allclose(features, worked_features(), atol=1e-12)
    np.testing.assert_array_equal(pixels, before)  # the caller's points are not touched
    by_name = dict(zip(FEATURE_NAMES, features))
    assert by_name["cx5"] == pytest.approx(-1) and by_name["cy5"] == pytest.approx(0, abs=1e-12)
    assert by_name["cx9"] == pytest.approx(0, abs=1e-12) and by_name["cy9"] == pytest.approx(-1)
    assert by_name["ext_thumb"] == pytest.approx(2) and by_name["ext_ring"] == pytest.approx(0.25)


@pytest.mark.parametrize("function", [canonical, extension_ratios, hand_features])
def test_degenerate_hand_raises_instead_of_returning_nan(function):
    with pytest.raises(ValueError):
        function(np.zeros((21, 2)))


def test_to_pixels_scales_x_by_width_and_y_by_height_and_drops_z():
    landmarks = np.tile([0.5, 0.25, 0.9], (21, 1))
    pixels = to_pixels(landmarks, 640, 480)
    assert pixels.shape == (21, 2)
    np.testing.assert_allclose(pixels, np.tile([320.0, 120.0], (21, 1)))


def test_to_pixels_removes_aspect_skew():
    hand = random_hand()  # one physical hand, in pixels that fit inside both frames
    in_wide_frame = np.column_stack([hand / (640, 480), np.zeros(21)])  # normalised as MediaPipe does: (x, y, z)
    in_tall_frame = np.column_stack([hand / (480, 640), np.zeros(21)])
    np.testing.assert_allclose(hand_features(to_pixels(in_wide_frame, 640, 480)),
                               hand_features(to_pixels(in_tall_frame, 480, 640)), atol=1e-9)


def test_finger_states_thresholds():
    for thumb in (0.0, 5.0):  # the thumb is not read
        assert finger_states([thumb, 1.2, 1.2, 0.9, 0.9]) == ("up", "up", "down", "down")
    assert finger_states([1.0] * 5) == ("unclear",) * 4
    assert finger_states([0, 1.15, 0.95, 1.151, 0.949]) == ("unclear", "unclear", "up", "down")  # strict comparisons
    assert finger_states([0, 1.01, 0.99, 1.5, 0.2], up=1.0, down=1.0) == ("up", "down", "up", "down")  # the single 1.0 threshold


def test_features_from_row_reads_pixel_columns():
    row = {"file": "a.png", "split": "train", "label": "rock", "width": 640, "height": 480}
    for point, (x, y) in enumerate(worked_pixels() / (640, 480)):
        row[f"x{point}"], row[f"y{point}"], row[f"z{point}"] = x, y, 0.0
    np.testing.assert_allclose(features_from_row(pd.Series(row)), worked_features(), atol=1e-12)


def test_features_from_csv_rows_finite():
    rows = pd.read_csv("data/hand_landmarks.csv", nrows=50)
    features = np.array([features_from_row(row) for _, row in rows.iterrows()])
    assert features.shape == (50, 47)
    assert np.isfinite(features).all()
