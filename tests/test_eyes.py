import base64
import contextlib
import hashlib
import os
import zipfile

import cv2
import numpy as np
import pandas as pd
import pytest
from mediapipe.tasks.python import vision
from sklearn.linear_model import LogisticRegression

from mindreader import eyes
from mindreader.hands import extension_ratios, features_from_row, hand_features, to_pixels

MODEL = "data/raw/hand_landmarker.task"  # real model, fetched once, gitignored
TEST_ZIP = "data/raw/rps-test-set.zip"
HAND_CSV = "data/hand_landmarks.csv"
GOOD_BYTES = b"pretend model"


@pytest.fixture
def local_model(tmp_path, monkeypatch):
    """Point ensure_model at a local file instead of the network; returns where it should land."""
    source = tmp_path / "source.task"
    source.write_bytes(GOOD_BYTES)
    monkeypatch.setattr(eyes, "MODEL_URL", source.as_uri())
    monkeypatch.setattr(eyes, "MODEL_SHA256", hashlib.sha256(GOOD_BYTES).hexdigest())
    return tmp_path / "hand.task"


def test_ensure_model_downloads_when_missing(local_model):
    assert eyes.ensure_model(str(local_model)) == str(local_model)
    assert local_model.read_bytes() == GOOD_BYTES


def test_ensure_model_keeps_a_good_file_without_downloading(local_model, monkeypatch):
    local_model.write_bytes(GOOD_BYTES)
    monkeypatch.setattr(eyes, "MODEL_URL", "file:///no/such/model")  # any download attempt would raise
    assert eyes.ensure_model(str(local_model)) == str(local_model)


@pytest.mark.parametrize("already_there", [False, True])
def test_ensure_model_rejects_wrong_bytes_and_leaves_no_file(local_model, monkeypatch, already_there):
    if already_there:
        local_model.write_bytes(b"truncated")  # an old half-finished copy
    else:
        monkeypatch.setattr(eyes, "MODEL_SHA256", "0" * 64)  # the fresh download will not match
    with pytest.raises(ValueError, match="hand model checksum mismatch"):
        eyes.ensure_model(str(local_model))
    assert list(local_model.parent.glob("hand.task*")) == []


@pytest.fixture(scope="module")
def model_path():
    return eyes.ensure_model(MODEL)


def test_blank_image_has_no_hand(model_path):
    blank = np.full((240, 320, 3), 128, np.uint8)
    with eyes.HandReader(model_path) as reader:
        assert reader.landmarks(blank) is None


needs_zip = pytest.mark.skipif(not os.path.exists(TEST_ZIP), reason="needs data/raw/rps-test-set.zip")


def cgi_rgb(member):
    with zipfile.ZipFile(TEST_ZIP) as z:
        png = z.read(member)
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


@needs_zip
def test_hand_image_gives_21_normalised_points(model_path):
    with eyes.HandReader(model_path) as reader:
        points = reader.landmarks(cgi_rgb("rps-test-set/rock/testrock01-00.png"))
    assert points.shape == (21, 3)
    assert ((points[:, :2] >= 0) & (points[:, :2] <= 1)).all()


@needs_zip
def test_non_contiguous_image_is_read_as_it_looks(model_path):
    mirrored_view = cgi_rgb("rps-test-set/paper/testpaper01-00.png")[:, ::-1]  # same pixels, odd memory layout
    with eyes.HandReader(model_path) as reader:
        from_view = reader.landmarks(mirrored_view)
        from_copy = reader.landmarks(np.ascontiguousarray(mirrored_view))
    np.testing.assert_allclose(from_view, from_copy, atol=1e-6)


@pytest.mark.parametrize("image", [
    np.zeros((50, 50), np.uint8),  # grey
    np.zeros((50, 50, 4), np.uint8),  # RGBA, e.g. straight from a browser canvas
    np.zeros((50, 50, 3), np.float32),  # 0-1 floats
])
def test_image_that_is_not_uint8_rgb_is_rejected(model_path, image):
    with eyes.HandReader(model_path) as reader:
        with pytest.raises(ValueError, match="uint8 image of shape"):
            reader.landmarks(image)


# ---- the live path: JPEG data URL -> frame -> landmarks -> gesture ----

def b64_url(raw: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")


def test_decode_jpeg_roundtrip():
    bgr = np.full((48, 64, 3), (200, 100, 50), np.uint8)  # blue, green and red all differ, so swapped channels would show
    url = eyes.encode_jpeg(bgr)
    assert url.startswith("data:image/jpeg;base64,")
    for text in (url, url.split(",", 1)[1]):  # with and without the data-URL prefix
        decoded = eyes.decode_jpeg(text)
        assert decoded.shape == (48, 64, 3) and decoded.dtype == np.uint8
        np.testing.assert_allclose(decoded, bgr, atol=3)  # JPEG is lossy, but a flat colour barely moves


def test_encode_jpeg_quality_changes_the_size():
    noise = np.random.default_rng(0).integers(0, 256, (96, 128, 3), dtype=np.uint8)
    low, default, high = (len(eyes.encode_jpeg(noise, **kw)) for kw in ({"quality": 30}, {}, {"quality": 95}))
    assert low < default < high


PNG_URL = "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", np.zeros((8, 8, 3), np.uint8))[1].tobytes()).decode("ascii")
GOOD_URL = eyes.encode_jpeg(np.zeros((8, 8, 3), np.uint8))
GARBAGE = {
    "stray character in a good JPEG": GOOD_URL[:60] + "!" + GOOD_URL[60:],  # strict base64 refuses it, a lenient decoder would skip it
    "line break in a good JPEG": GOOD_URL[:60] + "\n" + GOOD_URL[60:],
    "not a string": 12345,
    "bytes instead of text": b"\xff\xd8\xff",
    "empty": "",
    "prefix only": "data:image/jpeg;base64,",
    "not base64": "@@@ this is not base64 @@@",
    "text": b64_url(b"hello world"),
    "a PNG": PNG_URL,  # a good, decodable image, but only JPEG bytes may reach the decoder
    "JPEG marker, no image": b64_url(b"\xff\xd8\xff" + b"nonsense" * 20),
}


@pytest.mark.parametrize("max_side", [None, 1280])  # with a limit the header is read first, and must refuse the same things
@pytest.mark.parametrize("bad", list(GARBAGE.values()), ids=list(GARBAGE))
def test_decode_jpeg_rejects_garbage(bad, max_side):
    with pytest.raises(ValueError):
        eyes.decode_jpeg(bad, max_side)


def jpeg_bytes(width=64, height=48):
    """The bytes of a small JPEG, as a file would hold them."""
    return base64.b64decode(eyes.encode_jpeg(np.zeros((height, width, 3), np.uint8)).split(",", 1)[1])


def test_jpeg_size_reads_width_then_height_from_the_header():
    assert eyes.jpeg_size(jpeg_bytes(64, 48)) == (64, 48)
    assert eyes.jpeg_size(jpeg_bytes(48, 64)) == (48, 64)


def test_decode_jpeg_refuses_a_longer_side_than_max_side():
    url = eyes.encode_jpeg(np.zeros((30, 100, 3), np.uint8))
    assert eyes.decode_jpeg(url, max_side=100).shape == (30, 100, 3)  # the limit itself is allowed
    with pytest.raises(ValueError, match="100 x 30 pixels; the longest side may be 99"):
        eyes.decode_jpeg(url, max_side=99)


def test_a_header_cut_off_before_the_size_is_a_value_error():
    raw = jpeg_bytes()
    size_end = raw.index(b"\xff\xc0") + 9  # the frame header's marker, length, precision, height and width
    for cut in range(3, size_end):  # from just after the JPEG marker to one byte short of the size
        with pytest.raises(ValueError):
            eyes.jpeg_size(raw[:cut])
        with pytest.raises(ValueError):
            eyes.decode_jpeg(b64_url(raw[:cut]), max_side=1280)


@pytest.mark.parametrize("extra", [
    b"\xff\xfe\x00\x06abcd",  # a comment
    b"\xff\xdd\x00\x04\x00\x04",  # a restart interval
    b"\xff\xcc\x00\x04\x00\x00",  # arithmetic conditioning
    b"\xff\xe1\x00\x0a" + bytes(8),  # EXIF-like application data
    b"\xff\xe2\x01\x00" + bytes(254),  # ICC-profile-like application data (the length of 256 counts its own 2 bytes)
], ids=["comment", "restart interval", "arithmetic conditioning", "APP1", "APP2"])
def test_jpeg_size_hops_over_the_segments_real_files_have_before_the_frame_header(extra):
    raw = jpeg_bytes(64, 48)
    sof = raw.index(b"\xff\xc0")
    assert eyes.jpeg_size(raw[:sof] + extra + raw[sof:]) == (64, 48)


def test_jpeg_size_hops_over_a_huffman_table_that_comes_before_the_frame_header():
    raw = jpeg_bytes(64, 48)
    sof, table = raw.index(b"\xff\xc0"), raw.index(b"\xff\xc4")  # the table follows the frame header in these files
    table_end = table + 2 + int.from_bytes(raw[table + 2:table + 4], "big")
    assert eyes.jpeg_size(raw[:sof] + raw[table:table_end] + raw[sof:table] + raw[table_end:]) == (64, 48)


def test_jpeg_size_reads_progressive_and_restart_interval_files():
    bgr = np.zeros((48, 64, 3), np.uint8)
    for option in (cv2.IMWRITE_JPEG_PROGRESSIVE, cv2.IMWRITE_JPEG_RST_INTERVAL):
        assert eyes.jpeg_size(cv2.imencode(".jpg", bgr, [option, 1])[1].tobytes()) == (64, 48)


BIG_SOF = b"\xff\xc0\x00\x0b\x08\x4e\x20\x4e\x20\x01\x01\x11\x00"  # a frame header that claims 20000 x 20000
SMALL_SOF = b"\xff\xc0\x00\x0b\x08\x00\x10\x00\x10\x01\x01\x11\x00"  # one that claims 16 x 16
ODD_HEADERS = {
    "a segment shorter than its own length field": b"\xff\xd8\xff\xe0\x00\x01" + bytes(20),
    "a scan before any frame header": b"\xff\xd8\xff\xda\x00\x08" + bytes(20),
    "fill bytes before a marker": b"\xff\xd8\xff\xff\xe0\x00\x04" + bytes(20),
    "junk between segments": b"\xff\xd8\xff\xe0\x00\x04\x00\x00junk" + SMALL_SOF,
    "no frame header at all": b"\xff\xd8" + b"\xff\xe1\x00\x0a" + bytes(8) + b"\xff\xe1\x00\x0a" + bytes(8),
    # A restart marker (FF D0) has no length. The decoder goes on to BIG_SOF, but a reader that took the next two bytes
    # (FF C0 = 65472) as its length would skip ahead and report SMALL_SOF, so the size check would be fooled.
    "a restart marker hiding the real frame header": b"\xff\xd8\xff\xd0" + BIG_SOF + bytes(65476 - 4 - len(BIG_SOF)) + SMALL_SOF,
    # The decoder skips stray bytes up to the next FF, so it reads BIG_SOF here. A reader that did not insist on FF at the
    # start of each segment would take 00 DB FF C0 as a table 65472 bytes long and report SMALL_SOF.
    "stray bytes that look like a segment hiding the real frame header":
        b"\xff\xd8\xff\xfe\x00\x02" + b"\x00\xdb" + BIG_SOF + bytes(65480 - 8 - len(BIG_SOF)) + SMALL_SOF,
}


@pytest.mark.parametrize("raw", list(ODD_HEADERS.values()), ids=list(ODD_HEADERS))
def test_jpeg_size_refuses_an_odd_header(raw):
    with pytest.raises(ValueError):
        eyes.jpeg_size(raw)


def test_jpeg_size_only_ever_raises_value_error():
    rng = np.random.default_rng(0)
    real = jpeg_bytes()[:200]
    for _ in range(3000):
        damaged = bytearray(real)
        for place in rng.integers(2, len(damaged), 4):
            damaged[place] = rng.integers(0, 256)
        with contextlib.suppress(ValueError):  # any other exception fails the test
            eyes.jpeg_size(bytes(damaged))


def test_predict_gesture_uses_mirror_average():
    points = np.random.default_rng(0).uniform(0, 400, size=(21, 2))  # made-up pixel hand
    hand = hand_features(points)
    mirrored = hand.copy()
    mirrored[0:42:2] *= -1  # a mirror negates every canonical x and leaves the five ratios alone

    class Stub:
        classes_ = ["rock", "paper", "scissors"]  # deliberately not alphabetical

        def predict_proba(self, rows):
            assert isinstance(rows, np.ndarray)  # plain numpy, so scikit-learn has no feature names to complain about

            def answer(row):
                if np.allclose(row, hand):
                    return [1, 0, 0]  # rock for the hand
                if np.allclose(row, mirrored):
                    return [0, 1, 0]  # paper for its mirror
                return [0, 0, 1]  # scissors for anything else

            return np.array([answer(row) for row in rows])

    assert eyes.predict_gesture(Stub(), points) == ("rock", 0.5)


class ScriptedReader:
    """Stands in for HandReader: the red value of a frame (0, 1, 2 ...) says which scripted hand it holds, or None."""

    def __init__(self, hands):
        self.hands = hands

    def landmarks(self, rgb):
        return self.hands[int(rgb[0, 0, 0])]  # channel 0 is red only if read_throw swapped BGR to RGB


class ScriptedModel:
    """Stands in for a fitted classifier: each predict_proba call answers the next (label, prob) of the script."""
    classes_ = ["rock", "paper", "scissors"]  # deliberately not alphabetical

    def __init__(self, script):
        self.script = list(script)
        self.inputs = []

    def predict_proba(self, features):
        self.inputs.append(features)
        label, prob = self.script.pop(0)
        row = np.full(3, (1 - prob) / 2)
        row[self.classes_.index(label)] = prob
        return np.vstack([row, row])  # the hand and its mirror agree


def burst(*steps):
    """A 5-frame scene. A step is (label, prob) for a frame the model will read, or None for a frame with no hand."""
    hands = [None if step is None else np.random.default_rng(i).uniform(0.1, 0.9, size=(21, 3)) for i, step in enumerate(steps)]
    frames = [np.zeros((48, 64, 3), np.uint8) for _ in steps]
    for i, frame in enumerate(frames):
        frame[:, :, 2] = i  # BGR order: channel 2 is red
    return frames, ScriptedReader(hands), ScriptedModel([step for step in steps if step]), hands


def test_read_throw_accepts_3_of_5():
    frames, reader, model, hands = burst(None, ("rock", .9), ("rock", .8), ("rock", .7), ("paper", .9))
    result = eyes.read_throw(reader, model, frames)
    assert result["label"] == "rock"
    assert result["agree"] == 3
    assert result["prob"] == pytest.approx(0.8)
    assert result["best"] == 1  # the .9 rock, counted in frames (frame 0 had no hand)
    np.testing.assert_array_equal(result["landmarks"], hands[1])
    assert [type(result[key]) for key in ("label", "prob", "agree", "best")] == [str, float, int, int]  # JSON-ready


def test_read_throw_rejects_low_confidence():
    frames, reader, model, hands = burst(("rock", .6), ("rock", .8), ("rock", .4), ("paper", .9), None)
    result = eyes.read_throw(reader, model, frames)
    assert result["label"] is None  # three agree, but their mean is 0.6 and min_prob is 0.7
    assert result["agree"] == 3 and result["prob"] == pytest.approx(0.6)
    assert result["best"] == 1  # still describes the winner: its .8 frame, not the .9 paper
    np.testing.assert_array_equal(result["landmarks"], hands[1])


@pytest.mark.parametrize("need, expected", [(3, None), (2, "rock")])
def test_read_throw_needs_enough_agreeing_frames(need, expected):
    frames, reader, model, _ = burst(("rock", .9), ("rock", .9), ("paper", .9), ("scissors", .9), None)
    result = eyes.read_throw(reader, model, frames, need=need)
    assert result["label"] == expected and result["agree"] == 2


def test_read_throw_accepts_exactly_the_thresholds():
    frames, reader, model, _ = burst(("rock", 1.0), ("rock", .75), ("rock", .5), ("paper", 1.0), None)
    result = eyes.read_throw(reader, model, frames, need=3, min_prob=.75)  # the mean of 1, .75 and .5 is exactly .75
    assert result["label"] == "rock"


def test_read_throw_breaks_a_tie_for_the_label_seen_first():
    frames, reader, model, _ = burst(("scissors", .9), ("rock", .9), ("scissors", .9), ("rock", .9), None)
    assert eyes.read_throw(reader, model, frames, need=2)["label"] == "scissors"  # not rock, which is alphabetical and last


def test_read_throw_rejects_no_hand():
    frames, reader, model, _ = burst(None, None, None, None, None)
    assert eyes.read_throw(reader, model, frames) == {"label": None, "prob": 0.0, "agree": 0, "best": None, "landmarks": None}


def test_read_throw_measures_the_hand_in_pixels_of_its_own_frame():
    # Frame 100 wide, 50 high. The wrist is at the bottom; the index joint is 0.1 across = 10 px from it, the index tip
    # 0.4 up = 20 px, so the index ratio is 2.0. Swap width and height and it is 8.0; skip the pixels and it is 4.0.
    hand = np.tile([0.5, 0.9, 0.0], (21, 1))
    hand[[2, 9, 10, 14, 18]] += [0, -0.1, 0]  # the other joints and the middle knuckle, 5 px up
    hand[6] += [0.1, 0, 0]
    hand[8] += [0, -0.4, 0]
    model = ScriptedModel([("rock", .9)])
    eyes.read_throw(ScriptedReader([hand]), model, [np.zeros((50, 100, 3), np.uint8)])
    hand_row, mirror_row = model.inputs[0]
    assert hand_row[43] == pytest.approx(2.0) and mirror_row[43] == pytest.approx(2.0)  # feature 43 is ext_index


OPEN_HAND = np.array([  # a stylised open hand in a 320 x 240 frame, in pixels: wrist, then thumb, index, middle, ring, pinky
    (160, 220),
    (120, 195), (95, 165), (80, 135), (70, 105),
    (125, 150), (120, 110), (118, 80), (116, 50),
    (155, 145), (155, 100), (155, 65), (155, 30),
    (185, 150), (192, 108), (196, 78), (199, 48),
    (213, 165), (228, 130), (237, 105), (245, 80),
])


def test_draw_skeleton_draws():
    frame = np.zeros((240, 320, 3), np.uint8)
    hand = np.column_stack([OPEN_HAND / (320, 240), np.zeros(21)])  # normalised x, y, z
    drawn = eyes.draw_skeleton(frame, hand)
    assert not frame.any()  # the original is left alone
    assert drawn.shape == frame.shape and drawn.dtype == np.uint8
    for x, y in OPEN_HAND:
        assert drawn[y, x].any(), f"no joint at {x}, {y}"
    for a, b in eyes.CONNECTIONS:
        x, y = (OPEN_HAND[a] + OPEN_HAND[b]) // 2  # every bone is 26+ px long and 12+ px from any other line or joint
        assert drawn[y - 1:y + 2, x - 1:x + 2].any(), f"no bone between joints {a} and {b}"
    for tip in (4, 8, 12, 16, 20):  # each fingertip has one bone ending on it, so a blob there is the dot
        x, y = OPEN_HAND[tip]
        assert drawn[y - 3:y + 4, x - 3:x + 4].any(axis=2).sum() > 20, f"no dot on joint {tip}"
    assert not drawn[0, 0].any()  # nothing far from the hand


def test_connections_are_mediapipes_hand_bones():
    official = {(bone.start, bone.end) for bone in vision.HandLandmarksConnections.HAND_CONNECTIONS}
    assert set(eyes.CONNECTIONS) == official and len(eyes.CONNECTIONS) == 21


needs_live_data = pytest.mark.skipif(not (os.path.exists(TEST_ZIP) and os.path.exists(MODEL)),
                                     reason="needs data/raw/rps-test-set.zip and data/raw/hand_landmarker.task")


@pytest.fixture(scope="module")
def logreg():
    """A classifier fitted on the train rows as a plain numpy array (R12). It only turns hands into labels here."""
    rows = pd.read_csv(HAND_CSV)
    train = rows[rows["split"] == "train"]
    return LogisticRegression(max_iter=1000).fit(np.array([features_from_row(row) for _, row in train.iterrows()]), train["label"])


def csv_path(logreg, row):
    """What the offline pipeline made of an image: the stored landmarks -> (ratios, label)."""
    stored = np.array([[row[f"x{i}"], row[f"y{i}"], row[f"z{i}"]] for i in range(21)])
    points = to_pixels(stored, row["width"], row["height"])
    return extension_ratios(points), eyes.predict_gesture(logreg, points)[0]


def through_jpeg(bgr):
    return eyes.decode_jpeg(eyes.encode_jpeg(bgr))


def through_png(bgr):  # lossless, so any gap left is the pipeline's, not the codec's
    return cv2.imdecode(cv2.imencode(".png", bgr)[1], cv2.IMREAD_COLOR)


def live_path(reader, logreg, bgr, round_trip):
    """What the game makes of the same image: round trip -> landmarks -> (ratios, label), or None if no hand."""
    decoded = round_trip(bgr)
    landmarks = reader.landmarks(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
    if landmarks is None:
        return None
    points = to_pixels(landmarks, decoded.shape[1], decoded.shape[0])
    return extension_ratios(points), eyes.predict_gesture(logreg, points)[0]


def live_outcomes(logreg, round_trip, flip=False):
    """20 test images through the live path, as they are or flipped left to right.

    One (hand found, same label as the CSV path, largest ratio gap to the CSV row) per image."""
    rows = pd.read_csv(HAND_CSV)
    sample = rows[rows["split"] == "test"].sample(n=20, random_state=0)
    outcomes = []
    with zipfile.ZipFile(TEST_ZIP) as z, eyes.HandReader(MODEL) as reader:
        for _, row in sample.iterrows():
            png = z.read(f"rps-test-set/{row['label']}/{row['file']}")  # the image stays in memory, never on disk
            bgr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
            live = live_path(reader, logreg, cv2.flip(bgr, 1) if flip else bgr, round_trip)
            if live is None:
                outcomes.append((False, False, float("inf")))
            else:
                csv_ratios, csv_label = csv_path(logreg, row)
                outcomes.append((True, live[1] == csv_label, float(np.abs(live[0] - csv_ratios).max())))
    return outcomes


@pytest.fixture(scope="module", params=[False, True], ids=["as_is", "flipped"])
def jpeg_outcomes(request, logreg):
    return live_outcomes(logreg, through_jpeg, request.param)


# Labels only for the JPEG path. Measured on these 20 images: JPEG quality 80 moves the ratios by up to 0.103 (a 0.02 bar
# failed 8 of 20), and flipping moves them by up to 0.21 even as lossless PNG, since MediaPipe's landmarks on a flipped
# image are not the mirror of the original's. Do not add a ratio bar here; the lossless test below is the exact one.
@needs_live_data
def test_live_path_matches_csv(jpeg_outcomes):
    agree = sum(found and same for found, same, _ in jpeg_outcomes)
    assert agree >= 19, f"hand found with the CSV pipeline's label in only {agree} of 20 images"


@needs_live_data
def test_lossless_live_path_gives_the_csv_ratios(logreg):
    outcomes = live_outcomes(logreg, through_png)
    assert all(found for found, _, _ in outcomes), "a hand was not found in every image"
    assert max(gap for _, _, gap in outcomes) < 1e-3  # the CSV keeps 6 decimals; the measured largest gap is 5.6e-06
