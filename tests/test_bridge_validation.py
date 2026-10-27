import base64
import time

import cv2
import numpy as np
import pytest

from mindreader import bridge, eyes


def jpeg(width, height):
    """A black JPEG data URL of the given size."""
    return eyes.encode_jpeg(np.zeros((height, width, 3), np.uint8))


def declares(width, height):
    """A tiny valid JPEG whose frame header claims width x height pixels. Its pixel data is still 16 x 16."""
    raw = bytearray(base64.b64decode(jpeg(16, 16).split(",", 1)[1]))
    sof = raw.index(b"\xff\xc0")  # the frame header: marker, length, precision, then height and width as 2-byte numbers
    raw[sof + 5:sof + 9] = height.to_bytes(2, "big") + width.to_bytes(2, "big")
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")


@pytest.mark.parametrize("count", [3, 5])
def test_validate_frames_returns_the_decoded_frames_in_order(count):
    widths = [40 + 10 * i for i in range(count)]  # different widths tell the frames apart
    frames = bridge.validate_frames([jpeg(width, 30) for width in widths])
    assert [frame.shape for frame in frames] == [(30, width, 3) for width in widths]


@pytest.mark.parametrize("bad", [
    [], [jpeg(40, 30)] * 2, [jpeg(40, 30)] * 4, [jpeg(40, 30)] * 6,
    tuple([jpeg(40, 30)] * 3), "not a list", None,
], ids=["none", "two", "four", "six", "tuple", "string", "None"])
def test_validate_frames_count(bad):
    with pytest.raises(ValueError, match="3 or 5"):
        bridge.validate_frames(bad)


def test_validate_frames_size():
    noise = np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    fat = eyes.encode_jpeg(noise, quality=95)  # decodable and only 640 x 480, but too long to send
    assert len(fat) > bridge.MAX_FRAME_CHARS
    garbage = "A" * (bridge.MAX_FRAME_CHARS + 1)  # too long and not a JPEG either: the length is judged first
    for too_long in (fat, garbage):
        with pytest.raises(ValueError, match=f"frame 2.*{bridge.MAX_FRAME_CHARS}"):
            bridge.validate_frames([jpeg(40, 30), too_long, jpeg(40, 30)])


def test_validate_frames_size_limit_is_inclusive(monkeypatch):
    frame = jpeg(40, 30)
    monkeypatch.setattr(bridge, "MAX_FRAME_CHARS", len(frame))
    assert len(bridge.validate_frames([frame] * 3)) == 3
    monkeypatch.setattr(bridge, "MAX_FRAME_CHARS", len(frame) - 1)
    with pytest.raises(ValueError):
        bridge.validate_frames([frame] * 3)


@pytest.mark.parametrize("width, height, allowed", [
    (2000, 2000, False), (1281, 100, False), (100, 1281, False),  # the longest side counts, whichever side it is
    (1280, 720, True), (720, 1280, True),
])
def test_validate_frames_dimensions(width, height, allowed):
    big = jpeg(width, height)
    assert len(big) <= bridge.MAX_FRAME_CHARS  # so only the dimensions can be the problem
    frames = [jpeg(40, 30), big, jpeg(40, 30)]
    if allowed:
        assert bridge.validate_frames(frames)[1].shape == (height, width, 3)
    else:
        with pytest.raises(ValueError, match=f"frame 2.*{bridge.MAX_SIDE}"):
            bridge.validate_frames(frames)


@pytest.mark.parametrize("width, height", [(20000, 20000), (65535, 1), (1, 65535), (1281, 1)])
def test_validate_frames_refuses_a_declared_size_before_any_pixels_exist(monkeypatch, width, height):
    def no_decoding(*args, **kwargs):
        raise AssertionError("cv2.imdecode ran, so the pixel buffer would already exist")

    monkeypatch.setattr(cv2, "imdecode", no_decoding)
    liar = declares(width, height)
    assert len(liar) < 1000  # a few hundred characters that claim up to 4 billion pixels
    with pytest.raises(ValueError, match=f"frame 1 of 3: {width} x {height} pixels; the longest side may be {bridge.MAX_SIDE}"):
        bridge.validate_frames([liar, liar, liar])


def test_validate_frames_names_a_frame_cut_off_after_the_jpeg_marker():
    cut = "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8\xff").decode("ascii")
    with pytest.raises(ValueError, match="frame 2 of 3"):
        bridge.validate_frames([jpeg(40, 30), cut, jpeg(40, 30)])


@pytest.mark.parametrize("bad", [None, 42, b"bytes", "", "not base64 !", "data:image/jpeg;base64,AAAA"])
def test_validate_frames_names_the_bad_frame(bad):
    with pytest.raises(ValueError, match="frame 3"):
        bridge.validate_frames([jpeg(40, 30), jpeg(40, 30), bad])


@pytest.mark.parametrize("raw, expected", [
    ("  Krish  ", "Krish"),
    ("Kr\x00is\nh\t", "Krish"),  # control characters (category Cc) go
    ("\u202eKrish\u200b\ufeff", "Krish"),  # right-to-left override and zero-width characters (Cf) go
    ("A\ud800B\ue000C\u0378D", "ABCD"),  # a lone surrogate (Cs), private use (Co) and an unassigned code point (Cn) go
    ("Zoë 田中 🙂", "Zoë 田中 🙂"),  # letters, inner spaces and emoji stay
    ("a" * 25, "a" * 20),  # cut to 20 characters
    ("     " + "a" * 20, "a" * 20),  # leading spaces are stripped before the cut, so they do not use up the 20
    ("a" * 19 + " " + "bbbb", "a" * 19),  # the cut leaves a trailing space, which is stripped again
    ("A\u2028B\u2029C", "ABC"),  # line and paragraph separators (Zl, Zp) go: they split lines in a log or text file
    ("A\u00a0B\u3000C", "A\u00a0B\u3000C"),  # other spaces (Zs) stay
    ("\u202e\x00 ", ""),  # nothing left
    ("", ""),
])
def test_clean_name_strips_and_truncates(raw, expected):
    assert bridge.clean_name(raw) == expected


def test_clean_name_does_not_scan_a_huge_name():
    started = time.perf_counter()
    assert bridge.clean_name("a" * 10_000_000) == "a" * 20
    assert time.perf_counter() - started < 0.25


@pytest.mark.parametrize("not_text", [None, 5, b"Krish", ["Krish"]])
def test_clean_name_rejects_non_text(not_text):
    with pytest.raises(ValueError):
        bridge.clean_name(not_text)
