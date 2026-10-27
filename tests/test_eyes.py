import hashlib
import os
import zipfile

import cv2
import numpy as np
import pytest

from mindreader import eyes

MODEL = "data/raw/hand_landmarker.task"  # real model, fetched once, gitignored
TEST_ZIP = "data/raw/rps-test-set.zip"
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
