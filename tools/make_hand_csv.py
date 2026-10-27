"""Build data/hand_landmarks.csv from the two TensorFlow rock-paper-scissors zips.

Usage (from the repo root):
    python tools/make_hand_csv.py data/raw/rps.zip data/raw/rps-test-set.zip data/hand_landmarks.csv

Images are decoded straight from the zips and never written to disk. The CSV has one row per image where a hand
is found. Prints the detection rate per split and class (images without a hand still count in the denominator).
"""
import sys
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `python tools/make_hand_csv.py` can import mindreader
from mindreader.eyes import HandReader, ensure_model

LABELS = ("rock", "paper", "scissors")
COLUMNS = ["file", "split", "label", "width", "height"] + [f"{axis}{i}" for i in range(21) for axis in "xyz"]


def scan_zip(zip_path, reader):
    """Yield (file, label, width, height, landmarks or None) for every PNG in the zip."""
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(n for n in z.namelist() if n.endswith(".png")):
            label = name.split("/")[-2]
            if label not in LABELS:
                raise ValueError(f"{name}: folder {label!r} is not rock, paper or scissors")
            bgr = cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR)
            points = reader.landmarks(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            yield name.split("/")[-1], label, bgr.shape[1], bgr.shape[0], points


def print_rates(hits, total):
    for split in ("train", "test"):
        split_hits = sum(hits[split, label] for label in LABELS)
        split_total = sum(total[split, label] for label in LABELS)
        print(f"{split}: {split_hits}/{split_total} = {split_hits / split_total:.1%}")
        for label in LABELS:
            h, t = hits[split, label], total[split, label]
            print(f"  {label:9s} {h}/{t} = {h / t:.1%}")


def main(train_zip, test_zip, out_csv):
    rows, hits, total = [], Counter(), Counter()
    with HandReader(ensure_model(str(ROOT / "data/raw/hand_landmarker.task"))) as reader:
        for split, zip_path in (("train", train_zip), ("test", test_zip)):
            for file, label, width, height, points in scan_zip(zip_path, reader):
                total[split, label] += 1
                if points is not None:
                    hits[split, label] += 1
                    rows.append([file, split, label, width, height, *points.ravel()])
    pd.DataFrame(rows, columns=COLUMNS).to_csv(out_csv, index=False, float_format="%.6f")
    print_rates(hits, total)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
