#!/usr/bin/env bash
# Build notebook/MindReader.ipynb and .html from notebook/MindReader.py (the source of truth): convert, run every cell, check, export.
# PYTHON picks the interpreter (default python3). Never edit the .ipynb or .html by hand; rebuild them with this script.
set -euo pipefail
PYTHON="${PYTHON:-python3}"
export MR_SKIP_LIVE=1  # outside Colab the live camera cells print a placeholder
unset CDPATH  # a set CDPATH can make `cd` print the new directory and change where it goes
cd "$(dirname "$0")/.."

"$PYTHON" -m jupytext --to ipynb notebook/MindReader.py
"$PYTHON" -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=python3 notebook/MindReader.ipynb
"$PYTHON" tools/check_notebook_clean.py notebook/MindReader.ipynb
"$PYTHON" -m jupyter nbconvert --to html notebook/MindReader.ipynb
cp build/metrics.json notebook/metrics_snapshot.json  # Part 0 draws the last build's numbers (Part 2 only writes them later); commit this file
