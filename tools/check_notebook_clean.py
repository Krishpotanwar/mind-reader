"""Check that a built notebook is safe to share (D24): a cell tagged `live` has no outputs, and no output holds a camera frame
(an image/jpeg output, or a data:image/jpeg URI hidden in html or printed text).

Usage (from the repo root):
    python tools/check_notebook_clean.py notebook/MindReader.ipynb
Prints each problem and exits 1 if there are any.
"""
import json
import sys


def problems(notebook: dict) -> list[str]:
    """One line per problem found in the notebook (an .ipynb already loaded as a dict)."""
    found = []
    for number, cell in enumerate(notebook["cells"], start=1):
        outputs = cell.get("outputs", [])  # markdown cells have none
        if "live" in cell.get("metadata", {}).get("tags", []) and outputs:
            found.append(f"cell {number} is tagged live but has outputs")
        if "image/jpeg" in json.dumps(outputs):  # an output of that type, or a data:image/jpeg URI inside html or printed text
            found.append(f"cell {number} has image/jpeg in its output")
    return found


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python tools/check_notebook_clean.py notebook.ipynb")
    with open(sys.argv[1], encoding="utf-8") as f:
        issues = problems(json.load(f))
    for issue in issues:
        print(issue)
    sys.exit(1 if issues else 0)
