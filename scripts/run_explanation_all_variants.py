"""Run notebooks/06_explanation.ipynb for every variant in sequence.

Edits the VARIANT line in the notebook's config cell, executes via
`jupyter nbconvert --execute`, then restores the original VARIANT. Per-variant
artifacts (dt_model.pkl, shap/lime/cross-method JSONs) land in
`data/artifacts/{VARIANT}/` exactly as they would from manual execution.

Run from repo root:
    .venv/bin/python scripts/run_explanation_all_variants.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO     = Path(__file__).resolve().parent.parent
NB_PATH  = REPO / "notebooks" / "06_explanation.ipynb"
VARIANTS = ["L5B15", "CLUG", "BookingDotCom", "Cartrawler"]

# Substitute the value inside `VARIANT = "..."` while preserving everything else
# on the line — including the trailing newline that lives in the source list entry.
VARIANT_SUB = re.compile(r'(VARIANT\s*=\s*)"[^"]+"')


def set_variant_in_notebook(path: Path, variant: str) -> str:
    """Patch the VARIANT = "..." line in the notebook's config cell.
    Returns the original variant string so we can restore it after.
    """
    nb = json.loads(path.read_text())
    original = None
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = cell["source"]
        if isinstance(src, list):
            for i, line in enumerate(src):
                if VARIANT_SUB.search(line):
                    original = re.search(r'"([^"]+)"', line).group(1)
                    src[i] = VARIANT_SUB.sub(rf'\1"{variant}"', line)
                    break
            if original is not None:
                break
    if original is None:
        raise RuntimeError("Could not locate `VARIANT = \"...\"` in the notebook config cell.")
    path.write_text(json.dumps(nb, indent=1))
    return original


def execute_notebook(path: Path, variant: str) -> None:
    out = Path("/tmp") / f"06_explanation_{variant}.ipynb"
    print(f"[{variant}] executing → {out}")
    t0 = time.time()
    subprocess.run(
        [".venv/bin/jupyter", "nbconvert", "--to", "notebook",
         "--execute", str(path), "--output", str(out)],
        cwd=REPO, check=True,
    )
    print(f"[{variant}] done in {time.time() - t0:.1f}s")


def main() -> None:
    original = None
    try:
        for v in VARIANTS:
            previous = set_variant_in_notebook(NB_PATH, v)
            if original is None:
                original = previous  # capture before any patching
            execute_notebook(NB_PATH, v)
    finally:
        # Restore the original VARIANT so the notebook is unchanged on disk.
        if original is not None:
            set_variant_in_notebook(NB_PATH, original)
            print(f"\nRestored notebook VARIANT to {original!r}.")


if __name__ == "__main__":
    sys.exit(main())
