"""Build the uploadable v2 Edge experiment notebook from the archived v1 notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "project_xia_cs_shap_reproducibility.ipynb"
EXTENSION = ROOT / "scripts" / "project_xia_v2_edge_extension.py"
OUTPUT = ROOT / "notebooks" / "project_xia_v2_edge_experiments.ipynb"


def lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
# Stop after nested validation. Cells 75 onward perform the historical locked
# test evaluation and are intentionally excluded from the version-2 notebook.
notebook["cells"] = notebook["cells"][:75]
notebook["cells"].extend(
    [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": lines(
                "# Project XIA version 2: comparators, ablations and classifier transfer\n\n"
                "This extension uses the same deterministic outer folds and the "
                "fold-specific feature selections generated inside each outer "
                "training partition. It adds non-SHAP baselines, component "
                "ablations, classifier transfer and efficiency measurements. "
                "It does **not** access the historical locked test.\n"
            ),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines(EXTENSION.read_text(encoding="utf-8")),
        },
    ]
)
notebook.setdefault("metadata", {}).setdefault("project_xia", {})[
    "revision"
] = "v2-edge-comparators-ablations-classifier-transfer"
OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(OUTPUT)
