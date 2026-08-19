"""Build the standalone Kaggle notebook for TON_IoT external validation."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "project_xia_v2_toniot_external.py"
OUTPUT = ROOT / "notebooks" / "project_xia_v2_toniot_external_validation.ipynb"


def lines(text):
    return text.splitlines(keepends=True)


notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": lines(
                "# Project XIA v2 — TON_IoT external validation\n\n"
                "Independent replication of the CS-SHAP selection procedure on "
                "TON_IoT network data. The notebook performs a fresh leakage "
                "audit, exact deduplication, nested selection, ablation and "
                "matched conventional baselines. It does not transfer selected "
                "feature identities from Edge-IIoTset.\n"
            ),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines(SOURCE.read_text(encoding="utf-8")),
        },
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.7"},
        "project_xia": {"revision": "v2-toniot-external-validation"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(OUTPUT)
