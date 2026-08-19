"""Paired statistical analysis for the archived Project XIA nested-CV results.

This script does not retrain models. It provides an auditable analysis of the
five matched outer-fold results in v1.0.0. The small number of outer folds is
handled with an exact paired sign-flip test rather than an asymptotic t-test.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "project_xia_nested_cv_results.csv"
OUTPUT = ROOT / "results_v2"
METRICS = [
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "mcc",
    "fingerprinting_recall",
    "fingerprinting_f1",
    "training_seconds",
    "milliseconds_per_record",
]
REFERENCE = "CS_SHAP"


def exact_two_sided_sign_flip_pvalue(differences: np.ndarray) -> float:
    """Exact paired randomization p-value for the mean difference."""
    differences = np.asarray(differences, dtype=float)
    observed = abs(differences.mean())
    permuted = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        permuted.append(abs(np.mean(differences * np.asarray(signs))))
    permuted = np.asarray(permuted)
    return float(np.mean(permuted >= observed - 1e-15))


def paired_standardized_effect(differences: np.ndarray) -> float:
    """Cohen's dz; NaN when paired differences have zero variance."""
    sd = np.std(differences, ddof=1)
    return float(np.mean(differences) / sd) if sd > 0 else np.nan


def holm_adjust(pvalues: pd.Series) -> pd.Series:
    """Holm family-wise error correction."""
    order = np.argsort(pvalues.to_numpy())
    raw = pvalues.to_numpy()[order]
    adjusted_sorted = np.maximum.accumulate(
        np.minimum(1.0, raw * (len(raw) - np.arange(len(raw))))
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return pd.Series(adjusted, index=pvalues.index)


def main() -> None:
    data = pd.read_csv(INPUT)
    required = {"outer_fold", "method", *METRICS}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    methods = sorted(set(data["method"]) - {REFERENCE})
    records: list[dict[str, float | int | str]] = []

    for metric in METRICS:
        pivot = data.pivot(index="outer_fold", columns="method", values=metric)
        if pivot.isna().any().any():
            raise ValueError(f"Unmatched folds detected for {metric}")

        for comparator in methods:
            # Positive values favour CS-SHAP for predictive metrics. For time
            # and latency, positive values indicate that CS-SHAP is slower.
            differences = (pivot[REFERENCE] - pivot[comparator]).to_numpy()
            records.append(
                {
                    "metric": metric,
                    "reference": REFERENCE,
                    "comparator": comparator,
                    "n_outer_folds": len(differences),
                    "reference_mean": pivot[REFERENCE].mean(),
                    "comparator_mean": pivot[comparator].mean(),
                    "mean_paired_difference": differences.mean(),
                    "median_paired_difference": np.median(differences),
                    "paired_difference_sd": differences.std(ddof=1),
                    "cohen_dz": paired_standardized_effect(differences),
                    "exact_sign_flip_p": exact_two_sided_sign_flip_pvalue(
                        differences
                    ),
                    "reference_wins": int(np.sum(differences > 0)),
                    "ties": int(np.sum(differences == 0)),
                    "reference_losses": int(np.sum(differences < 0)),
                }
            )

    results = pd.DataFrame(records)
    results["holm_p_within_metric"] = results.groupby("metric")[
        "exact_sign_flip_p"
    ].transform(holm_adjust)

    OUTPUT.mkdir(exist_ok=True)
    results.to_csv(OUTPUT / "project_xia_v2_paired_statistics.csv", index=False)

    summary = data.groupby("method")[METRICS].agg(["mean", "std"])
    summary.to_csv(OUTPUT / "project_xia_v2_nested_summary.csv")

    print(results.to_string(index=False))
    print("\nInterpretation safeguards:")
    print("- Five outer folds provide low statistical resolution.")
    print("- Fold results are matched; comparisons must remain paired.")
    print("- Non-significance is not evidence of equivalence.")
    print("- Version 2 should use repeated nested CV and an external dataset.")


if __name__ == "__main__":
    main()
