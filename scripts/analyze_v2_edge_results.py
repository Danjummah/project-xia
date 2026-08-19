"""Validate and statistically analyse Project XIA v2 Edge-IIoTset outputs."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results_v2" / "edge_raw"
OUT = ROOT / "results_v2" / "edge_analysis"
RESULTS = RAW / "project_xia_v2_edge_results.csv"
SCALED_LINEAR_RESULTS = RAW / "project_xia_v2_edge_linear_scaled_results.csv"
REFERENCE = "CS_SHAP_Union"
VALID_CLASSIFIERS = [
    "XGBoost",
    "Random_Forest",
    "Linear_Logistic_SGD_Scaled",
]
METRICS = ["accuracy", "macro_f1", "balanced_accuracy", "mcc"]


def exact_sign_flip(diff: np.ndarray) -> float:
    observed = abs(float(np.mean(diff)))
    permuted = [
        abs(float(np.mean(diff * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(diff))
    ]
    return float(np.mean(np.asarray(permuted) >= observed - 1e-15))


def holm(values: pd.Series) -> pd.Series:
    order = np.argsort(values.to_numpy())
    ordered = values.to_numpy()[order]
    corrected_ordered = np.maximum.accumulate(
        np.minimum(1.0, ordered * (len(ordered) - np.arange(len(ordered))))
    )
    corrected = np.empty_like(corrected_ordered)
    corrected[order] = corrected_ordered
    return pd.Series(corrected, index=values.index)


def main() -> None:
    original = pd.read_csv(RESULTS)
    scaled_linear = pd.read_csv(SCALED_LINEAR_RESULTS)
    # The original unscaled SGD rows are invalid for a scale-sensitive linear
    # model and are replaced, not retained as an additional comparator.
    data = pd.concat(
        [
            original[~original["classifier"].eq("Linear_Logistic_SGD")],
            scaled_linear,
        ],
        ignore_index=True,
    )
    if data.duplicated().any():
        raise ValueError("Duplicate result rows detected")

    aggregate = data[data["record_type"].eq("aggregate")].copy()
    per_class = data[data["record_type"].eq("per_class")].copy()
    coverage = aggregate.groupby(["selector", "classifier"])["outer_fold"].agg(
        ["count", "nunique", "min", "max"]
    )
    if not (
        coverage["count"].eq(5)
        & coverage["nunique"].eq(5)
        & coverage["min"].eq(1)
        & coverage["max"].eq(5)
    ).all():
        raise ValueError("Incomplete selector/classifier fold coverage")

    valid = aggregate[aggregate["classifier"].isin(VALID_CLASSIFIERS)].copy()
    summary = (
        valid.groupby(["classifier", "selector"])[
            [
                "feature_count",
                "accuracy",
                "macro_f1",
                "balanced_accuracy",
                "mcc",
                "training_seconds",
                "milliseconds_per_record",
                "serialized_model_bytes",
            ]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )

    comparisons = []
    for classifier in VALID_CLASSIFIERS:
        classifier_data = valid[valid["classifier"].eq(classifier)]
        selectors = sorted(set(classifier_data["selector"]) - {REFERENCE})
        for metric in METRICS:
            pivot = classifier_data.pivot(
                index="outer_fold", columns="selector", values=metric
            )
            for comparator in selectors:
                diff = (pivot[REFERENCE] - pivot[comparator]).to_numpy()
                sd = diff.std(ddof=1)
                comparisons.append(
                    {
                        "classifier": classifier,
                        "metric": metric,
                        "reference": REFERENCE,
                        "comparator": comparator,
                        "reference_mean": pivot[REFERENCE].mean(),
                        "comparator_mean": pivot[comparator].mean(),
                        "mean_paired_difference": diff.mean(),
                        "cohen_dz": diff.mean() / sd if sd > 0 else np.nan,
                        "exact_sign_flip_p": exact_sign_flip(diff),
                        "reference_wins": int(np.sum(diff > 0)),
                        "ties": int(np.sum(diff == 0)),
                        "reference_losses": int(np.sum(diff < 0)),
                    }
                )
    comparisons = pd.DataFrame(comparisons)
    comparisons["holm_p_within_classifier_metric"] = comparisons.groupby(
        ["classifier", "metric"]
    )["exact_sign_flip_p"].transform(holm)

    class_summary = (
        per_class[per_class["classifier"].isin(VALID_CLASSIFIERS)]
        .groupby(["classifier", "selector", "class_name"])[
            ["class_precision", "class_recall", "class_f1", "class_support"]
        ]
        .agg(
            class_precision_mean=("class_precision", "mean"),
            class_recall_mean=("class_recall", "mean"),
            class_f1_mean=("class_f1", "mean"),
            total_support=("class_support", "sum"),
        )
        .reset_index()
    )

    OUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT / "project_xia_v2_edge_consolidated_results.csv", index=False)
    summary.to_csv(OUT / "project_xia_v2_edge_valid_summary.csv", index=False)
    comparisons.to_csv(
        OUT / "project_xia_v2_edge_paired_comparisons.csv", index=False
    )
    class_summary.to_csv(
        OUT / "project_xia_v2_edge_per_class_summary.csv", index=False
    )
    coverage.reset_index().to_csv(
        OUT / "project_xia_v2_edge_coverage_check.csv", index=False
    )

    print("Validation passed")
    print(f"Aggregate rows: {len(aggregate)}; per-class rows: {len(per_class)}")
    print("Unscaled Linear_Logistic_SGD removed; scaled rerun included")
    print("\nXGBoost macro-F1 means:")
    print(
        valid[valid["classifier"].eq("XGBoost")]
        .groupby("selector")["macro_f1"]
        .mean()
        .sort_values(ascending=False)
        .to_string()
    )


if __name__ == "__main__":
    main()
