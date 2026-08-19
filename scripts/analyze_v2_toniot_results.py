"""Validate and analyse Project XIA v2 TON_IoT external-validation outputs."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results_v2" / "toniot_raw"
OUT = ROOT / "results_v2" / "toniot_analysis"
REFERENCE = "CS_SHAP_Union"
METRICS = ["accuracy", "macro_f1", "balanced_accuracy", "mcc"]


def exact_sign_flip(diff):
    diff = np.asarray(diff, dtype=float)
    observed = abs(diff.mean())
    values = [
        abs(np.mean(diff * np.asarray(signs)))
        for signs in itertools.product((-1.0, 1.0), repeat=len(diff))
    ]
    return float(np.mean(np.asarray(values) >= observed - 1e-15))


def main():
    results = pd.read_csv(RAW / "project_xia_v2_toniot_results.csv")
    selections = pd.read_csv(RAW / "project_xia_v2_toniot_selections.csv")
    raw_audit = json.loads(
        (RAW / "project_xia_v2_toniot_audit.json").read_text()
    )
    if results.duplicated().any() or selections.duplicated().any():
        raise ValueError("Duplicate output rows detected")

    aggregate = results[results["record_type"].eq("aggregate")].copy()
    per_class = results[results["record_type"].eq("per_class")].copy()
    coverage = aggregate.groupby("selector")["outer_fold"].agg(
        ["count", "nunique", "min", "max"]
    )
    if len(aggregate) != 35 or not (
        coverage["count"].eq(5)
        & coverage["nunique"].eq(5)
        & coverage["min"].eq(1)
        & coverage["max"].eq(5)
    ).all():
        raise ValueError("Incomplete outer-fold coverage")
    if len(per_class) != 350:
        raise ValueError("Unexpected per-class row count")

    fold_support = (
        per_class[per_class["selector"].eq("Full_Controlled")]
        .groupby("outer_fold")["class_support"]
        .sum()
    )
    if fold_support.nunique() != 1:
        raise ValueError("Outer validation support differs across folds")
    controlled_rows = int(fold_support.iloc[0] * 5)
    expected_rows = (
        int(raw_audit["raw_rows"])
        - int(raw_audit["duplicates_removed"])
        - int(raw_audit["post_exclusion_duplicates_removed"])
    )
    if controlled_rows != expected_rows:
        raise ValueError("Controlled-row reconciliation failed")

    corrected_audit = dict(raw_audit)
    corrected_audit["rows_after_raw_deduplication"] = (
        int(raw_audit["raw_rows"]) - int(raw_audit["duplicates_removed"])
    )
    corrected_audit["controlled_rows"] = controlled_rows
    corrected_audit["audit_correction"] = (
        "controlled_rows reconciled after post-exclusion deduplication"
    )

    summary = aggregate.groupby("selector")[
        [
            "feature_count",
            "accuracy",
            "macro_f1",
            "balanced_accuracy",
            "mcc",
            "training_seconds",
            "milliseconds_per_record",
        ]
    ].agg(["mean", "std"])

    pivot_records = []
    for metric in METRICS:
        pivot = aggregate.pivot(
            index="outer_fold", columns="selector", values=metric
        )
        for comparator in sorted(set(pivot.columns) - {REFERENCE}):
            diff = (pivot[REFERENCE] - pivot[comparator]).to_numpy()
            sd = diff.std(ddof=1)
            pivot_records.append(
                {
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
    comparisons = pd.DataFrame(pivot_records)

    class_summary = per_class.groupby(["selector", "class_name"]).agg(
        precision_mean=("class_precision", "mean"),
        recall_mean=("class_recall", "mean"),
        f1_mean=("class_f1", "mean"),
        total_support=("class_support", "sum"),
    ).reset_index()

    stability_records = []
    for selector, group in selections.groupby("selector"):
        fold_sets = [
            set(group[group["outer_fold"].eq(fold)]["feature"])
            for fold in sorted(group["outer_fold"].unique())
        ]
        jaccards = [
            len(first & second) / len(first | second)
            for first, second in itertools.combinations(fold_sets, 2)
        ]
        frequencies = group.groupby("feature")["outer_fold"].nunique() / 5
        stability_records.append(
            {
                "selector": selector,
                "mean_feature_count": group.groupby("outer_fold")[
                    "feature_count"
                ].first().mean(),
                "mean_pairwise_jaccard": np.mean(jaccards),
                "union_feature_count": len(frequencies),
                "features_in_all_folds": int(frequencies.eq(1).sum()),
            }
        )
    stability = pd.DataFrame(stability_records)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "project_xia_v2_toniot_corrected_audit.json").write_text(
        json.dumps(corrected_audit, indent=2)
    )
    summary.to_csv(OUT / "project_xia_v2_toniot_valid_summary.csv")
    comparisons.to_csv(
        OUT / "project_xia_v2_toniot_paired_comparisons.csv", index=False
    )
    class_summary.to_csv(
        OUT / "project_xia_v2_toniot_per_class_summary.csv", index=False
    )
    stability.to_csv(
        OUT / "project_xia_v2_toniot_selection_stability.csv", index=False
    )
    coverage.reset_index().to_csv(
        OUT / "project_xia_v2_toniot_coverage_check.csv", index=False
    )

    print("TON_IoT validation passed")
    print("Controlled rows:", controlled_rows)
    print("Aggregate rows:", len(aggregate), "Per-class rows:", len(per_class))
    print(summary[["macro_f1", "training_seconds"]].to_string())


if __name__ == "__main__":
    main()
