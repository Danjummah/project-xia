"""Project XIA v2: leakage-safe Edge-IIoTset comparator and ablation study.

Run this file after the v1 reproducibility notebook has created
``X_development``, ``y_development``, ``label_encoder`` and
``create_subset_preprocessor``. The script uses the exact five outer splits
from v1 and the fold-specific selections produced only from each outer
training partition.

This extension deliberately does not touch the historical v1 locked test.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier


REQUIRED_GLOBALS = [
    "X_development",
    "y_development",
    "label_encoder",
    "create_subset_preprocessor",
]
missing_globals = [name for name in REQUIRED_GLOBALS if name not in globals()]
if missing_globals:
    raise RuntimeError(
        "Run the v1 notebook through nested validation before this extension. "
        f"Missing objects: {missing_globals}"
    )

WORKING = Path("/kaggle/working")
SELECTION_FILE = WORKING / "project_xia_nested_feature_selections.csv"
if not SELECTION_FILE.exists():
    raise FileNotFoundError(
        "Expected fold-specific selections at " + str(SELECTION_FILE)
    )

# Predeclared experiment settings
OUTER_SPLITS = 5
OUTER_RANDOM_STATE = 42
TOP_K = 19
RANDOM_STATE = 42

# Set to ["XGBoost"] for a short smoke test, then restore all three.
CLASSIFIERS_TO_RUN = ["XGBoost", "Random_Forest", "Linear_Logistic_SGD"]


def make_ordinal_ranker_preprocessor(reference: pd.DataFrame) -> ColumnTransformer:
    """Return one numeric output per original feature for ranking baselines."""
    numeric = [c for c in reference if reference[c].dtype != "object"]
    categorical = [c for c in reference if reference[c].dtype == "object"]
    transformers = []
    if numeric:
        transformers.append(
            ("numeric", SimpleImputer(strategy="median"), numeric)
        )
    if categorical:
        categorical_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "ordinal",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipe, categorical))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)


def transformed_feature_order(reference: pd.DataFrame) -> list[str]:
    numeric = [c for c in reference if reference[c].dtype != "object"]
    categorical = [c for c in reference if reference[c].dtype == "object"]
    return numeric + categorical


def select_mutual_information(
    X_train: pd.DataFrame, y_train: np.ndarray, k: int
) -> tuple[list[str], pd.DataFrame]:
    prep = make_ordinal_ranker_preprocessor(X_train)
    transformed = np.asarray(prep.fit_transform(X_train))
    scores = mutual_info_classif(
        transformed,
        y_train,
        discrete_features=False,
        random_state=RANDOM_STATE,
    )
    ranking = pd.DataFrame(
        {"feature": transformed_feature_order(X_train), "importance": scores}
    ).sort_values(["importance", "feature"], ascending=[False, True])
    return ranking.head(k)["feature"].tolist(), ranking


def select_random_forest(
    X_train: pd.DataFrame, y_train: np.ndarray, k: int
) -> tuple[list[str], pd.DataFrame]:
    prep = make_ordinal_ranker_preprocessor(X_train)
    transformed = np.asarray(prep.fit_transform(X_train))
    ranker = RandomForestClassifier(
        n_estimators=250,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    ranker.fit(transformed, y_train)
    ranking = pd.DataFrame(
        {
            "feature": transformed_feature_order(X_train),
            "importance": ranker.feature_importances_,
        }
    ).sort_values(["importance", "feature"], ascending=[False, True])
    return ranking.head(k)["feature"].tolist(), ranking


def make_classifier(name: str):
    if name == "XGBoost":
        return XGBClassifier(
            objective="multi:softprob",
            num_class=len(label_encoder.classes_),
            n_estimators=250,
            max_depth=6,
            learning_rate=0.10,
            subsample=0.80,
            colsample_bytree=0.80,
            min_child_weight=1,
            reg_alpha=0.0,
            reg_lambda=1.0,
            tree_method="hist",
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if name == "Random_Forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if name == "Linear_Logistic_SGD":
        # Scalable multinomial logistic-loss baseline for the large dataset.
        return SGDClassifier(
            # scikit-learn 1.0.2 uses the legacy name "log". It is the same
            # logistic loss later renamed to "log_loss".
            loss="log",
            penalty="l2",
            alpha=1e-4,
            class_weight="balanced",
            max_iter=2000,
            tol=1e-4,
            early_stopping=False,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown classifier: {name}")


def fold_feature_sets(
    selection_rows: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> tuple[dict[str, list[str]], list[dict[str, object]]]:
    def features_for(method: str) -> list[str]:
        return selection_rows.loc[
            selection_rows["method"].eq(method), "feature"
        ].tolist()

    cs_rows = selection_rows[selection_rows["method"].eq("CS_SHAP")]
    stable_core = cs_rows.loc[cs_rows["stable_balanced"], "feature"].tolist()
    stable_minority = cs_rows.loc[cs_rows["stable_minority"], "feature"].tolist()
    cs_union = features_for("CS_SHAP")

    mi_features, mi_ranking = select_mutual_information(X_train, y_train, TOP_K)
    rf_features, rf_ranking = select_random_forest(X_train, y_train, TOP_K)

    rank_records: list[dict[str, object]] = []
    for method, ranking in [
        ("Mutual_Information_19", mi_ranking),
        ("Random_Forest_Importance_19", rf_ranking),
    ]:
        for rank, row in enumerate(ranking.itertuples(index=False), start=1):
            rank_records.append(
                {
                    "selector": method,
                    "rank": rank,
                    "feature": row.feature,
                    "importance": row.importance,
                }
            )

    feature_sets = {
        "Full_42": list(X_train.columns),
        "Natural_Global_20": features_for("Natural_Global_20"),
        "Class_Balanced_15": features_for("Class_Balanced_15"),
        "Stable_Balanced_Core": stable_core,
        "Stable_Minority_Only": stable_minority,
        "CS_SHAP_Union": cs_union,
        "Mutual_Information_19": mi_features,
        "Random_Forest_Importance_19": rf_features,
    }
    for name, features in feature_sets.items():
        if not features:
            raise ValueError(f"Empty feature set generated for {name}")
    return feature_sets, rank_records


selection_table = pd.read_csv(SELECTION_FILE)
for col in ["stable_balanced", "stable_minority"]:
    selection_table[col] = selection_table[col].astype(str).str.lower().eq("true")

y_encoded = label_encoder.transform(y_development)
outer_cv = StratifiedKFold(
    n_splits=OUTER_SPLITS,
    shuffle=True,
    random_state=OUTER_RANDOM_STATE,
)

result_records: list[dict[str, object]] = []
selection_records: list[dict[str, object]] = []
baseline_ranking_records: list[dict[str, object]] = []

for outer_fold, (train_positions, validation_positions) in enumerate(
    outer_cv.split(X_development, y_encoded), start=1
):
    print(f"\n===== V2 OUTER FOLD {outer_fold}/{OUTER_SPLITS} =====")
    X_outer_train = X_development.iloc[train_positions]
    X_outer_validation = X_development.iloc[validation_positions]
    y_outer_train = y_encoded[train_positions]
    y_outer_validation = y_encoded[validation_positions]

    fold_rows = selection_table[selection_table["outer_fold"].eq(outer_fold)]
    feature_sets, ranking_records = fold_feature_sets(
        fold_rows, X_outer_train, y_outer_train
    )
    for record in ranking_records:
        record["outer_fold"] = outer_fold
        baseline_ranking_records.append(record)

    for selector_name, selected_features in feature_sets.items():
        for position, feature in enumerate(selected_features, start=1):
            selection_records.append(
                {
                    "outer_fold": outer_fold,
                    "selector": selector_name,
                    "position": position,
                    "feature": feature,
                    "feature_count": len(selected_features),
                }
            )

        for classifier_name in CLASSIFIERS_TO_RUN:
            print(
                f"Fold {outer_fold}: {selector_name} ({len(selected_features)}) "
                f"+ {classifier_name}"
            )
            prep = create_subset_preprocessor(selected_features, X_outer_train)
            pipeline_steps = [("preprocessor", prep)]
            if classifier_name == "Linear_Logistic_SGD":
                # Linear SGD is scale-sensitive. with_mean=False preserves the
                # sparse one-hot matrix created by the preprocessing pipeline.
                pipeline_steps.append(("scaler", StandardScaler(with_mean=False)))
            pipeline_steps.append(("model", make_classifier(classifier_name)))
            pipeline = Pipeline(pipeline_steps)

            start = time.perf_counter()
            pipeline.fit(X_outer_train[selected_features], y_outer_train)
            training_seconds = time.perf_counter() - start

            start = time.perf_counter()
            predictions = pipeline.predict(X_outer_validation[selected_features])
            inference_seconds = time.perf_counter() - start

            precision, recall, class_f1, support = precision_recall_fscore_support(
                y_outer_validation,
                predictions,
                labels=np.arange(len(label_encoder.classes_)),
                zero_division=0,
            )
            model_size_bytes = len(pickle.dumps(pipeline, protocol=4))

            result_records.append(
                {
                    "outer_fold": outer_fold,
                    "selector": selector_name,
                    "classifier": classifier_name,
                    "feature_count": len(selected_features),
                    "accuracy": accuracy_score(y_outer_validation, predictions),
                    "macro_f1": f1_score(
                        y_outer_validation, predictions, average="macro"
                    ),
                    "weighted_f1": f1_score(
                        y_outer_validation, predictions, average="weighted"
                    ),
                    "balanced_accuracy": balanced_accuracy_score(
                        y_outer_validation, predictions
                    ),
                    "mcc": matthews_corrcoef(y_outer_validation, predictions),
                    "training_seconds": training_seconds,
                    "inference_seconds": inference_seconds,
                    "milliseconds_per_record": inference_seconds
                    / len(validation_positions)
                    * 1000,
                    "serialized_model_bytes": model_size_bytes,
                }
            )

            for class_number, class_name in enumerate(label_encoder.classes_):
                result_records.append(
                    {
                        "outer_fold": outer_fold,
                        "selector": selector_name,
                        "classifier": classifier_name,
                        "feature_count": len(selected_features),
                        "record_type": "per_class",
                        "class_name": class_name,
                        "class_precision": precision[class_number],
                        "class_recall": recall[class_number],
                        "class_f1": class_f1[class_number],
                        "class_support": int(support[class_number]),
                    }
                )

results = pd.DataFrame(result_records)
results["record_type"] = results["record_type"].fillna("aggregate")
selections = pd.DataFrame(selection_records)
baseline_rankings = pd.DataFrame(baseline_ranking_records)

results.to_csv(WORKING / "project_xia_v2_edge_results.csv", index=False)
selections.to_csv(WORKING / "project_xia_v2_edge_selections.csv", index=False)
baseline_rankings.to_csv(
    WORKING / "project_xia_v2_edge_baseline_rankings.csv", index=False
)

aggregate = results[results["record_type"].eq("aggregate")]
summary = aggregate.groupby(["selector", "classifier"])[
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
].agg(["mean", "std"])
summary.to_csv(WORKING / "project_xia_v2_edge_summary.csv")

display(summary.round(5))
print("\nVersion-2 Edge-IIoTset extension completed.")
print("The historical locked test was not accessed.")
