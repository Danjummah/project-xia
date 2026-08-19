"""Project XIA v2 external validation on TON_IoT network data.

Standalone Kaggle script. It performs leakage control, exact deduplication,
five-fold nested CS-SHAP selection, conventional feature-selection baselines,
and outer-fold evaluation without using Edge-IIoTset feature identities.
"""

from __future__ import annotations

import glob
import inspect
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from xgboost import XGBClassifier


WORKING = Path("/kaggle/working")
RANDOM_STATE = 42
OUTER_FOLDS = 5
INNER_FOLDS = 5
CANDIDATE_K = 15
RECURRENCE_THRESHOLD = 0.80
MAX_EXPLANATION_ROWS_PER_CLASS = 120

matches = glob.glob(
    "/kaggle/input/**/train_test_network.csv",
    recursive=True,
)
if not matches:
    raise FileNotFoundError("train_test_network.csv was not found under /kaggle/input")
DATA_PATH = matches[0]

print("TON_IoT path:", DATA_PATH)
raw = pd.read_csv(DATA_PATH, low_memory=False)
print("Raw shape:", raw.shape)

required = {"label", "type"}
missing_required = required - set(raw.columns)
if missing_required:
    raise ValueError("Missing target columns: " + str(sorted(missing_required)))

# These fields directly identify endpoints or carry high-cardinality content.
# They are excluded before any split or model fitting under a predeclared rule.
EXCLUDED_SHORTCUT_FIELDS = [
    "src_ip",
    "dst_ip",
    "dns_query",
    "http_uri",
    "http_user_agent",
    "ssl_subject",
    "ssl_issuer",
    "weird_addl",
]

before_duplicates = len(raw)
data = raw.drop_duplicates().reset_index(drop=True)
duplicates_removed = before_duplicates - len(data)
print("Exact duplicate rows removed:", duplicates_removed)

duplicate_by_class = (
    raw.assign(_duplicate=raw.duplicated(keep="first"))
    .groupby("type")["_duplicate"]
    .agg(["sum", "count"])
    .rename(columns={"sum": "duplicate_copies", "count": "raw_rows"})
)
duplicate_by_class["duplicate_fraction"] = (
    duplicate_by_class["duplicate_copies"] / duplicate_by_class["raw_rows"]
)
duplicate_by_class.to_csv(
    WORKING / "project_xia_v2_toniot_duplicate_audit.csv"
)

label_consistency = pd.crosstab(data["type"], data["label"])
label_consistency.to_csv(
    WORKING / "project_xia_v2_toniot_label_consistency.csv"
)
if (label_consistency.gt(0).sum(axis=1) > 1).any():
    raise ValueError("At least one attack type maps to multiple binary labels")

target = data["type"].astype(str)
predictors = data.drop(
    columns=["label", "type"] + EXCLUDED_SHORTCUT_FIELDS,
    errors="ignore",
).copy()

# TON_IoT uses '-' as a categorical missing-value marker.
for column in predictors.select_dtypes(include="object").columns:
    predictors[column] = (
        predictors[column]
        .replace("-", np.nan)
        .fillna("__MISSING__")
    )

# Excluding identifiers/content can expose additional replicated controlled
# observations. Remove those copies with the target included in the key.
controlled = predictors.copy()
controlled["__target__"] = target.to_numpy()
before_controlled_duplicates = len(controlled)
controlled = controlled.drop_duplicates().reset_index(drop=True)
controlled_duplicates_removed = before_controlled_duplicates - len(controlled)
target = controlled.pop("__target__").astype(str)
predictors = controlled

constant_features = [
    column for column in predictors.columns
    if predictors[column].nunique(dropna=False) <= 1
]
predictors = predictors.drop(columns=constant_features)

print("Post-exclusion duplicate copies removed:", controlled_duplicates_removed)
print("Controlled records:", len(predictors))
print("Controlled predictors:", predictors.shape[1])
print("Constant predictors removed:", constant_features)
print("Class distribution after deduplication:")
print(target.value_counts().to_string())

audit = {
    "source_path": DATA_PATH,
    "raw_rows": int(before_duplicates),
    "rows_after_raw_deduplication": int(len(data)),
    "controlled_rows": int(len(predictors)),
    "duplicates_removed": int(duplicates_removed),
    "post_exclusion_duplicates_removed": int(controlled_duplicates_removed),
    "raw_predictor_count": int(raw.shape[1] - 2),
    "controlled_predictor_count": int(predictors.shape[1]),
    "excluded_shortcut_fields": EXCLUDED_SHORTCUT_FIELDS,
    "constant_features_removed": constant_features,
    "outer_folds": OUTER_FOLDS,
    "inner_folds": INNER_FOLDS,
    "candidate_k": CANDIDATE_K,
    "recurrence_threshold": RECURRENCE_THRESHOLD,
}
(WORKING / "project_xia_v2_toniot_audit.json").write_text(
    json.dumps(audit, indent=2)
)

encoder = LabelEncoder()
y = encoder.fit_transform(target)
class_names = list(encoder.classes_)


def make_prediction_preprocessor(features, reference):
    numeric = [f for f in features if reference[f].dtype != "object"]
    categorical = [f for f in features if reference[f].dtype == "object"]
    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric,
            )
        )
    if categorical:
        encoder_arguments = {"handle_unknown": "ignore"}
        if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
            encoder_arguments["sparse_output"] = True
        else:
            encoder_arguments["sparse"] = True
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(**encoder_arguments),
                        ),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers, remainder="drop")


def make_ranker_preprocessor(reference):
    numeric = [c for c in reference if reference[c].dtype != "object"]
    categorical = [c for c in reference if reference[c].dtype == "object"]
    transformers = []
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric))
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
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
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)


def ranker_feature_order(reference):
    numeric = [c for c in reference if reference[c].dtype != "object"]
    categorical = [c for c in reference if reference[c].dtype == "object"]
    return numeric + categorical


def make_xgb(seed):
    return XGBClassifier(
        objective="multi:softprob",
        num_class=len(class_names),
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
        random_state=seed,
        n_jobs=-1,
    )


def expanded_to_original_map(fitted_preprocessor, features, reference):
    numeric = [f for f in features if reference[f].dtype != "object"]
    categorical = [f for f in features if reference[f].dtype == "object"]
    mapping = list(numeric)
    if categorical:
        fitted_encoder = (
            fitted_preprocessor.named_transformers_["categorical"]
            .named_steps["encoder"]
        )
        for feature, categories in zip(categorical, fitted_encoder.categories_):
            mapping.extend([feature] * len(categories))
    return mapping


def aggregate_to_original(scores, mapping):
    if len(scores) != len(mapping):
        raise ValueError(
            "Transformed score length does not match original-feature mapping"
        )
    return pd.Series(scores).groupby(pd.Series(mapping)).sum()


def class_balanced_explanation_sample(X_valid, y_valid, seed):
    selected_positions = []
    rng = np.random.RandomState(seed)
    for class_number in range(len(class_names)):
        positions = np.flatnonzero(y_valid == class_number)
        take = min(MAX_EXPLANATION_ROWS_PER_CLASS, len(positions))
        if take:
            selected_positions.extend(
                rng.choice(positions, size=take, replace=False).tolist()
            )
    return np.asarray(selected_positions, dtype=int)


def stable_set(fold_rankings, method, candidate_k, threshold):
    subset = fold_rankings[fold_rankings["method"].eq(method)].copy()
    subset["selected"] = subset["rank"].le(candidate_k)
    evidence = subset.groupby("feature").agg(
        frequency=("selected", "mean"),
        mean_rank=("rank", "mean"),
        mean_importance=("importance", "mean"),
    )
    selected = evidence[evidence["frequency"].ge(threshold)].index.tolist()
    if not selected:
        selected = evidence.nsmallest(candidate_k, "mean_rank").index.tolist()
    return selected, evidence


outer_cv = StratifiedKFold(
    n_splits=OUTER_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

outer_results = []
outer_selections = []
inner_ranking_records = []
baseline_ranking_records = []

for outer_fold, (outer_train_pos, outer_valid_pos) in enumerate(
    outer_cv.split(predictors, y), start=1
):
    print("\n===== TON_IOT OUTER FOLD %d/%d =====" % (outer_fold, OUTER_FOLDS))
    X_outer_train = predictors.iloc[outer_train_pos]
    X_outer_valid = predictors.iloc[outer_valid_pos]
    y_outer_train = y[outer_train_pos]
    y_outer_valid = y[outer_valid_pos]

    outer_counts = pd.Series(y_outer_train).value_counts()
    median_count = float(outer_counts.median())
    minority_numbers = sorted(
        outer_counts[outer_counts < 0.10 * median_count].index.tolist()
    )
    if not minority_numbers:
        minority_numbers = [int(outer_counts.idxmin())]
    minority_names = [class_names[i] for i in minority_numbers]
    print("Training-defined minority classes:", minority_names)

    inner_cv = StratifiedKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE + outer_fold,
    )
    fold_records = []

    for inner_fold, (inner_train_pos, inner_valid_pos) in enumerate(
        inner_cv.split(X_outer_train, y_outer_train), start=1
    ):
        print("  ranking inner fold %d/%d" % (inner_fold, INNER_FOLDS))
        X_inner_train = X_outer_train.iloc[inner_train_pos]
        X_inner_valid = X_outer_train.iloc[inner_valid_pos]
        y_inner_train = y_outer_train[inner_train_pos]
        y_inner_valid = y_outer_train[inner_valid_pos]

        all_features = list(X_outer_train.columns)
        prep = make_prediction_preprocessor(all_features, X_inner_train)
        model = make_xgb(RANDOM_STATE + outer_fold * 10 + inner_fold)
        pipeline = Pipeline([("preprocessor", prep), ("model", model)])
        pipeline.fit(X_inner_train, y_inner_train)

        balanced_positions = class_balanced_explanation_sample(
            X_inner_valid, y_inner_valid, RANDOM_STATE + outer_fold * 100 + inner_fold
        )
        explanation_X = X_inner_valid.iloc[balanced_positions]
        explanation_y = y_inner_valid[balanced_positions]
        fitted_prep = pipeline.named_steps["preprocessor"]
        transformed = fitted_prep.transform(explanation_X)
        contributions = pipeline.named_steps["model"].get_booster().predict(
            xgb.DMatrix(transformed), pred_contribs=True
        )[:, :, :-1]
        mapping = expanded_to_original_map(
            fitted_prep, all_features, X_inner_train
        )

        natural_rng = np.random.RandomState(
            RANDOM_STATE + outer_fold * 1000 + inner_fold
        )
        natural_size = min(1200, len(X_inner_valid))
        natural_positions = natural_rng.choice(
            len(X_inner_valid), size=natural_size, replace=False
        )
        natural_transformed = fitted_prep.transform(
            X_inner_valid.iloc[natural_positions]
        )
        natural_contributions = pipeline.named_steps["model"].get_booster().predict(
            xgb.DMatrix(natural_transformed), pred_contribs=True
        )[:, :, :-1]
        natural_scores = np.abs(natural_contributions).mean(axis=(0, 1))
        class_rows = []
        for class_number in range(len(class_names)):
            mask = explanation_y == class_number
            if not np.any(mask):
                raise ValueError("Explanation sample omitted a class")
            class_rows.append(
                np.abs(contributions[mask, class_number, :]).mean(axis=0)
            )
        class_matrix = np.vstack(class_rows)
        balanced_scores = class_matrix.mean(axis=0)
        minority_scores = class_matrix[minority_numbers].mean(axis=0)

        score_groups = {
            "Natural_Global": natural_scores,
            "Class_Balanced": balanced_scores,
            "Minority_Diagnostic": minority_scores,
        }
        for method, transformed_scores in score_groups.items():
            original_scores = aggregate_to_original(transformed_scores, mapping)
            ranks = original_scores.rank(ascending=False, method="min")
            for feature in original_scores.index:
                record = {
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "method": method,
                    "feature": feature,
                    "importance": float(original_scores[feature]),
                    "rank": int(ranks[feature]),
                    "minority_classes": "|".join(minority_names),
                }
                fold_records.append(record)
                inner_ranking_records.append(record)

    fold_rankings = pd.DataFrame(fold_records)
    natural_features, natural_evidence = stable_set(
        fold_rankings, "Natural_Global", CANDIDATE_K, RECURRENCE_THRESHOLD
    )
    balanced_core, balanced_evidence = stable_set(
        fold_rankings, "Class_Balanced", CANDIDATE_K, RECURRENCE_THRESHOLD
    )
    minority_features, minority_evidence = stable_set(
        fold_rankings, "Minority_Diagnostic", CANDIDATE_K, RECURRENCE_THRESHOLD
    )
    cs_features = sorted(set(balanced_core) | set(minority_features))

    rank_prep = make_ranker_preprocessor(X_outer_train)
    rank_matrix = np.asarray(rank_prep.fit_transform(X_outer_train))
    rank_order = ranker_feature_order(X_outer_train)
    target_k = len(cs_features)

    mi_scores = mutual_info_classif(
        rank_matrix,
        y_outer_train,
        discrete_features=False,
        random_state=RANDOM_STATE,
    )
    mi_table = pd.DataFrame(
        {"feature": rank_order, "importance": mi_scores}
    ).sort_values(["importance", "feature"], ascending=[False, True])
    mi_features = mi_table.head(target_k)["feature"].tolist()

    rf_ranker = RandomForestClassifier(
        n_estimators=250,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_ranker.fit(rank_matrix, y_outer_train)
    rf_table = pd.DataFrame(
        {"feature": rank_order, "importance": rf_ranker.feature_importances_}
    ).sort_values(["importance", "feature"], ascending=[False, True])
    rf_features = rf_table.head(target_k)["feature"].tolist()

    for selector, table in [
        ("Mutual_Information_Matched", mi_table),
        ("Random_Forest_Importance_Matched", rf_table),
    ]:
        for rank, row in enumerate(table.itertuples(index=False), start=1):
            baseline_ranking_records.append(
                {
                    "outer_fold": outer_fold,
                    "selector": selector,
                    "rank": rank,
                    "feature": row.feature,
                    "importance": row.importance,
                }
            )

    feature_sets = {
        "Full_Controlled": list(X_outer_train.columns),
        "Natural_Global_Stable": natural_features,
        "Stable_Balanced_Core": balanced_core,
        "Stable_Minority_Only": minority_features,
        "CS_SHAP_Union": cs_features,
        "Mutual_Information_Matched": mi_features,
        "Random_Forest_Importance_Matched": rf_features,
    }

    for selector, features in feature_sets.items():
        if not features:
            raise ValueError("Empty feature set for " + selector)
        for position, feature in enumerate(features, start=1):
            outer_selections.append(
                {
                    "outer_fold": outer_fold,
                    "selector": selector,
                    "position": position,
                    "feature": feature,
                    "feature_count": len(features),
                    "minority_classes": "|".join(minority_names),
                }
            )

        prediction_prep = make_prediction_preprocessor(features, X_outer_train)
        prediction_model = make_xgb(RANDOM_STATE + outer_fold)
        prediction_pipeline = Pipeline(
            [("preprocessor", prediction_prep), ("model", prediction_model)]
        )
        start = time.perf_counter()
        prediction_pipeline.fit(X_outer_train[features], y_outer_train)
        training_seconds = time.perf_counter() - start
        start = time.perf_counter()
        predictions = prediction_pipeline.predict(X_outer_valid[features])
        inference_seconds = time.perf_counter() - start

        precision, recall, class_f1, support = precision_recall_fscore_support(
            y_outer_valid,
            predictions,
            labels=np.arange(len(class_names)),
            zero_division=0,
        )
        aggregate = {
            "outer_fold": outer_fold,
            "selector": selector,
            "feature_count": len(features),
            "record_type": "aggregate",
            "accuracy": accuracy_score(y_outer_valid, predictions),
            "macro_f1": f1_score(y_outer_valid, predictions, average="macro"),
            "weighted_f1": f1_score(y_outer_valid, predictions, average="weighted"),
            "balanced_accuracy": balanced_accuracy_score(y_outer_valid, predictions),
            "mcc": matthews_corrcoef(y_outer_valid, predictions),
            "training_seconds": training_seconds,
            "inference_seconds": inference_seconds,
            "milliseconds_per_record": inference_seconds / len(outer_valid_pos) * 1000,
            "minority_classes": "|".join(minority_names),
        }
        outer_results.append(aggregate)
        for class_number, class_name in enumerate(class_names):
            outer_results.append(
                {
                    "outer_fold": outer_fold,
                    "selector": selector,
                    "feature_count": len(features),
                    "record_type": "per_class",
                    "class_name": class_name,
                    "class_precision": precision[class_number],
                    "class_recall": recall[class_number],
                    "class_f1": class_f1[class_number],
                    "class_support": int(support[class_number]),
                    "minority_classes": "|".join(minority_names),
                }
            )

results = pd.DataFrame(outer_results)
selections = pd.DataFrame(outer_selections)
inner_rankings = pd.DataFrame(inner_ranking_records)
baseline_rankings = pd.DataFrame(baseline_ranking_records)

results.to_csv(WORKING / "project_xia_v2_toniot_results.csv", index=False)
selections.to_csv(WORKING / "project_xia_v2_toniot_selections.csv", index=False)
inner_rankings.to_csv(
    WORKING / "project_xia_v2_toniot_inner_rankings.csv", index=False
)
baseline_rankings.to_csv(
    WORKING / "project_xia_v2_toniot_baseline_rankings.csv", index=False
)

aggregate_results = results[results["record_type"].eq("aggregate")]
coverage = aggregate_results.groupby("selector")["outer_fold"].nunique()
if len(aggregate_results) != OUTER_FOLDS * 7 or not coverage.eq(OUTER_FOLDS).all():
    raise AssertionError("Incomplete external-validation aggregate results")
if aggregate_results[["accuracy", "macro_f1", "balanced_accuracy", "mcc"]].isna().any().any():
    raise AssertionError("Missing external-validation aggregate metrics")

summary = aggregate_results.groupby("selector")[
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
summary.to_csv(WORKING / "project_xia_v2_toniot_summary.csv")

display(summary.round(5))
print("\nTON_IoT external validation completed.")
print("Aggregate rows:", len(aggregate_results))
print("Per-class rows:", len(results[results["record_type"].eq("per_class")]))
print("The Edge-IIoTset feature set and historical locked test were not used.")
