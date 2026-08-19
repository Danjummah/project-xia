"""Verify principal Project XIA v2 manuscript values from archived CSVs."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EDGE = ROOT / "results_v2" / "edge_analysis"
TON = ROOT / "results_v2" / "toniot_analysis"


def close(actual, expected, tol=5e-7):
    assert abs(float(actual) - expected) <= tol, (actual, expected)


edge = pd.read_csv(EDGE / "project_xia_v2_edge_consolidated_results.csv")
edge_agg = edge[edge["record_type"].eq("aggregate")]
assert len(edge_agg) == 120
assert len(edge[edge["record_type"].eq("per_class")]) == 1800

means = edge_agg.groupby(["classifier", "selector"]).mean(numeric_only=True)
close(means.loc[("XGBoost", "CS_SHAP_Union"), "feature_count"], 18.6)
close(means.loc[("XGBoost", "CS_SHAP_Union"), "macro_f1"], 0.9697855246446029)
close(means.loc[("XGBoost", "Full_42"), "macro_f1"], 0.9692883102114866)
close(means.loc[("Random_Forest", "Natural_Global_20"), "macro_f1"], 0.9555303662035962)
close(means.loc[("Linear_Logistic_SGD_Scaled", "Full_42"), "macro_f1"], 0.7885097722283397)
close(means.loc[("Linear_Logistic_SGD_Scaled", "CS_SHAP_Union"), "macro_f1"], 0.740530375339927)

ton_results = pd.read_csv(ROOT / "results_v2" / "toniot_raw" / "project_xia_v2_toniot_results.csv")
ton_agg = ton_results[ton_results["record_type"].eq("aggregate")]
assert len(ton_agg) == 35
assert len(ton_results[ton_results["record_type"].eq("per_class")]) == 350
ton_means = ton_agg.groupby("selector").mean(numeric_only=True)
close(ton_means.loc["CS_SHAP_Union", "feature_count"], 14.8)
close(ton_means.loc["CS_SHAP_Union", "macro_f1"], 0.9664650471943605)
close(ton_means.loc["Full_Controlled", "macro_f1"], 0.9667043372020672)

stability = pd.read_csv(TON / "project_xia_v2_toniot_selection_stability.csv").set_index("selector")
close(stability.loc["CS_SHAP_Union", "mean_pairwise_jaccard"], 0.913039, tol=5e-6)
assert int(stability.loc["CS_SHAP_Union", "union_feature_count"]) == 17
assert int(stability.loc["CS_SHAP_Union", "features_in_all_folds"]) == 14

audit = json.loads((TON / "project_xia_v2_toniot_corrected_audit.json").read_text())
assert audit["raw_rows"] == 211043
assert audit["controlled_rows"] == 183560
assert audit["duplicates_removed"] == 20569
assert audit["post_exclusion_duplicates_removed"] == 6914
assert audit["controlled_predictor_count"] == 34

print("Project XIA v2 manuscript values verified successfully.")
