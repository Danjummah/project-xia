# Artifact codebook

## Version-2 additions

- `notebooks/project_xia_v2_edge_experiments.ipynb`: Edge-IIoTset nested
  comparators, ablations, and classifier-transfer experiment.
- `notebooks/project_xia_v2_toniot_external_validation.ipynb`: independently
  replicated TON_IoT experiment with dataset-specific audit and preprocessing.
- `results_v2/edge_analysis/`: consolidated Edge results, summaries, per-class
  metrics, coverage checks, and exact paired comparisons.
- `results_v2/toniot_analysis/`: corrected TON_IoT audit, summaries, per-class
  metrics, stability statistics, coverage checks, and paired comparisons.
- `results_v2/toniot_raw/`: direct Kaggle outputs retained without rewriting.
- `scripts/analyze_v2_edge_results.py` and `scripts/analyze_v2_toniot_results.py`:
  validation and summary builders.
- `scripts/verify_v2_manuscript_values.py`: assertions for principal values
  reported in the version-2 manuscript.
- `manuscript/CS_SHAP_V2_Two_Dataset_Manuscript.docx`: revised two-dataset
  research article associated with release v2.0.0.

| Artifact pattern | Description |
|---|---|
| `locked_test_results` | Aggregate performance and timing for each frozen method. |
| `locked_test_per_class` | Per-class precision, recall, F1, support, and average precision. |
| `locked_test_confusion_matrix` | Long-form confusion matrices. |
| `final_frozen_feature_sets` | Ordered features for final methods. |
| `nested_cv_results` / `nested_cv_summary` | Outer-fold results and aggregated statistics. |
| `nested_inner_rankings` / `nested_feature_selections` | Inner-fold ranks and outer-fold selections. |
| `cs_shap_selection_register` | Evidence and final CS-SHAP roles. |
| `foldwise_shap_rankings` / `shap_stability_results` | Fold ranks and stability measures. |
| `corrected_shap_rankings` / `class_specific_shap` | Ranking variants and class-specific importance. |
| `natural_shap_sample_distribution` / `shap_rank_correlations` | Explanation sample composition and rank correlations. |
| `.joblib` files | Fitted final pipeline and target label encoder; serialized executable content. |

On GitHub, `project_xia_final_cs_shap_pipeline.zip.part-*` contains a lossless,
split ZIP of the final pipeline. Run `scripts/reconstruct_model.py` to restore
and verify the original `.joblib` file.

Some CSVs retain an unnamed pandas index column from the original export. Values were not altered during packaging.
