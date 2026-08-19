# Project XIA v2.0.0

This release contains the extended two-dataset evaluation prepared in response
to editorial feedback requesting greater novelty, technical depth, and
comparative analysis.

## Added

- five-by-five nested validation on Edge-IIoTset and TON_IoT;
- conventional mutual-information and random-forest-importance comparators;
- class-balanced, minority-only, and stability component ablations;
- XGBoost, random-forest, and scaled-linear classifier evaluation;
- exact paired sign-flip tests and multiplicity-adjusted Edge comparisons;
- per-class metrics, efficiency measurements, and subset-stability analysis;
- TON_IoT duplicate, shortcut-field, and label-consistency audits;
- revised two-dataset manuscript and automated value-verification script.

## Main findings

- Edge-IIoTset XGBoost: CS-SHAP retained 18.6/42 predictors, macro-F1 0.969786,
  and reduced mean training time by approximately 25.4% relative to Full-42.
- TON_IoT XGBoost: CS-SHAP retained 14.8/34 predictors, macro-F1 0.966465,
  reduced mean training time by approximately 45.6%, and achieved mean pairwise
  subset Jaccard similarity 0.913.
- Classifier transfer is model-dependent: results were strongest for XGBoost,
  competitive for random forest, and weaker for scaled linear logistic regression.

## Interpretation boundary

The release supports stable, minority-aware feature compression for tree-based
multiclass IoT intrusion detection. It does not claim universal accuracy
superiority, classifier independence, real-time edge deployment, zero-day
detection, or direct train-on-one/test-on-another cross-dataset generalisation.

## Previous release

Version 1.0.0 remains archived at https://doi.org/10.5281/zenodo.22003473.
Version 2.0.0 is permanently archived at
https://doi.org/10.5281/zenodo.22018721.
