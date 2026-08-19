# Project XIA v2

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22018721.svg)](https://doi.org/10.5281/zenodo.22018721)

Reproducibility package for class-sensitive stable SHAP (CS-SHAP) feature
selection in multiclass IoT/IIoT intrusion detection. Version 2 adds nested
comparators, component ablations, classifier transfer, and independent
algorithmic replication on TON_IoT. The supported contribution is stable,
minority-aware feature compression for tree-based intrusion detection, not a
claim of universally superior accuracy or classifier-independent selection.

## Version-2 results

- **Edge-IIoTset / XGBoost:** CS-SHAP used 18.6 of 42 predictors on average,
  achieved macro-F1 **0.969786** versus **0.969288** for the full representation,
  and reduced mean training time from **47.44 s** to **35.38 s**.
- **TON_IoT / XGBoost:** CS-SHAP used 14.8 of 34 predictors, achieved macro-F1
  **0.966465** versus **0.966704** for the full representation, reduced mean
  training time from **52.26 s** to **28.42 s**, and had mean pairwise subset
  Jaccard similarity **0.913**.
- **Classifier boundary:** CS-SHAP was strongest with XGBoost, competitive but
  not best with random forest, and weaker with scaled linear logistic regression.

The archived v1 locked-test result remains historical and unchanged.

Version 2 is permanently archived at
[https://doi.org/10.5281/zenodo.22018721](https://doi.org/10.5281/zenodo.22018721).

## Contents

- `notebooks/`: complete cleaned analysis
- `results/`: original CSV outputs
- `results_v2/`: Edge-IIoTset and TON_IoT version-2 results and audit outputs
- `models/`: fitted pipeline and label encoder
- `scripts/verify_results.py`: verifies principal reported values
- `scripts/verify_v2_manuscript_values.py`: verifies the manuscript's key v2 values
- `manuscript/`: version-2 manuscript submitted with this release
- `CODEBOOK.md`: artifact descriptions
- `SHA256SUMS`: integrity hashes

## Reproduce on Kaggle

1. Create a Kaggle notebook and attach the Edge-IIoTset Cyber Security Dataset of IoT & IIoT.
2. Upload the notebook from `notebooks/`.
3. Confirm the dataset mount matches the `/kaggle/input/...` path in the first cells; edit only that path if Kaggle renamed it.
4. Use `requirements.txt` or `environment.yml`. Compatibility handling for
   `OneHotEncoder(sparse_output=...)` and logistic `log_loss`/`log` is included
   for different scikit-learn generations.
5. Run all cells in order. Nested analysis can take substantial time.
6. Download `/kaggle/working/` outputs and compare with `results/`.
7. From the repository root run `python scripts/verify_results.py` and
   `python scripts/verify_v2_manuscript_values.py`.

## Reconstruct the trained pipeline after downloading from GitHub

The connected publishing interface limits individual transfers, so the 7.3 MB
pipeline is stored losslessly as numbered ZIP parts. From the repository root,
run `python scripts/reconstruct_model.py`. This creates
`models/project_xia_final_cs_shap_pipeline.joblib` and verifies its SHA-256
digest. The smaller label encoder is stored normally.

## Data and methodological boundary

The source datasets are not redistributed; obtain them from their original
providers and follow their terms. Version 2 uses matched nested cross-validation.
Its TON_IoT experiment is method-level replication on a separate schema, not
training on one benchmark and testing on the other. The archived v1.0.0 package
is permanently available at
[https://doi.org/10.5281/zenodo.22003473](https://doi.org/10.5281/zenodo.22003473).

## Security

`joblib` files may execute code when loaded. Trust the source and check `SHA256SUMS` first.

## Author and licence

Abdul Hafiz Abdullai — Ghana Education Service, Asikuma-Odoben-Brakwa District
Education Directorate, Ghana — [ORCID](https://orcid.org/0009-0001-6995-0747).
Code is MIT licensed; dataset rights remain with their providers. Cite using
`CITATION.cff`.
