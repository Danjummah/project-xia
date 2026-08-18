# Project XIA

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22003473.svg)](https://doi.org/10.5281/zenodo.22003473)

Reproducibility package for class-sensitive SHAP (CS-SHAP) feature selection in multiclass IoT/IIoT intrusion detection. The full Kaggle analysis is preserved in a cleaned notebook with outputs removed; derived result tables and fitted final models are included.

## Main result

On the locked test set, `CS_SHAP_19` achieved accuracy **0.97751**, macro-F1 **0.96815**, balanced accuracy **0.96426**, and MCC **0.97553** with 19 features. Recorded training time was **37.236 s**, versus **51.557 s** for `Full_42` in the same experiment.

## Contents

- `notebooks/`: complete cleaned analysis
- `results/`: original CSV outputs
- `models/`: fitted pipeline and label encoder
- `scripts/verify_results.py`: verifies principal reported values
- `CODEBOOK.md`: artifact descriptions
- `SHA256SUMS`: integrity hashes

## Reproduce on Kaggle

1. Create a Kaggle notebook and attach the Edge-IIoTset Cyber Security Dataset of IoT & IIoT.
2. Upload the notebook from `notebooks/`.
3. Confirm the dataset mount matches the `/kaggle/input/...` path in the first cells; edit only that path if Kaggle renamed it.
4. Use `requirements.txt` or `environment.yml`. The original run recorded scikit-learn 1.0.2 and XGBoost 1.5.2.
5. Run all cells in order. Nested analysis can take substantial time.
6. Download `/kaggle/working/` outputs and compare with `results/`.
7. From the repository root run `python scripts/verify_results.py`.

## Reconstruct the trained pipeline after downloading from GitHub

The connected publishing interface limits individual transfers, so the 7.3 MB
pipeline is stored losslessly as numbered ZIP parts. From the repository root,
run `python scripts/reconstruct_model.py`. This creates
`models/project_xia_final_cs_shap_pipeline.joblib` and verifies its SHA-256
digest. The smaller label encoder is stored normally.

## Data and methodological boundary

The source dataset is not redistributed; obtain it from its original provider and follow its terms. The locked test set must never be used for tuning, selection, or method choice. The archived v1.0.0 package is permanently available at [https://doi.org/10.5281/zenodo.22003473](https://doi.org/10.5281/zenodo.22003473).

## Security

`joblib` files may execute code when loaded. Trust the source and check `SHA256SUMS` first.

## Author and licence

Abdul Hafiz Abdullai — Independent Researcher, Ghana — [ORCID](https://orcid.org/0009-0001-6995-0747). Code is MIT licensed; dataset rights remain with its provider. Cite using `CITATION.cff`.
