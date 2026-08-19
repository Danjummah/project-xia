# Project XIA v2 Edge-IIoTset Kaggle run guide

## Notebook

Upload `notebooks/project_xia_v2_edge_experiments.ipynb` to Kaggle and attach
the same Edge-IIoTset dataset used for version 1.

## Before running

1. Enable a Kaggle accelerator only if available; the XGBoost implementation
   currently uses CPU `hist` so GPU is not required.
2. Confirm the dataset path in the first loading cell. Change only the mount
   path if Kaggle assigned a different directory name.
3. Use a fresh Kaggle session. Do not preload version-1 result CSVs into
   `/kaggle/working`; the notebook must recreate them.
4. Keep `CLASSIFIERS_TO_RUN` unchanged for the final run. For a smoke test,
   temporarily use `["XGBoost"]`, restart the session afterwards, restore all
   three classifiers, and run the final notebook from the beginning.

## Expected final outputs

- `project_xia_v2_edge_results.csv`
- `project_xia_v2_edge_selections.csv`
- `project_xia_v2_edge_baseline_rankings.csv`
- `project_xia_v2_edge_summary.csv`

The results file contains aggregate and per-class rows. The selections file
records every feature used in every outer fold. The ranking file records the
fold-specific mutual-information and random-forest baseline rankings.

## Experiment matrix

Each of five matched outer folds evaluates:

- Full 42-feature representation
- Natural-global SHAP top 20
- Class-balanced SHAP top 15
- Stable class-balanced core only
- Stable minority set only
- CS-SHAP union
- Mutual-information top 19 selected inside the outer training partition
- Random-forest-importance top 19 selected inside the outer training partition

Each representation is evaluated with XGBoost, Random Forest, and scalable
linear logistic-loss classification. This produces 120 aggregate fits plus
per-class results and ranking fits.

## Integrity checks after running

1. The final cell must print `The historical locked test was not accessed.`
2. There must be five aggregate rows for every selector/classifier pair.
3. No aggregate metric may be missing.
4. Every selected feature must belong to the controlled 42-feature set.
5. Download all four output CSVs without editing them.

Do not interpret or select a preferred method from a partial smoke-test run.
