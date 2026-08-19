# Project XIA v2 revision protocol

## Reason for the revision

The JISA editorial decision found the topic in scope but judged the submission
to have insufficient novelty, technical depth, and comparative analysis. This
protocol converts those criticisms into predeclared experiments. Version 2 is
an extended study, not a relabelling of the archived v1.0.0 analysis.

## Evaluation boundary

The v1 locked test has already been inspected and reported. It is therefore a
historical test set in version 2 and must not be called untouched, unseen, or
prospective. Version-2 claims will rest on repeated nested cross-validation and
an independently sourced external benchmark.

## Formal CS-SHAP rule

For original feature set F, resampling folds r = 1,...,R, candidate size k,
and recurrence threshold tau:

1. Fit all preprocessing and the explanation model only on the analysis
   partition of fold r.
2. Compute class-specific absolute SHAP importance on the fold's held-out
   explanation partition.
3. Give each class equal weight to obtain the class-balanced ranking B_r.
4. Average only the predeclared rare-class rows to obtain minority ranking M_r.
5. Define recurrence frequencies
   p_B(j) = R^-1 sum_r I[j in top-k(B_r)] and
   p_M(j) = R^-1 sum_r I[j in top-k(M_r)].
6. Select S_CS = {j: p_B(j) >= tau} union {j: p_M(j) >= tau}.

All choices of rare classes, k, tau, folds, seeds, metrics, and classifiers must
be declared before evaluation on the corresponding outer validation fold.

## Required Edge-IIoTset experiments

### Comparators

- Full controlled feature set
- Natural global SHAP top-k
- Class-balanced SHAP top-k
- Mutual information top-k
- Random-forest permutation/impurity ranking top-k
- Recursive feature elimination where computationally feasible
- Stable class-balanced core only
- Stable minority set only
- Complete CS-SHAP union

### Component ablations

- CS-SHAP without minority additions
- CS-SHAP without the stability threshold
- CS-SHAP with natural-prevalence aggregation instead of equal class weights
- CS-SHAP across a small predeclared grid of k and tau, selected only inside
  the inner loop

### Classifier transfer

Evaluate each selected representation with at least:

- XGBoost
- Random Forest
- Logistic Regression (or linear SVM when convergence is demonstrably better)

Classifier-specific tuning must remain inside the inner loop. The feature
selector and downstream classifier are separate factors in the analysis.

### Statistics and efficiency

- Repeated nested stratified CV with matched outer splits
- Per-fold macro F1, balanced accuracy, MCC, and per-class recall/F1
- Paired exact/randomization comparisons with multiplicity control
- Selection frequency, Jaccard similarity, and Nogueira stability
- Training time, batch inference latency, transformed dimensionality, serialized
  model size, and peak memory where the environment permits reliable capture

## External validation

Apply the same algorithm independently to a second IoT/IIoT dataset. TON_IoT
network data is the preferred first candidate because it represents a distinct
testbed and has a documented IoT/IIoT security context. The external study is
method-level replication: features need not have identical names across
datasets. Report dataset-specific selected sets and compare reduction,
stability, macro performance, and minority-class behaviour.

If attack labels are harmonized, document the mapping before modelling. Do not
silently merge semantically different attacks. Dataset identity, capture IDs,
timestamps, raw addresses, payloads, and other likely shortcut fields require a
fresh leakage audit.

## Claims permitted after completion

The paper may claim a reusable framework only if the component ablations show
that minority conditioning and recurrence add measurable value, and the second
dataset shows comparable reduction/stability behaviour. It must not claim
real-time edge deployment without hardware measurements or zero-day detection
without an explicitly unseen-attack protocol.

## Release plan

- Preserve Zenodo v1.0.0 unchanged.
- Develop version 2 on a new repository branch.
- Record environment versions and seeds.
- Publish code and derived results only after internal checks pass.
- Create a new tagged release and Zenodo version when the manuscript is frozen.
