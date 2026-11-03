# Remediation evidence — 3 November 2026

Modified LAB5.ipynb and README.md; added pinned requirements, .python-version,
.gitignore and machine-readable results.json. The original educational PCA/LDA/
KPCA calculations and plots remain. Custom LDA now uses count-weighted scatter;
KPCA uses one definition with sqrt-eigenvalue scaling. Modern one-vs-rest logistic
regression preserves the lab's original classification strategy.

## Validation

CPython 3.12.16, NumPy 2.5.4, scikit-learn 1.9.1, SciPy 1.18.1.

- `python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 LAB5.ipynb`:
  fresh execution of every cell passed; saved figures and results.json.
- 14 code cells compiled and saved outputs contained no execution errors.
- Notebook assertions passed for PCA/LDA subspace equivalence, KPCA Gram-matrix
  equivalence, the 124/54 split, and the training-only scaler mean.
- `git diff --check`: passed.

Results in README come from the executed results.json, not earlier saved output.
PCA/LDA maximum subspace angles were approximately 1.16e-15 / 1.15e-15 radians;
KPCA maximum Gram-matrix difference was approximately 2.93e-15 (tolerance 1e-8).
Training five-fold accuracy mean/std and held-out accuracy are listed per model.

The original scaler fitted training data only; no leakage was demonstrated in
that step. New CV pipelines fit scaling and projections separately in each
training fold. No test data is used for parameter selection.

## Limits and claims

This is one small UCI Wine split, not a general performance estimate. The
synthetic KPCA plots use full datasets for exploration and do not evaluate
held-out predictions; the custom function has no out-of-sample transform.
Dense KPCA has quadratic memory cost. No runtime/speed benchmark was measured.
Only the pinned environment was tested. Windows blocked the venv launcher, so a
separate newly provisioned Python interpreter ran the checks; ordinary venv
activation was not fully exercised on this host.

Safe to describe: from-scratch dimensionality reduction validated numerically
against scikit-learn, with reproducible training-only CV and held-out evaluation.
Do not claim guaranteed superiority, general 100% accuracy, speed gains,
production deployment or successful CI.
