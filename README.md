# Dimensionality reduction: PCA, LDA and RBF kernel PCA

Open [LAB5.ipynb](LAB5.ipynb). It retains the educational matrix implementations,
compares them with scikit-learn, and evaluates fixed logistic-regression models.

## Reproduce (Python 3.12)

```sh
git clone https://github.com/MamilaGun23/Dimensionality-Reduction-Lab-PCA-LDA-and-KPCA.git
cd Dimensionality-Reduction-Lab-PCA-LDA-and-KPCA
python -m venv .venv
```

Activate the environment (PowerShell: `.venv/Scripts/Activate.ps1`;
macOS/Linux: `source .venv/bin/activate`), then:

```sh
python -m pip install -r requirements.txt
python -m notebook LAB5.ipynb
```

For a fresh noninteractive run of every cell:

```sh
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 LAB5.ipynb
```

This regenerates figures and [results.json](results.json). Numerical assertions
inside the notebook fail execution if the custom/reference checks disagree.
No credentials, remote dataset downloads or local machine paths are needed.

## Data and method

The Wine data is the offline copy bundled with `sklearn.datasets.load_wine`:
178 samples, 13 chemical measurements and three cultivars, originally from
[UCI Wine](https://archive.ics.uci.edu/dataset/109/wine). The notebook prints all
feature names and uses labels 1–3 to preserve the original lab convention.
See the dataset description returned by `load_wine().DESCR` for attribution.
UCI lists Wine under CC BY 4.0.

A stratified 70/30 train/test split uses seed 0 (124 training, 54 test samples).
The original scaler already fitted training data only; no leakage was found
there. Five-fold shuffled stratified CV (seed 0) now compares original features,
two-component PCA and two-component LDA on **the same training folds**, scoring
accuracy. Every fold fits its own scaler and projection inside a Pipeline.
One-vs-rest logistic regression uses lbfgs, seed 1 and max_iter=1000. Two
components are fixed in advance, and the held-out test set is never used to tune.

Custom PCA uses symmetric eigendecomposition. Custom LDA now uses unnormalised,
count-weighted within-class scatter and a symmetric generalised eigenproblem;
the previous sum of class covariance matrices weighted unequal classes equally.
PCA/LDA subspaces are checked against scikit-learn using principal angles.
Custom KPCA centres the RBF kernel and scales eigenvectors by sqrt(eigenvalue);
its score Gram matrix is checked against scikit-learn, avoiding sign ambiguity.
Duplicate KPCA definitions and a NumPy-incompatible generator were removed.

Synthetic moons (100 samples) and noisy circles (1,000 samples) use seed 123 and
gamma=15. These full-dataset plots are **exploratory visualisations**, not test
evaluations or evidence that gamma=15 is optimal. The custom KPCA function does
not implement out-of-sample projection.

## Measured results

Execution results are recorded below after running the notebook; see
[REMEDIATION.md](REMEDIATION.md) for commands and remaining limitations.

| Model | Training CV accuracy mean ± std | Held-out accuracy (54 samples) |
|---|---:|---:|
| Original (13 features) | 0.9753 ± 0.0332 | 1.0000 |
| PCA (2 components) | 0.9593 ± 0.0365 | 0.9259 |
| LDA (2 components) | 0.9757 ± 0.0199 | 1.0000 |

Maximum PCA/LDA subspace angles were 1.16e-15 / 1.15e-15 radians;
maximum KPCA Gram-matrix error was 2.93e-15 (asserted tolerance: 1e-8).

LDA uses labels to seek class separation; PCA seeks variance without labels.
LDA can help classification, but no method is guaranteed to win on other data.
A single 54-sample test partition does not establish general performance. CV
standard deviations describe fold variability, not confidence intervals.
No speed benchmark was run and no speed improvement is claimed.

Reference: Raschka and Mirjalili, *Python Machine Learning*, Chapter 5.
