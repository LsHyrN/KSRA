# KSRA: Experimental Settings and Data Preparation

This document supplements the runnable example in the repository README. It describes the dependencies, input format, and settings used to run the four KSRA variants. The README example is a usage demonstration; its `lambda_val = 0.01` is not a universal setting for all benchmark datasets.

## 1. Dependencies

The code uses Python 3 and the following packages:

```bash
python -m pip install numpy scipy scikit-learn pandas torch
```

NumPy and SciPy provide matrix operations, distance calculations, and MATLAB-file loading. Scikit-learn provides evaluation utilities, and pandas is used by the experiment script to export results. PyTorch is imported by the current implementation; the KSRA computations use NumPy and SciPy and do not require a GPU.

For a new run, record the installed versions with:

```bash
python --version
python -m pip freeze > environment.txt
```

These commands record the environment of that run, rather than reconstructing the environment of the originally reported experiments.

## 2. Input Data and Preprocessing

Each dataset is stored as a MATLAB `.mat` file containing the variable `trandata`, a two-dimensional array with one sample per row:

- All columns except the last contain input features.
- The last column contains binary evaluation labels: `1` for an anomaly and `0` for a normal sample.
- Labels are excluded from the input to `KSRA` and are used for evaluation and the parameter-selection procedure described below.

```python
trandata = loadmat(dataset_path)['trandata'].astype(np.float64)
data = trandata[:, :-1]
labels = trandata[:, -1].ravel()
```

Use the prepared benchmark files corresponding to the datasets listed in the manuscript. Their numerical attributes have already been scaled to the interval `[0, 1]`, and the prepared feature matrices contain no missing or infinite values. The `KSRA` function does not perform additional normalization or missing-value imputation. Replacing these files with raw source datasets therefore requires reproducing the preprocessing first.

The current implementation selects distance components using the stored feature values: a column whose entries are all at most `1 + 1e-9` is assigned to the Euclidean component; each remaining column is assigned to the Hamming component. This is an implementation convention for the prepared inputs, rather than a general-purpose semantic feature-type detector. In particular, a binary column containing only `0` and `1` is assigned to the Euclidean component.

For the Hamming component, categorical values are represented by numeric codes. The distance is the proportion of mismatched categorical attributes, using a comparison tolerance of `1e-9`; the magnitudes of unequal category codes do not determine their distance. No additional one-hot encoding is applied inside `KSRA`. An absent distance component contributes zero, and the mixed distance is the sum of the Euclidean and Hamming components.

For the README example, place `Example_mixed.mat` in `Datasets/` relative to the working directory, or adjust the example's file path to its actual location. The separate batch script `KSRA_experiments.py` instead expects a lowercase `datasets/` directory in the parent of the directory containing that script. It reads dataset names from `all_datalists_outlier.mat` (variable `datalists`) in that directory. Check these paths and explicitly select the manuscript's benchmark datasets before a batch run.

## 3. Model Variants

The function interface is:

```python
scores = KSRA(data, lambda_val, sigma=None, solver='v2')
```

| Manuscript variant | `solver` | Reconstruction loss | Coefficient regularizer |
| --- | --- | --- | --- |
| KSRA1 | `'v1'` | l2,1 norm | l2,1 norm |
| KSRA2 | `'v2'` | Squared Frobenius norm | Squared Frobenius norm |
| KSRA3 | `'v3'` | l2,1 norm | Squared Frobenius norm |
| KSRA4 | `'v4'` | Squared Frobenius norm | l2,1 norm |

Calling `KSRA(data, lambda_val)` uses KSRA2. To run another variant, set `solver` explicitly. Valid solver names are `'v1'`, `'v2'`, `'v3'`, and `'v4'`.

## 4. Parameter Settings

| Setting | Value or rule |
| --- | --- |
| Regularization parameter, lambda | Search over `10.0 ** np.arange(-8, 8)`, i.e., `1e-8` through `1e7` |
| Gaussian bandwidth, sigma | With `sigma=None`, the mean of all off-diagonal mixed distances, computed separately for each dataset |
| Restart probability, d | `0.85`; the propagation term has coefficient `1 - d = 0.15` |
| Initial propagation vector | Uniform, with each entry equal to `1 / n` |
| Propagation stopping tolerance | `1e-6`, measured by the Euclidean norm of consecutive vectors |
| Maximum propagation iterations | `1000` |
| KSRA1 and KSRA3 solver limits | At most `50` iterations; objective-change tolerance `1e-6` |
| KSRA2 solver | Direct regularized linear-system solution |
| KSRA4 solver limits | At most `100` iterations; primal and dual residual tolerances `1e-6`; ADMM penalty `rho = 1.0` |

Here, `n` is the number of samples. The default bandwidth rule requires a positive mean off-diagonal distance; an input containing only identical samples requires separate handling. The restart probability and propagation stopping settings are internal constants in the current code, rather than function arguments. Unless intentionally testing alternative settings, retain these defaults.

## 5. Evaluation and Reporting

The output is one anomaly score per sample; larger scores indicate greater abnormality. Compute ROC-AUC using the binary labels and the returned scores, as shown in the README. If reporting average precision (AP), use `sklearn.metrics.average_precision_score(labels, scores)`.

For the benchmark parameter search, lambda is selected separately for each dataset and variant by maximizing ROC-AUC over the stated grid. This selection uses evaluation labels and should be interpreted as label-informed benchmark tuning, rather than a label-independent deployment procedure. Report AP from the same selected run when pairing ROC-AUC and AP results. The value `0.01` in the README is only an example and does not replace this search.

Report the four variants separately. A best-across-variants result is a retrospective reference and should not be interpreted as the performance of a fixed, automatically selected variant. The manuscript's statistical comparisons, ablation studies, and parameter-sensitivity analyses use the fixed KSRA2 variant.

KSRA does not use random initialization or random sampling in its current implementation. Small numerical differences may nevertheless occur across numerical-library versions and platforms. Single-run benchmark results should not be interpreted as means over multiple random seeds.
