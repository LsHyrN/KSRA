# Reproducing the KSRA experiments

## Repository layout

These local release names distinguish the new files from the original research files.

    KSRA/
      Code/KSRA.py
      Code/KSRA_experiments.py
      Datasets/Example_mixed.mat
      Datasets/annealing_variant1.mat
      Datasets/...
      Results/
      Notes/REPRODUCIBILITY.md

Paths are relative to the code location. No preparation folders, manifests, or internal audit files are required.

## Dependencies and recorded environments

Only Python, NumPy, and SciPy are required. The main experiments used Python 3.11.4, NumPy 2.1.3, and SciPy 1.14.1. For example:

    python -m pip install numpy==2.1.3 scipy==1.14.1

The separate scalability benchmark used Python 3.8.20, NumPy 1.24.4, SciPy 1.10.1 and single-thread MKL on Windows 10, Intel Core i5-1035G1, approximately 16 GiB RAM. Its times should not be conflated with timings in the main experiments. Execution times depend on environment.

## Input

Each MAT file must contain a finite 2-D trandata matrix, with samples in rows, prepared features in all but the last column, and binary labels (normal=0, anomaly=1) in the final column. Both classes are required for evaluation.

Use the prepared benchmark files without extra normalization or imputation. Numerical columns are already range-normalized to [0,1]. Nominal columns contain integer codes. The benchmark convention treats a column as numerical when all entries are at most 1+1e-9. Remaining columns use mismatch distance. Binary nominal columns therefore follow this prepared-input convention. For new data, supply an explicit numeric_mask through the core API.

The runner defaults to the original 20 manuscript benchmarks, using full inputs. Example_mixed is an optional example and is excluded from paper averages. Exploratory full-catalog subsampling and skip lists are outside this entry point.

## Running

From the repository root:

    python Code/KSRA_experiments.py

Fixed KSRA2 only:

    python Code/KSRA_experiments.py --variants v2

Small example:

    python Code/KSRA_experiments.py --datasets Example_mixed --variants v2 --lambdas 0.01

Dataset arguments are filename stems. Use --output to specify an alternative result directory. A fresh subprocess runs each dataset/variant pair, with numerical-library threads fixed to one. Windows process commit is monitored against 4096 MiB. That native monitor is unavailable on other operating systems. Worker failure stops the batch and identifies its log.

Re-running an identical command skips successful and unconverged terminal records. Interrupted or failed configurations are recomputed. Incompatible code, data or solver identities require a new output directory. Dense solver checkpoints and representations are not saved.

## Models and numerical settings

The residual is E=K-KZ. All l2,1 norms sum Euclidean row norms.

| Variant | Loss | Regularizer |
|---|---|---|
| KSRA1 / v1 | Row l2,1 | Row l2,1 |
| KSRA2 / v2 | Squared Frobenius | Squared Frobenius |
| KSRA3 / v3 | Row l2,1 | Squared Frobenius |
| KSRA4 / v4 | Squared Frobenius | Row l2,1 |

- Lambda grid: 10^j for j=-8,...,7.
- Sigma: mean off-diagonal mixed distance.
- Mixed distance: numerical Euclidean distance plus the proportion of nominal mismatches, with mismatch tolerance 1e-9.
- Kernel construction row block size: 256.
- KSRA1/3 smoothed row-norm epsilon: 1e-4.
- Gradient and relative-step tolerances: 1e-7.
- ADMM absolute/relative tolerances: 1e-8 and 1e-6.
- Initial rho=1. The runner enables residual balancing every 25 iterations through iteration 1000, ratio threshold 10, multiplier 2, bounds [1e-8,1e8], and dual rescaling. The core API itself defaults to fixed rho.
- Maximum iterations: 20000, with continuation stages 2000, 5000 and 20000.
- Solver budget: 120 seconds per configuration, checked between iterations. One dense operation can exceed that time.
- Graph: magnitude symmetrization (abs(Z)+abs(Z).T)/2 with self-loops retained. Zero-degree rows transition uniformly.
- Restart weight d=0.85, propagation weight 0.15; uniform restart and initialization.
- Walk tolerance: 1e-12 in l2, maximum 1000 iterations.
- Structural score: min-max-normalized negative stationary probability. The core maps a constant vector to zero using its floating-point-aware rule.
- Final score: structural score multiplied by the local isolation weight.

## Selection and interpretation

Fitting and graph construction do not receive labels. Each dataset/variant selects the converged configuration with highest evaluation ROC-AUC; exact ties select smaller lambda. This uses evaluation labels and is not independent held-out parameter selection.

AP is non-interpolated average precision from the same selected score vector. It is not independently tuned. Dataset averages are unweighted across the same 20 tasks.

Unconverged configurations retain diagnostics, have no scores or metrics, and are excluded from selection. The grid_complete flag requires every requested lambda to succeed. A partial-grid best observed value does not establish a fully solved grid optimum.

Each solver configuration has a 120-second time budget. Hardware and numerical-library differences can therefore affect how many iterations finish within the budget and which configurations converge.

## Outputs

    Results/<dataset>/dataset.json
    Results/<dataset>/v2_lam_1e-02/result.json
    Results/<dataset>/v2_lam_1e-02/history.json
    Results/<dataset>/v2_lam_1e-02/progress.json
    Results/<dataset>/v2_lam_1e-02/scores.npy
    Results/<dataset>/v2_lam_1e-02/stationary.npy
    Results/<dataset>/v2_lam_1e-02/curves.npz
    Results/<dataset>_v2.log
    Results/summary.csv
    Results/summary.json

Scores follow original input row order. Successful results include ROC/PR arrays, parameters and convergence diagnostics. summary.csv contains one row per dataset/variant with selected lambda, ROC-AUC, AP and grid completeness. If an application locks the CSV, the latest table is left in summary.csv.tmp and fitting continues.

selected_configuration_seconds sums recorded profile construction, cumulative solver and scoring times. It excludes the whole parameter search and process startup. summary.json distinguishes batch completion from full numerical convergence.

Separate scalability, sensitivity, ablation, fusion and synthetic geometry experiments require their dedicated research scripts.

The repository includes per-sample anomaly scores for all converged configurations on the 20 manuscript benchmarks. The accompanying `summary.csv` identifies the configuration selected for each dataset and variant. Running the experiment script also generates diagnostic files in the output directory.

## Validation against archived results

We checked the public scripts against the archived experimental results on the Iris benchmark. The checks covered all four KSRA variants and included the full 16-value regularization grid for KSRA2. Across 27 successful runs, per-sample anomaly scores agreed within an absolute tolerance of 1e-12 and a relative tolerance of 1e-10. ROC-AUC and average precision agreed within an absolute tolerance of 1e-12.
