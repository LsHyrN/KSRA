# KSRA

## Abstract
Anomaly detection is a pivotal technology in fields such as data mining, information security, and image processing. Existing research, however, faces two major challenges: 1) traditional proximity-based detection methods are susceptible to the distance concentration effect in high-dimensional spaces, leading to a decline in detection performance. 2) most models are only compatible with single-attribute data (either numerical or nominal), making them struggle with the mixed-type data prevalent in real-world scenarios, which limits their applicability. To address these challenges, this paper proposes a Kernelized Self-Representation-based Anomaly detection (KSRA) method, an anomaly detection method that uses a mixed kernel to construct nonlinear similarity profiles from numerical and categorical attributes. These profiles provide a unified representation for subsequent self-representation learning. Building upon this, a generalized kernel self-representation framework is constructed. Four model variants are systematically derived based on different combinations of loss and regularization terms. A random walk on the learned graph produces a structural anomaly score, which is combined with a kernel-derived isolation weight to score each sample. On 20 benchmark datasets, the fixed variant KSRA$_2$ achieved the highest mean ROC-AUC among the fixed methods in the main comparison under the reported tuning protocol.

## Framework
![image](Paper/KSRA_EXSY/Optimal-Design-layout/images/Figures/framework_KSRA.PNG)

## Usage
From the repository root, run KSRA2 on the Iris dataset with λ = 0.01:
```bash
python Code/KSRA_experiments.py --datasets iris_Irisvirginica_11_variant1 --variants v2 --lambdas 0.01 --output example_results
```
This example gives ROC-AUC **0.9382** and average precision **0.9217**. The output file `example_results/summary.csv` reports both metrics and the path to the per-sample anomaly scores (`scores.npy`).
See [REPRODUCIBILITY.md](Notes/REPRODUCIBILITY.md) for dependencies and instructions for running the full benchmark.

## Contact
If you have any questions, please contact yangluoshu15717@foxmail.com or yuanzhong@scu.edu.cn.
