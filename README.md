# KSRA

## Abstract
Anomaly detection is a pivotal technology in fields such as data mining, information security, and image processing. Existing research, however, faces two major challenges: 1) traditional proximity-based detection methods are susceptible to the distance concentration effect in high-dimensional spaces, leading to a decline in detection accuracy. 2) most models are only compatible with single-attribute data (either numerical or nominal), making them struggle with the mixed-type data prevalent in real-world scenarios, which limits their applicability. To address these challenges, this paper proposes a Kernelized Self-Representation based Anomaly detection (KSRA) method. Specifically, the original mixed data is first mapped into a high-dimensional feature space using a mixed kernel function. This process not only captures non-linear relationships, but also achieves a unified representation and feature enhancement for different attribute types, effectively mitigating the distance concentration problem. Building upon this, a generalized kernel self-representation framework is constructed. Four model variants are systematically derived based on different combinations of loss and regularization terms. Finally, a random walk mechanism is introduced to achieve efficient global information diffusion and deep integration, based on which a final anomaly score is defined to accurately reflect the anomaly degree of each sample. To verify the effectiveness of the proposed method, extensive comparative experiments are conducted on 20 benchmark datasets. The results demonstrate that the KSRA method  significantly outperforms 11 mainstream anomaly detection algorithms.

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
