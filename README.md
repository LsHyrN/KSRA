# KSRA

## Abstract
Anomaly detection is a pivotal technology in fields such as data mining, information security, and image processing. Existing research, however, faces two major challenges: 1) traditional proximity-based detection methods are susceptible to the distance concentration effect in high-dimensional spaces, leading to a decline in detection accuracy. 2) most models are only compatible with single-attribute data (either numerical or nominal), making them struggle with the mixed-type data prevalent in real-world scenarios, which limits their applicability. To address these challenges, this paper proposes a Kernelized Self-Representation based Anomaly detection (KSRA) method. Specifically, the original mixed data is first mapped into a high-dimensional feature space using a mixed kernel function. This process not only captures non-linear relationships, but also achieves a unified representation and feature enhancement for different attribute types, effectively mitigating the distance concentration problem. Building upon this, a generalized kernel self-representation framework is constructed. Four model variants are systematically derived based on different combinations of loss and regularization terms. Finally, a random walk mechanism is introduced to achieve efficient global information diffusion and deep integration, based on which a final anomaly score is defined to accurately reflect the anomaly degree of each sample. To verify the effectiveness of the proposed method, extensive comparative experiments are conducted on 20 benchmark datasets. The results demonstrate that the KSRA method  significantly outperforms 11 mainstream anomaly detection algorithms.

## Framework
![image](Paper/KSRA_framework.png)

## Usage
You can run KSRA by:
```
if __name__ == '__main__':

    lam = 1000     # Set the regularization parameter for the self-learning representation

    data_name='audiology_variant1.mat'
    mat = loadmat(r'Datasets\\' + data_name)
    trandata = mat['trandata'].astype(np.float64)

    labels = trandata[:, -1].ravel()  # label
    data = trandata[:, :-1]           # data

    scores = KSRA(data, lam)
    print(f"samples_scores = {scores}")

    auc = roc_auc_score(labels, scores)
    print(f"AUC = {auc:.10f}")
```
You can get outputs as follows:
```
samples_scores = [0.03901524 0.03901524 0.05661084 0.09986081 0.03130849 0.03596569
 0.03943726 0.03943726 0.07241828 0.0453333  0.19702664 0.19678281
 0.21590567 0.17838202 0.04114497 0.04314381 0.05087851 0.21398877
 0.02818553 0.03587548 0.07576042 0.22431286 0.05087851 0.17573028
 0.11953246 0.23399348 0.00784547 0.01439658 0.07550898 0.05323101
 0.01937702 0.0191369  0.01284203 0.01264416 0.01588015 0.0448237
 0.0327011  0.01937702 0.04054508 0.02180903 0.0272029  0.03863627
 0.0624056  0.05439022 0.00450267 0.02700902 0.06540685 0.05072152
 0.08810665 0.00538759 0.04628345 0.06559822 0.06341031 0.04394836
 0.04030126 0.02691745 0.06184421 0.00508609 0.07582585 0.07774393
 0.11306623 0.02268043 0.10845526 0.03435836 0.11182033 0.03591687
 0.03702724 0.0257521  0.00281326 0.07532465 0.08400789 0.27324197
 0.08420109 0.03532668 0.03435836 0.09138536 0.09375518 0.04000779
 0.03435836 0.12387913 0.03062989 0.08411387 0.01933976 0.03558475
 0.02368155 0.10557463 0.08731453 0.06936739 0.07468577 0.08504194
 0.021559   0.01776795 0.00902538 0.10809127 0.08472708 0.06465409
 0.06263835 0.01419154 0.06111294 0.0675207  0.09365513 0.02189574
 0.0362641  0.08859728 0.09052213 0.0540809  0.12761503 0.06609863
 0.09533833 0.00281326 0.07954951 0.08439711 0.07099963 0.04569777
 0.17653917 0.01905353 0.17541196 0.23588104 0.11803462 0.05641782
 0.03853621 0.17075695 0.14875711 0.20188946 0.19080317 0.02204005
 0.15194635 0.14336358 0.06449327 0.1272794  0.09562656 0.04171258
 0.12930679 0.05746179 0.18089004 0.27923849 0.05115779 0.17573028
 0.02835962 0.10457647 0.09730803 0.13371762 0.29704511 0.26570839
 0.18188049 0.08939453 0.01674499 0.01937702 0.03863627 0.00784547
 0.06559822 0.04394836 0.02118946 0.01937702 0.10845526 0.00281326
 0.08420109 0.04000779 0.01901586 0.12554311 0.12554311 0.00698782
 0.         0.03732545 0.07414678 0.05878269 0.17541196 0.00723676
 0.00698782 0.05641782 0.00281326 0.07519625 0.06722884 0.01470884
 0.01470884 0.17877979 0.05973245 0.05289086 0.06633674 0.2465574
 0.20132641 0.02204005 0.0141389  0.07339275 0.04349434 0.06008769
 0.05783507 0.04137242 0.04351861 0.08213277 0.02627039 0.06609863
 0.13497655 0.14207313 0.13862451 0.09196727 0.09443152 0.06091964
 0.06259094 0.04171258 0.04180292 0.12711358 0.24866973 0.05574879
 0.05013039 0.05115779 0.04804443 0.01990683 0.02670666 0.01691096
 0.01691096 0.09010879 0.03366951 0.03426047 0.1254126  0.08420109
 0.04042362 0.09525589 0.09011813 0.05026431 0.05210362 0.09890372
 0.07750943 0.08939453 0.03596569 0.03096836]
AUC = 0.8736634486
```

## Contact
If you have any questions, please contact yangluoshu15717@foxmail.com or yuanzhong@scu.edu.cn.
