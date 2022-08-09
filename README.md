# TwiBot-22

This is the official repository of [TwiBot-22](https://twibot22.github.io/).

### Introduction

TwiBot-22 is the largest and most comprehensive Twitter bot detection benchmark to date. Specifically, [TwiBot-22](https://arxiv.org/abs/2206.04564) is designed to address the challenges of limited dataset scale, imcomplete graph structure, and low annotation quality in previous datasets. For more details, please refer to the [TwiBot-22 paper](https://arxiv.org/abs/2206.04564) and [statistics](descriptions/statistics.md).
![compare](./pics/compare.png)

### Dataset Format

Each dataset contains `node.json` (or `tweet.json`, `user.json`, `list.json`, and `hashtag.json` for TwiBot-22), `label.csv`, `split.csv` and `edge.csv` (for datasets with graph structure). See [here](descriptions/metadata.md) for a detailed description of these files.

### How to download TwiBot-22 dataset

Reviewers at the NeurIPS 2022 Datasets and Benchmarks Track: Feel free to download TwiBot-22 at [Google Drive](https://drive.google.com/drive/folders/1YwiOUwtl8pCd2GD97Q_WEzwEUtSPoxFs?usp=sharing).

`gdown --id 1YwiOUwtl8pCd2GD97Q_WEzwEUtSPoxFs`

### How to download other datasets

For TwiBot-20, visit the [TwiBot-20 github repository](https://github.com/BunsenFeng/TwiBot-20).

For other datasets, please visit the [Bot Repository](https://botometer.osome.iu.edu/bot-repository/datasets.html).

After downloading these datasets, you can transform them into the 4-file format detailed in "Dataset Format". Alternatively, you can directly download our preprocessed version:

For TwiBot-20, visit the [TwiBot-20 github repository](https://github.com/BunsenFeng/TwiBot-20), apply for TwiBot-20 access, and there will be a `TwiBot-20-Format22.zip` in the TwiBot-20 Google Drive link.

For other datasets, you can directly download them from [Google Drive](https://drive.google.com/drive/folders/1gXFZp3m7TTU-wyZRUiLHdf_sIZpISrze?usp=sharing). You should respect the license of each dataset, the "Content redistribution" section of the [Twitter Developer Agreement and Policy](https://developer.twitter.com/en/developer-terms/agreement-and-policy), the rules set by the [Bot Repository](https://botometer.osome.iu.edu/bot-repository/datasets.html), and only use these datasets for research purposes.

### Requirements

- pip: `pip install -r requirements.txt`
- conda : `conda install --yes --file requirements.txt `

### How to run baselines

1. clone this repo by running `git clone https://github.com/LuoUndergradXJTU/TwiBot-22.git`
2. make dataset directory `mkdir datasets` and download datasets to `./datasets`
3. change directory to `src/{name_of_the_baseline}`
4. run experiments under the guidance of corresponding `readme.md`

### Baseline Overview


| baseline                              | paper                                                                                  | acc on Twibot-22 | f1 on Twibot-22 | type  | tags                     |
| ------------------------------------- | -------------------------------------------------------------------------------------- | ---------------- | --------------- | ----- | ------------------------ |
| [Abreu et al.](src/Abreu/)            | [link](https://ieeexplore.ieee.org/abstract/document/9280525)                          | 0.7066           | 0.5344          | F     | `random forest`          |
| [Alhosseini et al.](src/Alhosseini/)  | [link](https://dl.acm.org/doi/fullHtml/10.1145/3308560.3316504)                        | 0.6910           | 0.4991          | F G   | `gcn`                    |
| [BGSRD](src/BGSRD/)                   | [link](https://www.mdpi.com/2073-8994/14/1/30)                                         | 0.7188           | 0.2114          | F     | `BERT GAT`               |
| [Bot Hunter](src/BotHunter/)          | [link](http://www.casos.cs.cmu.edu/publications/papers/LB_5.pdf)                       | 0.7279           | 0.2346          | F     | `random forest`          |
| [Botometer](src/Botometer/)           | [link](https://botometer.osome.iu.edu/)                                                | 0.4987           | 0.4257          | F T G |                          |
| [BotRGCN](src/BotRGCN/)               | [link](https://arxiv.org/abs/2106.13092)                                               | 0.7966           | 0.5750          | F T G | `BotRGCN`                |
| [Cresci et al.](src/Cresci/)          | [link](https://ieeexplore.ieee.org/abstract/document/7436643)                          | -                | -               | T     | `DNA`                    |
| [Dehghan et al.](src/Dehghan)         | [link](https://assets.researchsquare.com/files/rs-1428343/v1_covered.pdf?c=1647280648) | -                | -               | F T G | `Graph`                  |
| [Efthimion et al.](src/Efthimion/)    | [link](https://scholar.smu.edu/datasciencereview/vol1/iss2/5/)                         | 0.7408           | 0.2758          | F T   | `efthimion`              |
| [EvolveBot](src/EvolveBot/)           | [link](https://ieeexplore.ieee.org/abstract/document/6553246)                          | 0.7109           | 0.1409          | F T G | `random forest`          |
| [FriendBot](src/FriendBot)            | [link](https://link.springer.com/chapter/10.1007/978-3-030-41251-7_3)                  | -                | -               | F T G | `random forest`          |
| [Kipf et al.](src/GCN_GAT)            | [link](https://arxiv.org/abs/1609.02907)                                               | 0.7839           | 0.5496          | F T G | `Graph Neural Network`   |
| [Velickovic et al.](src/GCN_GAT)      | [link](https://arxiv.org/abs/1710.10903)                                               | 0.7948           | 0.5586          | F T G | `Graph Neural Network`   |
| [GraphHist](src/GraphHist/)           | [link](https://arxiv.org/abs/1910.01180)                                               | -                | -               | F T G | `random forest`          |
| [Hayawi et al.](src/Hayawi/)          | [link](https://link.springer.com/content/pdf/10.1007/s13278-022-00869-w.pdf)           | 0.7650           | 0.2474          | F     | `lstm`                   |
| [HGT](src/HGT_SimpleHGN/)             | [link](https://arxiv.org/abs/2003.01332)                                               | 0.7491           | 0.3960          | F T G | `Graph Neural Networks`  |
| [SimpleHGN](src/HGT_SimpleHGN/)       | [link](https://arxiv.org/abs/2112.14936)                                               | 0.7672           | 0.4544          | F T G | `Graph Neural Networks`  |
| [Kantepe et al.](src/Kantepe/)        | [link](https://ieeexplore.ieee.org/abstract/document/8093483)                          | 0.7640           | 0.5870          | F T   | `random forest`          |
| [Knauth et al.](src/Knauth/)          | [link](https://aclanthology.org/R19-1065/)                                             | 0.7125           | 0.3709          | F T G | `random forest`          |
| [Kouvela et al.](src/Kouvela/)        | [link](https://dl.acm.org/doi/abs/10.1145/3415958.3433075)                             | 0.7644           | 0.3003          | F T   | `random forest`          |
| [Kudugunta et al.](src/Kudugunta/)    | [link](https://arxiv.org/abs/1802.04289)                                               | 0.6587           | 0.5167          | F     | `SMOTENN, random forest` |
| [Lee et al.](src/Lee/)                | [link](https://ojs.aaai.org/index.php/ICWSM/article/view/14106)                        | 0.7628           | 0.3041          | F T   | `random forest`          |
| [LOBO](src/LOBO/)                     | [link](https://dl.acm.org/doi/10.1145/3274694.3274738)                                 | 0.7570           | 0.3857          | F T   | `random forest`          |
| [Miller et al.](src/Miller/)          | [link](https://dl.acm.org/doi/10.1016/j.ins.2013.11.016)                               | 0.3037           | 0.4529          | F T   | `k means`                |
| [Moghaddam et al.](src/Moghaddam/)    | [link](https://ieeexplore.ieee.org/abstract/document/9735340)                          | 0.7378           | 0.3207          | F G   | `random forest`          |
| [NameBot](src/NameBot/)               | [link](https://arxiv.org/pdf/1812.05932.pdf)                                           | 0.7061           | 0.0050          | F     | `Logistic Regression`    |
| [RGT](src/RGT/)                       | [link](https://arxiv.org/abs/2109.02927)                                               | 0.7647           | 0.4294          | F T G | `Graph Neural Networks`  |
| [RoBERTa](src/RoBERTa/)               | [link](https://arxiv.org/pdf/1907.11692.pdf)                                           | 0.7207           | 0.2053          | F T   | `RoBERTa`                |
| [Rodrguez-Ruiz](src/Rodrguez-Ruiz/)   | [link](https://www.sciencedirect.com/science/article/pii/S0167404820300031)            | 0.4936           | 0.5657          | F T G | `SVM`                    |
| [Santos et al.](src/Santos/)          | [link](https://dl.acm.org/doi/pdf/10.1145/3308560.3317599)                             | -                | -               | F T   | `decision tree`          |
| [SATAR](src/SATAR/)                   | [link](https://arxiv.org/abs/2106.13089)                                               | -                | -               | F T G |                          |
| [SGBot](src/SGBot/)                   | [link](https://arxiv.org/abs/1911.09179)                                               | 0.7508           | 0.3659          | F T   | `random forest`          |
| [T5](src/T5/)                         | [link](https://aaai.org/ocs/index.php/ICWSM/ICWSM17/paper/view/15587/14817)            | 0.7205           | 0.2027          | T     | `T5`                     |
| [Varol et al.](src/Varol)             | [link](https://aaai.org/ocs/index.php/ICWSM/ICWSM17/paper/view/15587/14817)            | 0.7392           | 0.2754          | F T   | `random forest`          |
| [Wei et al.](src/Wei/)                | [link](https://arxiv.org/pdf/2002.01336.pdf)                                           | 0.7020           | 0.5360          | T     |                          |

where `-` represents the baseline could not scale to TwiBot-22 dataset

|      Precision       | Botometer-feedback-2019 |    Cresci-2015     |    Cresci-2017    | Cresci-rtbust-2019 | Cresci-stock-2018  |   Gilani-2017    |   Midterm-2018   |    Twibot-20     |    Twibot-22     |
| :------------------: | :---------------------: | :----------------: | :---------------: | :----------------: | :----------------: | :--------------: | :--------------: | :--------------: | :--------------: |
|     Abreu et al.     |    63.63 </br>$_{3.60}$     |  99.05 </br> $_{0.21}$  | 98.34 </br> $_{0.13}$  |  78.57 </br> $_{1.44}$  |  75.45 </br> $_{0.45}$  | 76.82 </br> $_{1.20}$ | 97.28 </br> $_{0.07}$ | 72.20 </br> $_{0.52}$ | 50.92 </br> $_{0.10}$ |
|  Alhosseini et al.   |    ----- $\pm$ ----     |  90.71 $\pm$ 0.24  | ----- $\pm$ ----  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 58.95 $\pm$ 16.3 | 58.10 $\pm$ 1.42 |
|        BGSRD         |    27.50 $\pm$ 28.2     |  86.52 $\pm$ 0.64  | 75.85 $\pm$ 0.00  |  58.13 $\pm$ 11.1  |  52.78 $\pm$ 0.75  | 25.43 $\pm$ 23.2 | 84.40 $\pm$ 0.93 | 67.64 $\pm$ 2.26 | 22.55 $\pm$ 30.9 |
|      BotHunter       |    ----- $\pm$ ----     |  98.55 $\pm$ 0.56  | 98.65 $\pm$ 0.05  |  81.92 $\pm$ 2.04  |  84.29 $\pm$ 0.10  | 78.99 $\pm$ 0.96 | 99.44 $\pm$ 0.15 | 72.77 $\pm$ 0.25 | 68.09 $\pm$ 0.36 |
|      Botometer       |    21.05 $\pm$ ----     |  50.54 $\pm$ ----  | 93.35 $\pm$ ----  |  65.22 $\pm$ ----  |  68.50 $\pm$ ----  | 62.99 $\pm$ ---- | 31.18 $\pm$ ---- | 55.67 $\pm$ ---- | 30.81 $\pm$ ---- |
|       BotRGCN        |    ----- $\pm$ ----     |  95.51 $\pm$ 1.02  | ----- $\pm$ ----  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 84.52 $\pm$ 0.54 | 74.81 $\pm$ 2.22 |
|       Cresci         |    ----- $\pm$ ----     |   0.59 $\pm$ ----  | 12.96 $\pm$ ----  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- |  7.66 $\pm$ ---- | ----- $\pm$ ---- |
|    Dehghan et al.    |    ----- $\pm$ ----     |  96.15 $\pm$ 0.00  | ----- $\pm$ ----  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 94.72 $\pm$ 0.00 | ----- $\pm$ ---- |
|   Efthimion et al.   |     0.00 $\pm$ 0.00     |  93.82 $\pm$ 0.00  | 94.58 $\pm$ 0.00  |  68.29 $\pm$ 0.00  |  82.75 $\pm$ 0.00  | 37.50 $\pm$ 0.00 | 98.01 $\pm$ 0.00 | 64.20 $\pm$ 0.00 | 77.78 $\pm$ 0.00 |
|      EvolveBot       |    ----- $\pm$ ----     |  85.03 $\pm$ 3.77  | ----- $\pm$ ----  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 66.93 $\pm$ 0.60 | 56.38 $\pm$ 0.40 |
|      FriendBot       |    ----- $\pm$ ----     |  95.29 $\pm$ 1.62  | 77.55 $\pm$ 0.81  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 72.64 $\pm$ 0.52 | ----- $\pm$ ---- |
|         GCN          |    ----- $\pm$ ----     |  95.59 $\pm$ 0.69  | ----- $\pm$ ----  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 75.23 $\pm$ 3.08 | 71.19 $\pm$ 1.28 |
|         GAT          |    ----- $\pm$ ----     |  96.10 $\pm$ 0.71  | ----- $\pm$ ----  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 81.39 $\pm$ 1.18 | 76.23 $\pm$ 1.39 |
|      GraphHist       |    ----- $\pm$ ----     |  73.12 $\pm$ 0.10  | ----- $\pm$ ----  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 51.27 $\pm$ 0.20 | ----- $\pm$ ---- |
|    Hayawi et al.     |    25.00 $\pm$ 0.06     |  92.96 $\pm$ 0.03  | 95.47 $\pm$ 0.01  |  48.82 $\pm$ 0.01  |  50.73 $\pm$ 0.03  | 51.44 $\pm$ 0.05 | 85.30 $\pm$ 0.00 | 71.61 $\pm$ 0.01 | 80.00 $\pm$ 0.27 |
|         HGT          |    ----- $\pm$ ----     |  94.80 $\pm$ 0.49  | ----- $\pm$ ----  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 85.55 $\pm$ 0.31 | 68.22 $\pm$ 2.71 |
|      SimpleHGN       |    ----- $\pm$ ----     |  95.68 $\pm$ 0.90  | ----- $\pm$ ----  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 84.76 $\pm$ 0.46 | 72.57 $\pm$ 2.79 |
|    Kantepe et al.    |    ----- $\pm$ ----     |  81.30 $\pm$ 1.40  | 83.00 $\pm$ 0.90  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 63.40 $\pm$ 2.10 | 78.60 $\pm$ 1.80 |
|    Knauth et al.     |    57.41 $\pm$ 0.00     |  85.70 $\pm$ 0.00  | 91.56 $\pm$ 0.00  |  57.41 $\pm$ 0.00  |  99.89 $\pm$ 0.00  | 35.17 $\pm$ 0.00 | 99.91 $\pm$ 0.00 | 96.56 $\pm$ 0.00 | ----- $\pm$ ---- |
|    Kouvela et al.    |    48.00 $\pm$ 4.47     |  99.54 $\pm$ 0.18  | 99.24 $\pm$ 0.13  |  82.27 $\pm$ 2.00  |  82.17 $\pm$ 0.46  | 79.69 $\pm$ 1.09 | 97.56 $\pm$ 0.04 | 79.33 $\pm$ 0.44 | 69.30 $\pm$ 0.14 |
|   Kudugunta et al.   |    56.67 $\pm$ 10.8     |  100.0 $\pm$ 0.00  | 98.53 $\pm$ 0.19  |  66.09 $\pm$ 2.35  |  54.87 $\pm$ 0.47  | 85.44 $\pm$ 2.42 | 99.06 $\pm$ 0.16 | 80.40 $\pm$ 0.60 | 44.31 $\pm$ 0.00 |
|      Lee et al.      |    58.97 $\pm$ 3.29     |  98.65 $\pm$ 0.14  | 99.56 $\pm$ 0.08  |  79.37 $\pm$ 2.97  |  84.75 $\pm$ 0.42  | 77.58 $\pm$ 1.31 | 97.36 $\pm$ 0.07 | 76.60 $\pm$ 0.37 | 67.23 $\pm$ 0.29 |
|         LOBO         |    ----- $\pm$ ----     |  98.47 $\pm$ 0.63  | 99.30 $\pm$ 0.08  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 74.83 $\pm$ 0.08 | 75.43 $\pm$ 0.15 |
|    Miller et al.     |     0.00 $\pm$ 0.00     |  72.07 $\pm$ 0.00  | 77.21 $\pm$ 0.18  |  52.17 $\pm$ 0.00  |  54.78 $\pm$ 0.00  | 48.89 $\pm$ 0.00 | 83.85 $\pm$ 0.00 | 60.71 $\pm$ 0.20 | 29.46 $\pm$ 0.00 |
|   Moghaddam et al.   |    ----- $\pm$ ----     |  98.33 $\pm$ 0.26  | ----- $\pm$ ----  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 72.29 $\pm$ 0.67 | 67.61 $\pm$ 0.10 |
|       NameBot        |    45.45 $\pm$ 0.00     |  76.81 $\pm$ 0.00  | 80.39 $\pm$ 0.03  |  65.00 $\pm$ 0.00  |  58.34 $\pm$ 0.00  | 58.21 $\pm$ 0.00 | 86.93 $\pm$ 0.00 | 58.72 $\pm$ 0.00 | 67.73 $\pm$ 0.00 |
|         RGT          |    ----- $\pm$ ----     |  96.38 $\pm$ 0.59  | ----- $\pm$ ----  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 85.15 $\pm$ 0.28 | 75.03 $\pm$ 0.85 |
|       RoBERTa        |    ----- $\pm$ ----     |  97.58 $\pm$ 0.27  | 92.43 $\pm$ 0.99  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 73.88 $\pm$ 1.06 | 63.28 $\pm$ 0.90 |
| Rodrguez-Ruiz et al. |    ----- $\pm$ ----     |  78.64 $\pm$ 0.00  | 79.47 $\pm$ 0.00  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 61.60 $\pm$ 0.00 | 33.23 $\pm$ 0.00 |
|    Santos et al.     |    50.00 $\pm$ 0.00     |  72.86 $\pm$ 0.00  | 81.71 $\pm$ 0.00  |  75.68 $\pm$ 0.00  |  65.39 $\pm$ 0.00  | 32.26 $\pm$ 0.00 | 88.05 $\pm$ 0.00 | 62.73 $\pm$ 0.00 | ----- $\pm$ ---- |
|        SATAR         |    ----- $\pm$ ----     |  90.66 $\pm$ 0.67  | ----- $\pm$ ----  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 81.50 $\pm$ 1.45 | ----- $\pm$ ---- |
|        SGBot         |    59.70 $\pm$ 3.91     |  99.45 $\pm$ 0.20  | 98.26 $\pm$ 0.17  |  83.08 $\pm$ 2.60  |  83.90 $\pm$ 0.29  | 82.68 $\pm$ 1.88 | 99.35 $\pm$ 0.22 | 76.40 $\pm$ 0.40 | 73.11 $\pm$ 0.18 |
|         T5           |    ----- $\pm$ ----     |  91.04 $\pm$ 0.29  | 94.48 $\pm$ 0.65  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 72.19 $\pm$ 0.84 | 63.27 $\pm$ 0.71 |
|     Varol et al.     |    ----- $\pm$ ----     |  92.22 $\pm$ 0.66  | ----- $\pm$ ----  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 78.04 $\pm$ 0.61 | 75.74 $\pm$ 0.31 |
|      Wei et al.      |    ----- $\pm$ ----     |  91.70 $\pm$ 1.70  | 85.90 $\pm$ 1.90  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 61.00 $\pm$ 2.10 | 62.70 $\pm$ 1.80 |



|       Recall         | Botometer-feedback-2019 |    Cresci-2015     |    Cresci-2017    | Cresci-rtbust-2019 | Cresci-stock-2018  |   Gilani-2017    |   Midterm-2018   |    Twibot-20     |    Twibot-22     |
| :------------------: | :---------------------: | :----------------: | :---------------: | :----------------: | :----------------: | :--------------: | :--------------: | :--------------: | :--------------: |
|     Abreu et al.     |    46.66 $\pm$ 3.00     |  62.13 $\pm$ 0.97  | 91.97 $\pm$ 0.69  |  89.18 $\pm$ 1.40  |  75.67 $\pm$ 0.73  | 58.87 $\pm$ 2.75 | 98.63 $\pm$ 0.08 | 82.81 $\pm$ 0.51 | 11.73 $\pm$ 0.06 |
|  Alhosseini et al.   |    ----- $\pm$ ----     |  86.55 $\pm$ 0.98  | ----- $\pm$ ----  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 63.17 $\pm$ 7.21 | 53.78 $\pm$ 4.19 |
|        BGSRD         |     8.57 $\pm$ 8.52     |  95.56 $\pm$ 2.02  | 100.0 $\pm$ 0.00  |  35.14 $\pm$ 20.6  |  70.40 $\pm$ 26.1  | 60.00 $\pm$ 54.8 | 97.66 $\pm$ 3.66 | 73.19 $\pm$ 7.49 | 19.90 $\pm$ 27.2 |
|      BotHunter       |    ----- $\pm$ ----     |  91.48 $\pm$ 4.16  | 85.40 $\pm$ 0.19  |  83.02 $\pm$ 2.95  |  79.92 $\pm$ 0.54  | 62.29 $\pm$ 3.47 | 99.66 $\pm$ 0.06 | 86.75 $\pm$ 0.46 | 14.07 $\pm$ 0.12 |
|      Botometer       |    57.14 $\pm$ ----     |  98.95 $\pm$ ----  | 99.69 $\pm$ ----  |  100.0 $\pm$ ----  |  94.96 $\pm$ ----  | 89.91 $\pm$ ---- | 87.88 $\pm$ ---- | 50.82 $\pm$ ---- | 69.80 $\pm$ ---- |
|       BotRGCN        |    ----- $\pm$ ----     |  99.17 $\pm$ 0.25  | ----- $\pm$ ----  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 90.19 $\pm$ 1.72 | 46.80 $\pm$ 2.76 |
|       Cresci         |    ----- $\pm$ ----     |  66.67 $\pm$ ----  | 95.30 $\pm$ ----  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 67.47 $\pm$ ---- | ----- $\pm$ ---- |
|    Dehghan et al.    |    ----- $\pm$ ----     |  83.88 $\pm$ 0.00  | ----- $\pm$ ----  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 82.19 $\pm$ 0.00 | ----- $\pm$ ---- |
|   Efthimion et al.   |     0.00 $\pm$ 0.00     |  94.38 $\pm$ 0.00  | 89.23 $\pm$ 0.00  |  75.68 $\pm$ 0.00  |  58.02 $\pm$ 0.00  |  2.80 $\pm$ 0.00 | 94.04 $\pm$ 0.00 | 70.63 $\pm$ 0.00 | 16.76 $\pm$ 0.00 |
|      EvolveBot       |    ----- $\pm$ ----     |  95.83 $\pm$ 0.66  | ----- $\pm$ ----  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 72.81 $\pm$ 0.41 |  8.04 $\pm$ 0.05 |
|      FriendBot       |    ----- $\pm$ ----     |  100.0 $\pm$ 0.00  | 100.0 $\pm$ 0.00  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 88.94 $\pm$ 0.59 | ----- $\pm$ ---- |
|         GCN          |    ----- $\pm$ ----     |  98.81 $\pm$ 0.20  | ----- $\pm$ ----  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 87.62 $\pm$ 3.31 | 44.80 $\pm$ 1.71 |
|         GAT          |    ----- $\pm$ ----     |  99.11 $\pm$ 0.51  | ----- $\pm$ ----  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 89.53 $\pm$ 0.87 | 44.12 $\pm$ 1.65 |
|      GraphHist       |    ----- $\pm$ ----     |  100.0 $\pm$ 0.00  | ----- $\pm$ ----  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 99.05 $\pm$ 0.20 | ----- $\pm$ ---- |
|    Hayawi et al.     |    17.78 $\pm$ 0.06     |  79.31 $\pm$ 0.02  | 92.19 $\pm$ 0.03  |  81.25 $\pm$ 0.09  |  71.16 $\pm$ 0.07  | 28.00 $\pm$ 0.13 | 98.64 $\pm$ 0.00 | 83.50 $\pm$ 0.04 | 14.99 $\pm$ 0.05 |
|         HGT          |    ----- $\pm$ ----     |  99.11 $\pm$ 0.12  | ----- $\pm$ ----  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 91.00 $\pm$ 0.57 | 28.03 $\pm$ 2.60 |
|      SimpleHGN       |    ----- $\pm$ ----     |  99.29 $\pm$ 0.40  | ----- $\pm$ ----  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 92.06 $\pm$ 0.51 | 32.90 $\pm$ 1.64 |
|    Kantepe et al.    |    ----- $\pm$ ----     |  75.30 $\pm$ 1.20  | 76.10 $\pm$ 1.10  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 61.00 $\pm$ 1.90 | 46.80 $\pm$ 1.30 |
|    Knauth et al.     |    59.09 $\pm$ 0.00     |  97.40 $\pm$ 0.00  | 95.35 $\pm$ 0.00  |  51.24 $\pm$ 0.00  |  88.83 $\pm$ 0.00  | 44.00 $\pm$ 0.00 | 83.99 $\pm$ 0.00 | 76.30 $\pm$ 0.00 | ----- $\pm$ ---- |
|    Kouvela et al.    |    20.00 $\pm$ 4.71     |  96.79 $\pm$ 0.75  | 98.98 $\pm$ 0.18  |  80.00 $\pm$ 1.48  |  78.78 $\pm$ 0.18  | 57.20 $\pm$ 2.42 | 98.92 $\pm$ 0.06 | 95.17 $\pm$ 0.14 | 19.17 $\pm$ 0.04 |
|   Kudugunta et al.   |    45.33 $\pm$ 8.69     |  60.95 $\pm$ 0.21  | 85.88 $\pm$ 0.37  |  50.67 $\pm$ 1.21  |  47.54 $\pm$ 0.60  | 35.14 $\pm$ 1.70 | 90.24 $\pm$ 0.66 | 33.47 $\pm$ 1.30 | 61.98 $\pm$ 0.00 |
|      Lee et al.      |    44.00 $\pm$ 3.65     |  98.46 $\pm$ 0.14  | 99.13 $\pm$ 0.00  |  86.45 $\pm$ 1.44  |  80.30 $\pm$ 0.63  | 60.19 $\pm$ 2.15 | 98.37 $\pm$ 0.10 | 83.66 $\pm$ 0.69 | 19.65 $\pm$ 0.15 |
|         LOBO         |    ----- $\pm$ ----     |  99.05 $\pm$ 0.13  | 96.13 $\pm$ 0.39  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 87.81 $\pm$ 0.37 | 25.91 $\pm$ 0.20 |
|    Miller et al.     |     0.00 $\pm$ 0.00     |  100.0 $\pm$ 0.00  | 99.11 $\pm$ 0.11  |  37.50 $\pm$ 0.00  |  58.89 $\pm$ 0.00  | 77.19 $\pm$ 0.00 | 99.81 $\pm$ 0.00 | 97.44 $\pm$ 0.47 | 97.89 $\pm$ 0.01 |
|   Moghaddam et al.   |    ----- $\pm$ ----     |  59.23 $\pm$ 0.32  | ----- $\pm$ ----  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 84.38 $\pm$ 1.03 | 21.02 $\pm$ 0.07 |
|       NameBot        |    33.33 $\pm$ 0.00     |  91.12 $\pm$ 0.00  | 91.79 $\pm$ 0.00  |  70.27 $\pm$ 0.00  |  64.13 $\pm$ 0.00  | 36.45 $\pm$ 0.00 | 96.82 $\pm$ 0.00 | 70.47 $\pm$ 0.00 |  0.03 $\pm$ 0.00 |
|         RGT          |    ----- $\pm$ ----     |  99.23 $\pm$ 0.15  | ----- $\pm$ ----  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 91.06 $\pm$ 0.80 | 30.10 $\pm$ 0.17 |
|       RoBERTa        |    ----- $\pm$ ----     |  94.11 $\pm$ 0.58  | 96.27 $\pm$ 1.05  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 72.38 $\pm$ 2.05 | 12.27 $\pm$ 1.22 |
| Rodrguez-Ruiz et al. |    ----- $\pm$ ----     |  99.11 $\pm$ 0.00  | 92.88 $\pm$ 0.00  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 98.75 $\pm$ 0.00 | 81.32 $\pm$ 0.00 |
|    Santos et al.     |    13.33 $\pm$ 0.00     |  85.80 $\pm$ 0.00  | 84.40 $\pm$ 0.00  |  75.68 $\pm$ 0.00  |  64.95 $\pm$ 0.00  |  9.35 $\pm$ 0.04 | 97.24 $\pm$ 0.00 | 58.13 $\pm$ 0.00 | ----- $\pm$ ---- |
|        SATAR         |    ----- $\pm$ ----     |  99.88 $\pm$ 0.16  | ----- $\pm$ ----  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 91.22 $\pm$ 1.82 | ----- $\pm$ ---- |
|        SGBot         |    45.33 $\pm$ 2.98     |  63.67 $\pm$ 1.31  | 90.86 $\pm$ 0.39  |  81.62 $\pm$ 2.26  |  81.03 $\pm$ 0.90  | 63.62 $\pm$ 2.17 | 99.66 $\pm$ 0.20 | 94.91 $\pm$ 0.69 | 24.32 $\pm$ 0.09 |
|         T5           |    ----- $\pm$ ----     |  87.71 $\pm$ 0.66  | 90.26 $\pm$ 0.54  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 69.05 $\pm$ 1.46 | 12.09 $\pm$ 1.43 |
|     Varol et al.     |    ----- $\pm$ ----     |  97.40 $\pm$ 0.90  | ----- $\pm$ ----  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 84.37 $\pm$ 0.67 | 16.83 $\pm$ 0.21 |
|      Wei et al.      |    ----- $\pm$ ----     |  75.30 $\pm$ 1.50  | 72.10 $\pm$ 1.50  |  ----- $\pm$ ----  |  ----- $\pm$ ----  | ----- $\pm$ ---- | ----- $\pm$ ---- | 54.00 $\pm$ 2.70 | 46.80 $\pm$ 1.40 |





### Citation
Please cite [TwiBot-22](https://arxiv.org/abs/2206.04564) if you use the TwiBot-22 dataset or this repository
```
@article{feng2022twibot,
  title={TwiBot-22: Towards Graph-Based Twitter Bot Detection},
  author={Feng, Shangbin and Tan, Zhaoxuan and Wan, Herun and Wang, Ningnan and Chen, Zilong and Zhang, Binchi and Zheng, Qinghua and Zhang, Wenqian and Lei, Zhenyu and Yang, Shujie and Feng, Xinshun and Zhang, Qingyue and Wang, Hongrui and Liu, Yuhan and Bai, Yuyang and Wang, Heng and Cai, Zijian and Wang, Yanbo and Zheng, Lijing and Ma, Zihan and Li, Jundong and Luo, Minnan},
  journal={arXiv preprint arXiv:2206.04564},
  year={2022}
}
```

### How to contribute

1. New dataset: convert the original data to the [TwiBot-22 defined schema](descriptions/metadata.md).
2. New baseline: load well-formatted dataset from the dataset directory and define your model.

Welcome PR!

### Questions?

Feel free to open issues in this repository! Instead of emails, Github issues are much better at facilitating a conversation between you and our team to address your needs. You can also contact the project lead Shangbin Feng through ``shangbin at cs.washington.edu``.
