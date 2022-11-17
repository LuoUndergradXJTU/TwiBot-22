# TwiBot-22

This is the official repository of [TwiBot-22](https://twibot22.github.io/) @ NeurIPS 2022, Datasets and Benchmarks Track.

### Introduction

TwiBot-22 is the largest and most comprehensive Twitter bot detection benchmark to date. Specifically, [TwiBot-22](https://arxiv.org/abs/2206.04564) is designed to address the challenges of limited dataset scale, imcomplete graph structure, and low annotation quality in previous datasets. For more details, please refer to the [TwiBot-22 paper](https://arxiv.org/abs/2206.04564) and [statistics](descriptions/statistics.md).
![compare](./pics/compare.png)

### Dataset Format

Each dataset contains `node.json` (or `tweet.json`, `user.json`, `list.json`, and `hashtag.json` for TwiBot-22), `label.csv`, `split.csv` and `edge.csv` (for datasets with graph structure). See [here](descriptions/metadata.md) for a detailed description of these files.

### How to download TwiBot-22 dataset

TwiBot-22 is available at [Google Drive](https://drive.google.com/drive/folders/1YwiOUwtl8pCd2GD97Q_WEzwEUtSPoxFs?usp=sharing).

Please apply for access by contacting shangbin at cs.washington.edu with your **institutional email address** and clearly state your institution, your research advisor (if any), and your use case of TwiBot-22.

### How to download other datasets

For TwiBot-20, visit the [TwiBot-20 github repository](https://github.com/BunsenFeng/TwiBot-20).

For other datasets, please visit the [Bot Repository](https://botometer.osome.iu.edu/bot-repository/datasets.html).

After downloading these datasets, you can transform them into the 4-file format detailed in "Dataset Format". Alternatively, you can directly download our preprocessed version:

For TwiBot-20, visit the [TwiBot-20 github repository](https://github.com/BunsenFeng/TwiBot-20), apply for TwiBot-20 access, and there will be a `TwiBot-20-Format22.zip` in the TwiBot-20 Google Drive link.

For other datasets, you can directly download them from [Google Drive](https://drive.google.com/drive/folders/1gXFZp3m7TTU-wyZRUiLHdf_sIZpISrze?usp=sharing). You should adhere to the license of each dataset, the "Content redistribution" section of the [Twitter Developer Agreement and Policy](https://developer.twitter.com/en/developer-terms/agreement-and-policy), the rules set by the [Bot Repository](https://botometer.osome.iu.edu/bot-repository/datasets.html), and only use these datasets for research purposes.

### Requirements

- pip: `pip install -r requirements.txt`
- conda : `conda install --yes --file requirements.txt `

### How to run baselines

1. clone this repo by running `git clone https://github.com/LuoUndergradXJTU/TwiBot-22.git`
2. make dataset directory `mkdir datasets` and download datasets to `./datasets`
3. change directory to `src/{name_of_the_baseline}`
4. run experiments under the guidance of corresponding `readme.md`

### Baseline Overview


| baseline                              | paper                                                        | acc on Twibot-22 | f1 on Twibot-22 | type  | tags                     |
| ------------------------------------- | ------------------------------------------------------------ | ---------------- | --------------- | ----- | ------------------------ |
| [Abreu et al.](src/Abreu/)            | [link](https://ieeexplore.ieee.org/abstract/document/9280525) | 0.7066           | 0.5344          | F     | `random forest`          |
| [Alhosseini et al.](src/Alhosseini/)  | [link](https://dl.acm.org/doi/fullHtml/10.1145/3308560.3316504) | 0.4772           | 0.3810          | F G   | `gcn`                    |
| [BGSRD](src/BGSRD/)                   | [link](https://www.mdpi.com/2073-8994/14/1/30)               | 0.7188           | 0.2114          | F     | `BERT GAT`               |
| [Bot Hunter](src/BotHunter/)          | [link](http://www.casos.cs.cmu.edu/publications/papers/LB_5.pdf) | 0.7279           | 0.2346          | F     | `random forest`          |
| [Botometer](src/Botometer/)           | [link](https://botometer.osome.iu.edu/)                      | 0.4987           | 0.4257          | F T G |                          |
| [BotRGCN](src/BotRGCN/)               | [link](https://arxiv.org/abs/2106.13092)                     | 0.7966           | 0.5750          | F T G | `BotRGCN`                |
| [Cresci et al.](src/Cresci/)          | [link](https://ieeexplore.ieee.org/abstract/document/7436643) | -                | -               | T     | `DNA`                    |
| [Dehghan et al.](src/Dehghan)         | [link](https://assets.researchsquare.com/files/rs-1428343/v1_covered.pdf?c=1647280648) | -                | -               | F T G | `Graph`                  |
| [Efthimion et al.](src/Efthimion/)    | [link](https://scholar.smu.edu/datasciencereview/vol1/iss2/5/) | 0.7408           | 0.2758          | F T   | `efthimion`              |
| [EvolveBot](src/EvolveBot/)           | [link](https://ieeexplore.ieee.org/abstract/document/6553246) | 0.7109           | 0.1409          | F T G | `random forest`          |
| [FriendBot](src/FriendBot)            | [link](https://link.springer.com/chapter/10.1007/978-3-030-41251-7_3) | -                | -               | F T G | `random forest`          |
| [Kipf et al.](src/GCN_GAT)            | [link](https://arxiv.org/abs/1609.02907)                     | 0.7839           | 0.5496          | F T G | `Graph Neural Network`   |
| [Velickovic et al.](src/GCN_GAT)      | [link](https://arxiv.org/abs/1710.10903)                     | 0.7948           | 0.5586          | F T G | `Graph Neural Network`   |
| [GraphHist](src/GraphHist/)           | [link](https://arxiv.org/abs/1910.01180)                     | -                | -               | F T G | `random forest`          |
| [Hayawi et al.](src/Hayawi/)          | [link](https://link.springer.com/content/pdf/10.1007/s13278-022-00869-w.pdf) | 0.7650           | 0.2474          | F     | `lstm`                   |
| [HGT](src/HGT_SimpleHGN/)             | [link](https://arxiv.org/abs/2003.01332)                     | 0.7491           | 0.3960          | F T G | `Graph Neural Networks`  |
| [SimpleHGN](src/HGT_SimpleHGN/)       | [link](https://arxiv.org/abs/2112.14936)                     | 0.7672           | 0.4544          | F T G | `Graph Neural Networks`  |
| [Kantepe et al.](src/Kantepe/)        | [link](https://ieeexplore.ieee.org/abstract/document/8093483) | 0.7640           | 0.5870          | F T   | `random forest`          |
| [Knauth et al.](src/Knauth/)          | [link](https://aclanthology.org/R19-1065/)                   | 0.7125           | 0.3709          | F T G | `random forest`          |
| [Kouvela et al.](src/Kouvela/)        | [link](https://dl.acm.org/doi/abs/10.1145/3415958.3433075)   | 0.7644           | 0.3003          | F T   | `random forest`          |
| [Kudugunta et al.](src/Kudugunta/)    | [link](https://arxiv.org/abs/1802.04289)                     | 0.6587           | 0.5167          | F     | `SMOTENN, random forest` |
| [Lee et al.](src/Lee/)                | [link](https://ojs.aaai.org/index.php/ICWSM/article/view/14106) | 0.7628           | 0.3041          | F T   | `random forest`          |
| [LOBO](src/LOBO/)                     | [link](https://dl.acm.org/doi/10.1145/3274694.3274738)       | 0.7570           | 0.3857          | F T   | `random forest`          |
| [Miller et al.](src/Miller/)          | [link](https://dl.acm.org/doi/10.1016/j.ins.2013.11.016)     | 0.3037           | 0.4529          | F T   | `k means`                |
| [Moghaddam et al.](src/Moghaddam/)    | [link](https://ieeexplore.ieee.org/abstract/document/9735340) | 0.7378           | 0.3207          | F G   | `random forest`          |
| [NameBot](src/NameBot/)               | [link](https://arxiv.org/pdf/1812.05932.pdf)                 | 0.7061           | 0.0050          | F     | `Logistic Regression`    |
| [RGT](src/RGT/)                       | [link](https://arxiv.org/abs/2109.02927)                     | 0.7647           | 0.4294          | F T G | `Graph Neural Networks`  |
| [RoBERTa](src/RoBERTa/)               | [link](https://arxiv.org/pdf/1907.11692.pdf)                 | 0.7207           | 0.2053          | F T   | `RoBERTa`                |
| [Rodriguez-Ruiz](src/Rodriguez-Ruiz/) | [link](https://www.sciencedirect.com/science/article/pii/S0167404820300031) | 0.4936           | 0.5657          | F T G | `SVM`                    |
| [Santos et al.](src/Santos/)          | [link](https://dl.acm.org/doi/pdf/10.1145/3308560.3317599)   | -                | -               | F T   | `decision tree`          |
| [SATAR](src/SATAR/)                   | [link](https://arxiv.org/abs/2106.13089)                     | -                | -               | F T G |                          |
| [SGBot](src/SGBot/)                   | [link](https://arxiv.org/abs/1911.09179)                     | 0.7508           | 0.3659          | F T   | `random forest`          |
| [T5](src/T5/)                         | [link](https://aaai.org/ocs/index.php/ICWSM/ICWSM17/paper/view/15587/14817) | 0.7205           | 0.2027          | T     | `T5`                     |
| [Varol et al.](src/Varol)             | [link](https://aaai.org/ocs/index.php/ICWSM/ICWSM17/paper/view/15587/14817) | 0.7392           | 0.2754          | F T   | `random forest`          |
| [Wei et al.](src/Wei/)                | [link](https://arxiv.org/pdf/2002.01336.pdf)                 | 0.7020           | 0.5360          | T     |                          |

where `-` represents the baseline could not scale to TwiBot-22 dataset

#### Precision

|      Precision       | Botometer-feedback-2019 |    Cresci-2015     |    Cresci-2017    | Cresci-rtbust-2019 | Cresci-stock-2018  |   Gilani-2017    |   Midterm-2018   |    Twibot-20     |    Twibot-22     |
| :------------------: | :---------------------: | :----------------: | :---------------: | :----------------: | :----------------: | :--------------: | :--------------: | :--------------: | :--------------: |
|     Abreu et al.     |    63.63 </br>$_{3.60}$     |  99.05 </br> $_{0.21}$  | 98.34 </br> $_{0.13}$  |  78.57 </br> $_{1.44}$  |  75.45 </br> $_{0.45}$  | 76.82 </br> $_{1.20}$ | 97.28 </br> $_{0.07}$ | 72.20 </br> $_{0.52}$ | 50.92 </br> $_{0.10}$ |
|  Alhosseini et al.   |            -            |  87.69 </br> $_{1.23}$  |         -         |          -         |          -         |         -        |         -        | 57.81 </br> $_{0.43}$ | 29.99 </br> $_{3.08}$ |
|        BGSRD         |    27.50 </br> $_{28.2}$     |  86.52 </br> $_{0.64}$  | 75.85 </br> $_{0.00}$  |  58.13 </br> $_{11.1}$  |  52.78 </br> $_{0.75}$  | 25.43 </br> $_{23.2}$ | 84.40 </br> $_{0.93}$ | 67.64 </br> $_{2.26}$ | 22.55 </br> $_{30.9}$ |
|      BotHunter       |            -            |  98.55 </br> $_{0.56}$  | 98.65 </br> $_{0.05}$  |  81.92 </br> $_{2.04}$  |  84.29 </br> $_{0.10}$  | 78.99 </br> $_{0.96}$ | 99.44 </br> $_{0.15}$ | 72.77 </br> $_{0.25}$ | 68.09 </br> $_{0.36}$ |
|      Botometer       |    21.05 </br> -     |  50.54 </br> -  | 93.35 </br> -  |  65.22 </br> -  |  68.50 </br> -  | 62.99 </br> - | 31.18 </br> - | 55.67 </br> - | 30.81 </br> - |
|       BotRGCN        |            -            |  95.51 </br> $_{1.02}$  |         -         |          -         |          -         |         -        |         -        | 84.52 </br> $_{0.54}$ | 74.81 </br> $_{2.22}$ |
|       Cresci         |            -            |   0.59 </br> -  | 12.96 </br> -  |          -         |          -         |         -        |         -        |  7.66 </br> - |         -        |
|    Dehghan et al.    |            -            |  96.15 </br> $_{0.00}$  |         -         |          -         |          -         |         -        |         -        | 94.72 </br> $_{0.00}$ |         -        |
|   Efthimion et al.   |     0.00 </br> $_{0.00}$     |  93.82 </br> $_{0.00}$  | 94.58 </br> $_{0.00}$  |  68.29 </br> $_{0.00}$  |  82.75 </br> $_{0.00}$  | 37.50 </br> $_{0.00}$ | 98.01 </br> $_{0.00}$ | 64.20 </br> $_{0.00}$ | 77.78 </br> $_{0.00}$ |
|      EvolveBot       |            -            |  85.03 </br> $_{3.77}$  |         -         |          -         |          -         |         -        |         -        | 66.93 </br> $_{0.60}$ | 56.38 </br> $_{0.40}$ |
|      FriendBot       |            -            |  95.29 </br> $_{1.62}$  | 77.55 </br> $_{0.81}$  |          -         |          -         |         -        |         -        | 72.64 </br> $_{0.52}$ |         -        |
|         GCN          |            -            |  95.59 </br> $_{0.69}$  |         -         |          -         |          -         |         -        |         -        | 75.23 </br> $_{3.08}$ | 71.19 </br> $_{1.28}$ |
|         GAT          |            -            |  96.10 </br> $_{0.71}$  |         -         |          -         |          -         |         -        |         -        | 81.39 </br> $_{1.18}$ | 76.23 </br> $_{1.39}$ |
|      GraphHist       |            -            |  73.12 </br> $_{0.10}$  |         -         |          -         |          -         |         -        |         -        | 51.27 </br> $_{0.20}$ |         -        |
|    Hayawi et al.     |    25.00 </br> $_{0.06}$     |  92.96 </br> $_{0.03}$  | 95.47 </br> $_{0.01}$  |  48.82 </br> $_{0.01}$  |  50.73 </br> $_{0.03}$  | 51.44 </br> $_{0.05}$ | 85.30 </br> $_{0.00}$ | 71.61 </br> $_{0.01}$ | 80.00 </br> $_{0.27}$ |
|         HGT          |            -            |  94.80 </br> $_{0.49}$  |         -         |          -         |          -         |         -        |         -        | 85.55 </br> $_{0.31}$ | 68.22 </br> $_{2.71}$ |
|      SimpleHGN       |            -            |  95.68 </br> $_{0.90}$  |         -         |          -         |          -         |         -        |         -        | 84.76 </br> $_{0.46}$ | 72.57 </br> $_{2.79}$ |
|    Kantepe et al.    |            -            |  81.30 </br> $_{1.40}$  | 83.00 </br> $_{0.90}$  |          -         |          -         |         -        |         -        | 63.40 </br> $_{2.10}$ | 78.60 </br> $_{1.80}$ |
|    Knauth et al.     |    57.41 </br> $_{0.00}$     |  85.70 </br> $_{0.00}$  | 91.56 </br> $_{0.00}$  |  57.41 </br> $_{0.00}$  |  99.89 </br> $_{0.00}$  | 35.17 </br> $_{0.00}$ | 99.91 </br> $_{0.00}$ | 96.56 </br> $_{0.00}$ |         -        |
|    Kouvela et al.    |    48.00 </br> $_{4.47}$     |  99.54 </br> $_{0.18}$  | 99.24 </br> $_{0.13}$  |  82.27 </br> $_{2.00}$  |  82.17 </br> $_{0.46}$  | 79.69 </br> $_{1.09}$ | 97.56 </br> $_{0.04}$ | 79.33 </br> $_{0.44}$ | 69.30 </br> $_{0.14}$ |
|   Kudugunta et al.   |    56.67 </br> $_{10.8}$     |  100.0 </br> $_{0.00}$  | 98.53 </br> $_{0.19}$  |  66.09 </br> $_{2.35}$  |  54.87 </br> $_{0.47}$  | 85.44 </br> $_{2.42}$ | 99.06 </br> $_{0.16}$ | 80.40 </br> $_{0.60}$ | 44.31 </br> $_{0.00}$ |
|      Lee et al.      |    58.97 </br> $_{3.29}$     |  98.65 </br> $_{0.14}$  | 99.56 </br> $_{0.08}$  |  79.37 </br> $_{2.97}$  |  84.75 </br> $_{0.42}$  | 77.58 </br> $_{1.31}$ | 97.36 </br> $_{0.07}$ | 76.60 </br> $_{0.37}$ | 67.23 </br> $_{0.29}$ |
|         LOBO         |            -            |  98.47 </br> $_{0.63}$  | 99.30 </br> $_{0.08}$  |          -         |          -         |         -        |         -        | 74.83 </br> $_{0.08}$ | 75.43 </br> $_{0.15}$ |
|    Miller et al.     |     0.00 </br> $_{0.00}$     |  72.07 </br> $_{0.00}$  | 77.21 </br> $_{0.18}$  |  52.17 </br> $_{0.00}$  |  54.78 </br> $_{0.00}$  | 48.89 </br> $_{0.00}$ | 83.85 </br> $_{0.00}$ | 60.71 </br> $_{0.20}$ | 29.46 </br> $_{0.00}$ |
|   Moghaddam et al.   |            -            |  98.33 </br> $_{0.26}$  |         -         |          -         |          -         |         -        |         -        | 72.29 </br> $_{0.67}$ | 67.61 </br> $_{0.10}$ |
|       NameBot        |    45.45 </br> $_{0.00}$     |  76.81 </br> $_{0.00}$  | 80.39 </br> $_{0.03}$  |  65.00 </br> $_{0.00}$  |  58.34 </br> $_{0.00}$  | 58.21 </br> $_{0.00}$ | 86.93 </br> $_{0.00}$ | 58.72 </br> $_{0.00}$ | 67.73 </br> $_{0.00}$ |
|         RGT          |            -            |  96.38 </br> $_{0.59}$  |         -         |          -         |          -         |         -        |         -        | 85.15 </br> $_{0.28}$ | 75.03 </br> $_{0.85}$ |
|       RoBERTa        |            -            |  97.58 </br> $_{0.27}$  | 92.43 </br> $_{0.99}$  |          -         |          -         |         -        |         -        | 73.88 </br> $_{1.06}$ | 63.28 </br> $_{0.90}$ |
| Rodriguez-Ruiz et al. |            -            |  78.64 </br> $_{0.00}$  | 79.47 </br> $_{0.00}$  |          -         |          -         |         -        |         -        | 61.60 </br> $_{0.00}$ | 33.23 </br> $_{0.00}$ |
|    Santos et al.     |    50.00 </br> $_{0.00}$     |  72.86 </br> $_{0.00}$  | 81.71 </br> $_{0.00}$  |  75.68 </br> $_{0.00}$  |  65.39 </br> $_{0.00}$  | 32.26 </br> $_{0.00}$ | 88.05 </br> $_{0.00}$ | 62.73 </br> $_{0.00}$ |         -        |
|        SATAR         |            -            |  90.66 </br> $_{0.67}$  |         -         |          -         |          -         |         -        |         -        | 81.50 </br> $_{1.45}$ |         -        |
|        SGBot         |    59.70 </br> $_{3.91}$     |  99.45 </br> $_{0.20}$  | 98.26 </br> $_{0.17}$  |  83.08 </br> $_{2.60}$  |  83.90 </br> $_{0.29}$  | 82.68 </br> $_{1.88}$ | 99.35 </br> $_{0.22}$ | 76.40 </br> $_{0.40}$ | 73.11 </br> $_{0.18}$ |
|         T5           |            -            |  91.04 </br> $_{0.29}$  | 94.48 </br> $_{0.65}$  |          -         |          -         |         -        |         -        | 72.19 </br> $_{0.84}$ | 63.27 </br> $_{0.71}$ |
|     Varol et al.     |            -            |  92.22 </br> $_{0.66}$  |         -         |          -         |          -         |         -        |         -        | 78.04 </br> $_{0.61}$ | 75.74 </br> $_{0.31}$ |
|      Wei et al.      |            -            |  91.70 </br> $_{1.70}$  | 85.90 </br> $_{1.90}$  |          -         |          -         |         -        |         -        | 61.00 </br> $_{2.10}$ | 62.70 </br> $_{1.80}$ |

#### Recall

|       Recall         | Botometer-feedback-2019 |    Cresci-2015     |    Cresci-2017    | Cresci-rtbust-2019 | Cresci-stock-2018  |   Gilani-2017    |   Midterm-2018   |    Twibot-20     |    Twibot-22     |
| :------------------: | :---------------------: | :----------------: | :---------------: | :----------------: | :----------------: | :--------------: | :--------------: | :--------------: | :--------------: |
|     Abreu et al.     |    46.66 </br> $_{3.00}$     |  62.13 </br> $_{0.97}$  | 91.97 </br> $_{0.69}$  |  89.18 </br> $_{1.40}$  |  75.67 </br> $_{0.73}$  | 58.87 </br> $_{2.75}$ | 98.63 </br> $_{0.08}$ | 82.81 </br> $_{0.51}$ | 11.73 </br> $_{0.06}$ |
|  Alhosseini et al.   |            -            |  97.16 </br> $_{0.81}$  |         -         |          -         |          -         |         -        |         -        | 95.69 </br> $_{1.93}$ | 56.75 </br> $_{17.69}$ |
|        BGSRD         |     8.57 </br> $_{8.52}$     |  95.56 </br> $_{2.02}$  | 100.0 </br> $_{0.00}$  |  35.14 </br> $_{20.6}$  |  70.40 </br> $_{26.1}$  | 60.00 </br> $_{54.8}$ | 97.66 </br> $_{3.66}$ | 73.19 </br> $_{7.49}$ | 19.90 </br> $_{27.2}$ |
|      BotHunter       |            -            |  91.48 </br> $_{4.16}$  | 85.40 </br> $_{0.19}$  |  83.02 </br> $_{2.95}$  |  79.92 </br> $_{0.54}$  | 62.29 </br> $_{3.47}$ | 99.66 </br> $_{0.06}$ | 86.75 </br> $_{0.46}$ | 14.07 </br> $_{0.12}$ |
|      Botometer       |    57.14 </br> -     |  98.95 </br> -  | 99.69 </br> -  |  100.0 </br> -  |  94.96 </br> -  | 89.91 </br> - | 87.88 </br> - | 50.82 </br> - | 69.80 </br> - |
|       BotRGCN        |            -            |  99.17 </br> $_{0.25}$  |         -         |          -         |          -         |         -        |         -        | 90.19 </br> $_{1.72}$ | 46.80 </br> $_{2.76}$ |
|       Cresci         |            -            |  66.67 </br> -  | 95.30 </br> -  |          -         |          -         |         -        |         -        | 67.47 </br> - |         -        |
|    Dehghan et al.    |            -            |  83.88 </br> $_{0.00}$  |         -         |          -         |          -         |         -        |         -        | 82.19 </br> $_{0.00}$ |         -        |
|   Efthimion et al.   |     0.00 </br> $_{0.00}$     |  94.38 </br> $_{0.00}$  | 89.23 </br> $_{0.00}$  |  75.68 </br> $_{0.00}$  |  58.02 </br> $_{0.00}$  |  2.80 </br> $_{0.00}$ | 94.04 </br> $_{0.00}$ | 70.63 </br> $_{0.00}$ | 16.76 </br> $_{0.00}$ |
|      EvolveBot       |            -            |  95.83 </br> $_{0.66}$  |         -         |          -         |          -         |         -        |         -        | 72.81 </br> $_{0.41}$ |  8.04 </br> $_{0.05}$ |
|      FriendBot       |            -            |  100.0 </br> $_{0.00}$  | 100.0 </br> $_{0.00}$  |          -         |          -         |         -        |         -        | 88.94 </br> $_{0.59}$ |         -        |
|         GCN          |            -            |  98.81 </br> $_{0.20}$  |         -         |          -         |          -         |         -        |         -        | 87.62 </br> $_{3.31}$ | 44.80 </br> $_{1.71}$ |
|         GAT          |            -            |  99.11 </br> $_{0.51}$  |         -         |          -         |          -         |         -        |         -        | 89.53 </br> $_{0.87}$ | 44.12 </br> $_{1.65}$ |
|      GraphHist       |            -            |  100.0 </br> $_{0.00}$  |         -         |          -         |          -         |         -        |         -        | 99.05 </br> $_{0.20}$ |         -        |
|    Hayawi et al.     |    17.78 </br> $_{0.06}$     |  79.31 </br> $_{0.02}$  | 92.19 </br> $_{0.03}$  |  81.25 </br> $_{0.09}$  |  71.16 </br> $_{0.07}$  | 28.00 </br> $_{0.13}$ | 98.64 </br> $_{0.00}$ | 83.50 </br> $_{0.04}$ | 14.99 </br> $_{0.05}$ |
|         HGT          |            -            |  99.11 </br> $_{0.12}$  |         -         |          -         |          -         |         -        |         -        | 91.00 </br> $_{0.57}$ | 28.03 </br> $_{2.60}$ |
|      SimpleHGN       |            -            |  99.29 </br> $_{0.40}$  |         -         |          -         |          -         |         -        |         -        | 92.06 </br> $_{0.51}$ | 32.90 </br> $_{1.64}$ |
|    Kantepe et al.    |            -            |  75.30 </br> $_{1.20}$  | 76.10 </br> $_{1.10}$  |          -         |          -         |         -        |         -        | 61.00 </br> $_{1.90}$ | 46.80 </br> $_{1.30}$ |
|    Knauth et al.     |    59.09 </br> $_{0.00}$     |  97.40 </br> $_{0.00}$  | 95.35 </br> $_{0.00}$  |  51.24 </br> $_{0.00}$  |  88.83 </br> $_{0.00}$  | 44.00 </br> $_{0.00}$ | 83.99 </br> $_{0.00}$ | 76.30 </br> $_{0.00}$ |         -        |
|    Kouvela et al.    |    20.00 </br> $_{4.71}$     |  96.79 </br> $_{0.75}$  | 98.98 </br> $_{0.18}$  |  80.00 </br> $_{1.48}$  |  78.78 </br> $_{0.18}$  | 57.20 </br> $_{2.42}$ | 98.92 </br> $_{0.06}$ | 95.17 </br> $_{0.14}$ | 19.17 </br> $_{0.04}$ |
|   Kudugunta et al.   |    45.33 </br> $_{8.69}$     |  60.95 </br> $_{0.21}$  | 85.88 </br> $_{0.37}$  |  50.67 </br> $_{1.21}$  |  47.54 </br> $_{0.60}$  | 35.14 </br> $_{1.70}$ | 90.24 </br> $_{0.66}$ | 33.47 </br> $_{1.30}$ | 61.98 </br> $_{0.00}$ |
|      Lee et al.      |    44.00 </br> $_{3.65}$     |  98.46 </br> $_{0.14}$  | 99.13 </br> $_{0.00}$  |  86.45 </br> $_{1.44}$  |  80.30 </br> $_{0.63}$  | 60.19 </br> $_{2.15}$ | 98.37 </br> $_{0.10}$ | 83.66 </br> $_{0.69}$ | 19.65 </br> $_{0.15}$ |
|         LOBO         |            -            |  99.05 </br> $_{0.13}$  | 96.13 </br> $_{0.39}$  |          -         |          -         |         -        |         -        | 87.81 </br> $_{0.37}$ | 25.91 </br> $_{0.20}$ |
|    Miller et al.     |     0.00 </br> $_{0.00}$     |  100.0 </br> $_{0.00}$  | 99.11 </br> $_{0.11}$  |  37.50 </br> $_{0.00}$  |  58.89 </br> $_{0.00}$  | 77.19 </br> $_{0.00}$ | 99.81 </br> $_{0.00}$ | 97.44 </br> $_{0.47}$ | 97.89 </br> $_{0.01}$ |
|   Moghaddam et al.   |            -            |  59.23 </br> $_{0.32}$  |         -         |          -         |          -         |         -        |         -        | 84.38 </br> $_{1.03}$ | 21.02 </br> $_{0.07}$ |
|       NameBot        |    33.33 </br> $_{0.00}$     |  91.12 </br> $_{0.00}$  | 91.79 </br> $_{0.00}$  |  70.27 </br> $_{0.00}$  |  64.13 </br> $_{0.00}$  | 36.45 </br> $_{0.00}$ | 96.82 </br> $_{0.00}$ | 70.47 </br> $_{0.00}$ |  0.03 </br> $_{0.00}$ |
|         RGT          |            -            |  99.23 </br> $_{0.15}$  |         -         |          -         |          -         |         -        |         -        | 91.06 </br> $_{0.80}$ | 30.10 </br> $_{0.17}$ |
|       RoBERTa        |            -            |  94.11 </br> $_{0.58}$  | 96.27 </br> $_{1.05}$  |          -         |          -         |         -        |         -        | 72.38 </br> $_{2.05}$ | 12.27 </br> $_{1.22}$ |
| Rodriguez-Ruiz et al. |            -            |  99.11 </br> $_{0.00}$  | 92.88 </br> $_{0.00}$  |          -         |          -         |         -        |         -        | 98.75 </br> $_{0.00}$ | 81.32 </br> $_{0.00}$ |
|    Santos et al.     |    13.33 </br> $_{0.00}$     |  85.80 </br> $_{0.00}$  | 84.40 </br> $_{0.00}$  |  75.68 </br> $_{0.00}$  |  64.95 </br> $_{0.00}$  |  9.35 </br> $_{0.04}$ | 97.24 </br> $_{0.00}$ | 58.13 </br> $_{0.00}$ |         -        |
|        SATAR         |            -            |  99.88 </br> $_{0.16}$  |         -         |          -         |          -         |         -        |         -        | 91.22 </br> $_{1.82}$ |         -        |
|        SGBot         |    45.33 </br> $_{2.98}$     |  63.67 </br> $_{1.31}$  | 90.86 </br> $_{0.39}$  |  81.62 </br> $_{2.26}$  |  81.03 </br> $_{0.90}$  | 63.62 </br> $_{2.17}$ | 99.66 </br> $_{0.20}$ | 94.91 </br> $_{0.69}$ | 24.32 </br> $_{0.09}$ |
|         T5           |            -            |  87.71 </br> $_{0.66}$  | 90.26 </br> $_{0.54}$  |          -         |          -         |         -        |         -        | 69.05 </br> $_{1.46}$ | 12.09 </br> $_{1.43}$ |
|     Varol et al.     |            -            |  97.40 </br> $_{0.90}$  |         -         |          -         |          -         |         -        |         -        | 84.37 </br> $_{0.67}$ | 16.83 </br> $_{0.21}$ |
|      Wei et al.      |            -            |  75.30 </br> $_{1.50}$  | 72.10 </br> $_{1.50}$  |          -         |          -         |         -        |         -        | 54.00 </br> $_{2.70}$ | 46.80 </br> $_{1.40}$ |

#### F1

|         F1           | Botometer-feedback-2019 |    Cresci-2015     |    Cresci-2017    | Cresci-rtbust-2019 | Cresci-stock-2018  |   Gilani-2017    |   Midterm-2018   |    Twibot-20     |    Twibot-22     |
| :------------------: | :---------------------: | :----------------: | :---------------: | :----------------: | :----------------: | :--------------: | :--------------: | :--------------: | :--------------: |
|     Abreu et al.     |    53.84 </br> $_{3.03}$     |  76.36 </br> $_{0.72}$  | 95.04 </br> $_{0.30}$  |  83.54 </br> $_{1.04}$  |  76.93 </br> $_{0.58}$  | 66.66 </br> $_{0.10}$ | 97.95 </br> $_{0.03}$ | 77.14 </br> $_{0.46}$ | 53.44 </br> $_{0.09}$ |
|  Alhosseini et al.   |    -     |  92.17 </br> $_{0.36}$  | -  |  -  |  -  | - | - | 72.07 </br> $_{0.48}$ | 38.10 </br> $_{5.93}$ |
|        BGSRD         |    13.03 </br> $_{13.0}$     |  90.80 </br> $_{0.60}$  | 86.27 </br> $_{0.00}$  |  41.08 </br> $_{13.0}$  |  58.18 </br> $_{12.1}$  | 35.72 </br> $_{32.6}$ | 90.50 </br> $_{1.09}$ | 70.05 </br> $_{2.60}$ | 21.14 </br> $_{29.0}$ |
|      BotHunter       |    49.57 </br> $_{3.12}$     |  97.22 </br> $_{0.96}$  | 91.60 </br> $_{3.12}$  |  82.90 </br> $_{1.88}$  |  82.17 </br> $_{0.20}$  | 69.18 </br> $_{1.04}$ | 99.59 </br> $_{0.02}$ | 79.09 </br> $_{0.36}$ | 23.46 </br> $_{0.09}$ |
|      Botometer       |    30.77 </br> -     |  66.90 </br> -  | 96.12 </br> -  |  78.95 </br> -  |  79.59 </br> -  | 77.39 </br> - | 46.03 </br> - | 53.13 </br> - | 42.75 </br> - |
|       BotRGCN        |    -     |  97.30 </br> $_{0.53}$  | -  |  -  |  -  | - | - | 87.25 </br> $_{0.73}$ | 57.50 </br> $_{1.42}$ |
|       Cresci         |    -     |   1.17 </br> -  | 22.81 </br> -  |  -  |  -  | - | - | 13.69 </br> - | - |
|       Dehgan         |    -     |  88.34 </br> $_{0.00}$  | -  |  -  |  -  | - | - | 76.20 </br> $_{0.00}$ | - |
|   Efthimion et al.   |     0.00 </br> $_{0.00}$     |  94.10 </br> $_{0.00}$  | 91.83 </br> $_{0.00}$  |  71.79 </br> $_{0.00}$  |  68.21 </br> $_{0.00}$  | 05.22 </br> $_{0.00}$ | 95.98 </br> $_{0.00}$ | 67.26 </br> $_{0.00}$ | 27.58 </br> $_{0.00}$ |
|      EvolveBot       |    -     |  90.07 </br> $_{1.98}$  | -  |  -  |  -  | - | - | 69.75 </br> $_{0.50}$ | 14.09 </br> $_{0.08}$ |
|      FriendBot       |    -     |  97.58 </br> $_{0.84}$  | 87.35 </br> $_{0.52}$  |  -  |  -  | - | - | 79.97 </br> $_{0.34}$ | - |
|         GCN          |    -     |  97.17 </br> $_{0.43}$  | -  |  -  |  -  | - | - | 80.86 </br> $_{0.68}$ | 54.96 </br> $_{0.91}$ |
|         GAT          |    -     |  97.58 </br> $_{0.15}$  | -  |  -  |  -  | - | - | 85.25 </br> $_{0.38}$ | 55.86 </br> $_{1.01}$ |
|      GraphHist       |    -     |  84.47 </br> $_{8.23}$  | -  |  -  |  -  | - | - | 67.56 </br> $_{0.30}$ | - |
|    Hayawi et al.     |    20.49 </br> $_{0.06}$     |  85.56 </br> $_{0.01}$  | 93.78 </br> $_{0.01}$  |  60.87 </br> $_{0.03}$  |  60.75 </br> $_{0.06}$  | 34.67 </br> $_{0.11}$ | 91.48 </br> $_{0.00}$ | 77.05 </br> $_{0.02}$ | 24.74 </br> $_{0.08}$ |
|         HGT          |    -     |  96.93 </br> $_{0.24}$  | -  |  -  |  -  | - | - | 88.19 </br> $_{0.19}$ | 39.60 </br> $_{2.11}$ |
|      SimpleHGN       |    -     |  97.28 </br> $_{0.39}$  | -  |  -  |  -  | - | - | 88.25 </br> $_{0.18}$ | 45.44 </br> $_{1.65}$ |
|    Kantepe et al.    |    -     |  78.17 </br> $_{1.42}$  | 79.41 </br> $_{1.27}$  |  -  |  -  | - | - | 62.23 </br> $_{2.06}$ | 58.71 </br> $_{1.61}$ |
|    Knauth et al.     |    41.27 </br> $_{0.00}$     |  91.18 </br> $_{0.00}$  | 93.42 </br> $_{0.00}$  |  54.15 </br> $_{0.00}$  |  94.03 </br> $_{0.00}$  | 39.10 </br> $_{0.00}$ | 91.26 </br> $_{0.00}$ | 85.24 </br> $_{0.00}$ | 37.09 </br> $_{0.00}$ |
|    Kouvela et al.    |    28.10 </br> $_{5.27}$     |  98.15 </br> $_{0.38}$  | 99.11 </br> $_{0.06}$  |  81.10 </br> $_{1.03}$  |  80.44 </br> $_{0.23}$  | 66.57 </br> $_{1.72}$ | 98.23 </br> $_{0.05}$ | 86.53 </br> $_{0.26}$ | 30.03 </br> $_{0.04}$ |
|   Kudugunta et al.   |    49.61 </br> $_{8.20}$     |  75.74 </br> $_{0.16}$  | 91.74 </br> $_{0.17}$  |  49.22 </br> $_{1.28}$  |  50.94 </br> $_{0.38}$  | 49.75 </br> $_{2.10}$ | 94.45 </br> $_{0.32}$ | 47.26 </br> $_{1.35}$ | 51.67 </br> $_{0.00}$ |
|      Lee et al.      |    50.34 </br> $_{3.16}$     |  98.56 </br> $_{0.11}$  | 99.35 </br> $_{0.04}$  |  82.74 </br> $_{1.79}$  |  82.46 </br> $_{0.36}$  | 67.78 </br> $_{1.81}$ | 97.87 </br> $_{0.07}$ | 79.98 </br> $_{0.50}$ | 30.41 </br> $_{0.20}$ |
|         LOBO         |    -     |  98.76 </br> $_{0.26}$  | 97.69 </br> $_{0.18}$  |  -  |  -  | - | - | 80.80 </br> $_{0.20}$ | 38.57 </br> $_{0.23}$ |
|    Miller et al.     |     0.00 </br> $_{0.00}$     |  83.77 </br> $_{0.00}$  | 86.80 </br> $_{0.07}$  |  43.64 </br> $_{0.00}$  |  56.76 </br> $_{0.00}$  | 59.86 </br> $_{0.00}$ | 91.14 </br> $_{0.00}$ | 74.81 </br> $_{0.26}$ | 45.29 </br> $_{0.00}$ |
|   Moghaddam et al.   |    -     |  73.93 </br> $_{0.21}$  | -  |  -  |  -  | - | - | 77.87 </br> $_{0.71}$ | 32.07 </br> $_{0.03}$ |
|       NameBot        |    38.46 </br> $_{0.00}$     |  83.36 </br> $_{0.00}$  | 85.71 </br> $_{0.02}$  |  67.53 </br> $_{0.00}$  |  61.10 </br> $_{0.00}$  | 44.83 </br> $_{0.00}$ | 91.61 </br> $_{0.00}$ | 65.06 </br> $_{0.00}$ |  0.50 </br> $_{0.00}$ |
|         RGT          |    -     |  97.78 </br> $_{0.24}$  | -  |  -  |  -  | - | - | 88.01 </br> $_{0.41}$ | 42.94 </br> $_{1.85}$ |
|       RoBERTa        |    -     |  95.86 </br> $_{0.19}$  | 94.30 </br> $_{0.18}$  |  -  |  -  | - | - | 73.09 </br> $_{0.59}$ | 20.53 </br> $_{1.71}$ |
| Rodriguez-Ruiz et al. |    -     |  87.70 </br> $_{0.00}$  | 85.65 </br> $_{0.00}$  |  -  |  -  | - | - | 63.10 </br> $_{0.00}$ | 56.57 </br> $_{0.00}$ |
|    Santos et al.     |    21.05 </br> $_{0.00}$     |  78.80 </br> $_{0.00}$  | 83.03 </br> $_{0.00}$  |  75.68 </br> $_{0.00}$  |  65.17 </br> $_{0.00}$  | 14.49 </br> $_{0.00}$ | 92.42 </br> $_{0.00}$ | 60.34 </br> $_{0.00}$ | - |
|        SATAR         |    -     |  95.05 </br> $_{0.34}$  | -  |  -  |  -  | - | - | 86.07 </br> $_{0.70}$ | - |
|        SGBot         |    49.60 </br> $_{3.43}$     |  77.91 </br> $_{0.13}$  | 94.61 </br> $_{0.19}$  |  82.26 </br> $_{1.73}$  |  82.34 </br> $_{0.11}$  | 72.10 </br> $_{0.19}$ | 99.52 </br> $_{0.02}$ | 84.90 </br> $_{0.42}$ | 36.59 </br> $_{0.18}$ |
|         T5           |    -     |  89.35 </br> $_{0.26}$  | 92.32 </br> $_{0.11}$  |  -  |  -  | - | - | 70.57 </br> $_{0.39}$ | 20.27 </br> $_{2.03}$ |
|     Varol et al.     |    -     |  94.73 </br> $_{0.42}$  | -  |  -  |  -  | - | - | 81.08 </br> $_{0.48}$ | 27.54 </br> $_{0.26}$ |
|      Wei et al.      |    -     |  82.65 </br> $_{2.21}$  | 78.43 </br> $_{1.66}$  |  -  |  -  | - | - | 57.33 </br> $_{3.19}$ | 53.61 </br> $_{1.36}$ |

#### Test1

|model|Acc| F1 | precision | recall|
|:----:|:----:|:----:|:----:|:----:|
| Moghaddam et al.|89.41</br> $_{0.30}$|24.98</br> $_{2.72}$|16.57</br> $_{1.97}$|50.79</br> $_{4.25}$|
|SGBot|91.87</br> $_{0.11}$|47.43</br> $_{1.21}$|76.16</br> $_{2.31}$|34.48</br> $_{1.56}$|
|BotHunter|91.44</br> $_{0.12}$|40.39</br> $_{0.32}$|78.28</br> $_{3.11}$|27.24</br> $_{0.52}$|
|GAT|91.14</br> $_{0.45}$|47.00</br> $_{2.92}$|64.83</br> $_{4.31}$|36.95</br> $_{3.04}$|
|BotRGCN|88.74</br> $_{0.29}$|65.89</br> $_{1.62}$|79.82</br> $_{2.53}$|56.23</br> $_{3.24}$|
|RGT|92.8</br> $_{0.45}$|23.39</br> $_{4.61}$|58.33</br> $_{11.78}$|14.66</br> $_{2.98}$|

#### Test2

|model|Acc| F1 | precision | recall|
|:----:|:----:|:----:|:----:|:----:|
| Moghaddam et al.|83.93</br> $_{0.28}$|18.49</br> $_{0.95}$|11.58</br> $_{0.59}$|45.94</br> $_{3.35}$|
|SGBot|84.72</br> $_{0.31}$|26.00</br> $_{2.80}$|54.55</br> $_{2.80}$|17.11</br> $_{2.28}$|
|BotHunter|85.63</br> $_{0.31}$|23.38</br> $_{1.55}$|73.67</br> $_{9.81}$|13.95</br> $_{1.18}$|
|GAT|84.93</br> $_{0.23}$|30.47</br> $_{2.64}$|55.64</br> $_{2.02}$|21.05</br> $_{2.46}$|
|BotRGCN|85.59</br> $_{0.68}$|55.45</br> $_{2.77}$|67.45</br> $_{2.74}$|47.17</br> $_{3.65}$|
|RGT|87.1</br> $_{1.19}$|38.02</br> $_{7.21}$|58.50</br> $_{10.18}$|28.57</br> $_{6.68}$|

#### Test3

|model|Acc| F1 | precision | recall|
|:----:|:----:|:----:|:----:|:----:|
| Moghaddam et al.|87.61</br> $_{0.20}$|22.34</br> $_{1.78}$|14.48</br> $_{1.26}$|49.00</br> $_{2.74}$|
|SGBot|89.52</br> $_{0.13}$|38.96</br> $_{1.77}$|68.97</br> $_{1.57}$|27.18</br> $_{1.85}$|
|BotHunter|89.53</br> $_{0.12}$|33.77</br> $_{0.45}$|76.62</br> $_{2.45}$|21.66</br> $_{0.25}$|
|GAT|89.09</br> $_{0.38}$|40.58</br> $_{2.68}$|61.84</br> $_{3.50}$|30.28</br> $_{2.75}$|
|BotRGCN|87.92</br> $_{0.51}$|59.46</br> $_{2.36}$|76.88</br> $_{3.71}$|48.66</br> $_{3.76}$|
|RGT|89.6</br> $_{0.72}$|26.89</br> $_{4.71}$|56.49</br> $_{11.34}$|18.05</br> $_{3.63}$|

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

Feel free to open issues in this repository! Instead of emails, Github issues are much better at facilitating a conversation between you and our team to address your needs. You can also contact Zhaoxuan Tan through ``tanzhaoxuan at stu.xjtu.edu.cn``.
