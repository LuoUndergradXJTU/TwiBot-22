# TwiBot-22

This is the offical repository of [TwiBot-22](https://twibot22.github.io/).

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


| baseline                             | paper                     | acc on Twibot-22 | f1 on Twibot-22 | type  | tags                     |
| -------------------------------------- | ------------------ | ----------------- | ------- | -------------------------- | -------------------------- |
| [NameBot](src/NameBot/)              | [link](https://arxiv.org/pdf/1812.05932.pdf) | 0.7061           | 0.0050          | F     | `Logistic Regression`    |
| [Bot Hunter](src/BotHunter/)         | [link](http://www.casos.cs.cmu.edu/publications/papers/LB_5.pdf) | 0.7279           | 0.2346          | F     | `random forest`          |
| [Botometer](src/Botometer/)         | [link](https://botometer.osome.iu.edu/) | 0.4987           | 0.4257          | F T G |                          |
| [BotRGCN](src/BotRGCN/)              | [link](https://arxiv.org/abs/2106.13092) | 0.7691           | 0.4579          | F T G | `BotRGCN`                |
| [Cresci et al.](src/Cresci/)         | [link](https://ieeexplore.ieee.org/abstract/document/7436643) | -                | -               | T     | `DNA`                    |
| [efthimion](src/efthimion/)          | [link](https://scholar.smu.edu/datasciencereview/vol1/iss2/5/) | 0.7481           | 0.2758          | F T   | `efthimion`              |
| [Kipf et al.](src/GCN_GAT)           | [link](https://arxiv.org/abs/1609.02907) | 0.7489           | 0.2513          | F T G | `Graph Neural Network`   |
| [Velickovic et al.](src/V)           | [link](https://arxiv.org/abs/1710.10903) | 0.7585           | 0.4394          | F T G | `Graph Neural Network`   |
| [Alhosseini et al.](src/Alhosseini/) | [link](https://dl.acm.org/doi/fullHtml/10.1145/3308560.3316504) | 0.6103           | 0.5473          | F G   | `gcn`                    |
| [GraphHist](src/GraphHist/)          | [link](https://arxiv.org/abs/1910.01180) | -                | -               | F T G | `random forest`          |
| [BGSRD](src/BGSRD/)                  | [link](https://www.mdpi.com/2073-8994/14/1/30) | 0.7055           | 0.7200          | F     | `BERT GAT`               |
| [Hayawi et al.](src/Hayawi/)         | [link](https://link.springer.com/content/pdf/10.1007/s13278-022-00869-w.pdf) | 0.7187           | 0.1325          | F     | `lstm`                   |
| [HGT](src/HGT_SimpleHGN/)            | [link](https://arxiv.org/abs/2003.01332) | 0.7491           | 0.3960          | F T G | `Graph Neural Networks`  |
| [SimpleHGN](src/HGT_SimpleHGN/)      | [link](https://arxiv.org/abs/2112.14936) | 0.7672           | 0.4544          | F T G | `Graph Neural Networks`  |
| [Kantepe et al.](src/Kantepe/)       | [link](https://ieeexplore.ieee.org/abstract/document/8093483) | 0.764            | 0.587           | F T   | `random forest`          |
| [Knauth et al.](src/Knauth/)         | [link](https://aclanthology.org/R19-1065/) | -                | -               | F T G | `random forest`          |
| [Kouvela et al.](src/Kouvela/)       | [link](https://dl.acm.org/doi/abs/10.1145/3415958.3433075) | 0.7644           | 0.3003          | F T   | `random forest`          |
| [Kudugunta et al.](src/Kudugunta/)   | [link](https://arxiv.org/abs/1802.04289) | 0.6587           | 0.5167          | F     | `SMOTENN, random forest` |
| [Lee et al.](src/Lee/)               | [link](https://ojs.aaai.org/index.php/ICWSM/article/view/14106) | 0.7628           | 0.3041          | F T   | `random forest`          |
| [LOBO](src/LOBO/)                    | [link](https://dl.acm.org/doi/10.1145/3274694.3274738) | 0.7570           | 0.3857          | F T   | `random forest`          |
| [Miller et al.](src/Miller/)         | [link](https://dl.acm.org/doi/10.1016/j.ins.2013.11.016) | -                | -               | F T   | `k means`                |
| [Moghaddam et al.](src/Moghaddam/)   | [link](https://ieeexplore.ieee.org/abstract/document/9735340) | 0.7378           | 0.3207          | F G   | `random forest`          |
| [Abreu et al.](src/Abreu/)           | [link](https://ieeexplore.ieee.org/abstract/document/9280525) | 0.7066           | 0.5344          | F     | `random forest`          |
| [RGT](src/RGT/)                      | [link](https://arxiv.org/abs/2109.02927) | 0.7647           | 0.4294          | F T G | `Graph Neural Networks`  |
| [RoBERTa](src/RoBERTa/)              | [link](https://arxiv.org/pdf/1907.11692.pdf) | 0.7196           | 0.1915          | F T   | `RoBERTa`                |
| [Rodrguez-Ruiz](src/Rodrguez-Ruiz/)  | [link](https://www.sciencedirect.com/science/article/pii/S0167404820300031) | 0.7071           | 0.0008          | F T G | `SVM`                    |
| [Santos et al.](src/Santos/)         | [link](https://dl.acm.org/doi/pdf/10.1145/3308560.3317599) | -                | -               | F T   | `decision tree`          |
| [SATAR](src/SATAR/)                  | [link](https://arxiv.org/abs/2106.13089) | -                | -               | F T G |                          |
| [T5](src/Varol/)                     | [link](https://aaai.org/ocs/index.php/ICWSM/ICWSM17/paper/view/15587/14817) | 0.7170           | 0.1565          | T     | `T5`                     |
| [Varol et al.](src/Varol)            | [link](https://aaai.org/ocs/index.php/ICWSM/ICWSM17/paper/view/15587/14817) | 0.7392           | 0.2754          | F T   | `random forest`          |
| [Wei et al.](src/Wei/)               | [link](https://arxiv.org/pdf/2002.01336.pdf) | 0.7020           | 0.5360          | T     |                          |
| [SGBot](src/SGBot/)                  | [link](https://arxiv.org/abs/1911.09179) | 0.7392           | 0.2754          | F T   | `random forest`          |
| [EvolveBot](src/EvolveBot/)          | [link](https://ieeexplore.ieee.org/abstract/document/6553246) | 0.7109           | 0.1408          | F T G | `random forest`          |
| [FriendBot](src/FriendBot)| [link](https://link.springer.com/chapter/10.1007/978-3-030-41251-7_3)  |-|-|F T G|`random forest`|
| [Dehghan et al.](src/Dehghan)| [link](https://assets.researchsquare.com/files/rs-1428343/v1_covered.pdf?c=1647280648)  |-|-|F T G|`Graph`|

where `-` represents the baseline could not scale to TwiBot-22 dataset

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
