# Twibot-22 Baselines

This repo contains the official implementation of [TwiBot-22]() baselines.

### Introduction

TwiBot-22 is the largest and most comprehensive Twitter bot detection benchmark to date. Specifically, [TwiBot-22](https://dl.acm.org/doi/pdf/10.1145/3459637.3482019) is designed to address the challenges of limited dataset scale, imcomplete graph structure, and low annotation quality in previous datasets. For more details, please refer to the [TwiBot-22 paper]() and [statistics](descriptions/statistics.md).
![compare](./pics/compare.png)

### Dataset Format

Each dataset contains `node.json` (or `tweet.json` for TwiBot-22), `label.csv`, `split.csv` and `edge.csv` (for datasets with graph structure). See [here](descriptions/metadata.md) for a detailed description of these files.

### How to download TwiBot-22 dataset

Reviewers at the NeurIPS 2022 Datasets and Benchmarks Track: Feel free to download TwiBot-22 at [Google Drive](https://drive.google.com/drive/folders/1YwiOUwtl8pCd2GD97Q_WEzwEUtSPoxFs?usp=sharing).

`gdown --id 1YwiOUwtl8pCd2GD97Q_WEzwEUtSPoxFs`

### How to download other datasets

For TwiBot-20, visit the [TwiBot-20 github repository](https://github.com/BunsenFeng/TwiBot-20).

For other datasets, please visit the [Bot Repository](https://botometer.osome.iu.edu/bot-repository/datasets.html).

After downloading these datasets, you can transform them into the 4-file format detailed in "Dataset Format". Alternatively, you can directly download our preprocessed version:

For TwiBot-20, visit the [TwiBot-20 github repository](https://github.com/BunsenFeng/TwiBot-20), apply for TwiBot-20 access, and there will be a `TwiBot-20-Format22.zip` in the TwiBot-20 Google Drive link.

For other datasets, you can directly download them from [Google Drive](https://drive.google.com/drive/folders/1gXFZp3m7TTU-wyZRUiLHdf_sIZpISrze?usp=sharing). You should respect the "Content redistribution" section of the [Twitter Developer Agreement and Policy](https://developer.twitter.com/en/developer-terms/agreement-and-policy), the rules set by the [Bot Repository](https://botometer.osome.iu.edu/bot-repository/datasets.html), and only use these datasets for research purposes.

### Requirements

- pip: `pip install -r requirements.txt`
- conda : `conda install --yes --file requirements.txt `

### How to run baselines

1. clone this repo by running `git clone `
2. make dataset directory `mkdir datasets` and download datasets to `./datasets`
3. change directory to `src/{name_of_the_baseline}`
4. run experiments under the guidance of corresponding `readme.md`

### Baseline Overview


| baseline                             | acc on Twibot-22 | f1 on Twibot-22 | type  | tags                     |
| -------------------------------------- | ------------------ | ----------------- | ------- | -------------------------- |
| [NameBot](src/NameBot/)              | 0.7061           | 0.0050          | F     | `Logistic Regression`    |
| [Bot Hunter](src/BotHunter/)         | 0.7279           | 0.2346          | F     | `random forest`          |
| [Botomater](src/Botometer/)          | 0.4987           | 0.4257          | F T G |                          |
| [BotRGCN](src/BotRGCN/)              | 0.7691           | 0.4579          | F T G | `BotRGCN`                |
| [Cresci et al.](src/Cresci/)         | -                | -               | T     | `DNA`                    |
| [efthimion](src/efthimion/)          | 0.7481           | 0.2758          | F T   | `efthimion`              |
| [Kipf et al.](src)                   | 0.7489           | 0.2513          | F T G | `Graph Neural Network`   |
| [Velickovic et al.](src/V)           | 0.7585           | 0.4394          | F T G | `Graph Neural Network`   |
| [Alhosseini et al.](src/Alhosseini/) | 0.6103           | 0.5473          | F G   | `gcn`                    |
| [GraphHist](src/GraphHist/)          | -                | -               | F T G | `random forest`          |
| [BGSRD](src/BGSRD/)                  | 0.7055           | 0.7200          | F     | `BERT GAT`               |
| [Hayawi et al.](src/Hayawi/)         | 0.7187           | 0.1325          | F     | `lstm`                   |
| [HGT](src/HGT_SimpleHGN/)            | 0.7491           | 0.3960          | F T G | `Graph Neural Networks`  |
| [SimpleHGN](src/HGT_SimpleHGN/)      | 0.7672           | 0.4544          | F T G | `Graph Neural Networks`  |
| [Kantepe et al.](src/Kantepe/)       | 0.764            | 0.587           | F T   | `random forest`          |
| [Knauth et al](src/Knauth/)          | -                | -               | F T G | `random forest`          |
| [Kouvela et al.](src/Kouvela/)       | 0.7644           | 0.3003          | F T   | `random forest`          |
| [Kudugunta et al.](src/Kudugunta/)   | 0.6587           | 0.5167          | F     | `SMOTENN, random forest` |
| [Lee et al.](src/Lee/)               | 0.7628           | 0.3041          | F T   | `random forest`          |
| [LOBO](src/LOBO/)                    | 0.7570           | 0.3857          | F T   | `random forest`          |
| [Miller et al.](src/Miller/)         | -                | -               | F T   | `k means`                |
| [Moghaddam et al.](src/Moghaddam/)   | 0.7378           | 0.3207          | F G   | `random forest`          |
| [Abreu et al.](src/Abreu/)           | 0.7066           | 0.5344          | F     | `random forest`          |
| [RGT](src/RGT/)                      | 0.7647           | 0.4294          | F T G | `Graph Neural Networks`  |
| [RoBERTa](src/RoBERTa/)              | 0.7196           | 0.1915          | F T   | `RoBERTa`                |
| [Rodrguez-Ruiz](src/Rodrguez-Ruiz/)  | 0.7071           | 0.0008          | F T G | `SVM`                    |
| [Santos et al.](src/Santos/)         | -                | -               | F T   | `decision tree`          |
| [SATAR](src/SATAR/)                  | -                | -               | F T G |                          |
| [T5](src/Varol/)                     | 0.7170           | 0.1565          | T     | `T5`                     |
| [Varol et al.](src/Varol)            | 0.7392           | 0.2754          | F T   | `random forest`          |
| [Wei et al.](src/Wei/)               | 0.7020           | 0.5360          | T     |                          |
| [SGBot](src/SGBot/)                  | 0.7392           | 0.2754          | F T   | `random forest`          |
| [EvolveBot](src/EvolveBot/)          | 0.7109           | 0.1408          | F T G | `random forest`          |

where `-` represents the baseline could not scale to TwiBot-22 dataset

### How to contribute

1. New dataset: convert the original data to the [TwiBot-22 defined schema](descriptions/metadata.md).
2. New baseline: load well-formatted dataset from the dataset directory and define your model.

Welcome PR!

### Questions?

Feel free to open issues in this repository! Instead of emails, Github issues are much better at facilitating a conversation between you and our team to address your needs. You can also contact the project lead Shangbin Feng through ``shangbin at cs.washington.edu``.
