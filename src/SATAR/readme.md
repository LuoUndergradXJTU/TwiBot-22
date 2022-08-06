###  SATAR: A Self-supervised Approach to Twitter Account Representation Learning and its Application in Bot Detection
---
This is an unofficial implementation in PyTorch of SATAR. Coding by Herun Wan ([email address](wanherun@stu.xjtu.edu.cn))


- **authors**: Shangbin Feng, Herun Wan, Ningnan Wang, Jundong Li, Minnan Luo
- **link**: [https://arxiv.org/pdf/2106.13089.pdf](https://arxiv.org/pdf/2106.13089.pdf)
- **introduction**: SATAR is a self-supervised representation learning framework of Twitter users. SATAR jointly uses semantic, property and neighborhood information and adopts a co-influence module to aggregate these information.  SATAR considers the follower count as self-supervised label to pretrain parameters and fine-tunes parameters in bot detection task.
- **file structure**:

```python
├── dataset.py  # the file contains the dataset class
├── eval.py  # the file evaluates performance from trained parameters
├── get_neighbor_reps.py  # the file obtains the neighborhood vectors of each user 
├── get_reps.py  # the file obtains the representation of each user
├── model.py  # the file contain the SATAR model class
├── pretrain.py  # the code to pretrain model
├── train.py  # the code to train model
├── utils.py  # the file contain some utils class or methods
├── preprocess  # the files to preprocess datasets from raw data
│   ├── cresci-2015.py
│   ├── Twibot-20.py
│   └── Twibot-22.py
└── tmp  # other files
    ├── checkpoints  # save the trained parameters
    ├── cresci-2015  # the preprocessed data
    ├── Twibot-20
    └── Twibot-22
```

- **implement details**:  
  - Semantic Information. In practice, due to the GPU memory limitations, the number of tweets per user is limited to 128, the maximum length of each tweet is 64, and the length of words formed by all tweets of a user is at most 1024.
  - Neighborhood Information. In pre-train, we set the initial neighbor vectors of each user to 0. In fine-tune, we use the pre-trained model to get all users' representation and obtain neighbor vector of each user by averaging the neighbors' representation. 
  - Property Information. We adopt followed 15 properties: follower count, following count, tweet count, listed count, whether have withheld, whether have url, whether have profile image url, whether have pinned tweet id, wether have entities, whether have location, whether verified, whether protected, the length of description, the length of username, days difference between created time and collected time. For numerical properties, we adopt z-score normalization and 0-1 coding for true-or-false properties.
  - Due to dataset limitations, the model can only perform detection on Twibot-22, Twibot-20 and cresci-2015 dataset. Twibot-22 requires a lot of computing resources to perform.
    

### Getting Started

---

specify the dataset from ['Twibot-22', 'Twibot-20', 'cresci-2015'], 'Twibot-20' for example

#### Environment

- Python 3.7
- PyTorch == 1.9.1
- the rest of necessary Libraries
  

#### Data Preprocessing

first, preprocess raw dataset by running: 

```python
python preprocess/Twibot-20.py
```

make sure that the 'tmp/Twibot-20/' dictionary contains following files:

- vec.npy  # the word vectors
- tweets.npy  # the index of words in tweets 
- split.csv  # the dataset split from raw dataset
- properties.npy  # the properties vectors
- neighbors.npy  # the neighbors of users
- key_to_index.json  # the word index
- idx.json  # the user ids
- follower_labels.npy  # the self-supervised labels
- corpus.txt  # the tweets corpus
- bot_labels.npy # the bot labels
  

#### Pre-train

second, pretrain the model parameters by running:

```python
python pretrain.py --dataset Twibot-20
```

you could tune the following hyperparameters:

```python
parser.add_argument('--max_epoch', type=int, default=64)
parser.add_argument('--n_hidden', type=int, default=128)
parser.add_argument('--n_batch', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--max_tweet_count', type=int, default=128)
parser.add_argument('--max_tweet_length', type=int, default=64)
parser.add_argument('--max_words', type=int, default=1024)
```

for example, you can run pretrain code with 16 epoch, 0.3 dropout, n_batch 16 by running:

```python
python pretrain.py --dataset Twibot-20 --max_epoch 16 --dropout 0.3 --n_batch 16
```

after pretraining done, make sure that the 'tmp/Twibot-20/' dictionary contains following files:

- pretrain_weight.pt
  

#### Get user neighbor representations

third, get the neighbor representations of users by running:

```python
python get_reps.py --dataset Twibot-20 --n_hidden 128
```

make sure that the hidden dimensions are equal

after running done, make sure that the 'tmp/Twibot-20/' dictionary contains following files:

- reps.npy

```python
python get_neighbor_reps.py --dataset Twibot-20
```

after running done, make sure that the 'tmp/Twibot-20/' dictionary contains following files:

- neighbor_reps.npy
  

#### Train model

fourth, train the SATAR model by running:

```python
python train.py --dataset Twibot-20
```

you could tune the following hyperparameters:

```python
parser.add_argument('--max_epoch', type=int, default=64)
parser.add_argument('--n_hidden', type=int, default=128)
parser.add_argument('--n_batch', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--mode', type=int, default=0)
parser.add_argument('--max_tweet_count', type=int, default=128)
parser.add_argument('--max_tweet_length', type=int, default=64)
parser.add_argument('--max_words', type=int, default=1024)
```

mode means the train modes as following:

- 0: train without pretrain_weight
- 1: train with pretrain weight and fine tune
- 2: train with pretrain weight but not fine tune

#### Evaluation

last, evaluate trained model  by running:

```python
python eval.py --dataset Twibot-20
```





### Results

5 experiments were carried out and the results are as follows

|   dataset   | accuracy | f1-score | precision | recall |
| :---------: | :------: | :------: | :-------: | :----: |
| Cresci-2015 |  92.71   |  94.55   |   89.66   | 100.0  |
|             |  93.46   |  95.06   |   90.84   | 99.70  |
|             |  94.02   |  95.48   |   91.35   | 100.0  |
|             |  93.64   |  95.20   |   91.08   | 99.70  |
|             |  93.27   |  94.94   |   90.37   | 100.0  |
|    mean     |  93.42   |  95.05   |   90.66   | 99.88  |
|     std     |   0.48   |   0.34   |   0.67    |  0.16  |
|  Twibot-20  |  84.02   |  85.74   |   82.92   | 88.75  |
|             |  85.21   |  87.22   |   81.89   | 93.28  |
|             |  84.45   |  86.23   |   82.76   | 90.00  |
|             |  83.18   |  85.57   |   79.84   | 92.19  |
|             |  83.26   |  85.59   |   80.11   | 91.88  |
|    mean     |  84.02   |  86.07   |   81.50   | 91.22  |
|     std     |   0.85   |   0.70   |   1.45    |  1.82  |





| baseline | acc on Twibot-22 | f1 on Twibot-22 | type  | tags |
| :------: | :--------------: | :-------------: | :---: | :--: |
|  SATAR   |        -         |        -        | F T G |      |