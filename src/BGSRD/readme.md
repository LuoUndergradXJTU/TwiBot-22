### Social Bots Detection via Fusing BERT and Graph Convolutional Networks

---

- **authors**: Qinglang Guo, Haiyong Xie, Yangyang Li, Wen Ma, Chao Zhang

- **link**: https://www.mdpi.com/2073-8994/14/1/30

- **file structure**: 

```python
├── run.py         # train model
├── model
├── utils
├── preprocess.py  # load dataset and generate training data
├── result         # store the result file
├── data           # store the training data
└── Twibot-22      # store the training data
    ├── data           # store the training data
    ├── model
    └── run.py      # train model
```

- **implement details**: 
1. The original model use GatConv module from dgl, using ELU as the activate function of gat.
2. Due to memory limitation, we set the batch size as 32.
3. Due to memory limitation, the graph of Twibot-20 and midterm-2018 only consist of 3000 word(Top 3000 in order of word frequncy)
4. In the implementation of Twibot-22, we use pytorch geometric neighbor loader to sample graph data.

  

#### How to reproduce:

1. preprocess the the dataset by running 

   `python preprocess.py --source_path ${dataset}`

   this command will create related features in corresponding directory.

2. train the model by running:

   `python run.py --dataset ${dataset}`

   the final result will be saved into ${dataset}.txt



#### Result:

| dataset                   |      | acc    | precison | recall | f1     |
| ------------------------- | ---- | ------ | -------- | ------ | ------ |
| Twibot-22                 | mean | 0.7188 |  0.2255  | 0.1990 | 0.2114 |
| Twibot-22                 | std  | 0.0182 |  0.3088  | 0.2724 | 0.2895 |
| Twibot-20                 | mean | 0.6636 |  0.6764  | 0.7319 | 0.7005 |
| Twibot-20                 | std  | 0.0100 |  0.0226  | 0.0749 | 0.0260 |
| botometer-feedback-2019   | mean | 0.5962 |  0.2750  | 0.0857 | 0.1303 |
| botometer-feedback-2019   | std  | 0.0316 |  0.2820  | 0.0852 | 0.1301 |
| cresci-rtbust-2019        | mean | 0.5000 |  0.5813  | 0.3514 | 0.4108 |
| cresci-rtbust-2019        | std  | 0.0488 |  0.1112  | 0.2058 | 0.1300 |
| cresci-stock-2018         | mean | 0.5074 |  0.5278  | 0.7040 | 0.5818 |
| cresci-stock-2018         | std  | 0.0134 |  0.0075  | 0.2614 | 0.1205 |
| midterm-2018              | mean | 0.8287 |  0.8440  | 0.9766 | 0.9050 |
| midterm-2018              | std  | 0.0148 |  0.0093  | 0.0366 | 0.0109 |
| cresci-2017               | mean | 0.7585 |  0.7585  | 1.0000 | 0.8627 |
| cresci-2017               | std  | 0.0000 |  0.0000  | 0.0000 | 0.0000 |
| gilani-2017               | mean | 0.4847 |  0.2543  | 0.6000 | 0.3572 |
| gilani-2017               | std  | 0.0834 |  0.2321  | 0.5477 | 0.3261 |
| cresci-2015               | mean | 0.8778 |  0.8652  | 0.9556 | 0.9080 |
| cresci-2015               | std  | 0.0063 |  0.0064  | 0.0202 | 0.0060 |


| baseline  | acc on Twibot-22 | f1 on Twibot-22 | type |   tags   |
| --------  | ---------------- | --------------- | ---- | -------- |
| Guo et al |  0.7188          |     0.2114      |  F   |`BERT GAT`|

