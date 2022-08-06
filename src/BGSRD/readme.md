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
| Twibot-22                 | mean | 71.88  |  22.55   | 19.90  | 21.14  |
| Twibot-22                 | std  |  1.82  |  30.88   | 27.24  | 28.95  |
| Twibot-20                 | mean | 66.36  |  67.64   | 73.19  | 70.05  |
| Twibot-20                 | std  |  1.00  |   2.26   |  7.49  |  2.60  |
| botometer-feedback-2019   | mean | 59.62  |  27.50   |  8.57  | 13.03  |
| botometer-feedback-2019   | std  |  3.16  |  28.20   |  8.52  | 13.01  |
| cresci-rtbust-2019        | mean | 50.00  |  58.13   | 35.14  | 41.08  |
| cresci-rtbust-2019        | std  |  4.88  |  11.12   | 20.58  | 13.00  |
| cresci-stock-2018         | mean | 50.74  |  52.78   | 70.40  | 58.18  |
| cresci-stock-2018         | std  |  1.34  |   0.75   | 26.14  | 12.05  |
| midterm-2018              | mean | 82.87  |  84.40   | 97.66  | 90.50  |
| midterm-2018              | std  |  1.48  |   0.93   | 03.66  |  1.09  |
| cresci-2017               | mean | 75.85  |  75.85   |  0.00  | 86.27  |
| cresci-2017               | std  |  0.00  |   0.00   |  0.00  |  0.00  |
| gilani-2017               | mean | 48.47  |  25.43   | 60.00  | 35.72  |
| gilani-2017               | std  |  8.34  |  23.21   | 54.77  | 32.61  |
| cresci-2015               | mean | 87.78  |  86.52   | 95.56  | 90.80  |
| cresci-2015               | std  |  0.63  |   0.64   |  2.02  |  0.60  |



| baseline  | acc on Twibot-22 | f1 on Twibot-22 | type |   tags   |
| --------  | ---------------- | --------------- | ---- | -------- |
| Guo et all|  71.88           |     21.14       |  F   |`BERT GAT`|

