### Bot-hunter: A Tiered Approach to Detecting & Characterizing Automated Activity on Twitter

---

- **authors**: David M. Beskow,  Kathleen M. Carley

- **link**: http://www.casos.cs.cmu.edu/publications/papers/LB_5.pdf

- **file structure**: 

```python
├── train.py  # train model on cresci-2015
└── preprocess.py  # convert raw dataset into standard format and extract features

```

- **implement details**: 

The features are divided into user attributes, network attributes, contents and timing features. We ignore some features for lack of basic data in all the datasets. After extracting the features above, we choose the random forest model to train as the best performance baseline. Due to the difference of time format, we use diffenent modes to fit datasets.

#### How to reproduce:

1. Specify the dataset you want to reproduce;

2. Extract features and convert the raw dataset into standard format by running 

   `python preprocess.py --dataset DATASETNAME`

   This command will create related features in corresponding directory.

3. Using random forest model to train by running:

   `python train.py --dataset DATASETNAME`



#### Result:



| dataset      |      | acc    | precison | recall | f1     |
| ------------ | ---- | ------ | -------- | ------ | ------ |
| Cresci-2015  | mean | 0.9652 | 0.9855   | 0.9148 | 0.9722 |
| Cresci-2015  | std  | 0.0116 | 0.0056   | 0.0416 | 0.0096 |
| Twibot-20    | mean | 0.7522 | 0.7277   | 0.8675 | 0.7909 |
| Twibot-20    | std  | 0.0044 | 0.0025   | 0.0046 | 0.0036 |
| Twibot-22    | mean | 0.7279 | 0.6809   | 0.1407 | 0.2346 |
| Twibot-22    | std  | 0.0002 | 0.0036   | 0.0012 | 0.0009 |
| midterm-2018 | mean | 0.9931 | 0.9944   | 0.9966 | 0.9959 |
| midterm-2018 | std  | 0.0004 | 0.0015   | 0.0006 | 0.0002 |
| gilani-2017  | mean | 0.7638 | 0.7899   | 0.6229 | 0.6918 |
| gilani-2017  | std  | 0.0103 | 0.0096   | 0.0347 | 0.0104 |
| c-s-2018     | mean | 0.8118 | 0.8429   | 0.7992 | 0.8217 |
| c-s-2018     | std  | 0.0016 | 0.0010   | 0.0054 | 0.0020 |
| c-r-2019     | mean | 0.8147 | 0.8192   | 0.8302 | 0.8290 |
| c-r-2019     | std  | 0.0168 | 0.0204   | 0.0295 | 0.0188 |
| Cresci-2017  | mean | 0.8811 | 0.9865   | 0.8540 | 0.9160 |
| Cresci-2017  | std  | 0.0017 | 0.0005   | 0.0019 | 0.0013 |
| b-f-2019     | mean | 0.7472 | 0.5309   | 0.4133 | 0.4957 |
| b-f-2019     | std  | 0.0103 | 0.0053   | 0.0869 | 0.0312 |






| baseline | acc on Twibot-22 | f1 on Twibot-22 | type | tags|
| -------- | ---------------- | --------------- | ---- | --- |
| Bot Hunter |0.7279|0.2346|F|`random forest`|

