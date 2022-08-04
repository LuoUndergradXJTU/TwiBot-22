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



| dataset     |      | acc    | precison | recall | f1     |
| ----------- | ---- | ------ | -------- | ------ | ------ |
| Cresci-2015 | mean | 0.9652 | 0.9790   | 0.9645 | 0.9722 |
| Cresci-2015 | var  | 0.0001 | 0.0001   | 0.0000 | 0.0001 |
| Twibot-20   | mean | 0.7522 | 0.7279   | 0.8734 | 0.7909 |
| Twibot-20   | var  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |
| Twibot-22   | mean | 0.7279 | 0.6809   | 0.1416 | 0.2346 |
| Twibot-22   | var  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |
| midterm-2018| mean | 0.9931 | 0.9951   | 0.9967 | 0.9959 |
| midterm-2018| var  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |
| gilani-2017 | mean | 0.7638 | 0.8101   | 0.6095 | 0.6918 |
| gilani-2017 | var  | 0.0001 | 0.0001   | 0.0001 | 0.0001 |
| c-s-2018    | mean | 0.8118 | 0.8447   | 0.8057 | 0.8217 |
| c-s-2018    | var  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |
| c-r-2019    | mean | 0.8147 | 0.8611   | 0.8378 | 0.8290 |
| c-r-2019    | var  | 0.0001 | 0.0001   | 0.0001 | 0.0001 |
| Cresci-2017 | mean | 0.8811 | 0.9874   | 0.8558 | 0.9160 |
| Cresci-2017 | var  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |
| b-f-2019    | mean | 0.7472 | 0.5716   | 0.4633 | 0.4957 |
| b-f-2019    | var  | 0.0001 | 0.0003   | 0.0005 | 0.0009 |






| baseline | acc on Twibot-22 | f1 on Twibot-22 | type | tags|
| -------- | ---------------- | --------------- | ---- | --- |
| Bot Hunter |0.7279|0.2346|F|`random forest`|

