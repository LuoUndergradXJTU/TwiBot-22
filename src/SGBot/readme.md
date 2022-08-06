### Scalable and Generalizable Social Bot Detection through Data Selection

---

- **authors**: Kai-Cheng Yang, Onur Varol, Pik-Mai Hui, Filippo Menczer

- **link**:  https://arxiv.org/abs/1911.09179

- **file structure**: 

```python
├── train.py  # train model on cresci-2015
└── preprocess.py # convert raw dataset into standard format and extract features

```

- **implement details**: 

The features are divided into user metadata and derived features. We ignore some feature for lack of basic data in the datasets. The screen name likelihood feature is inspired by the work of Beskow and Carley (2019). We constructed the likelihood of all 3,969 possible bigrams. The likelihood of a screen name is defined by the geometric-mean likelihood of all bigrams in it. Screen name likelihood is a real value which describes the likelihood of the screen name.


  

#### How to reproduce:

1. Specify the dataset you want to reproduce ;

2. Extract features and convert the raw dataset into standard format by running 

   `python preprocess.py --dataset DATASETNAME`

   This command will create related features in corresponding directory.

3. Using random forest model to train by running:

   `python train.py --dataset DATASETNAME`




#### Result:



| dataset     |      | acc    | precison | recall | f1     |
| ----------- | ---- | ------ | -------- | ------ | ------ |
| Cresci-2015 | mean | 0.7708 | 0.9945   | 0.6367 | 0.7791 |
| Cresci-2015 | std  | 0.0021 | 0.0020   | 0.0131 | 0.0013 |
| Twibot-20   | mean | 0.8164 | 0.7640   | 0.9491 | 0.8490 |
| Twibot-20   | std  | 0.0046 | 0.0040   | 0.0069 | 0.0042 |
| Twibot-22   | mean | 0.7508 | 0.7311   | 0.2432 | 0.3659 |
| Twibot-22   | std  | 0.0005 | 0.0018   | 0.0009 | 0.0018 |
| midterm-2018| mean | 0.9919 | 0.9935   | 0.9966 | 0.9952 |
| midterm-2018| std  | 0.0004 | 0.0022   | 0.0020 | 0.0002 |
| gilani-2017 | mean | 0.7860 | 0.8268   | 0.6362 | 0.7210 |
| gilani-2017 | std  | 0.0077 | 0.0188   | 0.0217 | 0.0119 |
| c-s-2018    | mean | 0.8128 | 0.8390   | 0.8103 | 0.8234 |
| c-s-2018    | std  | 0.0006 | 0.0029   | 0.0090 | 0.0011 |
| c-r-2019    | mean | 0.8088 | 0.8308   | 0.8162 | 0.8226 |
| c-r-2019    | std  | 0.0147 | 0.0260   | 0.0226 | 0.0173 |
| Cresci-2017 | mean | 0.9212 | 0.9826   | 0.9086 | 0.9461 |
| Cresci-2017 | std  | 0.0027 | 0.0017   | 0.0039 | 0.0019 |
| b-f-2019    | mean | 0.7547 | 0.5970   | 0.4533 | 0.4960 |
| b-f-2019    | std  | 0.0189 | 0.0391   | 0.0298 | 0.0343 |




| baseline | acc on Twibot-22 | f1 on Twibot-22 | type | tags|
| -------- | ---------------- | --------------- | ---- | --- |
| Varol et al.|0.7508|0.3659|F T|`random forest`|

