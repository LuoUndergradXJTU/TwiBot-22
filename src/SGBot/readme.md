### Scalable and Generalizable Social Bot Detection through Data Selection

---

- **authors**: Kai-Cheng Yang, Onur Varol, Pik-Mai Hui, Filippo Menczer

- **link**:  https://arxiv.org/abs/1911.09179

- **file structure**: 

```python
├── train.py  # train model on cresci-2015
├── preprocess.py # convert raw dataset into standard format and extract features

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
| Cresci-2015 | mean | 0.7708 | 0.9222   | 0.9740 | 0.7791 |
| Cresci-2015 | var  | 0.0001 | 0.0066   | 0.0090 | 0.0002 |
| Twibot-20   | mean | 0.8164 | 0.7674   | 0.9483 | 0.8489 |
| Twibot-20   | var  | 0.0000 | 0.0001   | 0.0000 | 0.0000 |
| Twibot-22   | mean | 0.7508 | 0.7330   | 0.2449 | 0.3659 |
| Twibot-22   | var  | 0.0000 | 0.0001   | 0.0000 | 0.0000 |




| baseline | acc on Twibot-22 | f1 on Twibot-22 | type | tags|
| -------- | ---------------- | --------------- | ---- | --- |
| Varol et al.|0.7392|0.2754|P T|`random forest`|

