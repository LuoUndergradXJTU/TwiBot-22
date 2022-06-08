### Bot-hunter: A Tiered Approach to Detecting & Characterizing Automated Activity on Twitter

---

- **authors**: David M. Beskow， Kathleen M. Carley

- **link**: http://www.casos.cs.cmu.edu/publications/papers/LB_5.pdf

- **file structure**: 

```python
├── train.py  # train model on cresci-2015
├── preprocess.py # convert raw dataset into standard format and extract features

```

- **implement details**: 

The features are divided into user attributes, network attributes, contents and timing features. We ignore some features for lack of basic data in all the datasets. Shannon entropy is defined in 1, where possibility is the normalized count for each character found in the string. We choose the Random Forest model as the best performance baseline.

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
| Cresci-2015 | mean | 0.9652 | 0.9790   | 0.9645 | 0.9722 |
| Cresci-2015 | var  | 0.0001 | 0.0001   | 0.0000 | 0.0001 |
| Twibot-20   | mean | 0.7522 | 0.7279   | 0.8734 | 0.7909 |
| Twibot-20   | var  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |
| Twibot-22   | mean | 0.7279 | 0.6809   | 0.1416 | 0.2346 |
| Twibot-22   | var  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |






| baseline | acc on Twibot-22 | f1 on Twibot-22 | type | tags|
| -------- | ---------------- | --------------- | ---- | --- |
| Bot Hunter |0.7279|0.2346|P|`random forest`|

