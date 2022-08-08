### Its All in a Name: Detecting and Labeling Bots by Their Name

---

- **authors**: David M. Beskow and Kathleen M. Carley
- **link**:  https://arxiv.org/pdf/1812.05932.pdf
- **file structure**: 

```python
├── preprocess.py  # extract features and convert raw dataset into standard format
└── LRclassification.py  # train model and classify Bot
```

- **implement details**:

  The original paper did not have the specific practice of Feature Engineering, so we designed feature engineering myself
  
  - When calculating the parameter of large (small) letters, take the percentage of large (small) letters in
    the length of the user name as the standardized parameter
  - When calculating TF-IDF, the feature dimension is recorded as the length of all Bi grams
  - The effect of logistic regression in the original paper is the best, so this method is used for calculation

#### How to reproduce:

1. In line 161 of the file LRclassification.py, change  the name of dataset to what you want to train.

2. train Logistic Regression model by running:

   `python LRclassification.py`



#### Result:



| dataset                 |      | acc    | precison | recall | f1     |
| ----------------------- | ---- | ------ | -------- | ------ | ------ |
| botometer-feedback-2019 | mean | 0.6981 | 0.4545   | 0.3333 | 0.3846 |
| botometer-feedback-2019 | std  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |
| Cresci-2015             | mean | 0.7701 | 0.7681   | 0.9112 | 0.8336 |
| Cresci-2015             | std  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |
| cresci-2017             | mean | 0.7679 | 0.8039   | 0.9179 | 0.8571 |
| cresci-2017             | std  | 0.0003 | 0.0003   | 0.0000 | 0.0002 |
| cresci-rtbust-2019      | mean | 0.6324 | 0.6500   | 0.7027 | 0.6753 |
| cresci-rtbust-2019      | std  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |
| cresci-stock-2018       | mean | 0.5584 | 0.5834   | 0.6413 | 0.6110 |
| cresci-stock-2018       | std  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |
| gilani-2017             | mean | 0.6081 | 0.5821   | 0.3645 | 0.4483 |
| gilani-2017             | std  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |
| midterm-2018            | mean | 0.8510 | 0.8693   | 0.9682 | 0.9161 |
| midterm-2018            | std  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |
| Twibot-20               | mean | 0.5722 | 0.5872   | 0.7047 | 0.6506 |
| Twibot-20               | std  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |
| Twibot-22               | mean | 0.7061 | 0.6773   | 0.0003 | 0.0050 |
| Twibot-22               | std  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |











| baseline      | acc on Twibot-22 | f1 on Twibot-22 | type | tags                |
| ------------- | ---------------- | --------------- | ---- | ------------------- |
| NameBot  | 0.7061           | 0.0050          | F | `Logistic Regression` |

