### A one-class classification approach for bot detection on Twitter

---

- **authors**: JorgeRodríguez-Ruiza, Javier IsraelMata-Sánchezb, RaúlMonroyc, OctavioLoyola-Gonzálezd, ArmandoLópez-Cuevase

- **link**: https://www.sciencedirect.com/science/article/pii/S0167404820300031

- **file structure**: 

```python
├── dataset.py # convert raw dataset into standard format
├── preprocess.py # extract features 
├── classification.py #train model and classify Bot
```

- **implement details**:

  1. Classification using SVM, and using Logistic regression in Twibot-22
  2. The interval time is the difference between the data set collection time and the creation time, and
  then divided by the tweet count.
  3. The "create time" display in cresci-2017 dataset is abnormal, which is replaced by Thu Mar 06 02:37:29 +0000 2014
  
  | Feature         | description                      |                                |
  | --------------- | -------------------------------- | ------------------------------ |
  | retweets        | retweet count/tweet count        | /                              |
  | replies         | reply count/tweet count          | /                              |
  | favoriteC       | favorite count/tweet count       | /                              |
  | hashtag         | tag count/tweet count            | ✔(the number of "#")           |
  | url             | url count/tweet count            | ✔ (the number of"http")        |
  | mentions        | mention count/tweet count        | ✔  (the number of"@")          |
  | intertime       | as 2.                            | ✔                              |
  | ffratio         | friend count/follow count        | ✔                              |
  | favorites       | favorite tweet count             | /                              |
  | listed          | list count                       | ✔                              |
  | uniquehashtages | unique tag count/tweet count     | ✔(the unique number of "#")    |
  | uniqueMentions  | unique mention count/tweet count | ✔(the unique number of "@")    |
  | uniqueURL       | unique URL count/tweet count     | ✔(the unique number of "http") |
  
  

#### How to reproduce:

1. In line 143 of the file LRclassification.py, change  the name of dataset to what you want to train.

2. train Logistic Regression model by running:

 		 `python classification.py`



#### Result:

| dataset     |      | acc    | precison | recall | f1     |
| ----------- | ---- | ------ | -------- | ------ | ------ |
| Cresci-2015 | mean | 0.7346 | 0.7050   | 0.9970 | 0.8260 |
| Cresci-2015 | var  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |
| Cresci-2017 | mean | 0.7730 | 0.7779   | 0.9808 | 0.8676 |
| Cresci-2017 | var  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |
| Twibot-20   | mean | 0.6602 | 0.6160   | 0.9875 | 0.6310 |
| Twibot-20   | var  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |
| Twibot-22   | mean | 0.7071 | 0.7198   | 0.0004 | 0.0008 |
| Twibot-22   | var  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |







| baseline | acc on Twibot-22 | f1 on Twibot-22 | type | tags|
| -------- | ---------------- | --------------- | ---- | --- |
| Rodrguez-Ruiz|0.7071|0.0008|P T G|`SVM`|

