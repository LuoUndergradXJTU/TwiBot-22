### Bot-detective: An explainable Twitter bot detection service with crowdsourcing functionalities

---

- **authors**: 
Maria Kouvela, Ilias Dimitriadis, Athena Vakali

- **link**: https://dl.acm.org/doi/abs/10.1145/3415958.3433075

- **file structure**: 

```python
├── user_to_post.py # obtation tweets id posted by each user
├── user_features.py # user features
├── user_feature_extraction.py # calculate user features
├── merge_feature.py # merge features, labels and splits.
├── train.py # train model  on specific dataset
├── cresci-2015
│   ├── tweet_split.py  # split the initial tweets data on cresci-2015 to speed up calculation
│   ├── content_features.py  # content features
│   └── content_feature_extraction.py  # calculate content features on cresci-2015
├── cresci-2017 
│   ├── tweet_split.py 
│   ├── content_features.py 
│   └── content_feature_extraction.py  
├── Twibot-20    
│   ├── tweet_split.py  
│   ├── content_features.py 
│   └── content_feature_extraction.py  
└── Twibot-22
    ├── tweet_split.py  
    ├── content_features.py  
    └── content_feature_extraction.py  
```

- **implement details**: 
In Twibot-22, Twibot-20, cresci-2015 and cresci-2017, the id of the tweets posted by each user is obtained from the edge.csv file, and the top 20 tweets are selected to calculate the  content features.
In Twibot-22, “Favourite Tweets”, “Media”, “Sensitive Tweet” are discarded since required information is not included in the dataset;
In Twibot-20, cresci-2015 and cresci-2017, “Def. Image”, “Def. Profile”, “Favourite Tweets”, “Media”, “Sensitive Tweet”, “Times favourite”, “Times Retweeted” are discarded since required information is not included in the datasets.
In midterm-2018, gilani-2017, cresci-stock-2018, cresci-rtbust-2019 and botometer-feedback-2019, only user features can be obtained, so only 20 user features are selected for calculation.

  

#### How to reproduce:

For dataset Twibot-22, Twibot-20, cresci-2015 and cresci-2017 :

1. get the tweets id posted by each user by running

   `python user_to_post.py --datasets dataset_name`

   For example: `python user_to_post.py --datasets Twibot-22`
   
   this command will create a user_to_post.csv file which contains the tweets id posted by each user in corresponding directory.

2. get the user features by running

   `python user_feature_extraction.py --datasets dataset_name`

   this command will create a user_feature.csv in corresponding directory.

3. get the content features, first `cd dataset_name`, then run

   `python tweet_split.py`

   and 

   `python content_feature_extraction.py`

   the two commands will create a content_feature.csv in the directory.

4. go back to the root directory and merge the features, labels, and splits by running

   `python merge_feature.py --datasets dataset_name`

   this command will create a features.csv file in corresponding directory.

5. train random forest model by running

   `python train.py --datasets dataset_name`
   
   the final result will be saved into result.txt in corresponding directory.

For dataset midterm-2018, gilani-2017, cresci-stock-2018, cresci-rtbust-2019 and botometer-feedback-2019, you just need to do the 2nd, 4th and 5th steps to reproduce, since these datasets only have user information.



#### Result:

| dataset     |      | acc    | precison | recall | f1     |
| ----------- | ---- | ------ | -------- | ------ | ------ |
| cresci-2015 | mean | 97.76 | 99.54 | 96.79 | 98.15 |
| cresci-2015 | std  | 0.4523 | 0.1789 | 0.7517 | 0.3805 |
| gilani-2017 | mean | 74.73 | 79.69 | 57.2 | 66.57 |
| gilani-2017 | std  | 0.9474 | 1.0881 | 2.4191 | 1.7236 |
| cresci-2017 | mean | 98.42 | 99.24 | 98.98 | 99.11 |
| cresci-2017 | std  | 0.1096 | 0.1308 | 0.1813 | 0.0624 |
| midterm-2018 | mean | 97.01 | 97.56 | 98.92 | 98.23 |
| midterm-2018 | std  | 0.0828 | 0.0446 | 0.0632 | 0.0491 |
| cresci-stock-2018 | mean | 79.28 | 82.17 | 78.78 | 80.44 |
| cresci-stock-2018 | std  | 0.2893 | 0.4561 | 0.1772 | 0.2277 |
| cresci-rtbust-2019 | mean | 79.71 | 82.27 | 80.0 | 81.1 |
| cresci-rtbust-2019 | std  | 1.2304 | 2.0026 | 1.4803 | 1.0316 |
| botometer-feedback-2019 | mean | 71.32 | 48.0 | 20.0 | 28.1 |
| botometer-feedback-2019 | std  | 0.8438 | 4.4721 | 4.714 | 5.2663 |
| Twibot-20   | mean | 83.99 | 79.33 | 95.17 | 86.53 |
| Twibot-20   | std  | 0.3587 | 0.4426 | 0.1411 | 0.2627 |
| Twibot-22   | mean | 76.44 | 69.30 | 19.17 | 30.03 |
| Twibot-22   | std  | 0.014 | 0.1433 | 0.0389 | 0.0444 |







| baseline | acc on Twibot-22 | f1 on Twibot-22 | type | tags|
| -------- | ---------------- | --------------- | ---- | --- |
| Kouvela et al.|76.44|30.03|F T|`random forest`|

