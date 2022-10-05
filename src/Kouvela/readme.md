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
| cresci-2015 | mean | 0.9776 | 0.9954 | 0.9679 | 0.9815 |
| cresci-2015 | std  | 0.0045 | 0.0018 | 0.0076 | 0.0038 |
| gilani-2017 | mean | 0.7473 | 0.7969 | 0.5720 | 0.6657 |
| gilani-2017 | std  | 0.0095 | 0.0109 | 0.0242 | 0.0172 |
| cresci-2017 | mean | 0.9842 | 0.9924 | 0.9898 | 0.9911 |
| cresci-2017 | std  | 0.0011 | 0.0013 | 0.0018 | 0.0006 |
| midterm-2018 | mean | 0.9701 | 0.9756 | 0.9892 | 0.9823 |
| midterm-2018 | std  | 0.0008 | 0.0005 | 0.0006 | 0.0005 |
| cresci-stock-2018 | mean | 0.7928 | 0.8217 | 0.7878 | 0.8044 |
| cresci-stock-2018 | std  | 0.0029 | 0.0045 | 0.0018 | 0.0023 |
| cresci-rtbust-2019 | mean | 0.7970 | 0.8227 | 0.8000 | 0.8110 |
| cresci-rtbust-2019 | std  | 0.0123 | 0.0200 | 0.0148 | 0.0103 |
| botometer-feedback-2019 | mean | 0.7132 | 0.4800 | 0.2000 | 0.2810 |
| botometer-feedback-2019 | std  | 0.0085 | 0.0447 | 0.0472 | 0.0527 |
| Twibot-20 | mean | 0.8399 | 0.7933 | 0.9517 | 0.8654 |
| Twibot-20 | std  | 0.0036 | 0.0044 | 0.0014 | 0.0026 |
| Twibot-22 | mean | 0.7644 | 0.6930 | 0.1917 | 0.3004 |
| Twibot-22 | std  | 0.0002 | 0.0014 | 0.0004 | 0.0004 |







| baseline | acc on Twibot-22 | f1 on Twibot-22 | type | tags|
| -------- | ---------------- | --------------- | ---- | --- |
| Kouvela et al.|76.44|30.03|F T|`random forest`|

