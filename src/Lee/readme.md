### Seven Months with the Devils: A Long-Term Study of Content Polluters on Twitter

---

- **authors**: 
Kyumin Lee, Brian Eoff, James Caverlee

- **link**: https://ojs.aaai.org/index.php/ICWSM/article/view/14106

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
In Twibot-22, Twibot-20, cresci-2015 and cresci-2017, the id of the tweets posted by each user is obtained from the edge.csv file.
And in the Twibot-22, the top 20 tweets are selected to calculate the  content features, not the latest 200 tweets. In Twibot-20, cresci-2015 and cresci-2017, the top 200 tweets are selected to calculate the content feature, not the latest 200 tweets.
In In midterm-2018, gilani-2017, cresci-stock-2018, cresci-rtbust-2019 and botometer-feedback-2019, only user features can be obtained, so only user features are selected for calculation.
Meanwhile, the training parameters of the random forest model are not mentioned in the paper. When implementing, we use the validation set to tune the optimal parameters of the random forest. Finally we set the number of decision trees( n_estimators) to 76, and the other parameters are default values.
“the percentage of bidirectional friends”, “the standard deviation of unique numerical IDs of following”, “the standard deviation of unique numerical IDs of followers”, “the change rate of number of following obtained by a user’s temporal and historical information” are discarded since required information is not included in the datasets.

  

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

   `python merge_feature.py --datasets dataset_name`
   
   the final result will be saved into result.txt in corresponding directory.

For dataset midterm-2018, gilani-2017, cresci-stock-2018, cresci-rtbust-2019 and botometer-feedback-2019, you just need to do the 2nd, 4th and 5th steps to reproduce, since these datasets only have user information.



#### Result:

| dataset     |      | acc    | precison | recall | f1     |
| ----------- | ---- | ------ | -------- | ------ | ------ |
| cresci-2015 | mean | 0.9823 | 0.9865 | 0.9846 | 0.9856 |
| cresci-2015 | std  | 0.0014 | 0.0014 | 0.0014 | 0.0011 |
| gilani-2017 | mean | 0.7482 | 0.7758 | 0.6019 | 0.6778 |
| gilani-2017 | std  | 0.0121 | 0.0131 | 0.0215 | 0.0181 |
| cresci-2017 | mean | 0.9883 | 0.9956 | 0.9913 | 0.9934 |
| cresci-2017 | std  | 0.0008 | 0.0007 | 0.0000 | 0.0003 |
| midterm-2018 | mean | 0.9640 | 0.9736 | 0.9837 | 0.9786 |
| midterm-2018 | std  | 0.0012 | 0.0007 | 0.0010 | 0.0007 |
| cresci-stock-2018 | mean | 0.8153 | 0.8474 | 0.8030 | 0.8246 |
| cresci-stock-2018 | std  | 0.0035 | 0.0042 | 0.0063 | 0.0036 |
| cresci-rtbust-2019 | mean | 0.8353 | 0.7937 | 0.8645 | 0.8274 |
| cresci-rtbust-2019 | std  | 0.0192 | 0.0297 | 0.0144 | 0.0179 |
| botometer-feedback-2019 | mean | 0.7547 | 0.5897 | 0.4400 | 0.5034 |
| botometer-feedback-2019 | std  | 0.0134 | 0.0329 | 0.0365 | 0.0316 |
| Twibot-20 | mean | 0.7736 | 0.7660 | 0.8366 | 0.7998 |
| Twibot-20 | std  | 0.0053 | 0.0037 | 0.0069 | 0.0050 |
| Twibot-22 | mean | 0.7628 | 0.6723 | 0.1965 | 0.3041 |
| Twibot-22 | std  | 0.0005 | 0.0029 | 0.0015 | 0.0020 |







| baseline | acc on Twibot-22 | f1 on Twibot-22 | type | tags|
| -------- | ---------------- | --------------- | ---- | --- |
| Lee et al.|76.28|30.41|F T|`random forest`|

