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
│   └── tweet_split.py  # split the initial tweets data on cresci-2015 to speed up calculation
│   └── content_features.py  # content features
│   └── content_feature_extraction.py  # calculate content features on cresci-2015
├── cresci-2017 
│   └── tweet_split.py 
│   └── content_features.py 
│   └── content_feature_extraction.py  
├── Twibot-20    
│   └── tweet_split.py  
│   └── content_features.py 
│   └── content_feature_extraction.py  
├── Twibot-22
│   └── tweet_split.py  
│   └── content_features.py  
│   └── content_feature_extraction.py  
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
| cresci-2015 | mean | 98.23 | 98.65 | 98.46 | 98.56 |
| cresci-2015 | std  | 0.1389 | 0.142 | 0.1433 | 0.1133 |
| gilani-2017 | mean | 74.81 | 77.58 | 60.19 | 67.78 |
| gilani-2017 | std  | 1.2138 | 1.3134 | 2.1516 | 1.8089 |
| cresci-2017 | mean | 98.85 | 99.56 | 99.13 | 99.35 |
| cresci-2017 | std  | 0.0680 | 0.0766 | 0.0000 | midterm-2018 | mean | 96.4 | 97.36 | 98.37 | 97.87 |
| midterm-2018 | std  | 0.1231 | 0.0701 | 0.0985 | 0.0732 |
| cresci-stock-2018 | mean | 81.53 | 84.75 | 80.3 | 82.46 |
| cresci-stock-2018 | std  | 0.3508 | 0.4156 | 0.63 | 0.3637 |
| cresci-rtbust-2019 | mean | 83.53 | 79.37 | 86.45 | 82.74 |
| cresci-rtbust-2019 | std  | 1.9174 | 2.9712 | 1.4426 | 1.7944 |
| botometer-feedback-2019 | mean | 75.47 | 58.97 | 44.0 | 50.34 |
| botometer-feedback-2019 | std  | 1.3342 | 3.2872 | 3.6515 | 3.1569 |
| Twibot-20   | mean | 77.36 | 76.6 | 83.66 | 79.98 |
| Twibot-20   | std  | 0.5276 | 0.3736 | 0.6929 | 0.501 |
| Twibot-22   | mean | 76.28 | 67.23 | 19.65 | 30.41 |
| Twibot-22  | std  | 0.0479 | 0.2875 | 0.1517 | 0.2005 |







| baseline | acc on Twibot-22 | f1 on Twibot-22 | type | tags|
| -------- | ---------------- | --------------- | ---- | --- |
| Lee et al.|76.28|30.41|P T|`random forest`|

