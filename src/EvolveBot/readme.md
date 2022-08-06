### Empirical Evaluation and New Design for Fighting Evolving Twitter Spammers

---

- **authors**: Chao Yang, Robert Harkreader, and Guofei Gu

- **link**: https://ieeexplore.ieee.org/abstract/document/6553246

- **file structure**: 

```bash
└── baseline
	├── data_cresci15.py	# convert raw dataset into standard format on cresci-2015
	├── data_twibot20.py	# convert raw dataset into standard format on Twibot-20
	├── data_twibot22.py	# convert raw dataset into standard format on Twibot-22
	├── solve_cresci15.py	# train model on cresci-15
	├── solve_twibot20.py	# train model on Twibot-20
	├── solve_twibot22.py	# train model on Twibot-22
	├── solve_twibot20_nograph.py	# train model on Twibot-20 without graph features
	└── solve_twibot22_nograph.py	# train model on Twibot-22 without graph features
```

- **implement details**: 

  We only use 11 of 18 features. Firstly, we ignore the 3 automation features due to the lack of the information in our datasets. Secondly, fofo ratio and bi-directional links ratio referred in the paper are similar but different. The former means the number of friend divided by the number of followings according to the profile, while the latter means the number of friend on graph divided by the number of followings on graph. However, the number of friend in profile is not available in our datasets. Thus we only use bi-directional links ratio and ignore fofo ratio. Thirdly, 2 graph features, betweenness centrality and clustering coefficient, perform ordinarily and cost too much time. Therefore, we do not use them. Fourthly, tweet rate is not used due to the lack of the information.  
  
  Moreover, we use different methods to handle tweet similarity. The original paper use TF-IDF, but we use bert-tiny to produce features of each tweet and perform vectorial angle as similarity.
  
  For cresci-15, besides the original dataset, 'user_info.pt' is also needed for data preprocessing.
  
  

#### How to reproduce:

1. convert the raw dataset into standard format by running:(${dataset} is the dataset used)

   `python data_${dataset}.py`

2. train random forest model with random_seed by running:

   `python solve_${dataset}.py ${random_seed} `

   the final result will be output.

3. For example, we need to run:

   `python data_twibot20.py` and `python solve_twibot20.py 100 `to reproduce on Twibot-20 with random_seed = 100



#### Result:

random seed: 0, 100, 200, 300, 400

| dataset     |      | acc    | precison | recall | f1     |
| ----------- | ---- | ------ | -------- | ------ | ------ |
| Cresci-2015 | mean | 0.9218 | 0.8503   | 0.9583 | 0.9007 |
| Cresci-2015 | std  | 0.0174 | 0.0377   | 0.0066 | 0.0198 |
| Twibot-20   | mean | 0.6583 | 0.6693   | 0.7281 | 0.6975 |
| Twibot-20   | std  | 0.0063 | 0.0060   | 0.0041 | 0.0050 |
| Twibot-22   | mean | 0.7109 | 0.5638   | 0.0804 | 0.1409 |
| Twibot-22   | std  | 0.0003 | 0.0040   | 0.0005 | 0.0008 |







| baseline | acc on Twibot-22 | f1 on Twibot-22 | type | tags|
| -------- | ---------------- | --------------- | ---- | --- |
| Yang et al. |0.7109|0.1409|F T G|`random forest`|

