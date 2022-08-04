### Uncovering Social Media Bots:a Transparency-focused Approach


---

- **authors**: Eric F. Santos, Danilo S. Carvalho, Livia Ruback, Jonice Oliveira

- **link**: https://dl.acm.org/doi/pdf/10.1145/3308560.3317599

- **file structure**: 

```python
├── cresci-2015
├── dlc2015.py # convert raw dataset into standard format and save them
├── Twibot-20    
├── dltwi20.py # convert raw dataset into standard format and save them
├── cresci-2017
├── dlc2017.py # convert raw dataset into standard format and save them
├── midterm-2018
├── dlm2018.py # convert raw dataset into standard format and save them
├── gilani-2017    
├── dlg2017.py # convert raw dataset into standard format and save them
├── cresci-stock-2018
├── dlcs2018.py # convert raw dataset into standard format and save them
├── cresci-rtbust-2019
├── dlcr2019.py # convert raw dataset into standard format and save them
├── botometer-feedback-2019   
├── dlbf2019.py # convert raw dataset into standard format and save them
└── dt.py       # train a decision tree
```


- **implement details**: Since some datasets don’t contain contents of tweets which users posted, we extracted features from user’s description if the dataset we use doesn’t have content of tweets.

  

#### How to reproduce:(Twibot-20 For example)

1. convert the raw dataset into standard format by running 

   `python dltwi20.py`

   this command will create related features in corresponding directory.

2. open:

   `python gcntwi22.py`

   then change filename into features created by first command and change the path to datasets in codes in line31-line34
   
   




#### Result:


| dataset     |      | acc    | precison | recall | f1     |
| ----------- | ---- | ------ | -------- | ------ | ------ |
| Twibot-20   | mean | 0.5866 | 0.6273   | 0.5813 | 0.6034 |
| Twibot-20   | std  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |
| Cresci-2015 | mean | 0.7084 | 0.7286   | 0.8580 | 0.7880 |
| Cresci-2015 | std  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |
| Cresci-2017 | mean | 0.7384 | 0.8171   | 0.8440 | 0.8303 |
| Cresci-2017 | std  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |
| midterm-2018| mean | 0.8661 | 0.8805   | 0.9724 | 0.9242 |
| midterm-2018| std  | 0.0001 | 0.0000   | 0.0000 | 0.0000 |
| gilani-2017 | mean | 0.5144 | 0.3226   | 0.0935 | 0.1449 |
| gilani-2017 | std  | 0.0000 | 0.0000   | 0.0004 | 0.0000 |
| cresci-stock-2018| mean | 0.6245 | 0.6539   | 0.6495 | 0.6517 |
| cresci-stock-2018| std  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |
| cresci-rtbust-2019| mean | 0.7353 | 0.7568   | 0.7568 | 0.7568 |
| cresci-rtbust-2019| std  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |
| botometer-feedback-2019| mean | 0.7170 | 0.5000   | 0.1333 | 0.2105 |
| botometer-feedback-2019| std  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |








| baseline | acc on Twibot-22 | f1 on Twibot-22 | type | tags|
| -------- | ---------------- | --------------- | ---- | --- |
| Santos et al.|/|/|F T|`decision tree`|

