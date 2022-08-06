### You Are Known by Your Friends: Leveraging Network Metrics for Bot Detection in Twitter

---

- **authors**: David M. Beskow, Kathleen M. Carley

- **link**: https://link.springer.com/chapter/10.1007/978-3-030-41251-7_3

- **file structure**: 

```python
├── feature_engineering.py      # generate required features
├── feature_twibot22.py         # generate required features for Twibot-22 dataset
└── rand_forest.py              # train a random forest model on given dataset
```

- **implement details**: In all datasets except for Twibot-22, only following relationship is available so only one relationship i.e. following relationship is leveraged to construct ego networks.

  

#### How to reproduce:

1. specify the dataset b y running `dataset=Twibot-20` (Twibot-20 for example) ;

2. train random forest model by running:

   `python rand_forest.py --dataset ${dataset}`



#### Result:

random seed: 0, 100, 200, 300, 400

| dataset                  |      | acc    | precison | recall | f1     |
| ------------------------ | ---- | ------ | -------- | ------ | ------ |
| Cresci-2015              | mean | 0.9686 | 0.9529   | 1.0000 | 0.9758 |
| Cresci-2015              | std  | 0.0112 | 0.0162   | 0.0000 | 0.0084 |
| Cresci-2017              | mean | 0.7804 | 0.7755   | 1.0000 | 0.8735 |
| Cresci-2017              | std  | 0.0103 | 0.0081   | 0.0000 | 0.0052 |
| Twibot-20                | mean | 0.7589 | 0.7264   | 0.8894 | 0.7997 |
| Twibot-20                | std  | 0.0047 | 0.0052   | 0.0059 | 0.0034 |







| baseline              | acc on Twibot-22 | f1 on Twibot-22 | type | tags|
| --------------------- | ---------------- | --------------- | ---- | --- |
| FriendBot  |/|/|F T G|`random forest`|

