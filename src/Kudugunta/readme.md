### Deep neural networks for bot detection

---

- **authors**: Sneha Kudugunta, Emilio Ferrara

- **link**: https://arxiv.org/abs/1802.04289

- **file structure**: 

```python
└── train.py # train model on every dataset

```

- **implement details**: “Favorite count” is discarded since required information is not included in datasets.

  

#### How to reproduce:

1. train random forest model by and specify the dataset by running:

   `python train.py --datasets ${dataset} > result.txt`

   the final result will be saved into result.txt



#### Result:

random seed: 100, 200, 300, 400, 500

| dataset     |      | acc    | precison | recall | f1     |
| ----------- | ---- | ------ | -------- | ------ | ------ |
| cresci-2015 | mean | 0.7533 | 1.0000   | 0.6095 | 0.7574 |
| Cresci-2015 | std  | 0.0013 | 0.0000   | 0.0021 | 0.0016 |
| Twibot-20   | mean | 0.5959 | 0.8040   | 0.3347 | 0.4726 |
| Twibot-20   | std  | 0.0065 | 0.0060   | 0.0130 | 0.0135 |
| Twibot-22   | mean | 0.6587 | 0.4431   | 0.6198 | 0.5167 |
| Twibot-22   | std  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |
| cresci-2017 | mean | 0.8832 | 0.9853   | 0.8588 | 0.9174 |
| cresci-2017 | std  | 0.0021 | 0.0019   | 0.0037 | 0.0017 |
| cr-2019     | mean | 0.6294 | 0.6609   | 0.5067 | 0.4922 |
| cr-2019     | std  | 0.0081 | 0.0235   | 0.0121 | 0.0128 |
| bf-2019     | mean | 0.7396 | 0.5667   | 0.4533 | 0.4961 |
| bf-2019     | std  | 0.0470 | 0.1077   | 0.0869 | 0.0820 |
| cs-2018     | mean | 0.7753 | 0.5487   | 0.4754 | 0.5094 |
| cs-2018     | std  | 0.0014 | 0.0047   | 0.0060 | 0.0038 |
| midterm-2018     | mean | 0.9109 | 0.9906   | 0.9024 | 0.9445 |
| midterm-2018     | std  | 0.0049 | 0.0016   | 0.0066 | 0.0032 |
| gilani-2017     | mean | 0.7004 | 0.8544   | 0.3514 | 0.4975 |
| gilani-2017     | std  | 0.0105 | 0.0242   | 0.0170 | 0.0210 |







| baseline | acc on Twibot-22 | f1 on Twibot-22 | type | tags|
| -------- | ---------------- | --------------- | ---- | --- |
| Kudugunta et al.|0.6587|0.5167|F|`SMOTENN, random forest`|

