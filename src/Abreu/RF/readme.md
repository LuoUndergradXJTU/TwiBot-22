### Twitter Bot Detection with Reduced Feature Set

---

- **authors**: Jefferson Viana Fonseca Abreu,Celia Ghedini Ralha, Joao Jose Costa Gondim
- **link** : https://ieeexplore.ieee.org/abstract/document/9280525
- **file structure**: 

```python
└── all datasets
    └── twi.py  # train model on cresci-2015
```

- **implement details**: We choose the algorithm which performs best in the origin paper. And due to many datasets don't have the "favourite count", we don't take the feature into count.

  

#### How to reproduce:

1. specify the dataset  by changing `dataset=Twibot-22` in twi.py (Twibot-22 for example) ;

2. train random forest model by running:

   `python twi.py`

   the final result will be saved into ''dataset name''.txt



#### Result:

random seed: 100, 200, 300, 400, 500

| dataset                 |      | acc    | precison | recall | f1     |
| ----------------------- | ---- | ------ | -------- | ------ | ------ |
| Cresci-2015             | mean | 0.7570 | 0.9905   | 0.6213 | 0.7636 |
| Cresci-2015             | std  | 0.0058 | 0.0021   | 0.0097 | 0.0072 |
| Twibot-20               | mean | 0.7345 | 0.7220   | 0.8281 | 0.7714 |
| Twibot-20               | std  | 0.0056 | 0.0052   | 0.0051 | 0.0046 |
| Twibot-22               | mean | 0.7066 | 0.5092   | 0.1173 | 0.5344 |
| Twibot-22               | std  | 0.0001 | 0.0010   | 0.0006 | 0.0009 |
| Cresci-2017             | mean | 0.9273 | 0.9834   | 0.9197 | 0.9504 |
| Cresci-2017             | std  | 0.0040 | 0.0013   | 0.0069 | 0.0030 |
| cresci-rtbust-2019      | mean | 0.8088 | 0.7857   | 0.8918 | 0.8354 |
| cresci-rtbust-2019      | std  | 0.0130 | 0.0144   | 0.0140 | 0.0104 |
| cresci-stock-2018       | mean | 0.7545 | 0.7545   | 0.7567 | 0.7693 |
| cresci-stock-2018       | std  | 0.0057 | 0.0045   | 0.0073 | 0.0058 |
| midterm-2018            | mean | 0.9653 | 0.9728   | 0.9863 | 0.9795 |
| midterm-2018            | std  | 0.0006 | 0.0007   | 0.0008 | 0.0003 |
| gilani-2017             | mean | 0.7428 | 0.7682   | 0.5887 | 0.6666 |
| gilani-2017             | std  | 0.0065 | 0.0120   | 0.0275 | 0.001  |
| botometer-feedback-2019 | mean | 0.7735 | 0.6363   | 0.4666 | 0.5384 |
| botometer-feedback-2019 | std  | 0.0157 | 0.0360   | 0.0300 | 0.0303 |







| baseline     | acc on Twibot-22 | f1 on Twibot-22 | type | tags            |
| ------------ | ---------------- | --------------- | ---- | --------------- |
| Abreu et al. | 0.7066           | 0.5344          | F    | `random forest` |

