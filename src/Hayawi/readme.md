### DeeProBot: a hybrid deep neural network model for social bot detection based on user profile data

---

- **authors**: Kadhim Hayawi, Sujith Mathew, Neethu Venugopal, Mohammad M. Masud, Pin‑Han Ho

- **link**: https://link.springer.com/article/10.1007/s13278-022-00869-w

- **file structure**: 

```python
├── boto-19/
├── cresci-18/
├── cresci-17/
├── cresci-19/
├── twibot-22/
├── midterm-18/
├── cresci-15/
├── twibot-20/
├── gilani-17/
├── run.py 
└── Hayawi.md # README
```

- **implement details**: “Sentiment”, “Timing” features are discarded since required information is not included in datasets.



#### How to reproduce:

1. preprocess the dataset and train this model by running 

   ``python run.py {dataset name}``

   twibot-22 for example :

   `python run.py twibot-22`

   - dataset names:
     - boto-19
     - cresci-18
     - cresci-17
     - cresci-19
     - twibot-22
     - midterm-18
     - cresci-15
     - twibot-20
     - gilani-17



#### Result:

| dataset                 |      | acc    | precison | recall | f1     |
| ----------------------- | ---- | ------ | -------- | ------ | ------ |
| Cresci-2015             | mean | 0.8427 | 0.9296   | 0.7931 | 0.8556 |
| Cresci-2015             | std  | 0.0002 |0.0003 |0.0002 |0.0001 |
| Twibot-20               | mean | 0.7314 | 0.7161   | 0.8350 | 0.7705 |
| Twibot-20               | std  | 0.0001 |0.0001 |0.0004 |0.0002 |
| Twibot-22               | mean | 0.7650 | 0.8000   | 0.1499 | 0.2474 |
| Twibot-22               | std  | 0.0007 |0.0027 |0.0005 |0.0008 |
| Gilani-17               | mean | 0.5270 | 0.5144   | 0.2800 | 0.3467 |
| Gilani-17               | std  | 0.0002 |0.0005 |0.0013 |0.0011 |
| Cresci-2017             | mean | 0.9078 | 0.9547   | 0.9219 | 0.9378 |
| Cresci-2017             | std  | 0.0001 |0.0001 |0.0003 |0.0001 |
| Cresci-stock-2018       | mean | 0.5002 | 0.5073   | 0.7116 | 0.6075 |
| Cresci-stock-2018       | std  | 0.0002 |0.0003 |0.0007 |0.0006 |
| Cresci-rtbust-2019      | mean | 0.5118 | 0.4882   | 0.8125 | 0.6087 |
| Cresci-rtbust-2019      | std  | 0.0002 |0.0001 |0.0009 |0.0003 |
| Midterm-2018            | mean | 0.8459 | 0.8530   | 0.9864 | 0.9148 |
| Midterm-2018            | std  | 0.0000 |0.0000 |0.0000 |0.0000 |
| Botometer-feedback-2019 | mean | 0.7698 | 0.2500   | 0.1778 | 0.2049 |
| Botometer-feedback-2019 | std  | 0.0002 |0.0006 |0.0006 |0.0006 |







| baseline | acc on Twibot-22 | f1 on Twibot-22 | type | tags|
| -------- | ---------------- | --------------- | ---- | --- |
| Hayawi et al. |0.7650|0.2474|F|`lstm`|

