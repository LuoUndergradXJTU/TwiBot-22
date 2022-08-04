### RoBERTa

---

- **authors**: Liu et al.

- **link**: https://arxiv.org/pdf/1907.11692.pdf%5C

- **file structure**: 

```python
├── cresci-2015
│   ├── id_list.py
│   ├── label_list.py
│   ├── train.py
│   ├── des_embedding.py
│   ├── tweets_tensor.py
│   └── user_tweets_dict.py
├── cresci-2017
│   ├── id_list.py
│   ├── label_list.py
│   ├── train.py
│   ├── des_embedding.py
│   ├── tweets_tensor.py
│   └── user_tweets_dict.py
├── Twibot-20
│   ├── id_list.py
│   ├── label_list.py
│   ├── train.py
│   ├── des_
│   ├── tweets_tensor.py
│   └── user_tweets_dict.py
└── Twibot-22
    ├── id_list.py
    ├── label_list.py
    ├── des_embedding.py
    ├── train.py
    ├── tweets_tensor.py
    └── user_tweets_dict.py
```

- **implement details**: For Twibot-22, users' tweet counts could be cut to 20 for time consumption issue.

  

#### How to reproduce:

1. run id_list.py first to generate id_list.json and run tweets_tensor.py,des_embedding.py to generate tweets' and des' embeddings. run label_list.py to get label.

2. finetune RoBERTa by running train.py(need to change the path of data in the code)



#### Result:


| dataset     |      | acc    | precison | recall | f1     |
| ----------- | ---- | ------ | -------- | ------ | ------ |
| Cresci-2015 | mean | 0.9657 | 0.9735   | 0.9323 | 0.9525 |
| Cresci-2015 | std  | 0.0022 | 0.0002   | 0.0031 | 0.0029 |
| Cresci-2017 | mean | 0.9702 | 0.9153   | 0.9663 | 0.9401 |
| Cresci-2017 | std  | 0.0017 | 0.0078   | 0.0075 | 0.0033 |
| Twibot-20   | mean | 0.7529 | 0.7367   | 0.7188 | 0.7275 |
| Twibot-20   | std  | 0.0005 | 0.0089   | 0.0049 | 0.0013 |
| Twibot-22   | mean | 0.7196 | 0.6338   | 0.1130 | 0.1915 |
| Twibot-22   | std  | 0.0023 | 0.0090   | 0.0153 | 0.0220 |







| baseline | acc on Twibot-22 | f1 on Twibot-22 | type | tags|
| -------- | ---------------- | --------------- | ---- | --- |
| RoBERTa|0.7196|0.1915|F T|`RoBERTa`|
