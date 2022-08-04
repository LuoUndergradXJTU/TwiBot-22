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
| Cresci-2015 | mean | 0.9694 | 0.9771   | 0.9374 | 0.9577 |
| Cresci-2015 | std  | 0.0011 | 0.0029   | 0.0029 | 0.0014 |
| Cresci-2017 | mean | 0.9718 | 0.9245   | 0.9660 | 0.9428 |
| Cresci-2017 | std  | 0.0010 | 0.0114   | 0.0120 | 0.0019 |
| Twibot-20   | mean | 0.7551 | 0.7410   | 0.7182 | 0.7291 |
| Twibot-20   | std  | 0.0024 | 0.0135   | 0.0242 | 0.0067 |
| Twibot-22   | mean | 0.7213 | 0.6389   | 0.1231 | 0.2061 |
| Twibot-22   | std  | 0.0016 | 0.0051   | 0.0160 | 0.0222 |







| baseline | acc on Twibot-22 | f1 on Twibot-22 | type | tags|
| -------- | ---------------- | --------------- | ---- | --- |
| RoBERTa|0.7213|0.2061|F T|`RoBERTa`|
