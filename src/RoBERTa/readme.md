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
| Cresci-2015 | mean | 0.9701 | 0.9758   | 0.9411 | 0.9586 |
| Cresci-2015 | std  | 0.0013 | 0.0027   | 0.0058 | 0.0019 |
| Cresci-2017 | mean | 0.9719 | 0.9243   | 0.9627 | 0.9430 |
| Cresci-2017 | std  | 0.0009 | 0.0099   | 0.0105 | 0.0018 |
| Twibot-20   | mean | 0.7555 | 0.7388   | 0.7238 | 0.7309 |
| Twibot-20   | std  | 0.0018 | 0.0106   | 0.0205 | 0.0059 |
| Twibot-22   | mean | 0.7207 | 0.6328   | 0.1227 | 0.2053 |
| Twibot-22   | std  | 0.0016 | 0.0090   | 0.0122 | 0.0171 |








| baseline | acc on Twibot-22 | f1 on Twibot-22 | type | tags|
| -------- | ---------------- | --------------- | ---- | --- |
| RoBERTa|0.7207|0.2053|F T|`RoBERTa`|
