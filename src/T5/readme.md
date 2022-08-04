### T5

---

- **authors**: Raffel et al.

- **link**: https://www.jmlr.org/papers/volume21/20-074/20-074.pdf

- **file structure**: 

```python
├── cresci-2015
│   ├── concat.ipynb
│   ├── train.py
│   ├── des_embedding.py
│   ├── tweets_tensor.py
│   └── user_tweets_dict.py
├── cresci-2017
│   ├── concat.ipynb
│   ├── train.py
│   ├── des_embedding.py
│   ├── tweets_tensor.py
│   └── user_tweets_dict.py
├── Twibot-20
│   ├── id_list.py
│   ├── des_embedding.py
│   ├── label_list.py
│   ├── train.py
│   ├── tweets_tensor.py
│   └── user_tweets_dict.py
└── Twibot-22
    ├── concat.ipynb
    ├── des_embedding.py
    ├── id_list.py
    ├── label_list.py
    ├── train.py
    ├── twi0.py
    ├── twi1.py
    ├── twi10.py
    ├── twi2.py
    ├── twi3.py
    ├── twi4.py
    ├── twi5.py
    ├── twi6.py
    ├── twi7.py
    ├── twi8.py
    ├── twi9.py
    └── user_tweets_dict.py
```

- **implement details**: For Twibot-22, users' tweet counts could be cut to 20 for time consumption issue.

  

#### How to reproduce:

1. run id_list.py first to generate id_list.json and run twi0-10.py for Twibot-22 and tweets_tensor.py for others and run des_embedding.py to generate tweets' and des' embeddings. run label_list.py to get label.

2. finetune T5 by running train.py(need to change the path of data in the code)



#### Result:

| dataset     |      | acc    | precison | recall | f1     |
| ----------- | ---- | ------ | -------- | ------ | ------ |
| Cresci-2015 | mean | 0.9224 | 0.9114   | 0.8744 | 0.8925 |
| Cresci-2015 | std  | 0.0011 | 0.0024   | 0.0025 | 0.0015 |
| Cresci-2017 | mean | 0.9638 | 0.9415   | 0.9064 | 0.9236 |
| Cresci-2017 | std  | 0.0004 | 0.0016   | 0.0017 | 0.0009 |
| Twibot-20   | mean | 0.7667 | 0.7587   | 0.7219 | 0.7359 |
| Twibot-20   | std  | 0.0008 | 0.0133   | 0.0244 | 0.0008 |
| Twibot-22   | mean | 0.7170 | 0.6364   | 0.0894 | 0.1565 |
| Twibot-22   | std  | 0.0026 | 0.0133   | 0.0140 | 0.0218 |







| baseline | acc on Twibot-22 | f1 on Twibot-22 | type | tags|
| -------- | ---------------- | --------------- | ---- | --- |
| T5|0.7170|0.1565|T|`T5`|