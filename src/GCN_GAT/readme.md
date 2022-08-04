

### GCN and GAT

---

- **authors** **GCN** : **Thomas N. Kipf** , **Max Welling**
- **authors** **GAT**: **Petar Velickovic**, **Guillem Cucurull**, **Arantxa Casanova**, **Arantxa Casanova**, **Pietro Lio**, **Yoshua Bengio**
- **link**: https://arxiv.org/abs/1609.02907
- **link**: https://arxiv.org/abs/1710.10903
- **file structure**: 

```python
└── cresci-2015,Twibot-20,Twibot-22
    ├── main.py  # train model on the processed data
    ├── dataset.py  # preprocess data
    ├── utils.py  
    └── model.py  # load GCN/GAT model 
```

#### How to reproduce:

1. specify the dataset by running `dataset=Twibot-22` in Dataset.py (Twibot-22 for example) ;

2. change the model in the model.py

3. train model by running:

   `python main.py`

   



#### Result:

random seed: 100, 200, 300, 400, 500

GCN

| dataset     |      | acc    | precison | recall | f1     |
| ----------- | ---- | ------ | -------- | ------ | ------ |
| Cresci-2015 | mean | 0.9626 | 0.9622   | 0.9793 | 0.9707 |
| Cresci-2015 | var  | 0.001  | 0.001    | 0.001  | 0.001  |
| Twibot-20   | mean | 0.7582 | 0.769    | 0.7906 | 0.7797 |
| Twibot-20   | var  | 0.001  | 0.001    | 0.001  | 0.001  |
| Twibot-22   | mean | 0.7489 | 0.3702   | 0.7070 | 0.2513 |
| Twibot-22   | var  | 0.004  | 0.026    | 0.008  | 0.024  |

GAT

| dataset     |      | acc    | precison | recall | f1     |
| ----------- | :--- | ------ | -------- | ------ | ------ |
| Cresci-2015 | mean | 0.9645 | 0.9544   | 0.9911 | 0.9724 |
| Cresci-2015 | var  | 0.001  | 0.001    | 0.001  | 0.001  |
| Twibot-20   | mean | 0.7887 | 0.7686   | 0.8719 | 0.8170 |
| Twibot-20   | var  | 0.001  | 0.001    | 0.001  | 0.001  |
| Twibot-22   | mean | 0.7585 | 0.7585   | 0.3226 | 0.4394 |
| Twibot-22   | var  | 0.004  | 0.004    | 0.03   | 0.027  |

| baseline          | acc on Twibot-22 | f1 on Twibot-22 | type  | tags                   |
| ----------------- | ---------------- | --------------- | ----- | ---------------------- |
| Kipf et al.       | 0.7489           | 0.2513          | F T G | `Graph Neural Network` |
| Velickovic et al. | 0.7585           | 0.4394          | F T G | `Graph Neural Network` |

