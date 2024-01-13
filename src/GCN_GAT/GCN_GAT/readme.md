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
| Cresci-2015 | mean | 0.9637 | 0.9559   | 0.9881 | 0.9717 |
| Cresci-2015 | std  | 0.0057 | 0.0069   | 0.0020 | 0.0043 |
| Twibot-20   | mean | 0.7753 | 0.7523   | 0.8762 | 0.8086 |
| Twibot-20   | std  | 0.0173 | 0.0308   | 0.0331 | 0.0068 |
| Twibot-22   | mean | 0.7839 | 0.7119   | 0.4480 | 0.5496 |
| Twibot-22   | std  | 0.0009 | 0.0128   | 0.0171 | 0.0091 |

GAT

| dataset     |      | acc    | precison | recall | f1     |
| ----------- | :--- | ------ | -------- | ------ | ------ |
| Cresci-2015 | mean | 0.9689 | 0.9610   | 0.9911 | 0.9758 |
| Cresci-2015 | std  | 0.0021 | 0.0071   | 0.0051 | 0.0015 |
| Twibot-20   | mean | 0.8327 | 0.8139   | 0.8953 | 0.8525 |
| Twibot-20   | std  | 0.0056 | 0.0118   | 0.0087 | 0.0038 |
| Twibot-22   | mean | 0.7948 | 0.7623   | 0.4412 | 0.5586 |
| Twibot-22   | std  | 0.0009 | 0.0139   | 0.0165 | 0.0101 |

| baseline          | acc on Twibot-22 | f1 on Twibot-22 | type  | tags                   |
| ----------------- | ---------------- | --------------- | ----- | ---------------------- |
| Kipf et al.       | 0.7839           | 0.5496          | F T G | `Graph Neural Network` |
| Velickovic et al. | 0.7948           | 0.5586          | F T G | `Graph Neural Network` |

