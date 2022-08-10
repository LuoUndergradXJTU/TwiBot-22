### Graph-Hist: Graph Classification from Latent Feature Histograms With Application to Bot Detection

---

- **authors**: [Thomas Magelinski](https://arxiv.org/search/cs?searchtype=author&query=Magelinski%2C+T), [David Beskow](https://arxiv.org/search/cs?searchtype=author&query=Beskow%2C+D), [Kathleen M. Carley](https://arxiv.org/search/cs?searchtype=author&query=Carley%2C+K+M)

- **link**: https://arxiv.org/abs/1910.01180

- **file structure**: 

```python
├── model.py       # define pytorch model and histogram operator
├── preprocess.py  # preprocess the raw data and build graph
└── train.py       # training
```

- **implement details**: The embeddings are adapted from binchi zhang. The histogram operator are our own implementation since the official one is not provided.

  

#### How to reproduce:

1. train model by running:

   `python train.py with params.dataset=${dataset} >> ${dataset}/result.txt`

   the final result will be saved into result.txt



#### Result:

random seed: 100, 200, 300, 400, 500

| dataset   |      | acc    | precison | recall | f1     |
| --------- | ---- | ------ | -------- | ------ | ------ |
| cresci-2015 | mean | 0.7738 | 0.7312   | 1.0000 | 0.8447 |
| cresci-2015 | std  | 0.002  | 0.001    | 0.0    | 0.0823 |
| Twibot-20 | mean | 0.5133 | 0.5127   | 0.9905 | 0.6756 |
| Twibot-20 | std  | 0.003  | 0.002    | 0.002    | 0.003 |





| baseline  | acc on Twibot-22 | f1 on Twibot-22 | type  | tags            |
| --------- | ---------------- | --------------- | ----- | --------------- |
| GraphHist | /                | /               | F T G | `random forest` |
