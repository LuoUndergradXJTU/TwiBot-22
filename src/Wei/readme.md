### Twitter bot detection using bidirectional long short-term memory neural networks and word embeddings
---

- **authors**: Feng Wei, Uyen Trang Nguyen

- **link**: https://arxiv.org/pdf/2002.01336.pdf

- **file structure**: 

```
├── Twibot-20
│   ├── preprocess.py 
│   ├── bilstm_attention.py 
│   └── data_processor.py   
└── Twibot-22
    ├── preprocess.py 
    ├── bilstm_attention.py 
    └── data_processor.py 

```

- **implement details**: We set some parameters by ourselves. We set the embedding size to 200, the max length of sentence to 64, the hidden layer to 64, and the size of vocabulary to 5000.

  

#### How to reproduce:

1. convert the raw dataset into standard format by running 

   `python preprocess.py `

   this command will create related features in corresponding directory.

2. train bilstm model by running:

   `python bilstm_attention.py`

   the final result will be showed.



#### Result:

random seed: 0, 100, 200, 300, 400

| dataset     |      | acc   | precison| recall| f1    |
| ----------- | ---- | ----- | ------- | ----- | ----- |
| Cresci-2015 | mean | 0.961 | 0.917   | 0.753 | 0.827 |
| Cresci-2015 | std  | 0.014 | 0.017   | 0.015 | 0.022 |
| Cresci-2017 | mean | 0.893 | 0.859   | 0.721 | 0.784 |
| Cresci-2017 | std  | 0.007 | 0.019   | 0.015 | 0.017 |
| Twibot-20   | mean | 0.713 | 0.610   | 0.540 | 0.573 |
| Twibot-20   | std  | 0.016 | 0.021   | 0.027 | 0.031 |
| Twibot-22   | mean | 0.702 | 0.627   | 0.468 | 0.536 |
| Twibot-22   | std  | 0.012 | 0.018   | 0.014 | 0.013 |







| baseline | acc on Twibot-22 | f1 on Twibot-22 | type | tags|
| -------- | ---------------- | --------------- | ---- | --- |
| Varol et al.|0.702|0.536|T||

