### Detect Me If You Can: Spam Bot Detection Using Inductive Representation Learning

---

- **authors**: Seyed Ali Alhosseini, Raad Bin Tareaf, Pejman Najaf, Christoph Meinel

- **link**: https://dl.acm.org/doi/fullHtml/10.1145/3308560.3316504

- **file structure**: 

```python
├── cresci-2015
│   └── gcn2.py  # train model on cresci-2015
├── dataload2.py # convert raw dataset into standard format and save them
├── Twibot-20    
│   └── gcn.py  # train model on Twibot-20
├── dataload1.py # convert raw dataset into standard format and save them
├── Twibot-22
│   └── gcntwi22.py  # train model on Twibot-22
└── twi22.py # convert raw dataset into standard format and save them
```

- **implement details**: We ignored favourites_count due to the lack of the information in our datasets. When calculating the age of users in twibot22, we use may 1, 2022 minus the date when the user account was created as the age; When calculating the age of users in other datasets, we use October 1, 2020 minus the date when the user account was created as the age.

  

#### How to reproduce:(Twibot-22 for example)

1. convert the raw dataset into standard format by running 

   `python twi22.py`

   this command will create related features in corresponding directory.

2. train GCN model by running:

   `python gcntwi22.py`





#### Result:


| dataset     |      | acc    | precison | recall | f1     |
| ----------- | ---- | ------ | -------- | ------ | ------ |
| Cresci-2015 | mean | 0.8935 | 0.9071   | 0.8655 | 0.8800 |
| Cresci-2015 | std  | 0.0057 | 0.0024   | 0.0098 | 0.0075 |
| Twibot-20   | mean | 0.6446 | 0.5895   | 0.6317 | 0.6023 |
| Twibot-20   | std  | 0.0603 | 0.1626   | 0.0721 | 0.1290 |
| Twibot-22   | mean | 0.6910 | 0.5810   | 0.5378 | 0.4991 |
| Twibot-22   | std  | 0.0155 | 0.0142   | 0.0419 | 0.0625 |







| baseline | acc on Twibot-22 | f1 on Twibot-22 | type | tags|
| -------- | ---------------- | --------------- | ---- | --- |
| Alhosseini et al.|0.6910|0.4991|F G|`gcn`|

