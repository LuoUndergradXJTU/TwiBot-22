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
| Cresci-2015 | mean | 0.8957 | 0.8769   | 0.9716 | 0.9217 |
| Cresci-2015 | std  | 0.0060 | 0.0123   | 0.0081 | 0.0036 |
| Twibot-20   | mean | 0.5988 | 0.5781   | 0.9569 | 0.7207 |
| Twibot-20   | std  | 0.0059 | 0.0043   | 0.0193 | 0.0048 |
| Twibot-22   | mean | 0.4772 | 0.2999   | 0.5675 | 0.3810 |
| Twibot-22   | std  | 0.0871 | 0.0308   | 0.1769 | 0.0593 |







| baseline | acc on Twibot-22 | f1 on Twibot-22 | type | tags|
| -------- | ---------------- | --------------- | ---- | --- |
| Alhosseini et al.|0.4772|0.3810|F G|`gcn`|

