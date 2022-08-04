### Online Human-Bot Interactions: Detection, Estimation, and Characterization

---

- **authors**: Onur Varol, Emilio Ferrara, Clayton A. Davis, Filippo Menczer, Alessandro Flammini

- **link**: https://aaai.org/ocs/index.php/ICWSM/ICWSM17/paper/view/15587/14817 

- **file structure**: 

```python
├── cresci-2015
│   └── train.py  # train model on cresci-2015
├── preprocess.py # convert raw dataset into standard format
├── sta.py
├── Twibot-20    
│   └── train.py  # train model on Twibot-20
└── Twibot-22
    └── train.py  # train model on Twibot-22
```

- **implement details**: “Sentiment”, “Timing” features are discarded since required information is not included in datasets.

  

#### How to reproduce:

1. specify the dataset b y running `dataset=Twibot-22` (Twibot-22 for example) ;

2. convert the raw dataset into standard format by running 

   `python preprocess.py --datasets ${dataset}`

   this command will create related features in corresponding directory.

3. train random forest model by running:

   `cd ${dataset} && python train.py > result.txt`

   the final result will be saved into result.txt



#### Result:

random seed: 100, 200, 300, 400, 500

| dataset     |      | acc    | precison | recall | f1     |
| ----------- | ---- | ------ | -------- | ------ | ------ |
| Cresci-2015 | mean | 0.9316 | 0.9222   | 0.9740 | 0.9473 |
| Cresci-2015 | std  | 0.0054 | 0.0066   | 0.0090 | 0.0042 |
| Twibot-20   | mean | 0.7874 | 0.7804   | 0.8437 | 0.8108 |
| Twibot-20   | std  | 0.0055 | 0.0061   | 0.0067 | 0.0048 |
| Twibot-22   | mean | 0.7392 | 0.7574   | 0.1683 | 0.2754 |
| Twibot-22   | std  | 0.0002 | 0.0031   | 0.0021 | 0.0026 |







| baseline | acc on Twibot-22 | f1 on Twibot-22 | type | tags|
| -------- | ---------------- | --------------- | ---- | --- |
| Varol et al.|0.7392|0.2754|P T|`random forest`|

