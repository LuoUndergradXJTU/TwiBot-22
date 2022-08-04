### Friendship Preference: Scalable and Robust Category of Features for Social Bot Detection

---

- **authors**: Samaneh Hosseini Moghaddam, Maghsoud Abbaspour

- **link**: https://ieeexplore.ieee.org/abstract/document/9735340

- **file structure**: 

```python
├── process.py # convert raw dataset into standard format
└── train.py # train model on every dataset

```

- **implement details**: “Favorites count” is discarded since required information is not included in datasets.


#### How to reproduce:

1. process data, get the features and specify the dataset by running;
   
    `python process.py --datasets "dataset name"`

    there will be a .npy file, which is the processed feature

2. train random forest model and specify the dataset by running:

   `python train.py --datasets "dataset name" > result.txt`

   the final result will be saved into result.txt



#### Result:

random seed: 100, 200, 300, 400, 500

| dataset     |      | acc    | precison | recall | f1     |
| ----------- | ---- | ------ | -------- | ------ | ------ |
| cresci-2015 | mean | 0.7361 | 0.9833   | 0.5923 | 0.7393 |
| cresci-2015 | std  | 0.0016 | 0.0026   | 0.0032 | 0.0021 |
| Twibot-20   | mean | 0.7405 | 0.7229   | 0.8438 | 0.7787 |
| Twibot-20   | std  | 0.0080 | 0.0067   | 0.0103 | 0.0071 |
| Twibot-22   | mean | 0.7378 | 0.6761   | 0.2102 | 0.3207 |
| Twibot-22   | std  | 0.0001 | 0.0010   | 0.0007 | 0.0003 |




| baseline | acc on Twibot-22 | f1 on Twibot-22 | type | tags|
| -------- | ---------------- | --------------- | ---- | --- |
| Moghaddam et al.|0.7378|0.3207|F G|`random forest`|

