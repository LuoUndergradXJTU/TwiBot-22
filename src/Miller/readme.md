### Twitter spammer detection using data stream clustering

---

- **authors**: Zachary Miller, Brian Dickinson, William Deitrick, Wei Hu, Alex Hai Wang

- **link**: https://dl.acm.org/doi/10.1016/j.ins.2013.11.016

- **file structure**: 

```python
├── preprocess.py       # generate required features
└── stream_cluster.py   # train model on given dataset
```

- **implement details**: “Re-tweet count” feature is discarded since required information is not included in datasets. “Link count”, “Reply/mention count”, and “Hashtag count” are available only in “cresci-2015”, “cresci-2017”, “Twibot-20”, and “Twibot-22” datasets. 

  

#### How to reproduce:

1. specify the dataset by running `dataset=Twibot-22` (Twibot-22 for example) ;

2. generate required features from raw dataset by running: 

   `python preprocess.py --dataset ${dataset}`

3. train clustering model by running:

   `python stream_cluster.py --dataset ${dataset}`



#### Result:

random seed: 100, 200, 300, 400, 500

| dataset                  |      | acc    | precison | recall | f1     |
| ------------------------ | ---- | ------ | -------- | ------ | ------ |
| Cresci-2015              | mean | 0.7551 | 0.7207   | 1.0000 | 0.8377 |
| Cresci-2015              | std  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |
| Cresci-2017              | mean | 0.7713 | 0.7721   | 0.9911 | 0.8680 |
| Cresci-2017              | std  | 0.0017 | 0.0018   | 0.0011 | 0.0007 |
| Twibot-20                | mean | 0.6450 | 0.6071   | 0.9744 | 0.7481 |
| Twibot-20                | std  | 0.0035 | 0.0020   | 0.0047 | 0.0026 |
| Twibot-22                | mean | 0.3037 | 0.2946   | 0.9789 | 0.4529 |
| Twibot-22                | std  | 0.0001 | 0.0000   | 0.0001 | 0.0000 |
| botometer-feedback-2019  | mean | 0.7736 | 0.0000   | 0.0000 | 0.0000 |
| botometer-feedback-2019  | std  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |
| cresci-rtbust-2019       | mean | 0.5441 | 0.5217   | 0.3750 | 0.4364 |
| cresci-rtbust-2019       | std  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |
| cresci-stock-2018        | mean | 0.5253 | 0.5478   | 0.5889 | 0.5676 |
| cresci-stock-2018        | std  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |
| midterm-2018             | mean | 0.8372 | 0.8385   | 0.9981 | 0.9114 |
| midterm-2018             | std  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |
| gilani-2017              | mean | 0.5104 | 0.4889   | 0.7719 | 0.5986 |
| gilani-2017              | std  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |







| baseline     | acc on Twibot-22 | f1 on Twibot-22 | type | tags|
| ------------ | ---------------- | --------------- | ---- | --- |
| Miller et al.|0.3037|0.4529|F T|`k means`|

