### LOBO: Evaluation of Generalization Deficiencies in Twitter Bot Classifiers

---

- **authors**: Juan Echeverrï£¡a, Emiliano De Cristofaro, Nicolas Kourtellis, Ilias Leontiadis, Gianluca Stringhini, and Shi Zhou

- **link**: https://dl.acm.org/doi/10.1145/3274694.3274738

- **file structure**: 

```python
├── cresci-2015
│   └── train.py  # train model on cresci-2015
├── cresci-2017
│   └── train.py  # train model on cresci-2017
├── Twibot-20    
│   └── train.py  # train model on Twibot-20
├── Twibot-22
│   ├── semi-processed_dataset # semi-processed dataset for training
│   └── train.py  # train model on Twibot-22
└── preprocess.py # convert raw dataset into standard format
```

- **implement details**: 
  
1. Some features(# User favorites, Seconds active, Days active, # Geolocated tweets, % of Geolocated Tweets, # Tweets analyzed, # Favorites, Favorites (per tweet), # APIs used, # Retweets analyzed) that are either unavailable from the datasets or computational costly are not included in cresci-15, cresci-17, Twibot-20.
2. Due to the large scale of Twibot-22, only a subset of tweets is used for some users to generate their feature vectors.
  
- **preliminary**:
Due to the scale of Twibot-22 dataset, preprocessing may take a long time and is memory-intensive. So we do not recommend that you preprocess the data by yourself. You can download preprocessed data from https://drive.google.com/file/d/1-KF8qfW0F3a5L2HBE0bhzMAL9m76tVOM/view?usp=sharing and 
unzip it to Twibot-22/semi-processed_dataset before your reproduction on Twibot-22.

#### How to reproduce:

1. specify the dataset by running `dataset=Twibot-22` (Twibot-22 for example) ;

2. convert the raw dataset into standard format by running 

   `python preprocess.py --datasets ${dataset}`

   this command will create related features in corresponding directory.

3. change to corresponding directory by running 
   `cd ${dataset}`

4. train random forest model by running:

   `python train.py > result.txt`

   the final result will be saved into result.txt



#### Result:

random seed: 100, 200, 300, 400, 500

| dataset     |      | acc    | precison | recall | f1     |
| ----------- | ---- | ------ | -------- | ------ | ------ |
| Cresci-2015 | mean | 0.9843 | 0.9847   | 0.9905 | 0.9876 |
| Cresci-2015 | std  | 0.0034 | 0.0063   | 0.0013 | 0.0026 |
| Cresci-2017 | mean | 0.9655 | 0.9930   | 0.9613 | 0.9769 |
| Cresci-2017 | std  | 0.0026 | 0.0008   | 0.0039 | 0.0018 |
| Twibot-20   | mean | 0.7743 | 0.7483   | 0.8781 | 0.8080 |
| Twibot-20   | std  | 0.0020 | 0.0008   | 0.0037 | 0.0020 |
| Twibot-22   | mean | 0.7570 | 0.7543   | 0.2591 | 0.3857 |
| Twibot-22   | std  | 0.0005 | 0.0015   | 0.0020 | 0.0023 |







| baseline | acc on Twibot-22 | f1 on Twibot-22 | type | tags|
| -------- | ---------------- | --------------- | ---- | --- |
| LOBO |0.7570|0.3857|F T|`random forest`|

