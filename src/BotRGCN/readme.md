### BotRGCN: Twitter Bot Detection with Relational Graph Convolutional Networks

---

- **authors**: Shangbin Feng, Herun Wan, Ningnan Wang, Minnan Luo
- **link**: [https://arxiv.org/abs/2106.13092/](https://arxiv.org/abs/2106.13092)
- **file structure**: 

```python
├── twibot_22/
│   ├── raw_data/
│   ├── processed_data/
│   ├── utils.py
│   ├── model.py
│   ├── preprocess_1.py
│   ├── train.py # train BotRGCN model
│   ├── Dataset.py
│   ├── preprocess.py # preprocess the dataset
│   ├── dataset_tool.py
│   └── preprocess_2.py
├── cresci_15/
│   ├── raw_data/
│   ├── processed_data/
│   ├── utils.py
│   ├── model.py
│   ├── preprocess_1.py
│   ├── train.py # train BotRGCN model
│   ├── Dataset.py
│   ├── preprocess.py # preprocess the dataset
│   ├── dataset_tool.py
│   └── preprocess_2.py
├── twibot_20/
│   ├── raw_data/
│   ├── processed_data/
│   ├── utils.py
│   ├── model.py
│   ├── preprocess_1.py
│   ├── train.py # train BotRGCN model
│   ├── Dataset.py
│   ├── preprocess.py # preprocess the dataset
│   ├── dataset_tool.py
│   └── preprocess_2.py
└── readme.md
```

- **implement details**: 

   There are some changes in user numerical properties & user categorical properties due to the lack of relevant data
  
   1. numerical properties:
  
      - original: (dim=6)

         followers + followings + favorites + statuses + active_days + screen_name_length 

      - twibot-20/cresci-2015/twibot-22: (dim=5)
  
         followers + followings + statuses + active_days + screen_name_length
  
   2. categorical properties: 
  
      - original: (dim=11)

         protected + verified + default_profile_image + geo_enabled + contributors_enabled + is_translator + is_translation_enabled + profile_background_image + profile_user_background_image + has_extended_profile + default_profile

      - twibot-20/twibot-22: (dim=3)

         protected + verified + default_profile_image

      - cresci-2015: (dim=1)

         default_profile_image


#### How to reproduce:

1. specify the dataset by entering corresponding fold

   - cresci-15 : `cd cresci_15/`
   - twibot-20 : `cd twibot_20/`
   - twibot-22 : `cd twibot_22/`

2. preprocess the dataset by running

   `python preprocess.py`

3. train BotRGCN model by running:

   `python train.py`


#### Result:

| dataset     |      | acc    | precision | recall | f1     |
| ----------- | ---- | ------ | --------- | ------ | ------ |
| Cresci-2015 | mean | 0.9652 | 0.9551    | 0.9917 | 0.9730 |
| Cresci-2015 | std  | 0.0071 | 0.0102    | 0.0025 | 0.0053 |
| Twibot-20   | mean | 0.8575 | 0.8452    | 0.9019 | 0.8725 |
| Twibot-20   | std  | 0.0068 | 0.0054    | 0.0172 | 0.0073 |
| Twibot-22   | mean | 0.7966 | 0.7481    | 0.4680 | 0.5750 |
| Twibot-22   | std  | 0.0014 | 0.0222    | 0.0276 | 0.0142 |




</br>


| baseline | acc on Twibot-22 | f1 on Twibot-22 | type | tags|
| -------- | ---------------- | --------------- | ---- | --- |
| BotRGCN |0.7966| 0.5750          |F T G|`BotRGCN`|

