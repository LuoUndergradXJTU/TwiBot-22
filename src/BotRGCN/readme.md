### BotRGCN: Twitter Bot Detection with Relational Graph Convolutional Networks

---

- **authors**: Shangbin Feng, Herun Wan, Ningnan Wang, Minnan Luo
- **link**: [https://arxiv.org/abs/2106.13092/](https://arxiv.org/abs/2106.13092)
- **file structure**: 

```python
    |----twi_22\
    |    |----train.py
    |    |----preprocess.py
    |    |----raw_data\
    |    |----processed_data\
    |----cresci_15\
    |    |----train.py
    |    |----preprocess.py
    |    |----raw_data\
    |    |----processed_data\
    |----twi_20\
    |    |----train.py
    |    |----preprocess.py
    |    |----raw_data\
    |    |----processed_data\
    |----BotRGCN.md # README
```

- **implement details**: 

  - 1. There are some changes in user numerical properties & user categorical properties due to the lack of relevant data
  
       (1)  numerical properties :
  
       ​       original : (dim=6)
  
       ​               followers + followings + favorites + statuses + active_days + screen_name_length 
  
       twibot-20/cresci-2015/twibot-22 : (dim=5)
  
       ​                followers + followings + statuses + active_days + screen_name_length
  
       (2)  categorical properties : 
  
       ​       original : (dim=11)
  
       ​                  protected + verified + default_profile_image + geo_enabled + contributors_enabled + is_translator +                 is_translation_enabled + profile_background_image + profile_user_background_image + has_extended_profile + default_profile
  
       ​       twibot-20/twibot-22 : (dim=3)
  
       ​                    protected + verified + default_profile_image
  
       ​     cresci-2015 : (dim=1)
  
       ​                    default_profile_image
  
- 

#### How to reproduce:

1. specify the dataset by entering corresponding fold
   - cresci-15 : `cresci_15/`
   - twibot-20 : `twibot_20/`
   - twibot-22 : `twibot_22/`
2. preprocess the dataset by running
   `python preprocess.py`
3. train BotRGCN model by running:
   `python train.py`


#### Result:

| dataset     |      | acc    | precision | recall | f1     |
| ----------- | ---- | ------ | --------- | ------ | ------ |
| Cresci-2015 | mean | 0.9686 | 0.9563    | 0.9958 | 0.9757 |
| Cresci-2015 | var  | 0.0000 | 0.0001    | 0.0000 | 0.0000 |
| Twibot-20   | mean | 0.8551 | 0.8462    | 0.8956 | 0.8701 |
| Twibot-20   | var  | 0.0000 | 0.0001    | 0.0004 | 0.0000 |
| Twibot-22   | mean | 0.7691 | /         | /      | 0.4579 |
| Twibot-22   | var  | 0.0055 | /         | /      | 0.0338 |




</br>


| baseline | acc on Twibot-22 | f1 on Twibot-22 | type | tags|
| -------- | ---------------- | --------------- | ---- | --- |
| BotRGCN |0.7691| 0.4579          |P T G|`BotRGCN`|

