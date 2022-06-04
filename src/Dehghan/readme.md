The data has been preprocessed and stored in folders e.g./cresci-2015

first run all_fea.py to generate the total features used for training
remember to change the file path according to the dataset name

```python
dataset='Twibot-20'
```


then you can use the feature generated ,change the dataset name ,and run main_xgboost.py 
Check the results in results/dataset.log


#### Result:

| dataset                   |      | acc    | precison | recall | f1     |
| ------------------------- | ---- | ------ | -------- | ------ | ------ |
| Twibot-20_all	            | mean | 0.8604 | 0.9472   |	0.8219 |	0.8801 |
| Twibot-20_Deepwalk	    | mean | 0.8634 | 0.9400   |	0.8300 |	0.8816 |
| Twibot-20_Node2vec	    | mean | 0.8607 | 0.9425   |	0.8798 |	0.8718 |
| Twibot-20_Role2Vec	    | mean | 0.8607 | 0.9484   |	0.8261 |	0.8805 |
| Twibot-20_RolX   	        | mean | 0.8653 | 0.9313   |	0.8378 |	0.8820 |
| Twibot-20_Struc2Vec       | mean | 0.8617 | 0.9366   |	0.8298 |	0.8799 |
| Twibot-20_GraphWave       | mean | 0.8668 | 0.9331   |	0.6311 |	0.7620 |
| cresci-2015_GraphWave     | mean | 0.6206 | 0.9615   |	0.8388 |	0.8834 |
| cresci-2015_Node2Vec      | mean | 0.6318 | 0.9615   |	0.8388 |	0.7743 |
