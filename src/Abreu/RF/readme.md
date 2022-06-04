### Twitter Bot Detection with Reduced Feature Set

---

- **authors**: Jefferson Viana Fonseca Abreu,Celia Ghedini Ralha, Joao Jose Costa Gondim
- **link** : https://ieeexplore.ieee.org/abstract/document/9280525
- **file structure**: 

```python
├── all datasets
│   └── twi.py  # train model on cresci-2015
```

- **implement details**: We choose the algorithm which performs best in the origin paper. And due to many datasets don't have the "favourite count", we don't take the feature into count.

  

#### How to reproduce:

1. specify the dataset  by changing `dataset=Twibot-22` in twi.py (Twibot-22 for example) ;

2. train random forest model by running:

   `python twi.py `

   the final result will be saved into ''dataset name''.txt



#### Result:

random seed: 100, 200, 300, 400, 500

| dataset                 |      | acc    | precison | recall | f1     |
| ----------------------- | ---- | ------ | -------- | ------ | ------ |
| Cresci-2015             | mean | 0.7570 | 0.9905   | 0.6213 | 0.7636 |
| Cresci-2015             | var  | 0.001  | 0.001    | 0.001  | 0.001  |
| Twibot-20               | mean | 0.7345 | 0.7220   | 0.8281 | 0.7714 |
| Twibot-20               | var  | 0.001  | 0.001    | 0.001  | 0.001  |
| Cresci-2017             | mean | 0.9273 | 0.9834   | 0.9197 | 0.9504 |
| Cresci-2017             | var  | 0.001  | 0.001    | 0.001  | 0.001  |
| cresci-rtbust-2019      | mean | 0.8088 | 0.7857   | 0.8918 | 0.8354 |
| cresci-rtbust-2019      | var  | 0.001  | 0.001    | 0.001  | 0.001  |
| cresci-stock-2018       | mean | 0.7545 | 0.7545   | 0.7567 | 0.7693 |
| cresci-stock-2018       | var  | 0.001  | 0.001    | 0.001  | 0.001  |
| midterm-2018            | mean | 0.9653 | 0.9728   | 0.9863 | 0.9795 |
| midterm-2018            | var  | 0.001  | 0.001    | 0.001  | 0.001  |
| gilani-2017             | mean | 0.7428 | 0.7682   | 0.5887 | 0.6666 |
| gilani-2017             | var  | 0.001  | 0.001    | 0.001  | 0.001  |
| botometer-feedback-2019 | mean | 0.7735 | 0.6363   | 0.4666 | 0.5384 |
| botometer-feedback-2019 | var  | 0.001  | 0.001    | 0.001  | 0.001  |







| baseline     | acc on Twibot-22 | f1 on Twibot-22 | type | tags            |
| ------------ | ---------------- | --------------- | ---- | --------------- |
| Abreu et al. | 0.7066           | 0.5344          | P    | `random forest` |

