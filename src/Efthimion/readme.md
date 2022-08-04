### Supervised Machine Learning Bot Detection Techniques to Identify Social Twitter Bots

---

- **authors**: Efthimion et al.

- **link**: https://scholar.smu.edu/datasciencereview/vol1/iss2/5/

- **file structure**: 

```python
└── feature.py
```
- **implement details**: We abdicate the Levenshtein distance for time consumption problem.

  

#### How to reproduce:

1. run

    `python feature.py`
    
    different datasets are optional in the code.


#### Result:


| dataset     		      |      | acc    | precison | recall | f1     |
| ----------------------- | ---- | ------ | -------- | ------ | ------ |
| Botometer-feedback-2019 | mean | 0.6981 | 0.0000   | 0.0000 | 0.0000 |
| Botometer-feedback-2019 | std  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |
| Cresci-2015 		      | mean | 0.9252 | 0.9382   | 0.9438 | 0.9410 |
| Cresci-2015 		      | std  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |
| Cresci-2017             | mean | 0.8796 | 0.9458   | 0.8923 | 0.9183 |
| Cresci-2017             | std  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |
| Cresci-rtbust-2019      | mean | 0.6765 | 0.6829   | 0.7568 | 0.7179 |
| Cresci-rtbust-2019      | std  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |
| Cresci-stock-2018 	  | mean | 0.7076 | 0.8275   | 0.5802 | 0.6821 |
| Cresci-stock-2018 	  | std  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |
| gilani-2017 		      | mean | 0.5551 | 0.3750   | 0.0280 | 0.0522 |
| gilani-2017		      | std  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |
| midterm-2018   		  | mean | 0.9339 | 0.9801   | 0.9404 | 0.9598 |
| midterm-2018  		  | std  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |
| Twibot-20  		      | mean | 0.6281 | 0.6420   | 0.7063 | 0.6726 |
| Twibot-20   	    	  | std  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |
| Twibot-22  		      | mean | 0.7408 | 0.7778   | 0.1676 | 0.2758 |
| Twibot-22   		      | std  | 0.0000 | 0.0000   | 0.0000 | 0.0000 |




| baseline | acc on Twibot-22 | f1 on Twibot-22 | type | tags|
| -------- | ---------------- | --------------- | ---- | --- |
| efthimion|0.7408|0.2758|F T|`efthimion`|