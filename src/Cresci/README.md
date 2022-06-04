### DNA-inspired online behavioral modeling and its application to spambot detection

---

- **authors**: Stefano Cresci, Roberto Di Pietro, Marinella Petrocchi, Angelo Spognardi, and Maurizio Tesconi

- **link**: https://ieeexplore.ieee.org/abstract/document/7436643

- **file structure**: 

```python
├── preprocess.py
└── main.py
```

- **implement details**: The maximum length of a DNA is set to 998

  

#### How to reproduce:

1. extract DNAs from raw dataset

   `python preprocess.py`

   this command will create DNAs in corresponding directory. (This may take a bit of time)

2. utilize the DNA-based bot detection method:

   `python main.py`

   the final result will be printed on the screen



#### Result:

| dataset     |      | acc    | precison | recall | f1     |
| ----------- | ---- | ------ | -------- | ------ | ------ |
| Cresci-2015 | mean | 0.3701 | 0.0059   | 0.6667 | 0.0117 |
| Cresci-2015 | var  | 0      | 0        | 0      | 0      |
| Cresci-2017 | mean | 0.3349 | 0.1296   | 0.9530 | 0.2281 |
| Cresci-2017 | var  | 0      | 0        | 0      | 0      |
| Twibot-20   | mean | 0.4776 | 0.0766   | 0.6447 | 0.1369 |
| Twibot-20   | var  | 0      | 0        | 0      | 0      |







| baseline | acc on Twibot-22 | f1 on Twibot-22 | type | tags|
| -------- | ---------------- | --------------- | ---- | --- |
| Cresci et al.|-|-|T|`DNA`|

