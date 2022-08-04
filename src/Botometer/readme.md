###  Botometer 101: Social bot practicum for computational social scientists
---
This is an unofficial implementation Botometer. Coding by Herun Wan ([email address](wanherun@stu.xjtu.edu.cn))


- **authors**: Kai-Cheng Yang, Emilio Ferrara, Filippo Menczer
- **link**: [https://arxiv.org/abs/2201.01608](https://arxiv.org/abs/2201.01608)
- **introduction**: Botometer (formerly BotOrNot) is a public website to checks the activity of a Twitter account and gives it a score, where higher scores mean more bot-like activity. Botometer's classification system generates more 1,000 features using available meta-data and information extracted from interaction patterns and content.
- **file structure**:

```python
├── check.py  # check the score of Botometer API
├── combine.py  # combine the scores
├── eval.py  # get the bot detection metrices
├── MyBotomater.py  # the Botometer API class
├── preprocess.py  # get username from raw dataset
├── train.py  # get score from Botomater API
├── train_threads.py  # get score from Botomater API using muti-threads
├── scores.zip  # the score we collected   
└── tmp  # other files
    ├─output  # the metrics
    ├─score  # the score obtained from Botomater API, you can unzip scores.zip here
    ├─username # the username of users in different datasets
    └─key.json  # the twitter API and Botometer API key
```

- **implement details**:  
  
  We adopt the "english" ("english" or "universal") score returned by Botometer API to determine if a user is bot or not.  According to the research of [Lynnette Hui Xian Ng](https://www.sciencedirect.com/science/article/pii/S2468696422000027), we choose 0.75 as the threshold for judging bots. We remove the users which were unable to  get score from Botometer API because they are banned or they don't post any tweets.

### Getting Started

---

specify the dataset, 'Twibot-20' for example

#### Data Preprocessing

first, preprocess raw dataset by running: 

```python
python preprocess.py --dataset_name Twibot-20
```

#### Crawl Score

second, get score from API by running:

```python
python train.py --dataset_name Twibot-20
```

or you can use multi-threads to crawl data by running:

```python
python train_threads.py --dataset_name Twibot-20 --threads 5
```

if you are using VPN, you can setting proxies by running:

```python
parser.add_argument('--proxy', type=str)
```

for example, you can crawling using proxies by running:

``` python
python train.py --dataset_name Twibot-20 --proxy '127.0.0.1: 15236'
```

make sure you get the API key and save the key in dictionary 'tmp/key.json' as:

```python
{
    "rapid_api_key": "your rapid api key",
    "consumer_key": "your consumer key",
    "consumer_secret": "your consumer secret",
    "access_token": "your access token",
    "access_token_secret": "your access token secret"
}
```

or, you can use the score we collected in 'scores.zip', you can unzip it into specific dictionary. 



#### Combine score into one file

after getting score, combine the scores by running:

```python
python combine.py --dataset_name Twibot-20
```



#### Evaluation

last, evaluate trained model  by running:

```python
python eval.py --dataset Twibot-20
```

you can choose the hyperparameters as following:

```python
    parser.add_argument('--threshold', type=float, default=0.75)
    parser.add_argument('--type', type=str, default='english')
```

- threshold is the bot threshold score
- make sure type in ['english', 'universal']



### Results

|         dataset         | accuracy | f1-score | precision | recall |
| :---------------------: | :------: | :------: | :-------: | :----: |
| Botometer-feedback-2019 |  50.00   |  30.77   |   21.05   | 57.14  |
|       Cresci-2015       |  57.92   |  66.90   |   50.54   | 98.95  |
|       Cresci-2017       |  94.16   |  96.12   |   93.35   | 99.69  |
|   Cresci-rtbust-2019    |  69.23   |  78.95   |   65.22   | 100.0  |
|    Cresci-stock-2018    |  72.62   |  79.59   |   68.50   | 94.96  |
|       Gilani-2017       |  71.56   |  77.39   |   62.99   | 87.91  |
|      Midterm-2018       |  89.46   |  46.03   |   31.18   | 87.88  |
|        Twibot-20        |  53.09   |  53.13   |   55.67   | 50.82  |
|        Twibot-22        |  49.87   |  42.75   |   30.81   | 69.80  |





| baseline  | acc on Twibot-22 | f1 on Twibot-22 | type  | tags |
| :-------: | :--------------: | :-------------: | :---: | :--: |
| Botomater |      49.87       |      42.75      | F T G |      |