import re
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import networkx as nx
import ijson
from datetime import datetime

MONTH_DICT = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
CURRENT_TIME = datetime(2022, 6, 6, 13, 0, 0)  # NeurIPS 2022 datasets and benchmarks abstract submission deadline
MAX_FOLLOW = 1e10

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Twibot-20', help='Choose the dataset.')
arg = parser.parse_args()
DATASET = arg.dataset

if DATASET == 'Twibot-22':
    node = json.load(open('../../datasets/' + DATASET + '/node.json', 'r'))
else:
    node = json.load(open('../../datasets/' + DATASET + '/user.json', 'r'))
edge = pd.read_csv('../../datasets/' + DATASET + '/edge.csv')
X = []
id_map = dict()
num_user = 0
ind_first = True
for i in range(len(node)):
    id_map[node[i]['id']] = i
    if node[i]['id'][0] == 't' and ind_first:
        num_user = i
        ind_first = False
if num_user == 0:
    num_user = len(node)
follow = edge.values[edge['relation'].values == 'follow']
post = edge.values[edge['relation'].values == 'post']
tweet_map = dict()
if DATASET == 'Twibot-22':
    for i in range(num_user):
        tweet_map[i] = []
    for i in range(post.shape[0]):
        tweet_map[id_map[post[i, 0]]].append(str(node[id_map[post[i, 2]]]['text']))
else:
    id_tweet = json.load(open('../twibot22_Botrgcn_feature/id_tweet.json', 'r'))
    print([num_user, len(id_tweet)])
    for key in tqdm(id_tweet):
        tweet_map[int(key)] = id_tweet[key]
number_of_followers = []
number_of_friends = []
number_of_listed = []
number_of_tweet = []
lack_index_1 = []
lack_index_2 = []
lack_index_3 = []
lack_index_4 = []
for i in tqdm(range(num_user)):
    if node[i]['public_metrics']['followers_count'] is not None:
        number_of_followers.append(node[i]['public_metrics']['followers_count'])
    else:
        number_of_followers.append(0)
        lack_index_1.append(i)
    if node[i]['public_metrics']['following_count'] is not None:
        number_of_friends.append(node[i]['public_metrics']['following_count'])
    else:
        number_of_friends.append(0)
        lack_index_2.append(i)
    if node[i]['public_metrics']['listed_count'] is not None:
        number_of_listed.append(node[i]['public_metrics']['listed_count'])
    else:
        number_of_listed.append(0)
        lack_index_3.append(i)
    if node[i]['public_metrics']['tweet_count'] is not None:
        number_of_tweet.append(node[i]['public_metrics']['tweet_count'])
    else:
        number_of_tweet.append(0)
        lack_index_4.append(i)
mean_value_1 = int(sum(number_of_followers) / (len(number_of_followers) - len(lack_index_1)))
mean_value_2 = int(sum(number_of_friends) / (len(number_of_friends) - len(lack_index_2)))
mean_value_3 = int(sum(number_of_listed) / (len(number_of_listed) - len(lack_index_3)))
mean_value_4 = int(sum(number_of_tweet) / (len(number_of_tweet) - len(lack_index_4)))
for i in lack_index_1:
    number_of_followers[i] = mean_value_1
for i in lack_index_2:
    number_of_friends[i] = mean_value_2
for i in lack_index_3:
    number_of_listed[i] = mean_value_3
for i in lack_index_4:
    number_of_tweet[i] = mean_value_4
X.append(number_of_followers)
X.append(number_of_friends)
X.append(number_of_listed)
X.append(number_of_tweet)

verified = []
exist = 0
for i in tqdm(range(num_user)):
    if node[i]['verified'] is None:
        verified.append(0)
    else:
        verified.append(int(node[i]['verified']))
# for i in tqdm(range(num_user)):
#     if isinstance(node[i]['verified'], str):
#         exist = 1
#         if node[i]['verified'][0] == 'T':
#             verified.append(1)
#         else:
#             verified.append(0)
#     else:
#         verified.append(0)
print(verified[0])
X.append(verified)

account_age = []
lack_index_5 = []
for i in tqdm(range(num_user)):
    if node[i]['created_at'] is not None:
        time_list = node[i]['created_at'].split(' ')
        if len(time_list) < 2:
            account_age.append(0.0)
            lack_index_5.append(i)
            continue
        born_date = datetime(int(time_list[0].split('-')[0]), int(time_list[0].split('-')[1]), int(time_list[0].split('-')[2]),
                             int(time_list[1].split('+')[0].split(':')[0]), int(time_list[1].split('+')[0].split(':')[1]),
                             int(time_list[1].split('+')[0].split(':')[2]))
        now_date = CURRENT_TIME
        account_age.append((now_date - born_date).days)
    else:
        account_age.append(0.0)
        lack_index_5.append(i)
mean_value = sum(account_age) / (len(account_age) - len(lack_index_5))
for i in lack_index_5:
    account_age[i] = mean_value
X.append(account_age)

follower_ratio = []
lack_index_6 = []
for i in tqdm(range(num_user)):
    if node[i]['public_metrics']['followers_count'] is not None and node[i]['public_metrics']['following_count'] is not None:
        if node[i]['public_metrics']['following_count'] == 0:
            follower_ratio.append(MAX_FOLLOW)
        else:
            follower_ratio.append(node[i]['public_metrics']['followers_count'] / node[i]['public_metrics']['following_count'])
    else:
        follower_ratio.append(0.0)
        lack_index_6.append(i)
mean_value = sum(follower_ratio) / (len(follower_ratio) - len(lack_index_6))
for i in lack_index_6:
    follower_ratio[i] = mean_value
X.append(follower_ratio)

retweet_count = []
mention_count = []
hashtag_count = []
link_count = []
one_gram_count = [[] for _ in range(95)]
for i in tqdm(range(num_user)):
    if len(tweet_map[i]) > 0:
        rt_count = 0
        mt_count = 0
        ht_count = 0
        lk_count = 0
        gram_count = np.zeros(95)
        for j in range(len(tweet_map[i])):
            if isinstance(tweet_map[i][j], str):
                if len(tweet_map[i][j]) >= 4:
                    if tweet_map[i][j][0:4] == 'RT @':
                        rt_count += 1
                ht_count += len(re.findall('#', tweet_map[i][j]))
                mt_count += len(re.findall('@', tweet_map[i][j]))
                lk_count += len(re.findall('http', tweet_map[i][j]))
                for k in tweet_map[i][j]:
                    if 32 <= ord(k) <= 126:
                        gram_count[ord(k) - 32] += 1
        retweet_count.append(rt_count)
        mention_count.append(mt_count)
        hashtag_count.append(ht_count)
        link_count.append(lk_count)
        for j in range(95):
            one_gram_count[j].append(gram_count[j])
    else:
        retweet_count.append(0.0)
        mention_count.append(0.0)
        hashtag_count.append(0.0)
        link_count.append(0.0)
        for j in range(95):
            one_gram_count[j].append(0.0)
X.append(retweet_count)
X.append(mention_count)
X.append(hashtag_count)
X.append(link_count)
for j in range(95):
    X.append(one_gram_count[j])

X_T = np.array(X).T
X_file = pd.DataFrame(X_T)
print(X_file.shape)
X_file.to_csv('X_' + DATASET + '.csv')

