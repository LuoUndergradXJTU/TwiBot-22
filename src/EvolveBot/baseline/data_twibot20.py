import ijson
from pathlib import Path
import torch
import numpy as np
import time
import csv
import statistics
from urlextract import URLExtract
from transformers import (
    pipeline,
    BertTokenizerFast,
    BertModel,
)
import pandas as pd

root_dir = Path.cwd().parent.parent.parent
dataset_dir = root_dir / 'datasets'
twibot_20 = dataset_dir / 'Twibot-20'
json_dir = twibot_20 / 'node.json'
label = pd.read_csv(twibot_20 / 'label.csv')
split = pd.read_csv(twibot_20 / 'split.csv')

node_num = 9461
stop_time = 1601510400.0
r = np.zeros(shape=(node_num, 11))
# r represents the 11 features
r_help = np.zeros(shape = (node_num, 3 ))
# r_help represents the 3 features, which are number of followings on graph, number of friend on graph,
# number of followers
uid_to_user_index = {}
cnt = 0
for index, row in split.iterrows():
    if row['split'] == 'train' or row['split'] == 'test':
        uid_to_user_index[row['id']] = cnt
        cnt += 1
    if row['split'] == 'support':
        break
with open(json_dir, "r") as f:
    i = 0
    find_num = 0
    obj = ijson.items(f, "item")
    for x in obj:
        if not x['id'][0] == 'u':
            print(x)
            break
        id = x['id']
        if not id in uid_to_user_index:
            continue
        i = uid_to_user_index[id]
        find_num += 1
        r[i][0] = (x["public_metrics"]['following_count'] if x["public_metrics"]['following_count'] is not None else 0)
        r[i][2] = (x["public_metrics"]['tweet_count'] if x["public_metrics"]['tweet_count'] is not None else 0)
        r_help[i][2] = (x["public_metrics"]['followers_count'] if x["public_metrics"]['followers_count'] is not None else 0)
        time_seen = x["created_at"].strip()
        if time_seen is not None:
            if x['created_at'].find('L')!=-1:
                real_time = stop_time - int(x['created_at'][0:10])
                # print(int(x['created_at'][0:10]))
            else:
                real_time = stop_time - time.mktime(
                    time.strptime(time_seen, "%a %b %d %H:%M:%S +0000 %Y")
                )
        else:
            real_time = 0
        r[i][3] = real_time
        r[i][10] = (0 if real_time == 0 else r[i][0] / real_time)
        if find_num % 1000==0 and find_num!=0:
            print(find_num)
        if find_num == node_num:
            break
tokenizer_berttiny = BertTokenizerFast.from_pretrained("prajjwal1/bert-tiny")
model_berttiny = BertModel.from_pretrained("prajjwal1/bert-tiny")
nlp = pipeline(
    task="feature-extraction",
    model=model_berttiny,
    tokenizer=tokenizer_berttiny,
    device = 0
)
followers_of_following = [[] for i in range(node_num)]
num_of_tweet =np.zeros(shape=node_num)
tid_to_tweet_index = {}
edge_dir = twibot_20 / 'edge.csv'
with open(edge_dir) as csvfile:
    spamreader = csv.reader(csvfile, delimiter=",", quotechar="|")
    index = 0
    for row in spamreader:
        index += 1
        if row[1] == 'post':
            if row[0] not in uid_to_user_index:
                continue
            user_index_1 = uid_to_user_index[row[0]]
            if num_of_tweet[user_index_1] < 20:
                num_of_tweet[user_index_1] += 1
                tid_to_tweet_index[row[2]] = user_index_1

        if row[0] not in uid_to_user_index or row[2] not in uid_to_user_index:
            continue
        user_index_1 = uid_to_user_index[row[0]]
        user_index_2 = uid_to_user_index[row[2]]

        if row[1] == "friend":
            r_help[user_index_1][0] += 1
            r_help[user_index_1][1] += 1
            r[user_index_1][7] += r_help[user_index_2][2]
            r[user_index_1][8] += r[user_index_2][2]
            followers_of_following[user_index_1].append(r_help[user_index_2][2])
        elif row[1] == 'follow':
            r_help[user_index_1][0] += 1
            r[user_index_1][7] += r_help[user_index_2][2]
            r[user_index_1][8] += r[user_index_2][2]
            followers_of_following[user_index_1].append(r_help[user_index_2][2])

for i in range(node_num):
    r[i, 1] = 0 if r_help[i][0] == 0 else r_help[i][1] / r_help[i][0]
    r[i, 7] = 0 if r_help[i][0] == 0 else r[i][7] / r_help[i][0]
    r[i, 8] = 0 if r_help[i][0] == 0 else r[i][8] / r_help[i][0]
    r[i, 9] = 0 if len(followers_of_following[i]) == 0 else r[i, 0] / statistics.median(followers_of_following[i])

extractor = URLExtract()
num_of_url = np.zeros(node_num)
url_dict = [{} for i in range(node_num)]
tweet_vector = [[] for i in range(node_num)]

with open(json_dir, "r") as f:
    i = 0
    obj = ijson.items(f, "item")
    for x in obj:
        i += 1
        if i == 11826:
            break
    index = 0
    for x in obj:
        index += 1
        if index % 100000 == 0:
            print(index)
        if x['id'] not in tid_to_tweet_index:
            continue
        user_index_1 = tid_to_tweet_index[x['id']]
        if x["text"] is None:
            num_of_tweet[user_index_1] -= 1
        else:
            text = x["text"].strip()
            urls_list = extractor.find_urls(text)
            num_of_url[user_index_1] += len(urls_list)
            for x in urls_list:
                if x not in url_dict[user_index_1]:
                    url_dict[user_index_1] = x
            tweet_vector[user_index_1].append(nlp(text)[0][0])

for i in range(node_num):
    r[i][4] = 0 if num_of_tweet[i] == 0 else num_of_url[i] / num_of_tweet[i]
    r[i][5] = 0 if num_of_tweet[i] == 0 else len(url_dict[i]) / num_of_tweet[i]
    if num_of_tweet[i] <= 1:
         r[i][6] = 0
    else:
        for j in range(len(tweet_vector[i])):
            for k in range(j+1, len(tweet_vector[i])):
                x = tweet_vector[i][j]
                y = tweet_vector[i][k]
                r[i][6] += np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        r[i][6] =r[i][6] / (num_of_tweet[i] * (num_of_tweet[i] - 1) / 2)

# r_mean = np.mean(r, axis=0)
# r_std = np.std(r, axis=0)
# r = (r - r_mean) / r_std
np.save('twibot_20_feature', r)