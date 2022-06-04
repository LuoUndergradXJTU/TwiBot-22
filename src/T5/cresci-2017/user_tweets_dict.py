import os
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

path1 = Path('datasets/cresci-2017/node.json')
with open(path1, 'r') as f:
    nodes = json.loads(f.read())
    nodes = pd.DataFrame(nodes)
    users = nodes[nodes.id.str.contains('^u')]
    tweets = nodes[nodes.id.str.contains('^t')]
    
users_index_to_uid = list(users['id'])
tweets_index_to_tid = list(tweets['id'])
uid_to_users_index = {x: i for i, x in enumerate(users_index_to_uid)}
tid_to_tweets_index = {x: i for i, x in enumerate(tweets_index_to_tid)}

path2 = Path('datasets/cresci-2017/edge.csv')
edge_data = pd.read_csv(path2)
edge_data = edge_data[edge_data['relation'] == 'post']

edge_data['source_id'] = list(map(lambda x: uid_to_users_index[x], edge_data['source_id'].values))
edge_data['target_id'] = list(map(lambda x: tid_to_tweets_index[x], edge_data['target_id'].values))
edge_data = edge_data.reset_index(drop=True)
dict = {i: [] for i in range(len(users))}

for i in tqdm(range(len(edge_data))):
    try:
        user_index = edge_data['source_id'][i]
        dict[user_index].append(tweets['text'][edge_data['target_id'][i] + len(users)])
    except:
        continue

path3 = Path('src/T5/cresci-2017/')
np.save(path3 / 'user_tweets_dict.npy', dict)