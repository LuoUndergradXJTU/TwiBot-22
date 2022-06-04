import os
import json
import pandas as pd
from tqdm import tqdm


path = '../../datasets/'
datasets = os.listdir(path)

for dataset in datasets:
    acc = 0
    cnt = 0
    id0 = []  # list of test set
    label = {}

    if dataset == "Twibot-22":
        continue

    f1 = pd.read_csv(path + dataset + '/split.csv')
    for item_id, item in enumerate(tqdm(f1['split'], desc='Loading Split')):
        if item == 'test':
            id0.append(str(f1['id'][item_id]))
    del f1

    f2 = pd.read_csv(path + dataset + '/label.csv')
    for item_id, item in enumerate(tqdm(f2['id'], desc='Loading Labels')):
        if item in id0:
            label[item] = f2['label'][item_id]
    del f2
    cnt = len(label)

    f3 = pd.read_csv('xxx.csv')
    for item_id, item in enumerate(tqdm(f3['id'], desc='Testing')):
        try:
            acc += 1 if label[item] == f3['label'][item_id] else 0
        except:
            raise Exception('xxx!')
    print('{} \tacc :\t{}'.format(dataset, acc / cnt))

















