import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import json
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

dataset = 'Twibot-22'
idx = json.load(open('tmp/{}/idx.json'.format(dataset)))
idx = {item: index for index, item in enumerate(idx)}
features = np.load('tmp/{}/features.npy'.format(dataset), allow_pickle=True)
labels = np.load('tmp/{}/labels.npy'.format(dataset))

print('loading done')

user_idx = []
for index in range(10):
    data = json.load(open('../../datasets/{}/domain/user{}.json'.format(dataset, index)))
    user_id = [idx[item] for item in data]
    random.shuffle(user_id)
    user_idx.append(user_id)

if __name__ == '__main__':
    pbar = tqdm(total=100, ncols=0)
    fb = open('transfer_results.txt', 'w')
    for i in range(10):
        for j in range(10):
            cls = RandomForestClassifier(n_estimators=100)
            train_x, train_y = features[user_idx[i]], labels[user_idx[i]]
            test_x, test_y = features[user_idx[j]], labels[user_idx[j]]
            cls.fit(train_x, train_y)
            pred = cls.predict(test_x)
            acc = accuracy_score(test_y, pred)
            f1 = f1_score(test_y, pred)
            fb.write('{} train, {} test, acc: {}, f1: {}\n'.format(i, j, acc, f1))
            pbar.update()
            pbar.set_description('{} {} {} {}'.format(i, j, acc, f1))
    fb.close()



