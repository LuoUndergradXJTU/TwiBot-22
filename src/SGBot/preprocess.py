import json
import ijson
from argparse import ArgumentParser
import os
import os.path as osp
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()

dataset = args.dataset

assert dataset in ['Twibot-22', 'Twibot-20', 'midterm-2018', 'gilani-2017',
                   'cresci-stock-2018', 'cresci-rtbust-2019', 'cresci-2017',
                   'cresci-2015', 'botometer-feedback-2019']

collect_year = dataset.split('-')[-1]
if len(collect_year) == 2:
    collect_year = '20{}'.format(collect_year)

path = '../../datasets/{}'.format(dataset)
if not osp.exists(path):
    raise KeyError

bi_gram_likelihood = json.load(open('tmp/{}/bi_gram_likelihood.json'.format(dataset)))
label_data = pd.read_csv('../../datasets/{}/label.csv'.format(dataset))
label_index = {}
for index, item in label_data.iterrows():
    label_index[item['id']] = int(item['label'] == 'bot')
print(len(label_index))


def get_feature(value, segment=None):
    if value is None:
        return 0
    assert segment in ['likelihood', 'length', 'digits', 'profile_image_url', None]
    if segment == 'likelihood':
        value = value.strip()
        ans = 1
        for index in range(len(value) - 1):
            bi_gram = value[index] + value[index + 1]
            ans *= bi_gram_likelihood[bi_gram] ** (1.0 / (len(value) - 1))
        return ans
    if segment == 'length':
        return len(value.strip())
    if segment == 'digits':
        cnt = 0
        for i in value:
            if '0' <= i <= '9':
                cnt += 1
        return cnt
    if segment == 'profile_image_url':
        return int(item['profile_image_url'].find('default_profile_normal') == -1)
    if dataset == 'Twibot-20' and value in ['True ', 'False ']:
        value = (value == 'True ')
    if isinstance(value, bool):
        value = int(value)
    return value


def calc_age(created_at):
    if created_at is None:
        return 365 * 2
    created_at = created_at.strip()
    if dataset in ['Twibot-20', 'gilani-2017', 'cresci-stock-2018', 'cresci-rtbust-2019',
                   'cresci-2017', 'cresci-2015', 'botometer-feedback-2019']:
        mode = '%a %b %d %H:%M:%S %z %Y'
    elif dataset in ['Twibot-22']:
        mode = '%Y-%m-%d %H:%M:%S%z'
    elif dataset in ['midterm-2018']:
        mode = '%a %b %d %H:%M:%S %Y'
    else:
        raise KeyError
    if created_at.find('L') != -1:
        created_time = datetime.fromtimestamp(int(created_at.replace('000L', '')))
    else:
        created_time = datetime.strptime(created_at, mode)
    collect_time = datetime.strptime('{} Dec 31'.format(collect_year), '%Y %b %d')
    created_time = created_time.replace(tzinfo=None)
    collect_time = collect_time.replace(tzinfo=None)
    difference = collect_time - created_time
    return difference.days


if __name__ == '__main__':
    with open('{}/node.json'.format(path) if dataset != 'Twibot-22' else '{}/user.json'.format(path)) as f:
        data = ijson.items(f, 'item')
        features = []
        idx = []
        labels = []
        for item in tqdm(data, ncols=0):
            feature = []
            uid = item['id']
            if uid.find('u') == -1:
                break
            feature.append(get_feature(item['public_metrics']['tweet_count']))
            feature.append(get_feature(item['public_metrics']['followers_count']))
            feature.append(get_feature(item['public_metrics']['following_count']))
            feature.append(0)
            feature.append(get_feature(item['public_metrics']['listed_count']))
            feature.append(get_feature(item['profile_image_url'], 'profile_image_url'))
            feature.append(0)
            feature.append(get_feature(item['verified']))
            user_age = calc_age(item['created_at'])
            feature.append(feature[0] / user_age)
            feature.append(feature[1] / user_age)
            feature.append(feature[2] / user_age)
            feature.append(feature[3] / user_age)
            feature.append(feature[4] / user_age)
            if feature[2] == 0:
                feature.append(0)
            else:
                feature.append(feature[1] / feature[2])
            screen_name = item['username']
            name = item['name']
            feature.append(get_feature(screen_name, 'length'))
            feature.append(get_feature(screen_name, 'digits'))
            feature.append(get_feature(name, 'length'))
            feature.append(get_feature(name, 'digits'))
            feature.append(get_feature(item['description'], 'length'))
            feature.append(get_feature(screen_name, 'likelihood'))
            idx.append(uid)
            features.append(feature)
            if uid not in label_index:
                labels.append(2)
            else:
                labels.append(label_index[uid])
    features = np.array(features)
    labels = np.array(labels)
    print(len(idx))
    print(features.shape)
    print(labels.shape)
    json.dump(idx, open('tmp/{}/idx.json'.format(dataset), 'w'))
    np.save('tmp/{}/features.npy'.format(dataset), features)
    np.save('tmp/{}/labels.npy'.format(dataset), labels)

