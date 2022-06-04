import os
import os.path as osp
import pandas
from tqdm import tqdm
import ijson
import json
from datetime import datetime
import numpy as np
from gensim.models import word2vec
from nltk import tokenize

dataset = 'cresci-2015'
if not osp.exists('tmp/{}'.format(dataset)):
    os.makedirs('tmp/{}'.format(dataset))

properties_segments = ['created_at', 'description', 'entities', 'location',
                       'pinned_tweet_id', 'profile_image_url', 'protected',
                       'url', 'username', 'verified', 'withheld',
                       'public_metrics.followers_count', 'public_metrics.following_count',
                       'public_metrics.tweet_count', 'public_metrics.listed_count']


def calc_activate_days(created_at):
    created_at = created_at.strip()
    create_date = datetime.strptime(created_at, '%a %b %d %H:%M:%S %z %Y')
    crawl_date = datetime.strptime('2020 09 28 +0000', '%Y %m %d %z')
    delta_date = crawl_date - create_date
    return delta_date.days


def get_properties():
    print('processing properties')
    with open(osp.join(path, 'node.json')) as f:
        users = ijson.items(f, 'item')
        properties = []
        idx = []
        for user in tqdm(users, ncols=0):
            if user['id'].find('u') == -1:
                continue
            idx.append(user['id'])
            user_property = []
            for item in properties_segments:
                prop = user
                for key in item.split('.'):
                    prop = prop[key]
                    if prop is None:
                        continue
                if prop is None:
                    user_property.append(0)
                elif item in ['public_metrics.followers_count', 'public_metrics.following_count',
                              'public_metrics.tweet_count', 'public_metrics.listed_count']:
                    user_property.append(int(prop))
                elif item in ['withheld', 'url', 'profile_image_url',
                              'pinned_tweet_id', 'entities', 'location']:
                    user_property.append(1)
                elif item in ['verified', 'protected']:
                    user_property.append(int(prop))
                elif item in ['description', 'username']:
                    user_property.append(len(prop.strip()))
                elif item in ['created_at']:
                    user_property.append(calc_activate_days(prop.strip()))
            assert len(user_property) == 15
            properties.append(user_property)
        json.dump(idx, open('tmp/{}/idx.json'.format(dataset), 'w'))
        properties = np.array(properties)
        for i in range(properties.shape[1]):
            if np.max(properties[:, i]) == np.min(properties[:, i]):
                continue
            mean = np.mean(properties[:, i])
            std = np.std(properties[:, i])
            properties[:, i] = (properties[:, i] - mean) / std
        print(properties.shape)
        np.save('tmp/{}/properties.npy'.format(dataset), properties)


def get_neighbors():
    edge = pandas.read_csv(osp.join(path, 'edge.csv'), chunksize=10000000)
    user_idx = json.load(open('tmp/{}/idx.json'.format(dataset)))
    neighbors_index = {}
    for item in user_idx:
        neighbors_index[item] = {
            'follow': [],
            'friend': []
        }
    print(len(neighbors_index))
    for chunk in edge:
        for index, item in tqdm(chunk.iterrows(), ncols=0):
            source, relation, target = item['source_id'], item['relation'], item['target_id']
            if source.find('u') == 0 and target.find('u') == 0:
                if source not in user_idx or target not in user_idx:
                    continue
                neighbors_index[source][relation].append(target)
    neighbors = [neighbors_index[item] for item in user_idx]
    print(len(neighbors))
    neighbors = np.array(neighbors, dtype=object)
    np.save('tmp/{}/neighbors.npy'.format(dataset), neighbors)


def get_tweet_corpus():
    fb = open('tmp/{}/corpus.txt'.format(dataset), 'w')
    with open(osp.join(path, 'node.json')) as f:
        data = ijson.items(f, 'item')
        for item in tqdm(data, ncols=0):
            if item['id'].find('t') == -1:
                continue
            if item['text'] is None:
                continue
            fb.write(item['text'] + '\n')
    fb.close()


def get_word2vec_model():
    sentences = word2vec.Text8Corpus('tmp/{}/corpus.txt'.format(dataset))
    print('training word2vec model')
    model = word2vec.Word2Vec(sentences, vector_size=128, workers=8, min_count=5)
    vectors = model.wv.vectors
    key_to_index = model.wv.key_to_index
    print(vectors.shape)
    print(len(key_to_index))
    np.save('tmp/{}/vec.npy'.format(dataset), vectors)
    json.dump(key_to_index, open('tmp/{}/key_to_index.json'.format(dataset), 'w'))
    print('training done')


def get_tweets():
    edge = pandas.read_csv(osp.join(path, 'edge.csv'))
    author_idx = {}
    for index, item in tqdm(edge.iterrows(), ncols=0):
        if item['relation'] != 'post':
            continue
        author_idx[item['target_id']] = item['source_id']
    print(len(edge))
    key_to_index = json.load(open('tmp/{}/key_to_index.json'.format(dataset)))
    user_idx = json.load(open('tmp/{}/idx.json'.format(dataset)))
    tweets_index = {}
    for user in user_idx:
        tweets_index[user] = []
    with open(osp.join(path, 'node.json')) as f:
        data = ijson.items(f, 'item')
        for item in tqdm(data, ncols=0):
            if item['id'].find('t') == -1:
                continue
            if item['text'] is None:
                continue
            words = tokenize.word_tokenize(item['text'])
            tweet = []
            for word in words:
                if word in key_to_index:
                    tweet.append(key_to_index[word])
                else:
                    tweet.append(len(key_to_index))
            tweets_index[author_idx[item['id']]].append(tweet)
    tweets = [tweets_index[item] for item in user_idx]
    tweets = np.array(tweets, dtype=object)
    np.save('tmp/{}/tweets.npy'.format(dataset), tweets)


def get_bot_labels():
    user_idx = json.load(open('tmp/{}/idx.json'.format(dataset)))
    label_data = pandas.read_csv('../../datasets/{}/label.csv'.format(dataset))
    label_index = {}
    for index, item in tqdm(label_data.iterrows(), ncols=0):
        label_index[item['id']] = int(item['label'] == 'bot')
    bot_labels = [label_index[item] for item in user_idx]
    bot_labels = np.array(bot_labels)
    np.save('tmp/{}/bot_labels.npy'.format(dataset), bot_labels)


def get_follower_labels():
    follower_counts = []
    with open(osp.join(path, 'node.json')) as f:
        data = ijson.items(f, 'item')
        for item in tqdm(data, ncols=0):
            if item['id'].find('u') == -1:
                continue
            try:
                follower_counts.append(item['public_metrics']['followers_count'])
            except TypeError:
                follower_counts.append(0)
    follower_counts = np.array(follower_counts)
    print(follower_counts.shape)
    threshold = np.percentile(follower_counts, 80)
    print(threshold)
    follower_labels = []
    with open(osp.join(path, 'node.json')) as f:
        data = ijson.items(f, 'item')
        for item in tqdm(data, ncols=0):
            if item['id'].find('u') == -1:
                continue
            try:
                label = int(item['public_metrics']['followers_count'] >= threshold)
            except TypeError:
                label = 0
            follower_labels.append(label)
    follower_labels = np.array(follower_labels)
    print(follower_labels.shape)
    np.save('tmp/{}/follower_labels.npy'.format(dataset), follower_labels)


if __name__ == '__main__':
    path = '../../datasets/{}'.format(dataset)
    get_properties()
    get_neighbors()
    get_tweet_corpus()
    get_word2vec_model()
    get_tweets()
    get_bot_labels()
    get_follower_labels()
