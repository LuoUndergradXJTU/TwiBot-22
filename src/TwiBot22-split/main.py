import json
import os
import os.path as osp
import ijson
import pandas
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from queue import Queue
import random
random.seed(20220401)


def get_tag_features():
    key_to_index = json.load(open('data/key_to_index.json'))
    vec = np.load('data/vec.npy')
    print(vec.shape)
    tags = []
    features = []
    with open('../../datasets/Twibot-22/hashtag.json') as f:
        data = ijson.items(f, 'item')
        for item in tqdm(data, ncols=0):
            tag = item['tag_name']
            if tag in key_to_index:
                tags.append(tag)
                features.append(vec[key_to_index[tag]])
    features = np.array(features)
    print(features.shape)
    print(len(tags))
    json.dump(tags, open('data/tags.json', 'w'))
    np.save('data/tags_features.npy', features)


def get_kmeans_cluster():
    tags = json.load(open('data/tags.json'))
    features = np.load('data/tags_features.npy')
    print(features.shape)
    print(len(tags))
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(features)
    print(kmeans.labels_)
    np.save('data/kmeans_label.npy', kmeans.labels_)
    tags = json.load(open('data/tags.json'))
    labels = np.load('data/kmeans_label.npy')
    tags_label = [[] for _ in range(10)]
    for index, item in enumerate(tqdm(labels, ncols=0)):
        tags_label[item].append(tags[index])
    json.dump(tags_label, open('data/tags_cluster.json', 'w'))


def split1():
    tags_cluster = json.load(open('data/tags_cluster.json'))
    tag_index = {}
    for index in range(10):
        for tag in tags_cluster[index]:
            tag_index[tag] = index
    print(len(tag_index))
    for index in range(10):
        print(index, tags_cluster[index][:20])
    exit(0)
    users = [set() for _ in range(10)]
    for i in range(9):
        with open('../../datasets/Twibot-22/tweet_{}.json'.format(i)) as f:
            data = ijson.items(f, 'item')
            pbar = tqdm(total=10000000, ncols=0)
            pbar.set_description(''.format(i))
            for item in data:
                pbar.update()
                entities = item['entities']
                user_id = 'u{}'.format(item['author_id'])
                if entities is None or 'hashtags' not in entities:
                    continue
                hashtags = entities['hashtags']
                if not hashtags:
                    continue
                for hashtag in hashtags:
                    if 'text' in hashtag:
                        tag = hashtag['text']
                    elif 'tag' in hashtag:
                        tag = hashtag['tag']
                    else:
                        continue
                    if tag not in tag_index:
                        continue
                    users[tag_index[tag]].add(user_id)
    for index, item in enumerate(users):
        item = list(item)
        json.dump(item, open('split1/user_{}.json'.format(index), 'w'))


def add_edge(x, y, edges):
    if x not in edges:
        edges[x] = []
    edges[x].append(y)


def split_from_one_user(uid, edges):
    user_neighbors = set()
    user_queue = Queue()

    user_neighbors.add(uid)
    user_queue.put((uid, 0))
    max_neighbor_count = 0
    for index, item in edges.items():
        max_neighbor_count = max(max_neighbor_count, len(item))
    interval = max_neighbor_count // 120
    while not user_queue.empty():
        (user, depth) = user_queue.get()
        if user not in edges:
            continue
        neighbors = edges[user]
        random.shuffle(neighbors)
        neighbors = neighbors[: len(neighbors) // interval + 1]
        for neighbor in neighbors:
            if neighbor in user_neighbors:
                continue
            user_neighbors.add(neighbor)
            user_queue.put((neighbor, depth + 1))
    # print(len(user_neighbors))
    user_neighbors = list(user_neighbors)
    json.dump(user_neighbors, open('split2/user_{}.json'.format(uid), 'w'))


def split2():
    edges = {}
    with open('data/edge_user_ff_user.json') as f:
        data = ijson.items(f, 'item')
        for item in data:
            add_edge(item[0], item[2], edges)
            add_edge(item[2], item[0], edges)
    print('edge load done')
    users = []
    with open('../../datasets/Twibot-22/user.json') as f:
        data = ijson.items(f, 'item')
        for item in tqdm(data, ncols=0):
            users.append((item['id'], item['username'], item['public_metrics']['followers_count']))
    users.sort(key=lambda x: x[2], reverse=True)
    for item in tqdm(users[:20], ncols=0):
        split_from_one_user(int(item[0].replace('u', '')), edges)
    split_from_one_user(138840988, edges)


def sample_split1():
    labels = json.load(open('data/labels.csv'))
    path = 'split1'
    files = os.listdir(path)
    user_cnt = {}
    for file in files:
        data = json.load(open(osp.join(path, file)))
        for item in data:
            if item not in user_cnt:
                user_cnt[item] = 0
            user_cnt[item] += 1
    sample_user = set()
    for index, value in user_cnt.items():
        if value == 1:
            sample_user.add(index)
    file_list = []
    for file in files:
        data = json.load(open(osp.join(path, file)))
        random.shuffle(data)
        tmp = 0
        for item in data:
            if item not in sample_user:
                continue
            tmp += 1
        file_list.append((file, tmp))
    file_list.sort(key=lambda x: x[1])
    done_user = set()
    for file, _ in file_list:
        data = json.load(open(osp.join(path, file)))
        random.shuffle(data)
        bot = []
        human = []
        for item in data:
            if item in done_user:
                continue
            if len(bot) < 5000 and labels[item] == 1:
                bot.append(item)
                done_user.add(item)
            if len(human) < 5000 and labels[item] == 0:
                human.append(item)
                done_user.add(item)
        users = bot + human
        print(len(users))
        json.dump(users, open('final/user_1_{}.json'.format(file), 'w'))


def sample_split2():
    labels = json.load(open('data/labels.csv'))
    path = 'split2'
    files = os.listdir(path)
    user_cnt = {}
    for file in files:
        data = json.load(open(osp.join(path, file)))
        for item in data:
            if item not in user_cnt:
                user_cnt[item] = 0
            user_cnt[item] += 1
    sample_user = set()
    for index, value in user_cnt.items():
        if value == 1:
            sample_user.add(index)
    file_list = []
    for file in files:
        data = json.load(open(osp.join(path, file)))
        random.shuffle(data)
        tmp = 0
        for item in data:
            if item not in sample_user:
                continue
            tmp += 1
        file_list.append((file, tmp))
    file_list.sort(key=lambda x: x[1])
    done_user = {}
    for file, _ in file_list:
        data = json.load(open(osp.join(path, file)))
        data = ['u{}'.format(item) for item in data]
        random.shuffle(data)
        bot = []
        human = []
        for item in data:
            if item in done_user and done_user[item] == 3:
                continue
            if len(bot) < 5000 and labels[item] == 1:
                bot.append(item)
                if item not in done_user:
                    done_user[item] = 0
                done_user[item] += 1
            if len(human) < 5000 and labels[item] == 0:
                human.append(item)
                if item not in done_user:
                    done_user[item] = 0
                done_user[item] += 1
        # print(len(bot), len(human))
        users = bot + human
        print(len(users))
        json.dump(users, open('final/user_2_{}.json'.format(file), 'w'))


if __name__ == '__main__':
    split1()
    '''
    edges = {}
    with open('data/edge_user_ff_user.json') as f:
        data = ijson.items(f, 'item')
        for item in data:
            add_edge(item[0], item[2], edges)
            add_edge(item[2], item[0], edges)
    print('edge load done')
    users = []
    with open('../../datasets/Twibot-22/user.json') as f:
        data = ijson.items(f, 'item')
        for item in tqdm(data, ncols=0):
            users.append((item['id'], item['username'], item['public_metrics']['followers_count']))
    users.sort(key=lambda x: x[2], reverse=True)
    for item in tqdm(users[:20], ncols=0):
        print(item)
    '''





















