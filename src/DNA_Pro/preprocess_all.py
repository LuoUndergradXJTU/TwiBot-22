import os
import json
import pandas as pd
from tqdm import tqdm

dataset_path = '../../datasets'
dataset_list = os.listdir(dataset_path)
# dataset_list = ['Twibot-20']


class User:
    """Class of a User"""
    def __init__(self, userid:  str):
        self.userid = str(userid)
        self.dna = ''

    def add_tweet(self, content:  str) -> None:
        try:
            if content[0: 4] == 'RT @':
                self.dna += 'C'  # C is for a retweet
            elif content[0] == '@':
                self.dna += 'T'  # T is for a reply
            else:
                self.dna += 'A'  # A is for a simple tweet
        except:
            raise Exception('Error!', self.userid, content)

    def output(self, file):
        if len(self.dna) <= 200:
            file.write("{0} {1}\n".format(self.userid, self.dna))
        else:
            file.write("{0} {1}\n".format(self.userid, self.dna[0: 200]))
        return len(self.dna)

    def dna_len(self):
        return len(self.dna)


class Tweet:
    """Class of a Tweet"""
    def __init__(self, tweetid:  str, content:  str):
        self.tweetid = str(tweetid)
        self.content = str(content)

    def get_content(self) -> str:
        return self.content


if __name__ == '__main__':

    for dataset_name in dataset_list:
        user_id = []
        tweet_id = []
        node_class = {}
        cnt = 0

        print('Dataset {}'.format(dataset_name))

        if 'edge.csv' not in os.listdir(dataset_path + '/{}/'.format(dataset_name)):
            continue

        test_set_list = []
        f0 = pd.read_csv(dataset_path + '/{}/split.csv'.format(dataset_name))
        for item_id, item in enumerate(tqdm(f0['split'], desc='Loading User Lists')):
            if item != 'support':
                test_set_list.append(f0['id'][item_id])

        f1 = open(dataset_path + '/{}/node.json'.format(dataset_name))
        lines = json.load(f1)
        for line in tqdm(lines, desc='Loading Nodes'):
            if line['id'][0] == 'u':
                if line['id'] in test_set_list:
                    user_id.append(line['id'])
                    node_class[line['id']] = (User(line['id']))
            elif line['id'][0] == 't':
                tweet_id.append(line['id'])
                node_class[line['id']] = (Tweet(line['id'], line['text']))
            else:
                raise Exception('Unexpected node type!', line['id'])
        f1.close()

        f2 = pd.read_csv(dataset_path + '/{}/edge.csv'.format(dataset_name))
        for item_id, item in enumerate(tqdm(f2['relation'], desc='Loading Edges')):
            if item == 'post' and f2['source_id'][item_id] in test_set_list:
                node_class[f2['source_id'][item_id]].add_tweet(node_class[f2['target_id'][item_id]].get_content())
        del f2

        f3 = open('./tmp/datasets/{}_DNA.txt'.format(dataset_name), 'w', encoding='UTF-8')
        for item in tqdm(user_id, desc='Writing DNA'):
            cnt += node_class[item].output(f3)
        f3.close()
        print('cnt = {}'.format(cnt))
