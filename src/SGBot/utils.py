import ijson
from tqdm import tqdm
import json
import os
import os.path as osp


if __name__ == '__main__':
    for dataset in ['Twibot-22', 'Twibot-20', 'midterm-2018', 'gilani-2017',
                    'cresci-stock-2018', 'cresci-rtbust-2019', 'cresci-2017',
                    'cresci-2015', 'botometer-feedback-2019']:
        if not osp.exists('tmp/{}'.format(dataset)):
            os.makedirs('tmp/{}'.format(dataset))
        pbar = tqdm(ncols=0)
        pbar.set_description(dataset)
        path = '../../datasets/{}'.format(dataset)
        ch_list = [chr(i) for i in range(65, 91)] + \
                  [chr(i) for i in range(97, 123)] + \
                  [chr(i) for i in range(48, 58)] + ['_']
        bi_gram_count = {}
        for x in ch_list:
            for y in ch_list:
                tmp = x + y
                bi_gram_count[tmp] = 0
        with open('{}/node.json'.format(path) if dataset != 'Twibot-22' else '{}/user.json'.format(path)) as f:
            data = ijson.items(f, 'item')
            for item in data:
                pbar.update()
                uid = item['id']
                if uid.find('u') == -1:
                    break
                username = item['username']
                if username is None:
                    continue
                username = username.strip()
                for index in range(len(username) - 1):
                    bi_gram = username[index] + username[index + 1]
                    bi_gram_count[bi_gram] += 1
            bi_gram_sum = 0
            for value in bi_gram_count.values():
                bi_gram_sum += value
            for item in bi_gram_count:
                bi_gram_count[item] /= bi_gram_sum
            json.dump(bi_gram_count, open('tmp/{}/bi_gram_likelihood.json'.format(dataset), 'w'))
