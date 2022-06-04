import ijson
from argparse import ArgumentParser
import os
import json
from tqdm import tqdm
import pandas

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    args = parser.parse_args()
    dataset_path = '../../datasets/{}'.format(args.dataset_name)
    if not os.path.exists(dataset_path):
        raise ValueError
    splits = pandas.read_csv(os.path.join(dataset_path, 'split.csv'))
    user_test = set()
    for index, item in splits.iterrows():
        if item['split'] == 'test':
            user_test.add(item['id'])
    print(len(user_test))
    pbar = tqdm(total=len(user_test), ncols=0)
    users = []
    with open(os.path.join(dataset_path, 'node.json' if args.dataset_name != 'Twibot-22' else 'user.json')) as f:
        objects = ijson.items(f, 'item')
        while True:
            try:
                data = next(objects)
            except StopIteration:
                break
            if data['id'].find('u') == -1:
                continue
            if data['id'] not in user_test:
                continue
            users.append({'id': data['id'],
                          'username': data['username']})
            pbar.update()
    print(len(users))
    save_path = 'tmp/username/{}'.format(args.dataset_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    json.dump(users, open(os.path.join(save_path, 'users_test.json'), 'w'))
