import json
from tqdm import tqdm
from argparse import ArgumentParser
import os


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    args = parser.parse_args()
    dataset_name = args.dataset_name
    username_path = 'tmp/username/{}'.format(dataset_name)
    if not os.path.exists(username_path):
        raise ValueError
    usernames = json.load(open(os.path.join(username_path, 'users_test.json')))
    idx = set()
    for item in usernames:
        idx.add(item['id'])
    pbar = tqdm(total=len(usernames), ncols=0)
    save_path = 'tmp/scores/{}'.format(dataset_name)
    if not os.path.exists(save_path):
        raise ValueError
    file_list = os.listdir(save_path)
    cnt = 0
    for file in file_list:
        data = json.load(open(os.path.join(save_path, file)))
        if isinstance(data, str):
            os.remove(os.path.join(save_path, file))
            cnt += 1
        pbar.update()
    print(cnt)


