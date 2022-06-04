import json
import os
import os.path as osp
from argparse import ArgumentParser
from tqdm import tqdm


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    args = parser.parse_args()
    path = 'tmp/scores/{}'.format(args.dataset_name)
    if not osp.exists(path):
        raise ValueError
    scores = []
    for file in tqdm(os.listdir(path), ncols=0):
        try:
            data = json.load(open(osp.join(path, file)))
        except json.decoder.JSONDecodeError:
            continue
        if isinstance(data, str):
            continue
        item = data['cap']
        item['id'] = file.replace('.json', '')
        scores.append(item)
    save_path = 'tmp/output'.format(args.dataset_name)
    if not osp.exists(save_path):
        os.makedirs(save_path)
    print(len(scores))
    json.dump(scores, open(osp.join(save_path, '{}.json'.format(args.dataset_name)), 'w'))