import json
import csv
from tqdm import tqdm
import time
import os
import ijson
import pandas as pd

def to_csv(header, dic, filename):
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writerows([dic])

if __name__ == '__main__':
    filepath = "./tweet.csv"
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'text'])

    # df = pd.DataFrame(columns = ['id','text'])
    with open("./datasets/cresci-2015/node.json") as f:
        obj = ijson.items(f, 'item')
        while True:
            try:
                data = obj.__next__()
                if data['id'][0] == 't':
                    content = {}
                    content['id'] = data['id']
                    content['text'] = data['text']
                    to_csv(['id','text'], content, filepath)
                    # df.append(content, ignore_index=True)
            except StopIteration as e:
                break

    # df.sort_values(by="id", axis=0, ascending=True, inplace=True, ignore_index=True)
    final = pd.read_csv(filepath)
    final.sort_values(by="id", axis=0, ascending=True, inplace=True, ignore_index=True)

    num = 300000
    batch = int(len(final) / num)
    if not os.path.exists('./small_split_dataset'):
        os.mkdir('./small_split_dataset')
    for i in tqdm(range(batch)):
        begin = i * num
        tweet = final.loc[begin: begin + num - 1]
        tweet.to_csv('./small_split_dataset/tweet{}.csv'.format(i), index=False)