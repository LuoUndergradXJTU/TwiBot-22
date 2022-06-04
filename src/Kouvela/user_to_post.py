import pandas as pd
import csv
import sys
import time
import os
from tqdm import tqdm

def to_csv(dic, filename):
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id','post'])
        writer.writerows([dic])

def main(argv):
    if argv[1] == '--datasets':
        try:
            name = argv[2]
            return name
        except:
            return "Wrong command!"
    else:
        return "Wrong command!"

if __name__ == '__main__':
    dataset_name = main(sys.argv)
    if not os.path.exists('./{}'.format(dataset_name)):
        os.mkdir('./{}'.format(dataset_name))
    filepath = "{}/user_to_post.csv".format(dataset_name)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'post'])

    start = time.time()
    labels = pd.read_csv("./datasets/{}/label.csv".format(dataset_name))
    user_final = list(labels['id']) # user list with label

    df = pd.read_csv("./datasets/{}/edge.csv".format(dataset_name))
    df = df[df['relation'] == 'post']
    df = df[df['source_id'].isin(user_final)] # Filter users with label
    df.sort_values(by="source_id", axis=0, ascending=True, inplace=True, ignore_index=True)

    user_to_post = {}
    final = {}
    temp = df.loc[0]['source_id']
    user_to_post[temp] = []
    flag = False
    row_num = df.shape[0]
    for row in tqdm(df.iterrows(), total=row_num):
        if row[1][0] > temp:
            flag = True
            if flag:
                final['id'] = list(user_to_post.keys())[0]
                final['post'] = list(user_to_post.values())[0]
                to_csv(final, filepath)
                user_to_post = {}
                flag = False
            user_to_post[row[1][0]] = []
            user_to_post[row[1][0]].append(row[1][2])
            temp = row[1][0]
        else:
            user_to_post[row[1][0]].append(row[1][2])

    final['id'] = list(user_to_post.keys())[0]
    final['post'] = list(user_to_post.values())[0]
    to_csv(final, filepath)