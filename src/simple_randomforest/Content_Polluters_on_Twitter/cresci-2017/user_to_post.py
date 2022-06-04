import pandas as pd
import dask.dataframe as dd
import csv

labels = pd.read_csv("/data2/whr/czl/TwiBot22-baselines/datasets/cresci-2017/label.csv")
user_final = list(labels['id'])

df = dd.read_csv("/data2/whr/czl/TwiBot22-baselines/datasets/cresci-2017/edge.csv")
user_to_post = {}

for i, row in df.iterrows():
    if row[1] == 'post':
        if row[0] in user_final:
            if row[0] in user_to_post.keys():
                if len(user_to_post[row[0]]) >= 200:
                    continue
                user_to_post[row[0]].append(row[2])
            else:
                user_to_post[row[0]] = []
                user_to_post[row[0]].append(row[2])
                
with open("./users_post.csv", "w", newline='') as csv_file:
    writer=csv.writer(csv_file)
    for key, value in user_to_post.items():
        writer.writerow([key,value])