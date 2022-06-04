import pandas as pd
import dask.dataframe as dd
import csv

labels = pd.read_csv("/data2/whr/czl/TwiBot22-baselines/datasets/Twibot-20/label.csv")
user_final = []
for row in labels.iterrows():
    user_final.append(row[1][0])

df = dd.read_csv("/data2/whr/czl/TwiBot22-baselines/datasets/Twibot-20/edge.csv")
user_to_post = {}

for row in df.iterrows():
    if row[1][1] == 'post':
        if row[1][0] in user_final:
            if row[1][0] in user_to_post.keys():
                user_to_post[row[1][0]].append(row[1][2])
            else:
                user_to_post[row[1][0]] = []
                user_to_post[row[1][0]].append(row[1][2])
                
with open("./user_post.csv", "w", newline='') as csv_file:
    writer=csv.writer(csv_file)
    for key, value in user_to_post.items():
        writer.writerow([key,value])