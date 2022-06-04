import pandas as pd
import csv

labels = pd.read_csv("/data2/whr/czl/TwiBot22-baselines/datasets/cresci-2017/label.csv")
user_final = []
for row in labels.iterrows():
    user_final.append(row[1][0])

df = pd.read_csv("/data2/whr/czl/TwiBot22-baselines/datasets/cresci-2017/edge.csv")
user_to_post = {}

for row in df.iterrows():
    if row[1][1] == 'post':
        if row[1][0] in user_final:
            if row[1][0] in user_to_post.keys():
                if len(user_to_post[row[1][0]]) >=20:
                    continue
                user_to_post[row[1][0]].append(row[1][2])
            else:
                user_to_post[row[1][0]] = []
                user_to_post[row[1][0]].append(row[1][2])
            
                
with open("./user_post.csv", "w", newline='') as csv_file:
    writer=csv.writer(csv_file)
    for key, value in user_to_post.items():
        writer.writerow([key,value])