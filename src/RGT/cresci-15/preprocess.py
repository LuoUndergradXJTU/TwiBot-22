import csv
import pandas as pd
import torch
from tqdm import tqdm
import json

f = open('/data2/whr/czl/TwiBot22-baselines/datasets/cresci-2015/node.json')
users = json.load(f)
id2index = {}
k = 0

for user in users:

    id2index[str(user["id"])] = k
    k+=1

print("k: {}".format(k))

label_list = pd.read_csv("/data2/whr/czl/TwiBot22-baselines/datasets/cresci-2015/label.csv")
label = torch.zeros(label_list.shape[0])

for index, data in tqdm(label_list.iterrows()):
    if data["label"] == "human":
        label[id2index[data["id"]]] = 0
    if data["label"] == "bot":
        label[id2index[data["id"]]] = 1

torch.save(label.long(), "./label.pt")
print(len(label))

train_idx = []
val_idx = []
test_idx = []

split_list = pd.read_csv("/data2/whr/czl/TwiBot22-baselines/datasets/cresci-2015/split.csv")
for index, data in split_list.iterrows():
    if data["split"]=="train":
        train_idx.append(id2index[data["id"]])
    if data["split"]=="val":
        val_idx.append(id2index[data["id"]])
    if data["split"]=="test":
        test_idx.append(id2index[data["id"]])

torch.save(torch.tensor(train_idx), "./train_idx.pt")
torch.save(torch.tensor(val_idx), "./val_idx.pt")
torch.save(torch.tensor(test_idx), "./test_idx.pt")

follow_src = []
follow_dst = []
friend_src = []
friend_dst = []

edge = pd.read_csv("/data2/whr/czl/TwiBot22-baselines/datasets/cresci-2015/edge.csv")
follow = edge[edge["relation"] == "follow"]
friend = edge[edge["relation"] == "friend"]

for index, data in tqdm(follow.iterrows()):
    if data["source_id"] in id2index.keys() and data["target_id"] in id2index.keys():
        follow_src.append(id2index[data["source_id"]])
        follow_dst.append(id2index[data["target_id"]])


for index, data in tqdm(friend.iterrows()):
    if data["source_id"] in id2index.keys() and data["target_id"] in id2index.keys():
        friend_src.append(id2index[data["source_id"]])
        friend_dst.append(id2index[data["target_id"]])

follow_edge = torch.tensor([follow_src, follow_dst])
friend_edge = torch.tensor([friend_src, friend_dst])

print(follow_edge.size())
print(friend_edge.size())

edge_index = torch.cat((follow_edge, friend_edge), dim=1)
edge_type = torch.cat((torch.zeros(follow_edge.size(1)), torch.ones(friend_edge.size(1))))

print(edge_index.size())
print(edge_type.size())

torch.save(edge_index, "./edge_index.pt")
torch.save(edge_type, "./edge_type.pt")

# for i in tqdm(follow["source_id"]):

# edge_follow = torch.cat([follow["source_id"], friend["target_id"]], dim=0)
# edge_friend = torch.cat([friend["source_id"]])

# for index, row in tqdm(edge.iterrows()):
    
#     print(row)

    # if row["relation"] == "follow":
    #     edge_follow.append([id2index[row["source_id"]], id2index[row["target_id"]]])
    
    # if row["relation"] == "friend":
    #     edge_friend.append([id2index[row["source_id"]], id2index[["target_id"]]])

