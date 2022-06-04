import csv
import pandas as pd
import torch
from tqdm import tqdm

label_list = pd.read_csv("/data2/whr/czl/TwiBot22-baselines/datasets/Twibot-20/label.csv")
label = []

for i in label_list["label"]:
    if i == "bot":
        label.append(1)
    if i == "human":
        label.append(0)

label_tensor = torch.tensor(label)
torch.save(label_tensor, "./label.pt")
print(len(label))

id_list = pd.read_csv("/data2/whr/czl/TwiBot22-baselines/datasets/Twibot-20/split.csv")

id2index = {}
k = 0

for i in id_list["id"].tolist():
    id2index[i] = k
    k+=1

follow_src = []
follow_dst = []
friend_src = []
friend_dst = []

edge = pd.read_csv("/data2/whr/czl/TwiBot22-baselines/datasets/Twibot-20/edge.csv")
follow = edge[edge["relation"] == "follow"]
friend = edge[edge["relation"] == "friend"]

with tqdm(follow["source_id"].tolist(), desc="follow_src") as pbar:
    for i in pbar:
        follow_src.append(id2index[i])

for i in tqdm(follow["target_id"].tolist(), desc="follow_dst"):
    follow_dst.append(id2index[i])

for i in tqdm(friend["source_id"].tolist(), desc="friend_src"):
    friend_src.append(id2index[i])

for i in tqdm(friend["target_id"].tolist(), desc="friend_dst"):
    friend_dst.append(id2index[i])

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

