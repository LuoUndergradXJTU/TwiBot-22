from pathlib import Path
import pandas as pd
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
import torch
import numpy as np
import os
from torch_geometric.data import Data, HeteroData

def get_data_dir(server_id):
    if server_id == "206":
        return Path("/new_temp/Twibot22-baselines/datasets")
    elif server_id == "208":
        return Path("")
    elif server_id == "209":
        return Path("/data2/whr/TwiBot22-baselines/datasets")
    else:
        raise NotImplementedError

dataset_names = [
    'botometer-feedback-2019', 'botwiki-2019', 'celebrity-2019', 'cresci-2015', 'cresci-2017', 'cresci-rtbust-2019', 'cresci-stock-2018', 'gilani-2017', 'midterm-2018', 'political-bots-2019', 'pronbots-2019', 'vendor-purchased-2019', 'verified-2019', "Twibot-20"
]

def merge(dataset="Twibot-20", server_id="209"):
    assert dataset in dataset_names, f"Invalid dataset {dataset}"
    dataset_dir = get_data_dir(server_id) / dataset
    node_info = pd.read_csv(dataset_dir / "node.json")
    label = pd.read_csv(dataset_dir / "label.csv")
    node_info = pd.merge(node_info, label)
    return node_info

def split_user_and_tweet(df):
    """
    split user and tweet from df, and ignore entries whose id is `None`
    """
    df = df[df.id.str.len() > 0]
    return df[df.id.str.contains("^u")], df[df.id.str.contains("^t")]
    

def fast_merge(dataset="Twibot-20", server_id="209"):
    assert dataset in dataset_names, f"Invalid dataset {dataset}"
    dataset_dir = get_data_dir(server_id) / dataset
    node_info = pd.read_json(dataset_dir / "node.json")
    label = pd.read_csv(dataset_dir / "label.csv")
    split = pd.read_csv(dataset_dir / "split.csv")
    
    user, tweet = split_user_and_tweet(node_info)
    
    id_to_label = {}
    id_to_split = {}
    
    length = len(label)
    
    if len(label) != len(split):
        for i in range(length):
            id_to_label[label["id"][i]] = label["label"][i]
        
        length = len(split)
        for i in range(length):
            id_to_split[split["id"][i]] = split["split"][i]
    else:
        for i in range(length):
            id_to_label[label["id"][i]] = label["label"][i]
            id_to_split[split["id"][i]] = split["split"][i]
    
    length = len(user)
    
    user["label"] = "None"
    user["split"] = "None"
    
    for i in tqdm(range(length)):
        if user["id"][i] in id_to_label.keys():
            if i<len(label):
                user["label"][i] = id_to_label[user["id"][i]]
                user["split"][i] = id_to_split[user["id"][i]]
            else:
                user["label"][i] = np.nan
                user["split"][i] = id_to_split[user["id"][i]]
        
    # length = len(split)
    
    # for i in range(length):
    #     id_to_split[split["id"][i]] = split["split"][i]
    #     id_to_label[label["id"][i]] = label["label"][i]
        
    # tweet_id = list(tweet.id)
    # tweet["label"] = list(map(lambda x: id_to_label[x], tweet_id))
    # tweet["split"] = list(map(lambda x: id_to_split[x], tweet_id))
    #torch.save(user,'./user.pt')
    #torch.save(tweet,'./tweet.pt')
    return user, tweet
        
    


def merge_and_split(dataset="botometer-feedback-2019", server_id="209"):
    assert dataset in dataset_names, f"Invalid dataset {dataset}"
    dataset_dir = get_data_dir(server_id) / dataset
    node_info = pd.read_json(dataset_dir / "node.json")
    label = pd.read_csv(dataset_dir / "label.csv")
    split = pd.read_csv(dataset_dir / "split.csv")
    node_info = pd.merge(node_info, label)
    node_info = pd.merge(node_info, split)
    
    
    train = node_info[node_info["split"] == "train"]
    valid = node_info[node_info["split"] == "val"]
    test = node_info[node_info["split"] == "test"]
    
    
    return train, valid, test

@torch.no_grad()
def simple_vectorize(data):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')
    
    # public_metrics = list(data["public_metrics"])
    descriptions = list(data["description"])
    labels = list(data["label"])
    
    labels = list(map(lambda x: 0 if x == "human" else 1, labels))
    
    feats = []
    
    for text in tqdm(descriptions):
        if text is None: 
            feats.append(torch.randn(768))
            continue
        encoded_input = tokenizer(text, return_tensors='pt')
        feats.append(model(**encoded_input)["pooler_output"][0])
        
    feats = torch.stack(feats, dim=0)
    
    return feats.numpy(), np.array(labels, dtype=np.int32)
    
def homo_graph_vectorize(include_node_feature=False, dataset="Twibot-20", server_id="209"):
    dataset_dir = get_data_dir(server_id) / dataset
    
    data = merge(dataset, server_id)
    length = len(data)
    
    index2id = list(data["id"])
    
    id2index = {x: i for i, x in enumerate(index2id)}
    
    text = [a + " " + b for a, b in zip(list(data["description"], list(data["text"])))]
    
    edge = pd.read_csv(dataset_dir / "edge.csv")
    edge_types = set(edge["relation"])
    src = list(edge["source_id"])
    dst = list(edge["target_id"])
    
    src = list(map(lambda x: id2index[x], src))
    dst = list(map(lambda x: id2index[x], dst))
    
    edge_type_to_edge_index = {x: i for i, x in enumerate(edge_types)}
    edge_index = torch.LongTensor([src, dst])
    edge_type = torch.LongTensor(list(map(lambda x: edge_type_to_edge_index[x], edge["relation"])))
    
    
    if include_node_feature:
        tokenizer = RobertaTokenizer.from_pretained('roberta-base')
        model = RobertaModel.from_pretrained("roberta-base")
        
        text_feats = []
        for t in tqdm(text):
            encoded_input = tokenizer(t, return_tensors="pt")
            text_feats.append(model(**encoded_input)["pooler_ouput"][0])
        
        text_feats = torch.FloatTensor(text_feats)
        
        return Data(x=text_feats, edge_index=edge_index, edge_type=edge_type)
    else:
        
        return Data(edge_index=edge_index, edge_type=edge_type)
    



@torch.no_grad()
def hetero_graph_vectorize(include_node_feature=False, dataset="cresci-2015", server_id="209"):
    dataset_dir = get_data_dir(server_id) / dataset
    if not os.path.exists('./user.pt'):
        user, tweet = fast_merge(dataset, server_id)
    else:
        print('loading user.pt',end='   ')
        user=torch.load('./user.pt')
        print('Done.')
        print('loading tweet.pt',end='   ')
        tweet=torch.load('./tweet.pt')
        print('Done')
        
    unique_uid = set(user.id)
    unique_tid = set(tweet.id)
    user_index_to_uid = list(user.id)
    tweet_index_to_tid = list(tweet.id)
        
    uid_to_user_index = {x : i for i, x in enumerate(user_index_to_uid)}
    tid_to_tweet_index = {x : i for i, x in enumerate(tweet_index_to_tid)}
    
    '''
    user_text = [text for text in user.description]
    tweet_text = [text for text in tweet.text]
    
    
    assert len(user_text) == len(user_index_to_uid)
    assert len(tweet_text) == len(tweet_index_to_tid)
    '''
    edge = pd.read_csv(dataset_dir / "edge.csv")
    edge_types = set(edge["relation"])
    
    graph = HeteroData()
    
    for edge_type in edge_types:
        src = list(edge[edge["relation"] == edge_type]["source_id"])
        dst = list(edge[edge["relation"] == edge_type]["target_id"])

        
        if edge_type == "post":
            continue
            
            new_src = []
            new_dst = []
            
            for s, t in zip(src, dst):
                if s not in unique_uid or t not in unique_tid:
                    continue
                new_src.append(s)
                new_dst.append(t)
                
            src = new_src
            dst = new_dst
            
            src = list(map(lambda x: uid_to_user_index[x], src))
            dst = list(map(lambda x: tid_to_tweet_index[x], dst))
            
            ##

            graph["user", "post", "tweet"] = torch.LongTensor([src, dst])
            
        else:
            
            # src = list(map(lambda x: uid_to_user_index[x], src))
            # dst = list(map(lambda x: uid_to_user_index[x], dst))
            
            new_src = []
            new_dst = []
            
            for s, t in zip(src, dst):
                new_src.append(s)
                new_dst.append(t)
                
            src = new_src
            dst = new_dst
            
            src = list(map(lambda x: uid_to_user_index[x], src))
            dst = list(map(lambda x: uid_to_user_index[x], dst))
            
            graph["user", edge_type, "user"].edge_index = torch.LongTensor([src, dst])
            
    train_uid_with_label = user[user.split == "train"][["id", "split", "label"]]
    valid_uid_with_label = user[user.split == "val"][["id", "split", "label"]]
    test_uid_with_label = user[user.split == "test"][["id", "split", "label"]]
    
    if include_node_feature:
        if not os.path.exists('./graph.pt'):
            if not os.path.exists('../../data/des_tensor.pt'):
                print('loading roberta tokenizer & model')
                tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
                model = RobertaModel.from_pretrained("roberta-base")
                
                user_text_feats = []
                tweet_text_feats = []
                
                for text in tqdm(tweet_text):
                    if text is None:
                        tweet_text_feats.append(torch.zeros(768))
                        continue
                    if type(text)==float:
                        continue
                    encoded_input = tokenizer(text, return_tensors="pt")
                    tweet_text_feats.append(model(**encoded_input)["pooler_output"][0])
                
                for t in tqdm(user_text):
                    if t is None:
                        user_text_feats.append(torch.zeros(768))
                        continue
                    if type(t)==float:
                        continue
                    encoded_input = tokenizer(t, return_tensors="pt")
                    user_text_feats.append(model(**encoded_input)["pooler_output"][0])
                
                graph["user"].x = torch.stack(user_text_feats, dim=0).to('cpu')
                graph["tweet"].x = torch.stack(tweet_text_feats, dim=0).to('cpu')
                torch.save(graph,'./graph.pt')
            else:
                graph["user"].x=torch.load('../../data/des_tensor.pt')
                graph["tweet"].x=torch.load('../../data/tweets_tensor.pt')
                torch.save(graph,'./graph.pt')
        else:
            torch.load('./graph.pt')
        return graph, uid_to_user_index, tid_to_tweet_index, train_uid_with_label, valid_uid_with_label, test_uid_with_label
        
        
    else:
        
        return graph, uid_to_user_index, tid_to_tweet_index, train_uid_with_label, valid_uid_with_label, test_uid_with_label,user_index_to_uid,tweet_index_to_tid


@torch.no_grad()
def homo_graph_vectorize_only_user(include_node_feature=False, dataset="cresci-2015", server_id="209"):
    dataset_dir = get_data_dir(server_id) / dataset
    
    user, _ = fast_merge(dataset, server_id)
    labels = list(user.label)
    labels = list(map(lambda x: 1 if x == "human" else 0, labels))
    labels = torch.LongTensor(labels)
    
    unique_uid = set(user.id)
    
    user_index_to_uid = list(user.id)
    uid_to_user_index = {x : i for i, x in enumerate(user_index_to_uid)}
    
    user_text = [text for text in user.description]
    
    assert len(user_text) == len(user_index_to_uid)
    
    edge = pd.read_csv(dataset_dir / "edge.csv")

    src = list(edge["source_id"])
    dst = list(edge["target_id"])          
        
    # src = list(map(lambda x: uid_to_user_index[x], src))
    # dst = list(map(lambda x: uid_to_user_index[x], dst))
    
    new_src = []
    new_dst = []
    
    for s, t in zip(src, dst):
        if s not in unique_uid or t not in unique_uid:
            continue
        new_src.append(s)
        new_dst.append(t)
        
    src = new_src
    dst = new_dst
    
    src = list(map(lambda x: uid_to_user_index[x], src))
    dst = list(map(lambda x: uid_to_user_index[x], dst))
    
    edge_index = torch.LongTensor([src, dst])
            
    train_uid_with_label = user[user.split == "train"][["id", "split", "label"]]
    valid_uid_with_label = user[user.split == "val"][["id", "split", "label"]]
    test_uid_with_label = user[user.split == "test"][["id", "split", "label"]]
    
    if include_node_feature:
        if "user_info.pt" not in os.listdir(dataset_dir):
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            model = RobertaModel.from_pretrained("roberta-base")
            
            user_text_feats = []
            
            for t in tqdm(user_text):
                if t is None:
                    user_text_feats.append(torch.zeros(768))
                    continue
                encoded_input = tokenizer(t, return_tensors="pt")
                user_text_feats.append(model(**encoded_input)["pooler_output"][0])
                
            user_text_feats = torch.stack(user_text_feats, dim=0)
                
            user_info = {
                "user_text_feats": user_text_feats,
                "edge_index": edge_index,
                "labels": labels,
                "uid_to_user_index": uid_to_user_index,
                "train_uid_with_label": train_uid_with_label,
                "valid_uid_with_label": valid_uid_with_label,
                "test_uid_with_label": test_uid_with_label,
            }
            
            torch.save(user_info, dataset_dir / "user_info.pt")
        
            return user_text_feats, edge_index, labels, uid_to_user_index, train_uid_with_label, valid_uid_with_label, test_uid_with_label
        
        else:
            return tuple(torch.load(dataset_dir / "user_info.pt").values())
        
        
    else:
        return user_text, edge_index, labels, uid_to_user_index, train_uid_with_label, valid_uid_with_label, test_uid_with_label

def df_to_mask(uid_with_label, uid_to_user_index, phase="train"):
    user_list = list(uid_with_label[uid_with_label.split == phase].id)
    phase_index = list(map(lambda x: uid_to_user_index[x], user_list))
    return torch.LongTensor(phase_index)

## debug
if __name__ == "__main__":
    # homo_graph_vectorize()
    # hetero_graph_vectorize(dataset="cresci-2015", include_node_feature=True)
    homo_graph_vectorize_only_user(True)