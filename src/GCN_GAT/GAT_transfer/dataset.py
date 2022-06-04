import torch
import os.path as osp
from torch_geometric.data import Data


def get_data():
    path = '/data2/whr/czl/TwiBot22-baselines/src/BotRGCN/data_twi22/'
    labels = torch.load(osp.join(path, 'label.pt'))
    des_embedding = torch.load(osp.join(path, 'des_tensor.pt'))
    tweet_embedding = torch.load(osp.join(path, 'tweets_tensor.pt'))
    num_property_embedding = torch.load(osp.join(path, 'num_properties_tensor.pt'))
    cat_property_embedding = torch.load(osp.join(path, 'cat_properties_tensor.pt'))
    edge_index = torch.load(osp.join(path, 'edge_index.pt'))
    return Data(edge_index=edge_index,
                y=labels,
                des_embedding=des_embedding,
                tweet_embedding=tweet_embedding,
                num_property_embedding=num_property_embedding,
                cat_property_embedding=cat_property_embedding)

