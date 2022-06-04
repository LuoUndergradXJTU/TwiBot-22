import os

import torch
import os.path as osp
from torch_geometric.data import Data


def get_transfer_data():
    path = '../../BotRGCN/twibot_22/processed_data'
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
                cat_property_embedding=cat_property_embedding,
                num_nodes=labels.shape[0])


data_index = {
    'cresci-2015': 'cresci_15',
    'Twibot-22': 'twibot_22',
    'Twibot-20': 'twibot_20'
}


def get_train_data(dataset_name):
    path = '../../BotRGCN/{}/processed_data'.format(data_index[dataset_name])
    if not osp.exists(path):
        raise KeyError
    labels = torch.load(osp.join(path, 'label.pt'))
    des_embedding = torch.load(osp.join(path, 'des_tensor.pt'))
    tweet_embedding = torch.load(osp.join(path, 'tweets_tensor.pt'))
    num_property_embedding = torch.load(osp.join(path, 'num_properties_tensor.pt'))
    cat_property_embedding = torch.load(osp.join(path, 'cat_properties_tensor.pt'))
    edge_type = torch.load(osp.join(path, 'edge_type.pt'))
    edge_index = torch.load(osp.join(path, 'edge_index.pt'))
    if dataset_name == 'Twibot-20':
        labels = torch.cat([labels, torch.empty(217754, dtype=torch.long).fill_(2)])
        train_idx = torch.arange(0, 8278)
        val_idx = torch.arange(8278, 8278 + 2365)
        test_idx = torch.arange(8278+2365, 8278 + 2365 + 1183)
    else:
        train_idx = torch.load(osp.join(path, 'train_idx.pt'))
        val_idx = torch.load(osp.join(path, 'val_idx.pt'))
        test_idx = torch.load(osp.join(path, 'test_idx.pt'))
    return Data(edge_index=edge_index,
                edge_type=edge_type,
                y=labels,
                des_embedding=des_embedding,
                tweet_embedding=tweet_embedding,
                num_property_embedding=num_property_embedding,
                cat_property_embedding=cat_property_embedding,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                num_nodes=des_embedding.shape[0])


