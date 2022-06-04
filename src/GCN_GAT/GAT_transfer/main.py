from tqdm import tqdm
from dataset import get_data
from model import BotGAT
import json
import torch
import torch.nn as nn
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data


dataset = 'Twibot-22'

idx = json.load(open('idx.json'))

user_idx = []
for index in range(10):
    data = json.load(open('../../../datasets/{}/domain/user{}.json'.format(dataset, index)))
    user_idx.append([idx[item] for item in data])

data = get_data()

print('load done.')

hidden_dim = 128
dropout = 0.5
lr = 1e-4
weight_decay = 1e-5
max_epoch = 300
batch_size = 128


def forward_one_epoch():
    train_loader = NeighborLoader()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = nn.CrossEntropyLoss()
    for i in range(10):
        for j in range(10):
            train(i, j)


