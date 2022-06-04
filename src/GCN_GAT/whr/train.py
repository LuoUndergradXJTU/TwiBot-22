import os
from argparse import ArgumentParser
from utils import null_metrics, calc_metrics, is_better
import torch
from dataset import get_train_data
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
import torch.nn as nn
from model import BotGAT, BotGCN, BotRGCN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--mode', type=str, default='GCN')
parser.add_argument('--visible', type=bool, default=False)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--max_epoch', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--no_up', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--dropout', type=float, default=0.3)
args = parser.parse_args()

dataset_name = args.dataset
mode = args.mode
visible = args.visible

assert mode in ['GCN', 'GAT', 'RGCN']
assert dataset_name in ['cresci-2015', 'Twibot-20', 'Twibot-22']

data = get_train_data(dataset_name)

hidden_dim = args.hidden_dim
dropout = args.dropout
lr = args.lr
weight_decay = args.weight_decay
max_epoch = args.max_epoch
batch_size = args.batch_size
no_up = args.no_up


def forward_one_epoch(epoch, model, optimizer, loss_fn, train_loader, val_loader):
    model.train()
    all_label = []
    all_pred = []
    ave_loss = 0.0
    cnt = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        n_batch = batch.batch_size
        out = model(batch.des_embedding,
                    batch.tweet_embedding,
                    batch.num_property_embedding,
                    batch.cat_property_embedding,
                    batch.edge_index,
                    batch.edge_type)
        label = batch.y[:n_batch]
        out = out[:n_batch]
        all_label += label.data
        all_pred += out
        loss = loss_fn(out, label)
        ave_loss += loss.item() * n_batch
        cnt += n_batch
        loss.backward()
        optimizer.step()
    ave_loss /= cnt
    ave_loss /= cnt
    all_label = torch.stack(all_label)
    all_pred = torch.stack(all_pred)
    metrics, plog = calc_metrics(all_label, all_pred)
    plog = 'Epoch-{} train loss: {:.6}'.format(epoch, ave_loss) + plog
    if visible:
        print(plog)
    val_metrics = validation(epoch, 'validation', model, loss_fn, val_loader)
    return val_metrics


@torch.no_grad()
def validation(epoch, name, model, loss_fn, loader):
    model.eval()
    all_label = []
    all_pred = []
    ave_loss = 0.0
    cnt = 0.0
    for batch in loader:
        batch = batch.to(device)
        n_batch = batch.batch_size
        out = model(batch.des_embedding,
                    batch.tweet_embedding,
                    batch.num_property_embedding,
                    batch.cat_property_embedding,
                    batch.edge_index,
                    batch.edge_type)
        label = batch.y[:n_batch]
        out = out[:n_batch]
        all_label += label.data
        all_pred += out
        loss = loss_fn(out, label)
        ave_loss += loss.item() * n_batch
        cnt += n_batch
    ave_loss /= cnt
    all_label = torch.stack(all_label)
    all_pred = torch.stack(all_pred)
    metrics, plog = calc_metrics(all_label, all_pred)
    plog = 'Epoch-{} {} loss: {:.6}'.format(epoch, name, ave_loss) + plog
    if visible:
        print(plog)
    return metrics


def train():
    print(data)
    train_loader = NeighborLoader(data,
                                  num_neighbors=[256] * 4,
                                  batch_size=batch_size,
                                  input_nodes=data.train_idx,
                                  shuffle=True)
    val_loader = NeighborLoader(data,
                                num_neighbors=[256] * 4,
                                batch_size=batch_size,
                                input_nodes=data.val_idx)
    test_loader = NeighborLoader(data,
                                 num_neighbors=[256] * 4,
                                 batch_size=batch_size,
                                 input_nodes=data.test_idx)
    if mode == 'GAT':
        model = BotGAT(hidden_dim=hidden_dim,
                       dropout=dropout,
                       num_prop_size=data.num_property_embedding.shape[-1],
                       cat_prop_size=data.cat_property_embedding.shape[-1]).to(device)
    elif mode == 'GCN':
        model = BotGCN(hidden_dim=hidden_dim,
                       dropout=dropout,
                       num_prop_size=data.num_property_embedding.shape[-1],
                       cat_prop_size=data.cat_property_embedding.shape[-1]).to(device)
    elif mode == 'RGCN':
        model = BotRGCN(hidden_dim=hidden_dim,
                        dropout=dropout,
                        num_prop_size=data.num_property_embedding.shape[-1],
                        cat_prop_size=data.cat_property_embedding.shape[-1],
                        num_relations=data.edge_type.max().item() + 1).to(device)
    else:
        raise KeyError
    best_val_metrics = null_metrics()
    best_state_dict = None
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    pbar = tqdm(range(max_epoch), ncols=0)
    cnt = 0
    for epoch in pbar:
        val_metrics = forward_one_epoch(epoch, model, optimizer, loss_fn, train_loader, val_loader)
        if is_better(val_metrics, best_val_metrics):
            best_val_metrics = val_metrics
            best_state_dict = model.state_dict()
            cnt = 0
        else:
            cnt += 1
        pbar.set_postfix_str('val acc {} no up cnt {}'.format(val_metrics['acc'], cnt))
        if cnt == no_up:
            break
    model.load_state_dict(best_state_dict)
    test_metrics = validation(max_epoch, 'test', model, loss_fn, test_loader)
    torch.save(best_state_dict, 'checkpoints/{}_{}.pt'.format(dataset_name, test_metrics['acc']))
    for key, value in test_metrics.items():
        print(key, value)


if __name__ == '__main__':
    train()