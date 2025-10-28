from tqdm import tqdm
from dataset import get_transfer_data
from model import BotGAT
import json
import torch
import torch.nn as nn
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import accuracy_score, f1_score
import random


dataset = 'Twibot-22'

idx = json.load(open('idx.json'))

user_idx = []
for index in range(10):
    data = json.load(open('../../../datasets/{}/domain/user{}.json'.format(dataset, index)))
    user_id = [idx[item] for item in data]
    random.shuffle(user_id)
    user_idx.append(user_id)

data = get_transfer_data()

print('load done.')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hidden_dim = 128
dropout = 0.3
lr = 1e-3
weight_decay = 1e-5
max_epoch = 1000
no_up = 50
batch_size = 1024


def forward_one_epoch(model, optimizer, loss_fn, train_loader, test_loader):
    model.train()
    labels = []
    preds = []
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
                    batch.edge_index)
        label = batch.y[:n_batch]
        out = out[:n_batch]
        labels += label.to('cpu').data
        preds += out.argmax(-1).to('cpu').data
        loss = loss_fn(out, label)
        ave_loss += loss.item() * n_batch
        cnt += n_batch
        loss.backward()
        optimizer.step()
    ave_loss /= cnt
    test_loss, test_acc, test_f1 = validation(model, loss_fn, test_loader)
    return ave_loss, test_loss, accuracy_score(labels, preds), test_acc, test_f1


@torch.no_grad()
def validation(model, loss_fn, loader):
    model.eval()
    labels = []
    preds = []
    ave_loss = 0.0
    cnt = 0.0
    for batch in loader:
        batch = batch.to(device)
        n_batch = batch.batch_size
        out = model(batch.des_embedding,
                    batch.tweet_embedding,
                    batch.num_property_embedding,
                    batch.cat_property_embedding,
                    batch.edge_index)
        label = batch.y[:n_batch]
        out = out[:n_batch]
        labels += label.to('cpu').data
        preds += out.argmax(-1).to('cpu').data
        loss = loss_fn(out, label)
        ave_loss += loss.item() * n_batch
        cnt += n_batch
    ave_loss /= cnt
    return ave_loss, accuracy_score(labels, preds), f1_score(labels, preds)


def train(train_id, test_id):
    mx = 0
    mx_f1 = 0
    train_idx = torch.tensor(user_idx[train_id], dtype=torch.long)
    test_idx = torch.tensor(user_idx[test_id], dtype=torch.long)
    train_loader = NeighborLoader(data,
                                  num_neighbors=[256] * 4,
                                  batch_size=batch_size,
                                  input_nodes=train_idx)
    test_loader = NeighborLoader(data,
                                 num_neighbors=[256] * 4,
                                 batch_size=batch_size,
                                 input_nodes=test_idx)
    model = BotGAT(hidden_dim=hidden_dim, dropout=dropout).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    pbar = tqdm(range(max_epoch), ncols=0)
    pbar.set_description('{} {}'.format(train_id, test_id))
    cnt = 0
    for _ in pbar:
        train_loss, test_loss, train_acc, test_acc, test_f1 = forward_one_epoch(model,
                                                                                optimizer,
                                                                                loss_fn,
                                                                                train_loader,
                                                                                test_loader)
        pbar.set_postfix_str('test acc {:4f} '
                             'train acc {:4f} '
                             'test loss {:4f} '
                             'train loss {:4f} '
                             'no up cnt {}'.format(test_acc, train_acc, test_loss, train_loss, cnt))
        if test_acc >= mx:
            mx = test_acc
            mx_f1 = test_f1
            cnt = 0
        else:
            cnt += 1
        if cnt == no_up:
            return mx, mx_f1
    return mx, mx_f1


if __name__ == '__main__':
    fb = open('transfer_results.txt', 'w')
    for i in range(10):
        for j in range(10):
            acc, f1 = train(i, j)
            fb.write('{} train, {} test, acc: {}, f1: {}\n'.format(i, j, acc, f1))
    fb.close()


