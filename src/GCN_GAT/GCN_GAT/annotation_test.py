import pandas
import json
import torch
import os.path as osp
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from model import BotGAT, BotRGCN
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hidden_dim = 128
batch_size = 1024
lr = 1e-3
weight_decay = 1e-5
dropout = 0.3
no_up_limit = 4


def metrics(truth, preds):
    return accuracy_score(truth, preds), \
           f1_score(truth, preds), \
           precision_score(truth, preds), \
           recall_score(truth, preds)


def train_one_epoch():
    model.train()
    pbar = tqdm(train_loader, ncols=0)
    for batch in pbar:
        optimizer.zero_grad()
        batch = batch.to(device)
        out = model(batch.des_embedding,
                    batch.tweet_embedding,
                    batch.num_property_embedding,
                    batch.cat_property_embedding,
                    batch.edge_index,
                    batch.edge_type)
        out = out[:batch.batch_size]
        label = batch.y[:batch.batch_size]
        loss = loss_fn(out, label)
        loss.backward()
        optimizer.step()
        pbar.set_postfix_str('{:.6f}'.format(loss))


@torch.no_grad()
def validation(loader):
    model.eval()
    all_truth = []
    all_preds = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.des_embedding,
                    batch.tweet_embedding,
                    batch.num_property_embedding,
                    batch.cat_property_embedding,
                    batch.edge_index,
                    batch.edge_type)
        out = out[:batch.batch_size]
        label = batch.y[:batch.batch_size]
        all_truth.append(label.to('cpu'))
        all_preds.append(out.argmax(dim=-1).to('cpu'))
    all_truth = torch.cat(all_truth).numpy()
    all_preds = torch.cat(all_preds).numpy()
    return metrics(all_truth, all_preds)


if __name__ == '__main__':
    split = pandas.read_csv('../../../datasets/Twibot-22/annotation_test/split_new.csv')
    labels = pandas.read_csv('../../../datasets/Twibot-22/annotation_test/label_new.csv')
    idx = json.load(open('idx.json'))
    tmp = [0 for _ in range(len(labels))]
    for item in tqdm(labels.itertuples(), ncols=0, total=len(labels)):
        tmp[idx[item[1]]] = int(item[2] == 'bot')
    labels = torch.tensor(tmp, dtype=torch.long)
    train_mask = []
    val_mask = []
    test_mask_1 = []
    test_mask_2 = []
    test_mask_3 = []
    for item in tqdm(split.itertuples(), ncols=0, total=len(split)):
        try:
            if item[2] == 'train':
                train_mask.append(idx[item[1]])
            elif item[2] == 'val':
                val_mask.append(idx[item[1]])
            elif item[2] == 'test1':
                test_mask_1.append(idx[item[1]])
            elif item[2] == 'test2':
                test_mask_2.append(idx[item[1]])
            elif item[2] == 'test3':
                test_mask_3.append(idx[item[1]])
            else:
                print(item[2])
                exit(0)
        except KeyError:
            continue
    print(len(train_mask), len(val_mask))
    print(len(test_mask_1), len(test_mask_2), len(test_mask_3))
    train_mask = torch.tensor(train_mask, dtype=torch.long)
    val_mask = torch.tensor(val_mask, dtype=torch.long)
    test_mask_1 = torch.tensor(test_mask_1, dtype=torch.long)
    test_mask_2 = torch.tensor(test_mask_2, dtype=torch.long)
    test_mask_3 = torch.tensor(test_mask_3, dtype=torch.long)
    path = '../../BotRGCN/twibot_22/processed_data'
    des_embedding = torch.load(osp.join(path, 'des_tensor.pt'))
    tweet_embedding = torch.load(osp.join(path, 'tweets_tensor.pt'))
    num_property_embedding = torch.load(osp.join(path, 'num_properties_tensor.pt'))
    cat_property_embedding = torch.load(osp.join(path, 'cat_properties_tensor.pt'))
    edge_index = torch.load(osp.join(path, 'edge_index.pt'))
    edge_type = torch.load(osp.join(path, 'edge_type.pt'))

    data = Data(edge_index=edge_index,
                edge_type=edge_type,
                y=labels,
                des_embedding=des_embedding,
                tweet_embedding=tweet_embedding,
                num_property_embedding=num_property_embedding,
                cat_property_embedding=cat_property_embedding,
                num_nodes=labels.shape[0])

    train_loader = NeighborLoader(data,
                                  num_neighbors=[256] * 4,
                                  batch_size=batch_size,
                                  input_nodes=train_mask,
                                  shuffle=True)
    val_loader = NeighborLoader(data,
                                num_neighbors=[256] * 4,
                                batch_size=batch_size,
                                input_nodes=val_mask)
    test_loader_1 = NeighborLoader(data,
                                   num_neighbors=[256] * 4,
                                   batch_size=batch_size,
                                   input_nodes=test_mask_1)
    test_loader_2 = NeighborLoader(data,
                                   num_neighbors=[256] * 4,
                                   batch_size=batch_size,
                                   input_nodes=test_mask_2)
    test_loader_3 = NeighborLoader(data,
                                   num_neighbors=[256] * 4,
                                   batch_size=batch_size,
                                   input_nodes=test_mask_3)

    while True:
        model = BotRGCN(hidden_dim=hidden_dim,
                        dropout=dropout,
                        num_prop_size=data.num_property_embedding.shape[-1],
                        cat_prop_size=data.cat_property_embedding.shape[-1]).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        best_state = model.state_dict()
        best_acc = 0
        no_up = 0
        while True:
            train_one_epoch()
            val_metrics = validation(val_loader)
            if val_metrics[0] > best_acc:
                best_acc = val_metrics[0]
                best_state = model.state_dict()
                no_up = 0
            else:
                no_up += 1
            print('acc {:.6f} f1 {:.6f} pre {:.6f} rec {:.6f} no up {}'.
                  format(val_metrics[0], val_metrics[1], val_metrics[2], val_metrics[3], no_up))
            if no_up == no_up_limit:
                break
        model.load_state_dict(best_state)
        f = open('annotation_metrics_RGCN.txt', 'a')
        test_metrics = validation(test_loader_1)
        f.write('test1 : acc {:.6f} f1 {:.6f} pre {:.6f} rec {:.6f}\n'.
                format(test_metrics[0], test_metrics[1], test_metrics[2], test_metrics[3]))
        test_metrics = validation(test_loader_2)
        f.write('test2 : acc {:.6f} f1 {:.6f} pre {:.6f} rec {:.6f}\n'.
                format(test_metrics[0], test_metrics[1], test_metrics[2], test_metrics[3]))
        test_metrics = validation(test_loader_3)
        f.write('test3 : acc {:.6f} f1 {:.6f} pre {:.6f} rec {:.6f}\n'.
                format(test_metrics[0], test_metrics[1], test_metrics[2], test_metrics[3]))
        f.close()
