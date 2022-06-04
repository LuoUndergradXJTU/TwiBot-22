from argparse import ArgumentParser
from dataset import SATARDataset
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import SATAR, FollowersClassifier
from tqdm import tqdm
from utils import null_metrics, calc_metrics, is_better

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--max_epoch', type=int, default=64)
parser.add_argument('--n_hidden', type=int, default=128)
parser.add_argument('--n_batch', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--max_tweet_count', type=int, default=128)
parser.add_argument('--max_tweet_length', type=int, default=64)
parser.add_argument('--max_words', type=int, default=1024)
args = parser.parse_args()
dataset_size = {
    'cresci-2015': 5301,
    'Twibot-20': 229580,
    'Twibot-22': 1000000
}

dataset_name = args.dataset
assert dataset_name in ['Twibot-22', 'Twibot-20', 'cresci-2015']
path = 'tmp/{}'.format(dataset_name)

best_val_metrics = null_metrics()
best_state_dict = None

max_epoch = args.max_epoch
n_hidden = args.n_hidden
n_batch = args.n_batch
lr = args.lr
weight_decay = args.weight_decay
dropout = args.dropout
max_words = args.max_words
max_tweet_count = args.max_tweet_count
max_tweet_length = args.max_tweet_length


begin_time = time.time()
data = {
    'tweets': np.load('{}/tweets.npy'.format(path), allow_pickle=True),
    'properties': np.load('{}/properties.npy'.format(path)),
    'neighbor_reps': np.zeros((dataset_size[dataset_name], n_hidden * 2)),
    'bot_labels': np.load('{}/bot_labels.npy'.format(path)),
    'follower_labels': np.load('{}/follower_labels.npy'.format(path))
}

word_vec = np.load('{}/vec.npy'.format(path))
word_vec = torch.tensor(word_vec)
words_size = len(word_vec)
blank_vec = torch.zeros((1, word_vec.shape[-1]))
word_vec = torch.cat((word_vec, blank_vec), dim=0)
num_embeddings = word_vec.shape[0]
embedding_dim = word_vec.shape[-1]
embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
embedding_layer.weight.data = word_vec
embedding_layer.weight.requires_grad = False
embedding_layer.to(device)
print('loading done in {}s'.format(time.time() - begin_time))


def forward_one_batch(batch):
    return classifier(model(batch))


def forward_one_epoch(epoch):
    model.train()
    classifier.train()
    pbar = tqdm(train_loader, ncols=0)
    pbar.set_description('train {} epoch'.format(epoch))
    all_label = []
    all_pred = []
    ave_loss = 0
    cnt = 0
    for batch in pbar:
        optimizer.zero_grad()
        batch_size = batch['follower_labels'].shape[0]
        out = forward_one_batch({
            'words': embedding_layer(batch['words'].to(device)),
            'tweets': embedding_layer(batch['tweets'].to(device)),
            'neighbor_reps': batch['neighbor_reps'].to(device),
            'properties': batch['properties'].to(device)
        })
        labels = batch['follower_labels'].to(device)
        loss = loss_fn(out, labels)
        ave_loss += loss.item() * batch_size
        cnt += batch_size
        loss.backward()
        optimizer.step()
        all_label += labels.data
        all_pred += out
        pbar.set_postfix(loss='{:.5f}'.format(loss.cpu().detach().numpy()))
    ave_loss /= cnt
    all_label = torch.stack(all_label)
    all_pred = torch.stack(all_pred)
    metrics, plog = calc_metrics(all_label, all_pred)
    plog = 'Epoch-{} train loss: {:.6}'.format(epoch, ave_loss) + plog
    print(plog)
    val_metrics = validation(epoch, 'validation', val_loader)
    global best_val_metrics
    global best_state_dict
    if is_better(val_metrics, best_val_metrics):
        best_val_metrics = val_metrics
        best_state_dict = model.state_dict()
        torch.save(best_state_dict, 'tmp/{}/pretrain_weight.pt'.format(dataset_name))


@torch.no_grad()
def validation(epoch, name, loader):
    model.eval()
    classifier.eval()
    all_label = []
    all_pred = []
    ave_loss = 0
    cnt = 0
    for batch in loader:
        batch_size = batch['follower_labels'].shape[0]
        out = forward_one_batch({
            'words': embedding_layer(batch['words'].to(device)),
            'tweets': embedding_layer(batch['tweets'].to(device)),
            'neighbor_reps': batch['neighbor_reps'].to(device),
            'properties': batch['properties'].to(device)
        })
        labels = batch['follower_labels'].to(device)
        loss = loss_fn(out, labels)
        ave_loss += loss.item() * batch_size
        cnt += batch_size
        all_label += labels.data
        all_pred += out
    ave_loss /= cnt
    all_label = torch.stack(all_label)
    all_pred = torch.stack(all_pred)
    metrics, plog = calc_metrics(all_label, all_pred)
    plog = 'Epoch-{} {} loss: {:.6}'.format(epoch, name, ave_loss) + plog
    print(plog)
    return metrics


if __name__ == '__main__':
    train_set = SATARDataset(dataset_name,
                             split=['train'] if dataset_name != 'Twibot-20' else ['train', 'support'],
                             data=data,
                             padding_value=num_embeddings - 1,
                             max_words=max_words,
                             max_tweet_count=max_tweet_count,
                             max_tweet_length=max_tweet_length
                             )
    val_set = SATARDataset(dataset_name,
                           split=['val'],
                           data=data,
                           padding_value=num_embeddings - 1,
                           max_words=max_words,
                           max_tweet_count=max_tweet_count,
                           max_tweet_length=max_tweet_length
                           )
    train_loader = DataLoader(train_set, batch_size=n_batch, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=n_batch, shuffle=False)

    model = SATAR(hidden_dim=n_hidden, embedding_dim=embedding_dim, dropout=dropout).to(device)
    classifier = FollowersClassifier(in_dim=n_hidden, out_dim=2).to(device)
    optimizer = torch.optim.Adam(set(model.parameters()) |
                                 set(classifier.parameters()),
                                 lr=lr,
                                 weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    for i in range(max_epoch):
        forward_one_epoch(i)
    print('the best val acc is {}'.format(best_val_metrics['acc']))
    torch.save(best_state_dict, 'tmp/{}/pretrain_weight.pt'.format(dataset_name))
