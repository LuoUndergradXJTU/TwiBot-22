from argparse import ArgumentParser
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import SATAR
from tqdm import tqdm
import json


class MyDataset(Dataset):
    def __init__(self, data, padding_value,
                 max_tweet_count=128, max_tweet_length=64, max_words=1024):
        self.data = data
        self.padding_value = padding_value
        self.max_tweet_count = max_tweet_count
        self.max_tweet_length = max_tweet_length
        self.max_words = max_words

    def __getitem__(self, index):
        tweets = self.data['tweets'][index]
        tweets = tweets[:self.max_tweet_count]
        tweets_cache = []
        words = []
        for tweet in tweets:
            words += tweet
            cache = tweet[:self.max_tweet_length]
            for _ in range(len(tweet), self.max_tweet_length):
                cache.append(self.padding_value)
            tweets_cache.append(cache)
        for _ in range(len(tweets), self.max_tweet_count):
            tweets_cache.append([self.padding_value] * self.max_tweet_length)
        tweets = torch.tensor(tweets_cache, dtype=torch.long)
        words_cache = words[:self.max_words]
        for _ in range(len(words), self.max_words):
            words_cache.append(self.padding_value)
        words = torch.tensor(words_cache, dtype=torch.long)
        properties = torch.tensor(self.data['properties'][index], dtype=torch.float)
        neighbor_reps = torch.tensor(self.data['neighbor_reps'][index], dtype=torch.float)
        bot_labels = torch.tensor(self.data['bot_labels'][index], dtype=torch.long)
        follower_labels = torch.tensor(self.data['follower_labels'][index], dtype=torch.long)
        return {
            'words': words,
            'tweets': tweets,
            'properties': properties,
            'neighbor_reps': neighbor_reps,
            'bot_labels': bot_labels,
            'follower_labels': follower_labels,
        }

    def __len__(self):
        return self.data['bot_labels'].shape[0]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--n_hidden', type=int, default=128)
args = parser.parse_args()
dataset_size = {
    'cresci-2015': 5301,
    'Twibot-20': 229580,
    'Twibot-22': 1000000
}

dataset_name = args.dataset
assert dataset_name in ['Twibot-22', 'Twibot-20', 'cresci-2015']
path = 'tmp/{}'.format(dataset_name)

begin_time = time.time()

n_hidden = args.n_hidden

idx = json.load(open('tmp/{}/idx.json'.format(dataset_name)))
data = {
    'tweets': np.load('{}/tweets.npy'.format(path), allow_pickle=True),
    'properties': np.load('{}/properties.npy'.format(path)),
    'neighbor_reps': np.zeros((dataset_size[dataset_name], n_hidden * 2)),
    'bot_labels': np.load('{}/bot_labels.npy'.format(path)),
    'follower_labels': np.load('{}/follower_labels.npy'.format(path)),
    'neighbors': np.load('{}/neighbors.npy'.format(path), allow_pickle=True)
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

if __name__ == '__main__':
    dataset = MyDataset(data, padding_value=num_embeddings - 1)
    print(len(dataset))
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model = SATAR(hidden_dim=n_hidden, embedding_dim=embedding_dim)
    pretrain_weight = torch.load('tmp/{}/pretrain_weight.pt'.format(dataset_name), map_location='cpu')
    model.load_state_dict(pretrain_weight)
    model = model.to(device)
    reps = []
    with torch.no_grad():
        for batch in tqdm(loader, ncols=0):
            out = model({
                'words': embedding_layer(batch['words'].to(device)),
                'tweets': embedding_layer(batch['tweets'].to(device)),
                'neighbor_reps': batch['neighbor_reps'].to(device),
                'properties': batch['properties'].to(device)
            })
            reps.append(out.to('cpu').detach())
    reps = torch.cat(reps, dim=0)
    reps = reps.numpy()
    print(reps.shape)
    np.save('tmp/{}/reps.npy'.format(dataset_name), reps)

