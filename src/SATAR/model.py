import torch
import torch.nn as nn
import torch.nn.functional as func


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.weight_vector = nn.Parameter(torch.randn(hidden_dim, 1))

    def forward(self, x):
        weight = torch.tanh(self.fc(x)).matmul(self.weight_vector)
        weight = func.softmax(weight, dim=0)
        result = x.mul(weight)
        result = result.sum(dim=-2)
        return result


class SemanticWord(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        # self.split_size = split_size
        self.hidden_dim = hidden_dim // 2
        self.lstm = nn.LSTM(input_size=in_dim,
                            hidden_size=self.hidden_dim,
                            bidirectional=True,
                            batch_first=True)

    def init_hidden(self, batch_size, device):
        return (
            torch.FloatTensor(2, batch_size, self.hidden_dim).fill_(0).to(device),
            torch.FloatTensor(2, batch_size, self.hidden_dim).fill_(0).to(device)
        )

    def forward(self, texts):
        # texts = embedding_layer(texts)
        batch_size, _, _ = texts.shape
        hidden = self.init_hidden(batch_size, device=texts.device)
        texts, _ = self.lstm(texts, hidden)
        return texts


class SemanticTweet(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim // 2
        self.lstm = nn.LSTM(input_size=in_dim,
                            hidden_size=self.hidden_dim,
                            bidirectional=True,
                            batch_first=True)

    def init_hidden(self, batch_size, device):
        return (
            torch.FloatTensor(2, batch_size, self.hidden_dim).fill_(0).to(device),
            torch.FloatTensor(2, batch_size, self.hidden_dim).fill_(0).to(device)
        )

    def forward(self, x):
        batch_size, _, _ = x.shape
        hidden = self.init_hidden(batch_size=batch_size, device=x.device)
        result, _ = self.lstm(x, hidden)
        return result


class SemanticVector(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, dropout):
        super().__init__()
        self.word_level_model = SemanticWord(in_dim=embedding_dim,
                                             hidden_dim=hidden_dim // 2)
        self.tweet_low_level_model = SemanticWord(in_dim=embedding_dim,
                                                  hidden_dim=hidden_dim)
        self.tweet_high_level_model = SemanticTweet(in_dim=hidden_dim,
                                                    hidden_dim=hidden_dim // 2)
        self.word_attn = Attention(hidden_dim=hidden_dim // 2)
        self.tweet_low_attn = Attention(hidden_dim=hidden_dim)
        self.tweet_high_attn = Attention(hidden_dim=hidden_dim // 2)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, user):
        words = user['words']
        words_rep = self.word_level_model(words)
        words_rep = self.word_attn(words_rep)
        words_rep = self.dropout(words_rep)
        tweets = user['tweets']
        batch_size, _, _, _ = tweets.shape
        tweets_high_rep = []
        for index in range(batch_size):
            tweets_low_rep = self.tweet_low_level_model(tweets[index])
            tweets_low_rep = self.tweet_low_attn(tweets_low_rep)
            tweets_high_rep.append(tweets_low_rep)
            # del tweets_low_rep
            # torch.cuda.empty_cache()
            # torch.cuda.synchronize()
        tweets_high_rep = torch.stack(tweets_high_rep)
        tweets_high_rep = self.tweet_high_level_model(tweets_high_rep)
        tweets_rep = self.tweet_high_attn(tweets_high_rep)
        tweets_rep = self.dropout(tweets_rep)
        return torch.cat([words_rep, tweets_rep], dim=1)


class PropertyVector(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout):
        super().__init__()
        self.fc = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, user):
        return self.dropout(self.act(self.fc(user['properties'])))


class NeighborVector(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout):
        super().__init__()
        self.fc = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, user):
        return self.dropout(self.act(self.fc(user['neighbor_reps'])))


class CoInfluence(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.Wsp = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.Wpn = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.Wns = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.Ws = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.Wp = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.Wn = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.Wh = nn.Parameter(torch.randn(3 * hidden_dim, hidden_dim))

    def forward(self, user):
        rs, rp, rn = user[0], user[1], user[2]
        fsp = torch.tanh(torch.einsum('ij, ij -> i', rs.matmul(self.Wsp), rp))
        fpn = torch.tanh(torch.einsum('ij, ij -> i', rp.matmul(self.Wpn), rn))
        fns = torch.tanh(torch.einsum('ij, ij -> i', rn.matmul(self.Wns), rs))
        hs = torch.tanh(rs.matmul(self.Ws) +
                        torch.einsum('i, ij -> ij', fsp, rp.matmul(self.Wp)) +
                        torch.einsum('i, ij -> ij', fns, rn.matmul(self.Wn)))
        hp = torch.tanh(rp.matmul(self.Wp) +
                        torch.einsum('i, ij -> ij', fsp, rs.matmul(self.Ws)) +
                        torch.einsum('i, ij -> ij', fpn, rn.matmul(self.Wn)))
        hn = torch.tanh(rn.matmul(self.Wn) +
                        torch.einsum('i, ij -> ij', fpn, rp.matmul(self.Wp)) +
                        torch.einsum('i, ij -> ij', fns, rs.matmul(self.Ws)))
        h = torch.cat([hs, hp, hn], dim=-1)
        return torch.tanh(h.matmul(self.Wh))


class SATAR(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, property_dim=15, dropout=0.5):
        super().__init__()
        self.semantic_encoder = SemanticVector(hidden_dim=hidden_dim,
                                               embedding_dim=embedding_dim,
                                               dropout=dropout)
        self.property_encoder = PropertyVector(in_dim=property_dim,
                                               hidden_dim=hidden_dim,
                                               dropout=dropout)
        self.neighbor_encoder = NeighborVector(in_dim=hidden_dim * 2,
                                               hidden_dim=hidden_dim,
                                               dropout=dropout)
        self.co_influence = CoInfluence(hidden_dim=hidden_dim)

    def forward(self, user):
        semantic_rep = self.semantic_encoder({'words': user['words'], 'tweets': user['tweets']})
        property_rep = self.property_encoder({'properties': user['properties']})
        neighbors_rep = self.neighbor_encoder({'neighbor_reps': user['neighbor_reps']})
        user = [semantic_rep, property_rep, neighbors_rep]
        rep = self.co_influence(user)
        return rep


class BotClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.cls = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.cls(self.dropout(x))


class FollowersClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.cls = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.cls(self.dropout(x))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SATAR(hidden_dim=128, embedding_dim=128).to(device)
    batch = {
        'words': torch.randn(32, 1024, 128).to(device),
        'tweets': torch.randn(32, 16, 128, 128).to(device),
        'properties': torch.randn(32, 15).to(device),
        'neighbor_reps': torch.randn(32, 128 * 2).to(device)
    }
    out = model(batch)
    print(out.shape)