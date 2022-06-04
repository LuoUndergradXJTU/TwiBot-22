import torch.nn as nn
import torch
from torch_geometric.nn.conv.gat_conv import GATConv


class BotGAT(nn.Module):
    def __init__(self, hidden_dim, des_size=768, tweet_size=768, num_prop_size=5, cat_prop_size=3, dropout=0.3):
        super(BotGAT, self).__init__()
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, hidden_dim // 4),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, hidden_dim // 4),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, hidden_dim // 4),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, hidden_dim // 4),
            nn.LeakyReLU()
        )

        self.linear_relu_input = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dim, 2)

        self.gat1 = GATConv(hidden_dim, hidden_dim // 4, heads=4)
        self.gat2 = GATConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, des, tweet, num_prop, cat_prop, edge_index):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((d, t, n, c), dim=1)

        x = self.linear_relu_input(x)
        x = self.gat1(x, edge_index)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)
        return x