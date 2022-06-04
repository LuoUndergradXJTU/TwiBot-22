from torch_geometric.nn import GCNConv

import torch
import torch.nn as nn

ALPHA = 2.

def uniform_init(params_list):
    for param in params_list:
        nn.init.uniform_(param)

class Hist(torch.autograd.Function):
    @staticmethod
    def forward(ctx, c, bin_centers, hist_l, hist_r, n_bins):
        ctx.save_for_backward(c)
        ctx.constant = bin_centers
        hists = []
        for i in range(c.size(1)):
            hists.append(torch.histc(c[:, i], n_bins, hist_l, hist_r))

        ret = torch.stack(hists, dim=0)
        # ret.requires_grad = True
        return ret

    @staticmethod
    def backward(ctx, grad_output):
        c, = ctx.saved_tensors
        bin_centers = ctx.constant
        # print(c.device)
        d = bin_centers.unsqueeze(1).unsqueeze(
            1).repeat(1, c.size(0), c.size(1)) - c
        sum_i_d = (- ALPHA * d.abs()).exp().sum(dim=0)
        ret = torch.einsum("ilj,ji->lj", (- ALPHA * d.abs()
                                          ).exp() * torch.sign(d), grad_output)

        return ret / sum_i_d, None, None, None, None


class GraphHistEncoder(nn.Module):
    def __init__(self, input_dim, n_gcns, hist_l, hist_r, n_bins, hidden_dim):
        super().__init__()
        self.gcns = nn.ModuleList(GCNConv(input_dim, hidden_dim)
                                  for _ in range(n_gcns))

        bin_edges = torch.linspace(hist_l, hist_r, n_bins + 1)
        self.bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
        self.register_buffer("model_bin_centers", self.bin_centers)
        self.hist_l = hist_l
        self.hist_r = hist_r
        self.n_bins = n_bins
        self.bin_sep = self.bin_centers[1] - self.bin_centers[0]
        self.mlp1 = nn.Linear(n_gcns * hidden_dim, n_gcns * hidden_dim)
        self.mlp2 = nn.Linear(n_gcns * hidden_dim, n_gcns * hidden_dim)
        self.activation = nn.Tanh()
        # uniform_init(self.parameters())

    def forward(self, x, edge_index):
        h = self.activation(
            torch.cat([gcn(x, edge_index) for gcn in self.gcns], dim=1))
        c = self.activation(self.mlp2(self.activation(self.mlp1(h))))
        # histgram = Hist.apply(c, self.bin_centers, )
        return c


class GraphHistDecoder(nn.Module):
    def __init__(self, n_bins, in_channels, input_dim):
        super().__init__()
        self.n_bins = n_bins
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64, 3, padding="same"),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 4, padding="same"),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 5, padding="same"),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 6, padding="same"),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU(),
        )

        self.cnn_res = nn.Sequential(
            nn.Conv1d(in_channels, 96, input_dim)
        )

        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, input_dim)
            # print(dummy_input.shape)
            dummy_ret = self.cnn(dummy_input)

        feature_in_channels = dummy_ret.size(0) * dummy_ret.size(1)

        self.mlp = nn.Sequential(
            nn.Linear(feature_in_channels + 96, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        
        # uniform_init(self.parameters())

    def forward(self, x):
        x1 = self.cnn(x)
        x2 = self.cnn_res(x)
        x1 = nn.Flatten()(x1)
        x2 = nn.Flatten()(x2)
        x = torch.cat([x1, x2], dim=1)
        return self.mlp(x)
