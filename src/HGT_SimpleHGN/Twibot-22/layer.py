import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
import torch.nn.functional as F

class SimpleHGN(MessagePassing):
    def __init__(self, in_channels, out_channels, num_edge_type, rel_dim, beta=None, final_layer=False):
        super(SimpleHGN, self).__init__(aggr = "add", node_dim=0)
        self.W = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.W_r = torch.nn.Linear(rel_dim, out_channels, bias=False)
        self.a = torch.nn.Linear(3*out_channels, 1, bias=False)
        self.W_res = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.rel_emb = torch.nn.Embedding(num_edge_type, rel_dim)
        self.beta = beta
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.ELU = torch.nn.ELU()
        self.final = final_layer
        
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                    
    def forward(self, x, edge_index, edge_type, pre_alpha=None):
        
        node_emb = self.propagate(x=x, edge_index=edge_index, edge_type=edge_type, pre_alpha=pre_alpha)
        output = node_emb + self.W_res(x)
        output = self.ELU(output)
        if self.final:
            output = F.normalize(output, dim=1)
            
        return output, self.alpha.detach()
      
    def message(self, x_i, x_j, edge_type, pre_alpha, index, ptr, size_i):
        out = self.W(x_j)
        rel_emb = self.rel_emb(edge_type)
        alpha = self.leaky_relu(self.a(torch.cat((self.W(x_i), self.W(x_j), self.W_r(rel_emb)), dim=1)))
        alpha = softmax(alpha, index, ptr, size_i)
        if pre_alpha is not None and self.beta is not None:
            self.alpha = alpha*(1-self.beta) + pre_alpha*(self.beta)
        else:
            self.alpha = alpha
        out = out * alpha.view(-1,1)
        return out

    def update(self, aggr_out):
        return aggr_out
