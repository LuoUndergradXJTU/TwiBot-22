import torch
import torch.nn as nn

class twoLayerMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, activation="relu"):
        super().__init__()
        self.mlp1 = nn.Linear(input_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU() if activation == "relu" else nn.Sigmoid()
        
    def forward(self, x):
        return self.mlp2(self.activation(self.mlp1(x)))