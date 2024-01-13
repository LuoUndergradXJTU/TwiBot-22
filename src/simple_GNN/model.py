import torch_geometric
import torch
from torch_geometric.nn import GCNConv, FastRGCNConv

class GNN_classifier_1_layer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, activate="relu"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.activation = torch.nn.ReLU() if activate == "relu" else torch.nn.Sigmoid()
        
        self.conv1 = GCNConv(in_channels=input_dim, out_channels=output_dim)
        
    def forward(self, x, edge_index):
        x = self.conv1(x ,edge_index)
        return x
    
class GNN_classifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, activate="relu"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.activation = torch.nn.ReLU() if activate == "relu" else torch.nn.Sigmoid()
        
        self.conv1 = GCNConv(in_channels=input_dim, out_channels=self.hidden_dim)
        self.conv2 = GCNConv(in_channels=hidden_dim, out_channels=output_dim)
        
    def forward(self, x, edge_index):
        x = self.conv1(x ,edge_index)
        x = self.activation(x)
        x = self.conv2(x, edge_index)
        return x

class GNN_classifier_3_layer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, activate="relu"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.activation = torch.nn.ReLU() if activate == "relu" else torch.nn.Sigmoid()
        
        self.conv1 = GCNConv(in_channels=input_dim, out_channels=self.hidden_dim)
        self.conv2 = GCNConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim)
        self.conv3 = GCNConv(in_channels=hidden_dim, out_channels=output_dim)
        
    def forward(self, x, edge_index):
        x = self.conv1(x ,edge_index)
        x = self.activation(x)
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        x = self.conv3(x, edge_index)
        return x
class GNN_classifier_4_layer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, activate="relu"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.activation = torch.nn.ReLU() if activate == "relu" else torch.nn.Sigmoid()
        
        self.conv1 = GCNConv(in_channels=input_dim, out_channels=self.hidden_dim)
        self.conv2 = GCNConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim)
        self.conv3 = GCNConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim)
        self.conv4 = GCNConv(in_channels=hidden_dim, out_channels=output_dim)
        
    def forward(self, x, edge_index):
        x = self.conv1(x ,edge_index)
        x = self.activation(x)
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        x = self.conv3(x, edge_index)
        x = self.activation(x)
        x = self.conv4(x, edge_index)
        return x
class GNN_classifier_5_layer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, activate="relu"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.activation = torch.nn.ReLU() if activate == "relu" else torch.nn.Sigmoid()
        
        self.conv1 = GCNConv(in_channels=input_dim, out_channels=self.hidden_dim)
        self.conv2 = GCNConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim)
        self.conv3 = GCNConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim)
        self.conv4 = GCNConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim)
        self.conv5 = GCNConv(in_channels=hidden_dim, out_channels=output_dim)
        
    def forward(self, x, edge_index):
        x = self.conv1(x ,edge_index)
        x = self.activation(x)
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        x = self.conv3(x, edge_index)
        x = self.activation(x)
        x = self.conv4(x, edge_index)
        x = self.activation(x)
        x = self.conv5(x, edge_index)
        return x
    
class GNN_classifier_6_layer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, activate="relu"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.activation = torch.nn.ReLU() if activate == "relu" else torch.nn.Sigmoid()
        
        self.conv1 = GCNConv(in_channels=input_dim, out_channels=self.hidden_dim)
        self.conv2 = GCNConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim)
        self.conv3 = GCNConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim)
        self.conv4 = GCNConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim)
        self.conv5 = GCNConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim)
        self.conv6 = GCNConv(in_channels=hidden_dim, out_channels=output_dim)
        
    def forward(self, x, edge_index):
        x = self.conv1(x ,edge_index)
        x = self.activation(x)
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        x = self.conv3(x, edge_index)
        x = self.activation(x)
        x = self.conv4(x, edge_index)
        x = self.activation(x)
        x = self.conv5(x, edge_index)
        x = self.activation(x)
        x = self.conv6(x, edge_index)
        return x

class GNN_classifier_7_layer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, activate="relu"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.activation = torch.nn.ReLU() if activate == "relu" else torch.nn.Sigmoid()
        
        self.conv1 = GCNConv(in_channels=input_dim, out_channels=self.hidden_dim)
        self.conv2 = GCNConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim)
        self.conv3 = GCNConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim)
        self.conv4 = GCNConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim)
        self.conv5 = GCNConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim)
        self.conv6 = GCNConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim)
        self.conv7 = GCNConv(in_channels=hidden_dim, out_channels=output_dim)
        
    def forward(self, x, edge_index):
        x = self.conv1(x ,edge_index)
        x = self.activation(x)
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        x = self.conv3(x, edge_index)
        x = self.activation(x)
        x = self.conv4(x, edge_index)
        x = self.activation(x)
        x = self.conv5(x, edge_index)
        x = self.activation(x)
        x = self.conv6(x, edge_index)
        x = self.activation(x)
        x = self.conv7(x, edge_index)
        return x

class GNN_classifier_8_layer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, activate="relu"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.activation = torch.nn.ReLU() if activate == "relu" else torch.nn.Sigmoid()
        
        self.conv1 = GCNConv(in_channels=input_dim, out_channels=self.hidden_dim)
        self.conv2 = GCNConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim)
        self.conv3 = GCNConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim)
        self.conv4 = GCNConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim)
        self.conv5 = GCNConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim)
        self.conv6 = GCNConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim)
        self.conv7 = GCNConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim)
        self.conv8 = GCNConv(in_channels=hidden_dim, out_channels=output_dim)
        
    def forward(self, x, edge_index):
        x = self.conv1(x ,edge_index)
        x = self.activation(x)
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        x = self.conv3(x, edge_index)
        x = self.activation(x)
        x = self.conv4(x, edge_index)
        x = self.activation(x)
        x = self.conv5(x, edge_index)
        x = self.activation(x)
        x = self.conv6(x, edge_index)
        x = self.activation(x)
        x = self.conv7(x, edge_index)
        x = self.activation(x)
        x = self.conv8(x, edge_index)
        return x
    
class HGNN_classifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_relations, num_bases, hidden_dim=256, activation="relu"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.activation = torch.nn.ReLU() if activation == "relu" else torch.nn.Sigmoid()
        
        self.conv1 = torch_geometric.nn.FastRGCNConv()
        self.conv2 = torch_geometric.nn.FastRGCNConv(hidden_dim, output_dim)
        
    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = self.activation(x)
        x = self.conv2(x, edge_index, edge_type)
        
        return x
    
class GNN_classifier_3_layer_viz(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, activate="relu"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.activation = torch.nn.ReLU() if activate == "relu" else torch.nn.Sigmoid()
        
        self.conv1 = GCNConv(in_channels=input_dim, out_channels=self.hidden_dim)
        self.conv2 = GCNConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim)
        self.conv3 = GCNConv(in_channels=hidden_dim, out_channels=output_dim)
        
    def forward(self, x, edge_index):
        x = self.conv1(x ,edge_index)
        x = self.activation(x)
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        embedding = x
        x = self.conv3(x, edge_index)
        return x, embedding