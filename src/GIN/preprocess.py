import torch
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data, HeteroData
import sys
sys.path.append("..")
from utils.dataset import hetero_graph_vectorize, homo_graph_vectorize, homo_graph_vectorize_only_user, df_to_mask

def generate_homo_graph_loader_only_user(num_layers, num_neighbors, batch_size, include_node_feature=True, dataset="cresci-2015", server_id="209"):
    """
    return pyg Data
    """
    user_text_feats, edge_index, labels, uid_to_user_index, train_uid_with_label, valid_uid_with_label, test_uid_with_label = homo_graph_vectorize_only_user(include_node_feature, dataset, server_id)
    
    train_mask = df_to_mask(train_uid_with_label, uid_to_user_index, "train")
    valid_mask = df_to_mask(valid_uid_with_label, uid_to_user_index, "val")
    test_mask = df_to_mask(test_uid_with_label, uid_to_user_index, "test")
    
    graph = Data(x=user_text_feats, edge_index=edge_index, y=labels)
    
    train_loader = NeighborLoader(graph, num_neighbors=[num_neighbors] * num_layers, input_nodes=train_mask, batch_size=batch_size)
    valid_loader = NeighborLoader(graph, num_neighbors=[num_neighbors] * num_layers, input_nodes=valid_mask, batch_size=batch_size)
    test_loader = NeighborLoader(graph, num_neighbors=[num_neighbors] * num_layers, input_nodes=test_mask, batch_size=batch_size)
    
    return train_loader, valid_loader, test_loader

def generate_hetero_graph_loader(num_layers, num_neighbor_users, num_neighbor_tweets, batch_size, include_node_feature, dataset, server_id):
    
    graph, uid_to_user_index, tid_to_tweet_index, train_uid_with_label, valid_uid_with_label, test_uid_with_label = hetero_graph_vectorize(include_node_feature, dataset, server_id)
    
    train_mask = df_to_mask(train_uid_with_label, uid_to_user_index, "train")
    valid_mask = df_to_mask(valid_uid_with_label, uid_to_user_index, "val")
    test_mask = df_to_mask(test_uid_with_label, uid_to_user_index, "test")
    
    train_loader = NeighborLoader(graph, num_neighbors={"user": [num_neighbor_users] * num_layers, "tweet": [num_neighbor_tweets] * num_layers}, input_nodes=train_mask, batch_size=batch_size)
    valid_loader = NeighborLoader(graph, num_neighbors={"user": [num_neighbor_users] * num_layers, "tweet": [num_neighbor_tweets] * num_layers}, input_nodes=valid_mask, batch_size=batch_size)
    test_loader = NeighborLoader(graph, num_neighbors={"user": [num_neighbor_users] * num_layers, "tweet": [num_neighbor_tweets] * num_layers}, input_nodes=test_mask, batch_size=batch_size)
    
    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    loader, _, _ = generate_homo_graph_loader_only_user(num_layers=2, num_neighbors=20, batch_size=128)
    for i in loader:
        print(i)
        break
