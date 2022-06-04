import json
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader


def generate_graph_loader(dataset="Twibot-20", num_neighbors=40, num_layers=3, batch_size=256):
    # node = json.load(open(f"../../datasets/{dataset}/node.json", "r"))
    edge = pd.read_csv(f"../../datasets/{dataset}/edge.csv")
    label = pd.read_csv(f"../../datasets/{dataset}/label.csv")
    split = pd.read_csv(f"../../datasets/{dataset}/split.csv")

    edge = edge[(edge.relation == "follow") | (edge.relation == "friend")]

    # id_map = dict()
    # for i in range(len(node)):
    #     id_map[node[i]["id"]] = i

    # torch.save(id_map, f"./{dataset}/id_map.pt")
    id_map = torch.load(f"./{dataset}/id_map.pt")

    # src = list(map(lambda x: id_map[x], edge["source_id"]))
    # dst = list(map(lambda x: id_map[x], edge["target_id"]))
    
    # src = []
    # dst = []
    # for s, d in zip(edge["source_id"], edge["target_id"]):
    #     try:
    #         s = id_map[s]
    #         d = id_map[d]
    #     except KeyError:
    #         continue
    #     src.append(s)
    #     dst.append(d)

    # edge_index = torch.LongTensor([src, dst])

    # torch.save(edge_index, f"./{dataset}/edge_index.pt")
    edge_index = torch.load(f"./{dataset}/edge_index.pt")

    num_user = label.shape[0]
    y = (label["label"].values == "bot").astype(int)

    feature_file_name = "feature_matrix.csv" if dataset == "Twibot-20" else "feature_matrix_cresci_15.csv"
    X = pd.read_csv(feature_file_name).values[:]

    num_nodes = X.shape[0]
    labels = torch.zeros(num_nodes, dtype=torch.int32)
    labels[: num_user] = torch.LongTensor(y[:])

    for i in range(X.shape[0]):
        X[i][np.isnan(X[i])] = np.nanmean(X[i])

    X = torch.FloatTensor(X)

    graph = Data(x=X, edge_index=edge_index, y=labels)

    train_split = split["split"].values[0: num_user] == "train"
    valid_split = split["split"].values[0: num_user] == "val"
    test_split = split["split"].values[0: num_user] == "test"

    train_mask = torch.LongTensor(
        [id_map[i] for i in split["id"].values[0: num_user][train_split]])
    valid_mask = torch.LongTensor(
        [id_map[i] for i in split["id"].values[0: num_user][valid_split]])
    test_mask = torch.LongTensor(
        [id_map[i] for i in split["id"].values[0: num_user][test_split]])

    train_loader = NeighborLoader(graph, num_neighbors=[
                                  num_neighbors] * num_layers, input_nodes=train_mask, batch_size=batch_size, shuffle=True)
    valid_loader = NeighborLoader(graph, num_neighbors=[
                                  num_neighbors] * num_layers, input_nodes=valid_mask, batch_size=batch_size)
    test_loader = NeighborLoader(graph, num_neighbors=[
                                 num_neighbors] * num_layers, input_nodes=test_mask, batch_size=batch_size)

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    generate_graph_loader()
