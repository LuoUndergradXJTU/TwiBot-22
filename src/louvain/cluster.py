

import sys
sys.path.append("..")
from utils.dataset import hetero_graph_vectorize, homo_graph_vectorize, homo_graph_vectorize_only_user, df_to_mask
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
from community import community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx

def leiden_cluster():
    # Does not work because the graph is a directed graph, not undirected

    user_text_feats, edge_index, labels, uid_to_user_index, train_uid_with_label, valid_uid_with_label, test_uid_with_label = homo_graph_vectorize_only_user(include_node_feature=True, dataset="cresci-2015", server_id="209")

    # train_mask = df_to_mask(train_uid_with_label, uid_to_user_index, "train")
    # valid_mask = df_to_mask(valid_uid_with_label, uid_to_user_index, "val")
    # test_mask = df_to_mask(test_uid_with_label, uid_to_user_index, "test")

    graph = Data(x=user_text_feats, edge_index=edge_index, y=labels)

    G = to_networkx(graph)

    # train, valid, test = merge_and_split("cresci-2015", server_id="209")
    # self.train_loader, self.valid_loader, self.test_loader = generate_homo_graph_loader_only_user(num_layers, num_neighbors, batch_size, True, dataset, server_id)

    # compute the best partition
    partition = community_louvain.best_partition(G)

    # draw the graph
    pos = nx.spring_layout(G)
    # color the nodes according to their partition
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)

    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
                           cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()

    plt.savefig(f'./output/louvain_1.png')

if __name__ == "__main__":
    louvain_cluster()