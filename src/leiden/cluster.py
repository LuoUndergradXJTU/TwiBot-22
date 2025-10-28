

import sys
sys.path.append("..")
from utils.dataset import hetero_graph_vectorize, homo_graph_vectorize, homo_graph_vectorize_only_user, df_to_mask
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
import igraph as ig
import leidenalg as la
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import torch
import random
import numpy as np
import datetime
from torch_geometric import transforms as T


def leiden_cluster():
    
    # Does not work because the graph is a directed graph, not undirected

    user_text_feats, edge_index, labels, uid_to_user_index, train_uid_with_label, valid_uid_with_label, test_uid_with_label = homo_graph_vectorize_only_user(include_node_feature=True, dataset="cresci-2015", server_id="209")

    train_mask = df_to_mask(train_uid_with_label, uid_to_user_index, "train")
    valid_mask = df_to_mask(valid_uid_with_label, uid_to_user_index, "val")
    test_mask = df_to_mask(test_uid_with_label, uid_to_user_index, "test")

    graph = Data(x=user_text_feats, edge_index=edge_index, y=labels)

    # Convert labels (torch tensor) to a list
    # labels = labels.cpu().detach().numpy()

    # find a random sample of 500 ID from 5301 train_uid_with_label
    random.seed(50)
    NUM_SAMPLED = 1500
    full_length = len(user_text_feats) #3708
    num_list = random.sample(range(0, full_length), NUM_SAMPLED) 

    # selected_uid = random.sample(train_uid_with_label['id'], 500)
    # selected_labels = np.asarray(labels)[num_list]
    
    # labels = [1 if label=='bot' else 0 for label in selected_labels]
    
    # Create a torch bool
    subset = torch.zeros((full_length), dtype=torch.bool)
    subset[num_list] = 1
    subgraph = graph.subgraph(subset)

    # Removing isolated nodes that have no edges
    transform = T.Compose([T.remove_isolated_nodes.RemoveIsolatedNodes()])
    subgraph = transform(subgraph)
    selected_labels = subgraph.y.cpu().detach().numpy()

    G_intermediate = to_networkx(subgraph)
    g = ig.Graph.from_networkx(G_intermediate)
    # Add label as a vertex attribute
    # https://stackoverflow.com/questions/19290209/adding-vertex-labels-to-plot-in-igraph
    # g.vs['label']=labels
    # color_dict = {0: "#0928ae33", 1: "#00280033"} # blue and green
    # g.vs['color']=[color_dict[gender] for gender in labels]

    # train, valid, test = merge_and_split("cresci-2015", server_id="209")
    # self.train_loader, self.valid_loader, self.test_loader = generate_homo_graph_loader_only_user(num_layers, num_neighbors, batch_size, True, dataset, server_id)

    # compute the best partition
    partition = la.find_partition(g, la.ModularityVertexPartition)

    # draw the graph
    # pos = nx.spring_layout(G)
    # color the nodes according to their partition
    # cmap = cm.get_cmap('viridis', max(partition.values()) + 1)

    # nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
    #                        cmap=cmap, node_color=list(partition.values()))
    # nx.draw_networkx_edges(G, pos, alpha=0.5)
    # plt.show()
    # shape_dict = {0: "circle", 1: "rectangle"} # blue and green
    # shape_list = [shape_dict[label] for label in labels]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    layout = partition.graph.layout("kamada_kawai") # kamada_kawai # layout_fruchterman_reingold
    
    #https://python.igraph.org/en/stable/tutorial.html#tutorial-layouts-plotting
    visual_style = {"vertex_size": 20,
                    "vertex_shape": 'circle',
                    "vertex_label": selected_labels,
                    "vertex_label_color": '#919096FF',
                    "layout": layout,
                    "edge_width": 1,
                    "edge_arrow_size": 0.7,
                    "edge_arrow_width": 0.7,
                    "edge_color": '#00000020',
                    "mark_groups": False
                    }


    ig.plot(partition, 
            target=ax,
            **visual_style
            )

    # print datetime now
    

    now = datetime.datetime.now()

    plt.savefig(f'output/leiden_{now}.png') 
    #7: kamada_kawai
    #8:auto
    #9:fruchterman_reingold
    #10: lgl
    #11:drl
    #12:drl+sampled
    #12:kamada_kawai+sampled

    

if __name__ == "__main__":
    leiden_cluster()