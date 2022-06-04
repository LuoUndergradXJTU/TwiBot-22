'''Set of functions to construct a graph as a combination of
smaller subgraphs (of aparticular shape, defined in the shapes.py file)
'''
import graphwave
import math
import networkx as nx
import numpy as np
from shapes import *
from graphwave.utils.utils import *


def build_structure(width_basis, basis_type, list_shapes, start=0,
                    rdm_basis_plugins =False, add_random_edges=0,
                    plot=False, savefig=False):
    '''This function creates a basis (torus, string, or cycle)
    and attaches elements of the type in the list randomly along the basis.
    Possibility to add random edges afterwards.
    INPUT:
    --------------------------------------------------------------------------------------
    width_basis      :      width (in terms of number of nodes) of the basis
    basis_type       :      (torus, string, or cycle)
    shapes           :      list of shape list (1st arg: type of shape,
                            next args:args for building the shape,
                            except for the start)
    start            :      initial nb for the first node
    rdm_basis_plugins:      boolean. Should the shapes be randomly placed
                            along the basis (True) or regularly (False)?
    add_random_edges :      nb of edges to randomly add on the structure
    plot,savefig     :      plotting and saving parameters
    OUTPUT:
    --------------------------------------------------------------------------------------
    basis            :       a nx graph with the particular shape
    colors           :       labels for each role
    '''
    basis, role_id = eval(basis_type)(start, width_basis)
    n_basis, n_shapes = nx.number_of_nodes(basis), len(list_shapes)
    start += n_basis        # indicator of the id of the next node

    # Sample (with replacement) where to attach the new motives
    if rdm_basis_plugins is True:
        plugins = np.random.choice(n_basis, n_shapes, replace=False)
    else:
        spacing = math.floor(width_basis / n_shapes)
        plugins = [int(k * spacing) for k in range(n_shapes)]
    communities = [0] * n_basis
    seen_shapes = {'basis': [0, n_basis]}
    for p in plugins:
        role_id[p] += 1

    for shape_id, shape in enumerate(list_shapes):
        shape_type = shape[0]
        args = [start]
        if len(shape)>1:
            args += shape[1:]
        args += [0]
        graph_s, roles_graph_s = eval(shape_type)(*args)
        n_s = nx.number_of_nodes(graph_s)
        try:
            col_start = seen_shapes[shape_type][0]
        except:
            col_start = np.max(role_id) + 1
            seen_shapes[shape_type] = [col_start, n_s]
        # Attach the shape to the basis
        basis.add_nodes_from(graph_s.nodes())
        basis.add_edges_from(graph_s.edges())
        basis.add_edges_from([(start, plugins[shape_id])])
        role_id[plugins[shape_id]] += (-2 - 10 * seen_shapes[shape_type][0])
        communities += [shape_id] * n_s
        temp_labels = [r + col_start for r in roles_graph_s]
        temp_labels[0] += 100 * seen_shapes[shape_type][0]
        role_id += temp_labels
        start += n_s

    if add_random_edges > 0:
        # add random edges between nodes:
        for p in range(add_random_edges):
            src, dest = np.random.choice(nx.number_of_nodes(basis),
                                         2, replace=False)
            print src, dest
            basis.add_edges_from([(src, dest)])
    if plot is True: plot_networkx(basis, role_id)

    return basis, communities, plugins, role_id


def build_lego_structure(list_shapes, start=0, plot=False, savefig=False,
                         bkbone_graph_type='nx.connected_watts_strogatz_graph',
                         bkbone_graph_args=[4, 0.4], save2text='', add_node=10):
    '''This function creates a graph from a list of building blocks on top
    of a backbone graph
     INPUT:
    ---------------------------------------------------------------------------------
    list_shapes           :   list of shape list (1st arg: type of shape,
                              next args: args for building the shape, except
                              for the start)
    bkbone_graph_type     :   which type of backbone graph
                              (default= 'nx.connected_watts_strogatz_graph')
    add_nodes             :   number of "empty nodes" to add to the graph structures, ie,
                              nodes in the graph that do not belong to a
                              particular clique
    bkbone_graph_args     :   arguments for generating the backbone graph
                              (except from nb of nodes, which
                              is automatically computef)
    start                 :   initial node nb
    plot, savefig,save2txt:   plotting and saving parameters

    OUTPUT:
    ---------------------------------------------------------------------------------
    graph                 :   a nx graph (association of cliques/motifs
                              planted along a backbone structure)
    communities           :   motif Id
    role_labels           :   role id
    label_shape           :   label/class of the motif. This induces
                              different levels of similarities among motifs
    '''
    graph = nx.Graph()
    shape_id = []            # labels for the different shapes
    role_labels = []         # labels for the different shapes
    communities = []         # roles in the network
    seen_shapes = {}

    start = graph.number_of_nodes()
    for nb_shape, shape in enumerate(list_shapes):
        shape_type = shape[0]
        try:
            role_start, shape_id_start = seen_shapes[shape_type]
        except:
            if len(role_labels) > 0:
                seen_shapes[shape_type] = [np.max(role_labels) + 1, np.max(shape_id) + 1]
                role_start, shape_id_start = seen_shapes[shape_type]
            else:
                seen_shapes[shape_type] = [0, 0]
                role_start, shape_id_start = 0, 0
        args = [start]
        args += shape[1:]
        args += [role_start]
        graph_s, roles = eval(shape_type)(*args)
        # Attach the shape to the basis
        graph.add_nodes_from(graph_s.nodes())
        graph.add_edges_from(graph_s.edges())

        communities += [nb_shape] * nx.number_of_nodes(graph_s)
        role_labels += roles
        shape_id += [shape_id_start] * nx.number_of_nodes(graph_s)
        start += graph_s.number_of_nodes()
    # Now we link the different shapes by attaching them to the underlyin
    # graph structure:
    n_nodes, n_shapes = graph.number_of_nodes(), len(list_shapes)
    graph.add_nodes_from(range(n_nodes, n_nodes + add_node))
    role_labels += [n_shapes + 1] * add_node
    communities += range(n_shapes, n_shapes + add_node)
    shape_id += [-1] * add_node

    # generate back_bone Graph
    bkbone_graph_args.insert(0, n_shapes + add_node)
    bkbone_graph = eval(bkbone_graph_type)(*bkbone_graph_args)
    for e in bkbone_graph.edges():
        ii = np.random.choice(np.where(np.array(communities) == e[0])[0], 1)[0]
        jj = np.random.choice(np.where(np.array(communities) == e[1])[0], 1)[0]
        graph.add_edges_from([(ii, jj)])

    if plot is True: plot_networkx(graph, role_labels)
    if len(save2text) > 0: saveNet2txt(graph, colors=role_labels, name='net', path=save2text)
    return graph, communities, role_labels, shape_id


def create_bigger_network(nb_cells, width_cell, list_cell_shapes,
                          rdm_basis_plugins=True, cell_type="cycle"):
    ''' Automatically creates a big network by linking several instances of a
    graph created by build_structure(width_basis, basis_type, list_shapes,..)
    '''
    width_basis, basis_type = width_cell[0]
    list_shapes = list_cell_shapes[0]
    graph, roles, plugins = build_structure(width_basis, basis_type,
                                            list_shapes, start=0,
                                            rdm_basis_plugins=rdm_basis_plugins,
                                            add_random_edges=0, plot=False)
    start = graph.number_of_nodes()
    for i in range(1, nb_cells):
        width_basis, basis_type = width_cell[i]
        list_shapes = list_cell_shapes[i]
        graph_i, roles_i, plugins_i = build_structure(width_basis,
                                                      basis_type,
                                                      list_shapes,
                                                      start=start,
                                                      add_random_edges=0,
                                                      plot=False)
        graph.add_nodes_from(graph_i.nodes())
        graph.add_edges_from(graph_i.edges())
        graph.add_edges_from([(start, start + 1)])
        start += graph_i.number_of_nodes()
        roles += roles_i
        plugins += plugins_i
    return graph, roles, plugins
