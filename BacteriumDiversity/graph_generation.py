
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import random

nodes = []
cols = []
edges = {}

def node_feature_extraction():
    global nodes
    global edges
    global cols
    df = pd.read_csv('Raw_data_normalized.csv', sep=",")

    numpy_data = df.to_numpy()
    cols = df.columns[1:]

    adjacency = numpy_data[0:, 1:]

    graph = nx.from_numpy_matrix(adjacency, create_using=nx.DiGraph)
#    nx.draw(graph)

    in_degrees = graph.in_degree()
    out_degrees = graph.out_degree()
    clustering_coefficients = nx.clustering(graph)

    plt.hist(clustering_coefficients)
    plt.show()

    # in_degree_freq =
    # out_degree_freq = nx.degree_histogram(graph, out_degree=True)
    # degrees = range(len(in_degree_freq))
    # plt.figure(figsize=(12, 8))
    # plt.loglog(range(len(in_degrees)), in_degrees, 'go-', label='in-degree')
    # plt.loglog(range(len(out_degrees)), out_degrees, 'bo-', label='out-degree')
    # plt.xlabel('Degree')
    # plt.ylabel('Frequency')
    #
    # plt.show()

    #PSR = 0, SR = 1, R = 2, PR = 3
    profile = {}

     # PSR: At least one out degree, at least one in-degree and in degree is not equal to other total number of bacteria - 1
    psr = dict(filter(lambda elem: elem[1] != 0 and out_degrees[elem[0]] != 0 and in_degrees[elem[0]] != len(in_degrees)-1, graph.in_degree()))
    for (key, value) in psr.items():
        profile[key] = 0


    # SR: Out degree is 0 and at least one in degree and in degree is not equal to other total number of bacteria - 1
    sr = dict(filter(lambda elem: elem[1] != 0 and out_degrees[elem[0]] == 0 and in_degrees[elem[0]] != len(in_degrees) - 1, graph.in_degree()))
    for key, value in sr.items():
        profile[key] = 1

    # R: Both in degree and out degree is 0
    r = dict(filter(lambda elem: elem[1] == 0 and out_degrees[elem[0]] == 0,graph.in_degree()))
    for key, value in r.items():
        profile[key] = 2

    # PR: At least one out degree and no indegree
    pr = dict(filter(lambda elem: elem[1] == 0 and out_degrees[elem[0]] != 0, graph.in_degree()))
    for key, value in pr.items():
        profile[key] = 3


#    cam_net_components = nx.connected_component_subgraphs(graph)

    graph = graph.to_undirected()
    A = (graph.subgraph(c) for c in nx.connected_components(graph))
    cam_net_mc = list(A)[0]

    print(cam_net_mc)
#    cam_net_mc = cam_net_components[0]

    # Betweenness centrality
    bet_cen = nx.betweenness_centrality(cam_net_mc)
#    print(bet_cen)
    # Closeness centrality
    clo_cen = nx.closeness_centrality(cam_net_mc)
#    print(clo_cen)
    # Eigenvector centrality
    eig_cen = nx.eigenvector_centrality(cam_net_mc)
#    print(eig_cen)

    new_numpy_data = []

    for element in in_degrees:
        # In degree, out degree, clustering co-efficient, PSR profile
        if element[0] not in bet_cen.keys():
            new_numpy_data.append([int(element[0]), element[1], out_degrees[element[0]], clustering_coefficients[element[0]],
             profile[element[0]], 0, 0, 0])
        else:
            new_numpy_data.append([int(element[0]), element[1], out_degrees[element[0]], clustering_coefficients[element[0]],
             profile[element[0]], bet_cen[element[0]], clo_cen[element[0]], eig_cen[element[0]]])

    # Defining nodes and edges
    nodes = new_numpy_data

    #    print("Nodes: ", nodes)
    for i in range(len(adjacency)):
        for j in range(len(adjacency[i])):
            #            if int(adjacency[i][j]) != 0:
            edges[(i, j)] = adjacency[i][j]


    np.savetxt("node_features.csv", new_numpy_data, fmt = '%1.3f')



node_feature_extraction()


#
# import torch
# import torch.nn as nn
# from torch.optim import Adam
#
#
# #2 nodes and 4 features at any time
# network = nn.Sequential(
#     nn.Linear(2*4, 64),
#     nn.ReLU(True),
#     nn.Linear(64, 64),
#     nn.ReLU(True),
#     nn.Linear(64, 1),
# )
# optimizer = Adam(network.parameters())
# loss_func = nn.MSELoss()
#
# # ---------------------------------------
# # Setup dataset
# # ---------------------------------------
#
# from torch.utils.data import Dataset
#
#
# class GraphData(Dataset):
#     def __init__(self, nodes, edges) -> None:
#         super().__init__()
#         self.nodes = nodes
#         self.edges = edges
#
#     def __len__(self):
#         return int(1e6)
#
#     def __getitem__(self, index):
#         node_1 = random.randrange(len(nodes))
#         node_2 = random.randrange(len(nodes))
#
#         feature1 = torch.tensor(self.nodes[node_1])
#         feature2 = torch.tensor(self.nodes[node_2])
#         edge = torch.tensor(float(self.edges[(node_1, node_2)]))
#
#         features = [feature1, feature2]
#         features = torch.cat(features)
#
#         return features, edge
#
#
# # ---------------------------------------
# # Train the neural network
# # ---------------------------------------
#
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
#
# #print("Nodes: ", nodes)
# ds = GraphData(nodes, edges)
# dl = DataLoader(ds, 8)
#
# n_steps = 1000
# for step_id, batch in enumerate(dl):
#     if step_id > n_steps:
#         break
#
#     # calc loss
#     features, connected = batch
#     estimated = network(features).squeeze(1)
# #    print("Estimated: ", estimated)
#     loss = loss_func(estimated, connected)
#
#     # update network params
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
# #    print("Loss: ", loss)
#
# # ---------------------------------------
# # Generate new graphs
# # ---------------------------------------
#
# from torch.distributions import Categorical
#
# nodes1 = torch.tensor(nodes)
# mu = nodes1.mean(0)
# std = nodes1.std(0)
#
# num_graphs = 1
#
# for _ in range(num_graphs):
#     new_nodes = torch.randn_like(nodes1) * std + mu
#
#     keys = []
#     features = []
#     for i in range(len(nodes1)):
#         for j in range(len(nodes1)):
#             features.append(torch.cat((new_nodes[i], new_nodes[j])))
#             keys.append((i, j))
#     features = torch.stack(features)
#
#     with torch.no_grad():
#         probs = network(features)
#
#
# #    print(probs)
#
#     new_edges = {
#         keys[i]: (probs[i])
#         for i in range(len(keys))
#     }
#     new_graph = np.zeros(shape=(len(nodes), len(nodes)), dtype=float)
#
#     for edge in new_edges:
#         print(edge)
#         print(new_edges[edge])
#         new_graph[edge[0]][edge[1]] = new_edges[edge] if new_edges[edge] > 300 else 0
#
#     print(new_graph)
#     np.savetxt("new_graph.csv", new_graph, fmt = '%1.8f', delimiter=',', header=",".join(cols))
