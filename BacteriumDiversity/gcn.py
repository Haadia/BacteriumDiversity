

import tensorflow
from tensorflow.keras import layers, optimizers, losses, metrics, Model
import stellargraph as sg

from sklearn import preprocessing, model_selection


from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GCN
from tensorflow.keras import layers, optimizers, losses, metrics, Model
import numpy as np
import sys
import pandas as pd
import random
import itertools
import networkx as nx
import csv
import matplotlib
import matplotlib.pyplot as plt
from node2vec import Node2Vec
import matplotlib.patches as mpatches
import seaborn as sns

TRAINING_NODES = 100
VALUATION_NODES = 200
TESTING_NODES = 500

def gcn(G, classes):

    #Training
    train_classes, test_classes = model_selection.train_test_split(
        classes, train_size=TRAINING_NODES, test_size=None, stratify=classes
    )
    val_classes, test_classes = model_selection.train_test_split(
        test_classes, train_size=VALUATION_NODES, test_size=None, stratify=test_classes
    )

    target_encoding = preprocessing.LabelBinarizer()

    train_targets = target_encoding.fit_transform(train_classes)
    val_targets = target_encoding.transform(val_classes)
    test_targets = target_encoding.transform(test_classes)


    generator = FullBatchNodeGenerator(G, method="gcn")
    train_gen = generator.flow(train_classes.index, train_targets)

    gcn = GCN(
        layer_sizes=[16, 16], activations=["relu", "relu"], generator=generator, dropout=0.5
    )

    x_inp, x_out = gcn.in_out_tensors()

    predictions = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)

    print(predictions)

def gcn2():
    g = nx.read_adjlist('BacteriumDiversity/adjacency_graph.csv', delimiter=",", create_using=nx.DiGraph())
    nx.draw(g, with_labels=True)

    df_node2vec = pd.read_csv('BacteriumDiversity/numpy_data_node2vec.csv', header=None, sep=",", skiprows=1)

    numpy_data_node2vec = df_node2vec.iloc[0:].to_numpy()

    squareg = sg.StellarGraph.from_networkx(g)
    print(squareg.info())
    g_feature_attr = g.copy()

    def compute_features(node_id):
        # in general this could compute something based on other features, but for this example,
        # we don't have any other features, so we'll just do something basic with the node_id
#        print("BUNNY ", numpy_data_node2vec[np.where(numpy_data_node2vec[:,0] == node_id)])
        return numpy_data_node2vec[np.where(numpy_data_node2vec[:,0] == node_id)][1:3]

    features = pd.DataFrame(columns=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    for row in numpy_data_node2vec:
        print(row)
        features.loc[row[0]] = row[1:11]

    print(features)

    # for node_id, node_data in g_feature_attr.nodes(data=True):
    #     print("NODE ID: ", node_id)
    #     print("NODE DATA: ", node_data)
    #     node_data["feature"] = compute_features(node_id)

    square = sg.StellarGraph.from_networkx(g_feature_attr, node_features=features)#, node_features="feature") #, node_features=numpy_data_node2vec[:, 1:11])
    print(square.info())


gcn2()