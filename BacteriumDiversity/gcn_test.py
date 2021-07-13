import stellargraph as sg
from stellargraph.mapper import CorruptedGenerator, FullBatchNodeGenerator
from stellargraph.layer import GCN, DeepGraphInfomax

import pandas as pd
from sklearn import model_selection, preprocessing
from IPython.display import display, HTML

import tensorflow as tf
from tensorflow.keras import Model, layers, optimizers, callbacks

dataset = sg.datasets.Cora()
display(HTML(dataset.description))
G, node_classes = dataset.load()

#print(G.info())

print(dataset)
print("Node classes ")
print(node_classes)