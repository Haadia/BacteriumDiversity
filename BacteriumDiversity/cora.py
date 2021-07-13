import pandas as pd
import os

import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GCN

from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, model_selection
#from IPython.display import display, HTML
import matplotlib.pyplot as plt
#%matplotlib inline

dataset = sg.datasets.Cora()
display(HTML(dataset.description))
G, node_subjects = dataset.load()

print(G.info())

# node_subjects.value_counts().to_frame()
#
# train_subjects, test_subjects = model_selection.train_test_split(
#     node_subjects, train_size=140, test_size=None, stratify=node_subjects
# )
# val_subjects, test_subjects = model_selection.train_test_split(
#     test_subjects, train_size=500, test_size=None, stratify=test_subjects
# )
#
#
# target_encoding = preprocessing.LabelBinarizer()
#
# train_targets = target_encoding.fit_transform(train_subjects)
# val_targets = target_encoding.transform(val_subjects)
# test_targets = target_encoding.transform(test_subjects)
#
# generator = FullBatchNodeGenerator(G, method="gcn")
#
#
# train_gen = generator.flow(train_subjects.index, train_targets)
#
# gcn = GCN(
#     layer_sizes=[16, 16], activations=["relu", "relu"], generator=generator, dropout=0.5
# )
#
# x_inp, x_out = gcn.in_out_tensors()
# predictions = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)
#
# model = Model(inputs=x_inp, outputs=predictions)
# model.compile(
#     optimizer=optimizers.Adam(lr=0.01),
#     loss=losses.categorical_crossentropy,
#     metrics=["acc"],
# )
#
# val_gen = generator.flow(val_subjects.index, val_targets)
#
# from tensorflow.keras.callbacks import EarlyStopping
#
# es_callback = EarlyStopping(monitor="val_acc", patience=50, restore_best_weights=True)
#
# history = model.fit(
#     train_gen,
#     epochs=200,
#     validation_data=val_gen,
#     verbose=2,
#     shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
#     callbacks=[es_callback],
# )
#
# sg.utils.plot_history(history)
# plt.savefig('BacteriumDiversity/cora.png')
#
# test_gen = generator.flow(test_subjects.index, test_targets)
#
# test_metrics = model.evaluate(test_gen)
# print("\nTest Set Metrics:")
# for name, val in zip(model.metrics_names, test_metrics):
#     print("\t{}: {:0.4f}".format(name, val))
