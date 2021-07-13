
import numpy as np
import sys
import pandas as pd
import random
import itertools
import networkx as nx
from matplotlib import colors
import csv
import matplotlib
import matplotlib.pyplot as plt
from node2vec import Node2Vec
#import matplotlib.patches as mpatches
# import seaborn as sns
# import stellargraph as sg
# from sklearn import model_selection
from sklearn_som.som import SOM
from matplotlib.colors import ListedColormap
from sklearn.utils import class_weight
# from tensorflow.keras.callbacks import EarlyStopping
# from keras import backend as K
# import tensorflow as tf
print("hey")
# import stellargraph as sg
from sklearn.model_selection import StratifiedShuffleSplit



from sklearn import preprocessing, model_selection


# from stellargraph.mapper import FullBatchNodeGenerator
# from stellargraph.layer import GCN
# from tensorflow.keras import layers, optimizers, losses, metrics, Model
# Matrix Degree = Size of grid
# Num of bacteria = Type of bacterial strains
# Num of strains = Strains of each population in grid
# Simulation type = 1 - Bacteriostatic, 0 - Bacteriocidal

print("Hello")
np.set_printoptions(threshold=sys.maxsize)
#tf.config.run_functions_eagerly(True)

def simulate(matrix_degree, num_bacteria, num_strains, graphs, num_iterations = 600, inhibition_degree_per_cycle = 5, simulationType = 1):
    GRAPHS = []
    FEATURES = pd.DataFrame(columns=['1', '2'])
    FEATURES2 = pd.DataFrame(columns=['1', '2'])
    DIVERSITY_CLASSES = pd.DataFrame(columns=['Diversity_Class'])
    DIVERSITY_CLASSES2 = pd.DataFrame(columns=['Diversity_Class'])

    bacteria1 = []
    bacteria2 = []
    bacteria3 = []
    bacteria4 = []
    bacteria5 = []
    bacteria6 = []
    bacteria7 = []
    bacteria8 = []
    bacteria9 = []
    bacteria10 = []

    bacteria_pop = {}

    for l in range(graphs):

        print("GRAPH No. ", l)
        # 1. Read csv and get col names and numbers
        df = pd.read_csv('newData_normalized.csv', sep=",")
        col_names = list(df.columns[1:])
        # print("COL Names: ", col_names)
    #    df.drop(['0'], axis=1, inplace=True)
        numpy_data = df.iloc[0:].to_numpy()

        # print("NUMPY DATA: ")
        # print(numpy_data)
        # 2. Create a matrix of length matrix_degree * matrix_degree
        matrix = np.zeros(shape=(matrix_degree, matrix_degree), dtype=float)

        df2 = pd.read_csv('reproduction_rate.csv', sep=",")
    #    numpy_data2 = df2.iloc[0:].to_numpy()
    #     print(df2.columns)
        df2.set_index("ID", drop=True, inplace=True)
        dictionary = df2.to_dict(orient="index")
    #    print(numpy_data2)
    #    print(dictionary)
        initNodes = []
        # 3. Randomly add bacteriocins from (1) at different locations in 2
        random_bacteria = random.sample(col_names, num_bacteria)
        for b in range(num_bacteria):

            node = random_bacteria[b]
            initNodes.append(int(float(random_bacteria[b])))

            print("Init nodes: ", len(initNodes))

            for _ in range(num_strains):
                selected_cell = -1
                indexRow = 0
                indexCol = 0
                while selected_cell != 0:
                    indexRow = random.randint(0, matrix_degree - 1)
                    indexCol = random.randint(0, matrix_degree - 1)
                    selected_cell = matrix[indexRow][indexCol]
                matrix[indexRow][indexCol] = int(float(node))

    #    print(matrix)
    #     print("Init nodes: ", len(initNodes), initNodes.sort())
    #     sortedNodes = initNodes.sort()
    #     print("Sorted nodes: ", initNodes)
        G = nx.DiGraph()
        for node in initNodes:
            G.add_node(str(node))

        print("G nodes: ", len(G.nodes))

        # print("G nodes are:")
        # print(G.nodes)

        for node1 in initNodes:
            for node2 in initNodes:
                if node1 != node2:
                    # print(node1)
                    # print(node2)
                    if numpy_data[node1-1][node2] != 0.0:
                        G.add_edge(str(node1), str(node2), weight=numpy_data[node1-1][node2])

        pos = nx.shell_layout(G)
        nx.draw_networkx(G, pos)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.show()

        print("g nodes are: ", len(G.nodes))
        # Node Embeddings
        node2vec = Node2Vec(G, dimensions=2, walk_length=30, num_walks=200,
                            workers=4)

        # Embed nodes
        model = node2vec.fit(window=10, min_count=1,
                             batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)


        # Save embeddings for later use
        model.wv.save_word2vec_format("node_embeddings.csv")

        unique, counts = np.unique(matrix, return_counts=True)
        occurances = dict(zip(unique, counts))
        occurances = {k: v if k == 0.0 else round(dictionary[int(float(k))]['Value'] * 10000 * v / 100) for k, v in
                      occurances.items()}

    #    print(occurances)

        # plt.imshow(matrix, origin="lower", cmap='Purples_r', interpolation='nearest')
        # plt.colorbar()
    #    plt.show()

        diversity_matrix = []
        for r in range(num_iterations):
            new_matrix = matrix.copy()

            # repr_rate = dictionary[int(float(node))]['Value'] * 10000
            # print(repr_rate)

            for i in range(matrix_degree):
                for j in range(matrix_degree):
                    if matrix[i][j] != 0 and matrix[i][j] != -1 and occurances[matrix[i][j]] != 0:
                        to_range = 1
                        from_range = -1
                        if i == 0 or j == 0:
                            from_range = 0
                        if j == matrix_degree-1 or i == matrix_degree-1:
                            to_range = 0
                        random_index_1 = random.randint(from_range, to_range)
                        random_index_2 = random.randint(from_range, to_range)
                        neighbour_cell = matrix[i+random_index_1][j+random_index_2]
                        if neighbour_cell == 0:
                            # Setting this one as booked
                            matrix[i + random_index_1][j + random_index_2] = -1
                            new_matrix[i + random_index_1][j + random_index_2] = matrix[i][j]
                        else:
                            if simulationType == 0:
                                # -1 = Booked
                                if neighbour_cell != -1:
                                    forward_inhibition_score = numpy_data[int(matrix[i][j]) - 1][int(neighbour_cell) - 1]
                                    backward_inhibition_score = numpy_data[int(neighbour_cell) - 1][int(matrix[i][j]) - 1]
                                    accumulated_score = forward_inhibition_score - backward_inhibition_score
                                    if accumulated_score > 0:
                                        # Setting this one as booked
                                        matrix[i + random_index_1][j + random_index_2] = -1
                                        new_matrix[i + random_index_1][j + random_index_2] = matrix[i][j]
                                elif neighbour_cell == -1:
                                    if matrix[i][j] > new_matrix[i + random_index_1][j + random_index_2]:
                                        matrix[i + random_index_1][j + random_index_2] = -1
                                        new_matrix[i + random_index_1][j + random_index_2] = matrix[i][j]

    #                print("Matrix", matrix)
    #                print(matrix[i][j])
                    if matrix[i][j] != -1:
                        occurances[matrix[i][j]] -= 1
    #                print(occurances)
            matrix = new_matrix

            # bounds = G.nodes
            # norm = 'viridis'.BoundaryNorm(bounds, cmap.N)]
            cmap = matplotlib.cm.get_cmap("viridis", 10)
            plt.imshow(matrix, origin="lower", cmap=cmap, interpolation='nearest')
            plt.colorbar()
            plt.show()
    #        print(matrix)
            unique, counts = np.unique(matrix, return_counts=True)
            occurances = dict(zip(unique, counts))


            if 0.0 not in occurances:
    #            print("Finalized")
#                print("Setting diversity matrix: ", occurances)
                diversity_matrix = occurances

                if simulationType == 1:
                    break
                else:
                    continue
                break


            diversity_matrix2 = occurances
            occurances = {k: v if k == 0.0 else round(dictionary[int(float(k))]['Value'] * 10000 * v/100) for k, v in occurances.items()}
            percentage2 = {k: v / (matrix_degree * matrix_degree) * 100 for k, v in diversity_matrix2.items()}
            #    print("Percentage ", percentage)

            for key, value in percentage2.items():
                if key in bacteria_pop.keys():
                    print("Appending")
                    bacteria_pop[key].append(value)
                else:
                    print("Adding new")
                    bacteria_pop[key] = [value]

            #
            # plt.imshow(matrix, origin="lower", cmap='viridis', interpolation='nearest')
            # plt.colorbar()
            #
            # plt.bar(range(len(occurances)), list(occurances.values()), align='center')
            # plt.xticks(range(len(occurances)), list(occurances.keys()))
            # plt.xlabel('Bacteria')
            # plt.ylabel('Population')
            # plt.show()

            #plt.show()

#        print("Final diversity Matrix ", diversity_matrix)


        percentage = {k: v / (matrix_degree*matrix_degree) * 100 for k, v in diversity_matrix.items()}
    #    print("Percentage ", percentage)
        classes = {k: "High" if (v / (matrix_degree*matrix_degree) * 100) > 70 else "Medium" if  (v / (matrix_degree*matrix_degree) * 100) > 30 else "Low" for k, v in diversity_matrix.items()}
    #    print("Classes ", classes)
    #     for key, value in percentage.items():
    #         if key in bacteria_pop.keys():
    #             print("Appending")
    #             bacteria_pop[key].append(value)
    #         else:
    #             print("Adding new")
    #             bacteria_pop[key] = [value]

        print(percentage.items())
        data = list(percentage.items())
        np_array = np.array(data)
        # Self organizing maps
        # print("Hey")
        # print(np_array)
        som = SOM(m=3, n=1, dim=1)
        np_array2 = np_array[:, 1:]

        # print("Hey")
        # print(np_array2)
        som.fit(np_array2)

        print(np_array2)
        # Assign each datapoint to its predicted cluster
        predictions = som.predict(np_array2)
        print(predictions)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 7))
        x = np_array[:, 0]
        y = np_array2[:, 0]
        colors = ['red', 'green', 'blue']

        ax.scatter(x, y, c=predictions, cmap=ListedColormap(colors))
        ax.title.set_text('SOM Predictions')
        plt.xlabel("Bacteria Label")
        plt.ylabel("Percentage")
        filename = "BacteriumDiversity/SOM_Figures_15/" + str(iter) + ".png"
        plt.show()

        print(bacteria_pop.keys())
        del bacteria_pop[0.0]
        print(bacteria_pop)
        for k, v in bacteria_pop.items():
            plt.plot(range(1, len(v) + 1), v, '.-', label=k)
            # NOTE: changed `range(1, 4)` to mach actual values count
        plt.legend()  # To draw legend
        plt.show()

        df_node2vec = pd.read_csv('node_embeddings.csv', header=None, sep='\s+', skiprows=1)
        df_node2vec.sort_values(by=0, inplace=True)

        print("Node Embeddings: ", len(df_node2vec))

        df_node2vec.insert(3, 'Diversity_Class', predictions)
       # df.loc[0] = str(i) + "-" + df.loc[0].astype(str)


        df_node2vec = df_node2vec.set_index(0)
   #     df.reset_index(drop=True, inplace=True)

        diversity_class = pd.DataFrame(columns=['Diversity_Class'])
        i = 0
        for index, row in df_node2vec.iterrows():
            if l == 0:
                DIVERSITY_CLASSES.loc[index] = row['Diversity_Class']  #0 if i % 3 == 0 else 1 if i % 3 == 1 else 2
            DIVERSITY_CLASSES2.loc[index] = row['Diversity_Class']      #0 if i % 3 == 0 else 1 if i % 3 == 1 else 2
            diversity_class.loc[index] = row['Diversity_Class']  #0 if i % 3 == 0 else 1 if i % 3 == 1 else 2
            i += 1


        diversity_class = diversity_class['Diversity_Class'].astype(int)

        # Feature Matrix for nodes
        features = df_node2vec.drop(columns="Diversity_Class")


        #Append Graph
        GRAPHS.append(G)

        if l == 0:
            FEATURES = features

        #Append Features
        for index, row in features.iterrows():
            # if l == 0:
            #     FEATURES.loc[index] = [row[1], row[2]]
            FEATURES2.loc[index] = [row[1], row[2]]
        #            FEATURES.append({'1': row[1], '2': row[2]}, ignore_index=True)

#        print("diversity class: ", diversity_class)

        # if l == 0:
        #     stellargraph = sg.StellarGraph.from_networkx(G,
#                                                         node_features=features)
            # print(len(diversity_class))
            # print(len(features))
            # train_classes, test_classes = model_selection.train_test_split(
            #     diversity_class, train_size=28, test_size=None, stratify=diversity_class
            # )
            # # print(len(train_classes))
            # # print(len(test_classes))
            # val_classes, test_classes = model_selection.train_test_split(
            #     test_classes, train_size=6, test_size=None, stratify=test_classes
            # )

            # gcn(stellargraph, diversity_class, train_classes, test_classes, val_classes)#, train_size=6, validation_size=6)
            # gcn(stellargraph, diversity_class, train_classes, test_classes, val_classes)#, train_size=6, validation_size=6)

    # Create graph using featured nodes
    # print("Graphs: ", GRAPHS)
    # print("Features: ", FEATURES)
    # print("DIVERSITY CLASSES: ", DIVERSITY_CLASSES)


    #
    # for key, values in bacteria_pop.items():
    #     dates_str, values = zip(*data_list)  # Unpack
    #     dates = convert_str_to_dates(dates_str)  # Not handled in example
    #     plt.plot(dates, values, label=key)
    # plt.legend()
    # plt.show()



    graph = nx.union_all(GRAPHS)

    # pos = nx.spring_layout(graph)
    # nx.draw(graph, pos)
    # pos = nx.spring_layout(graph, scale=2)
    # nx.draw(graph, pos)
    # plt.savefig('BacteriumDiversity/plotgraph.svg')


    # square2 = sg.StellarGraph.from_networkx(graph,
    #                                        node_features=FEATURES2)
    #
    # gcn(square2, DIVERSITY_CLASSES2)#, train_classes, test_classes, val_classes)


# def gcn(G, classes): #, train_size = 9, validation_size = 28):
#
#     #Training
#     #For now, out of 50, 10 nodes for training, 15 for valuation and rest for testing
#     # train_classes, test_classes = model_selection.train_test_split(
#     #     classes, train_size=train_size, test_size=None, stratify=classes
#     # )
#     # val_classes, test_classes = model_selection.train_test_split(
#     #     test_classes, train_size=validation_size, test_size=None, stratify=test_classes
#     # )
#
#     print("Experiment env: ")
#     print(classes.value_counts().to_frame())
#
#     train_classes, test_classes = model_selection.train_test_split(
#         classes, train_size=int(80/100*len(classes)), test_size=None, stratify=classes
#     )
#     # print(len(train_classes))
#     # print(len(test_classes))
#     val_classes, test_classes = model_selection.train_test_split(
#         test_classes, train_size=int(50/100*len(test_classes)), test_size=None, stratify=test_classes
#     )
#
#     print(train_classes.value_counts().to_frame())
#     print(val_classes.value_counts().to_frame())
#     print(test_classes.value_counts().to_frame())
#
#
#     # print(G.info())
# #    print(classes)
#
#     target_encoding = preprocessing.LabelBinarizer()
#
#     train_targets = target_encoding.fit_transform(train_classes)
#     val_targets = target_encoding.transform(val_classes)
#     test_targets = target_encoding.transform(test_classes)
#
#
#     generator = FullBatchNodeGenerator(G, method="gcn")
#     train_gen = generator.flow(train_classes.index, train_targets)
#
#     gcn = GCN(
#         layer_sizes=[16, 16], activations=["relu", "relu"], generator=generator, dropout=0.5
#     )
#
#     x_inp, x_out = gcn.in_out_tensors()
#
#     tf.config.run_functions_eagerly(True)
#     predictions = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)
#
#
#     class_weights = class_weight.compute_class_weight(
#         'balanced',
#         classes=np.unique(train_classes['Diversity_Class']),
#         y=train_classes['Diversity_Class'])
#
#     class_weights = {i: class_weights[i] for i in range(3)}
#
#     print("Class Weights: ", class_weights)
#     #    print(predictions)
#     model = Model(inputs=x_inp, outputs=predictions)
#     model.compile(
#         optimizer=optimizers.Adam(lr=0.01),
#         loss=weightedLoss(losses.categorical_crossentropy, class_weights),
#         weighted_metrics=["acc"],
#         run_eagerly=True
#     )
#
#
#     val_gen = generator.flow(val_classes.index, val_targets)
#     es_callback = EarlyStopping(monitor="val_acc", patience=50, restore_best_weights=True)
#
#     history = model.fit(
#         train_gen,
#         batch_size=32,
#         epochs=int(sys.argv[6]),
#         validation_data=val_gen,
#         verbose=2,
#         shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
# #        callbacks=[es_callback],
#     )
#
#     sg.utils.plot_history(history)
#     plt.savefig('BacteriumDiversity/' + sys.argv[5])
#
#     test_gen = generator.flow(test_classes.index, test_targets)
#     test_metrics = model.evaluate(test_gen)
#     print("\nTest Set Metrics:")
#     for name, val in zip(model.metrics_names, test_metrics):
#         print("\t{}: {:0.4f}".format(name, val))
#
#     # all_nodes = classes.index
#     # print(all_nodes)
#     # all_gen = generator.flow(all_nodes)
#     # all_predictions = model.predict(all_gen)
#
#     # df = pd.DataFrame(columns=['Actual', 'Predicted'])
#     # for node in all_nodes:
#     #     gen = generator.flow(node)
#     #     prediction = model.predict(gen)
#     #     print("PREDICTION: ", prediction[0][0])
#     #     node_prediction = target_encoding.inverse_transform(prediction[0][0].squeeze())
#     #     print("ACTUAL: ", classes[node])
#     #     print("PREDICTED: ", node_prediction)
#
# #    model.predict()
#
#     # print("ALL PREDICTIONS")
#     # print(all_predictions)
#     # node_predictions = target_encoding.inverse_transform(all_predictions.squeeze())
#     # print("NODE PREDICTIONS")
#     # print(node_predictions)
#     # df = pd.DataFrame({"Predicted": node_predictions})
#     # print(df)
#
#     # df2 = pd.DataFrame({"Actual": classes}, index=classes.index)
#     # print(df2)
#
# def weightedLoss(originalLossFunc, weightsList):
#
#     def lossFunc(true, pred):
#
#         axis = -1 #if channels last
#         #axis=  1 #if channels first
#
#
#         #argmax returns the index of the element with the greatest value
#         #done in the class axis, it returns the class index
#         classSelectors = K.argmax(true, axis=axis)
#             #if your loss is sparse, use only true as classSelectors
#
#         #considering weights are ordered by class, for each class
#         #true(1) if the class index is equal to the weight index
#         classSelectors = [K.equal(i, classSelectors) for i in range(len(weightsList))]
#
#         #casting boolean to float for calculations
#         #each tensor in the list contains 1 where ground true class is equal to its index
#         #if you sum all these, you will get a tensor full of ones.
#         classSelectors = [K.cast(x, K.floatx()) for x in classSelectors]
#
#         #for each of the selections above, multiply their respective weight
#         weights = [sel * w for sel,w in zip(classSelectors, weightsList)]
#
#         #sums all the selections
#         #result is a tensor with the respective weight for each element in predictions
#         weightMultiplier = weights[0]
#         for i in range(1, len(weights)):
#             weightMultiplier = weightMultiplier + weights[i]
#
#
#         #make sure your originalLossFunc only collapses the class axis
#         #you need the other axes intact to multiply the weights tensor
#         loss = originalLossFunc(true,pred)
#         loss = loss * weightMultiplier
#
#         return loss
#     return lossFunc
#
#
# def gcn2(G, classes):
#     # Use scikit-learn to compute training and test sets
#     train_targets, test_targets = model_selection.train_test_split(classes, train_size=0.5)
#
#     generator = sg.mapper.FullBatchNodeGenerator(G, method="gcn")
#
#     # two layers of GCN, each with hidden dimension 16
#     gcn = sg.layer.GCN(layer_sizes=[16, 16], generator=generator)
#     x_inp, x_out = gcn.in_out_tensors()  # create the input and output TensorFlow tensors
#
#     # use TensorFlow Keras to add a layer to compute the (one-hot) predictions
#     predictions = tf.keras.layers.Dense(units=len(classes.columns), activation="softmax")(x_out)
#
#     # use the input and output tensors to create a TensorFlow Keras model
#     model = tf.keras.Model(inputs=x_inp, outputs=predictions)
#
# #for i in range(35,41):
# # print("Args: ", sys.argv, len(sys.argv))
# #
# # simulate(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))

simulate(100, 10, 5, 1)

# simulate(500, 3, 5, 5000)
# simulate(1000, 3, 5, 5000)
# simulate(500, 5, 5, 5000)
# simulate(500, 5, 5, 10000)


# simulate(1000, 10, 5, 10000)


# Initial:
# 3 bacteria, 5 strains / bacteria,

# 5000 grap

#Arguments
# Grid Size, number of bacteria, No of strains, No of Graphs, Output plot name, No. of Epochs