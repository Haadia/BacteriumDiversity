
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
#import matplotlib.patches as mpatches
#import seaborn as sns
#import stellargraph as sg
#from sklearn import model_selection
from sklearn_som.som import SOM
from matplotlib.colors import ListedColormap

#import tensorflow as tf
print("hey")
#import stellargraph as sg

# Matrix Degree = Size of grid
# Num of bacteria = Type of bacterial strains
# Num of strains = Strains of each population in grid
# Simulation type = 1 - Bacteriostatic, 0 - Bacteriocidal

print("Hello")
np.set_printoptions(threshold=sys.maxsize)


def simulate(matrix_degree, num_bacteria, num_strains, iter, num_iterations = 100, inhibition_degree_per_cycle = 5, simulationType = 1):

    # 1. Read csv and get col names and numbers
    df = pd.read_csv('BacteriumDiversity/newData_normalized.csv', sep=",")
    col_names = list(df.columns)

#    df.drop(['0'], axis=1, inplace=True)
    numpy_data = df.iloc[0:].to_numpy()

    # 2. Create a matrix of length matrix_degree * matrix_degree
    matrix = np.zeros(shape=(matrix_degree, matrix_degree), dtype=float)

    df2 = pd.read_csv('BacteriumDiversity/reproduction_rate.csv', sep=",")
#    numpy_data2 = df2.iloc[0:].to_numpy()
    df2.set_index("ID", drop=True, inplace=True)
    dictionary = df2.to_dict(orient="index")
#    print(numpy_data2)
#    print(dictionary)
    initNodes = []
    # 3. Randomly add bacteriocins from (1) at different locations in 2
    for b in range(num_bacteria):
        node = random.choice(col_names)
        initNodes.append(int(float(node)))


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

    G = nx.DiGraph()
    for node in initNodes:
        G.add_node(node)


    for node1 in initNodes:
        for node2 in initNodes:
            if node1 != node2:
                # print(node1)
                # print(node2)
                if numpy_data[node1][node2] != 0.0:
                    G.add_edge(node1, node2, weight=numpy_data[node1][node2])

    pos = nx.shell_layout(G)
    nx.draw_networkx(G, pos)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)


    # Node Embeddings
    node2vec = Node2Vec(G, dimensions=10, walk_length=30, num_walks=200,
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
#                print("Matrix", matrix)
#                print(matrix[i][j])
                if matrix[i][j] != -1:
                    occurances[matrix[i][j]] -= 1
#                print(occurances)
        matrix = new_matrix

#        print(matrix)
        unique, counts = np.unique(matrix, return_counts=True)
        occurances = dict(zip(unique, counts))

        if 0.0 not in occurances:
#            print("Finalized")
            diversity_matrix = occurances
            if simulationType == 1:
                break
            else:
                continue
            break


        occurances = {k: v if k == 0.0 else round(dictionary[int(float(k))]['Value'] * 10000 * v/100) for k, v in occurances.items()}

        plt.imshow(matrix, origin="lower", cmap='Purples_r', interpolation='nearest')
        plt.colorbar()

        # plt.bar(range(len(occurances)), list(occurances.values()), align='center')
        # plt.xticks(range(len(occurances)), list(occurances.keys()))
        # plt.xlabel('Bacteria')
        # plt.ylabel('Population')
#        plt.show()

        #plt.show()

#    print("Final diversity Matrix ", diversity_matrix)
    percentage = {k: v / (matrix_degree*matrix_degree) * 100 for k, v in diversity_matrix.items()}
#    print("Percentage ", percentage)
    classes = {k: "High" if (v / (matrix_degree*matrix_degree) * 100) > 70 else "Medium" if  (v / (matrix_degree*matrix_degree) * 100) > 30 else "Low" for k, v in diversity_matrix.items()}
#    print("Classes ", classes)

    data = list(percentage.items())
    np_array = np.array(data)
    # Self organizing maps
#    print("Hey")
#    print(np_array)
    som = SOM(m=3, n=1, dim=1)
    np_array2 = np_array[:, 1:]

    # print("Hey")
    # print(np_array2)
    som.fit(np_array2)

    # Assign each datapoint to its predicted cluster
    predictions = som.predict(np_array2)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 7))
    x = np_array[:, 0]
    y = np_array2[:, 0]
    colors = ['red', 'green', 'blue']

    ax.scatter(x, y, c=predictions, cmap=ListedColormap(colors))
    ax.title.set_text('SOM Predictions')
    plt.xlabel("Bacteria Label")
    plt.ylabel("Percentage")
#    filename = "BacteriumDiversity/SOM_Figures_15/" + str(iter) + ".png"
#    plt.savefig(filename)

    df_node2vec = pd.read_csv('node_embeddings.csv', header=None, sep=" ", skiprows=1)

    df_node2vec.sort_values(by=0, inplace=True)


#    for index, row in df_node2vec.iterrows():
    df_node2vec.insert(11, 'Diversity_Class', predictions)
    #    df.drop(['0'], axis=1, inplace=True)
    numpy_data_node2vec = df_node2vec.iloc[0:].to_numpy()


    print(numpy_data_node2vec[:, 1:11])

    # square = StellarGraph.from_networkx(g, node_features=nodefeatures)
    # print(square.info())



#for i in range(35,41):
simulate(40, 15, 3, 1)