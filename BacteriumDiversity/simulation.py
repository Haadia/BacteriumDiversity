
import numpy as np
import pandas as pd
import random
import itertools
import networkx as nx
import csv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


def simulate(matrix_degree, graph_degree, num_iterations = 20, inhibition_degree_per_cycle = 5, simulationType = "Local"):

    print("Hello")
    # 1. Read csv and get col names and numbers
    df = pd.read_csv('BacteriumDiversity/Raw_data_73.csv', sep=",")
    col_names = list(df.columns)

#    df.drop(['0'], axis=1, inplace=True)
    numpy_data = df.iloc[0:].to_numpy()

    # 2. Create a matrix of length matrix_degree * matrix_degree
    matrix = np.zeros(shape=(matrix_degree, matrix_degree), dtype=float)

    # 3. Randomly add bacteriocins from (1) at different locations in 2
    for b in range(graph_degree):
        node = random.choice(col_names)
        indexRow = random.randint(0,matrix_degree-1)
        indexCol = random.randint(0,matrix_degree-1)
        matrix[indexRow][indexCol] = int(node)

    print("Initial population: ")
    print(matrix)

    # Make a loop til num_iteractions
    for j in range(num_iterations):
        #In every iteraction, randomly check some inhibitions
        random_pair_indices_row = random.sample(range(0, matrix_degree), 5)
        random_pair_indices_col = random.sample(range(0, matrix_degree), 5)
        random_pairs = []

        for i in range(5):
            random_pairs.append([random_pair_indices_row[i],random_pair_indices_col[i]])

        print(random_pairs)


        for pair in random_pairs:
            bacteriaOne = matrix[pair[0]][pair[1]]

            if simulationType == "Local":

                dict = {}
                #Take all of its neighboring bacterias and check their accumulated inhibition score

                if pair[1] != 0:
                    bacteriaLeft = matrix[pair[0]][pair[1]-1]
                    if bacteriaOne != 0.0 and bacteriaLeft != 0.0:
                        forward_inhibition_score = numpy_data[int(bacteriaOne)-1][int(bacteriaLeft)-1]
                        backward_inhibition_score = numpy_data[int(bacteriaLeft)-1][int(bacteriaOne)-1]
                        accumulated_score = forward_inhibition_score - backward_inhibition_score
                        dict["Left"] = accumulated_score

                if pair[1] != matrix_degree-1:
                    bacteriaRight = matrix[pair[0]][pair[1]+1]
                    if bacteriaOne != 0.0 and bacteriaRight != 0.0:
                        forward_inhibition_score = numpy_data[int(bacteriaOne)-1][int(bacteriaRight)-1]
                        backward_inhibition_score = numpy_data[int(bacteriaRight)-1][int(bacteriaOne)-1]
                        accumulated_score = forward_inhibition_score - backward_inhibition_score
                        dict["Right"] = accumulated_score

                if pair[0] != 0:
                    bacteriaTop = matrix[pair[0]-1][pair[1]]
                    if bacteriaOne != 0.0 and bacteriaTop != 0.0:
                        forward_inhibition_score = numpy_data[int(bacteriaOne)-1][int(bacteriaTop)-1]
                        backward_inhibition_score = numpy_data[int(bacteriaTop)-1][int(bacteriaOne)-1]
                        accumulated_score = forward_inhibition_score - backward_inhibition_score
                        dict["Top"] = accumulated_score

                if pair[0] != matrix_degree-1:
                    bacteriaBottom = matrix[pair[0]+1][pair[1]]
                    if bacteriaOne != 0.0 and bacteriaBottom != 0.0:
                        forward_inhibition_score = numpy_data[int(bacteriaOne)-1][int(bacteriaBottom)-1]
                        backward_inhibition_score = numpy_data[int(bacteriaBottom)-1][int(bacteriaOne)-1]
                        accumulated_score = forward_inhibition_score - backward_inhibition_score
                        dict["Bottom"] = accumulated_score

                if pair[0] != 0 and pair[1] != 0:
                    bacteriaTopLeft = matrix[pair[0]-1][pair[1]-1]
                    if bacteriaOne != 0.0 and bacteriaTopLeft != 0.0:
                        forward_inhibition_score = numpy_data[int(bacteriaOne)-1][int(bacteriaTopLeft)-1]
                        backward_inhibition_score = numpy_data[int(bacteriaTopLeft)-1][int(bacteriaOne)-1]
                        accumulated_score = forward_inhibition_score - backward_inhibition_score
                        dict["TopLeft"] = accumulated_score

                if pair[0] != matrix_degree - 1 and pair[1] != 0:
                    bacteriaBottomLeft = matrix[pair[0]+1][pair[1]-1]
                    if bacteriaOne != 0.0 and bacteriaBottomLeft != 0.0:
                        forward_inhibition_score = numpy_data[int(bacteriaOne)-1][int(bacteriaBottomLeft)-1]
                        backward_inhibition_score = numpy_data[int(bacteriaBottomLeft)-1][int(bacteriaOne)-1]
                        accumulated_score = forward_inhibition_score - backward_inhibition_score
                        dict["BottomLeft"] = accumulated_score

                if pair[0] != 0 and pair[1] != matrix_degree-1:
                    bacteriaTopRight = matrix[pair[0]-1][pair[1]+1]
                    if bacteriaOne != 0.0 and bacteriaTopRight != 0.0:
                        forward_inhibition_score = numpy_data[int(bacteriaOne)-1][int(bacteriaTopRight)-1]
                        backward_inhibition_score = numpy_data[int(bacteriaTopRight)-1][int(bacteriaOne)-1]
                        accumulated_score = forward_inhibition_score - backward_inhibition_score
                        dict["TopRight"] = accumulated_score

                if pair[1] != matrix_degree-1 and pair[0] != matrix_degree-1:
                    bacteriaBottomRight = matrix[pair[0]+1][pair[1]+1]
                    if bacteriaOne != 0.0 and bacteriaBottomRight != 0.0:
                        forward_inhibition_score = numpy_data[int(bacteriaOne)-1][int(bacteriaBottomRight)-1]
                        backward_inhibition_score = numpy_data[int(bacteriaBottomRight)-1][int(bacteriaOne)-1]
                        accumulated_score = forward_inhibition_score - backward_inhibition_score
                        dict["BottomRight"] = accumulated_score

                if bacteriaOne != 0.0 and len(dict) != 0:
                    maximumValue = max(dict.values())
                    maximumKey = max(dict, key=dict.get)
                    if maximumKey == "Left":
                        if maximumValue > 0:
                            matrix[pair[0]][pair[1]-1] = int(bacteriaOne)
                        else:
                            matrix[pair[0]][pair[1]] = matrix[pair[0]][pair[1]-1]

                    if maximumKey == "Right":
                        if maximumValue > 0:
                            matrix[pair[0]][pair[1]+1] = int(bacteriaOne)
                        else:
                            matrix[pair[0]][pair[1]] = matrix[pair[0]][pair[1]+1]

                    if maximumKey == "Top":
                        if maximumValue > 0:
                            matrix[pair[0]-1][pair[1]]= int(bacteriaOne)
                        else:
                            matrix[pair[0]][pair[1]] = matrix[pair[0]-1][pair[1]]

                    if maximumKey == "Bottom":
                        if maximumValue > 0:
                            matrix[pair[0]+1][pair[1]] = int(bacteriaOne)
                        else:
                            matrix[pair[0]][pair[1]] = matrix[pair[0]+1][pair[1]]

                    if maximumKey == "TopLeft":
                        if maximumValue > 0:
                            matrix[pair[0]-1][pair[1]-1] = int(bacteriaOne)
                        else:
                            matrix[pair[0]][pair[1]] = matrix[pair[0]-1][pair[1]-1]

                    if maximumKey == "BottomLeft":
                        if maximumValue > 0:
                            matrix[pair[0]+1][pair[1]-1] = int(bacteriaOne)
                        else:
                            matrix[pair[0]][pair[1]] = matrix[pair[0]+1][pair[1]-1]

                    if maximumKey == "TopRight":
                        if maximumValue > 0:
                            matrix[pair[0]-1][pair[1]+1] = int(bacteriaOne)
                        else:
                            matrix[pair[0]][pair[1]] = matrix[pair[0]-1][pair[1]+1]

                    if maximumKey == "BottomRight":
                        if maximumValue > 0:
                            matrix[pair[0]+1][pair[1]+1] = int(bacteriaOne)
                        else:
                            matrix[pair[0]][pair[1]] = matrix[pair[0]+1][pair[1]+1]
            else:
                #Each random bacteria's innihibition with every other bacteria
                print("Global")

        print("Simulation step: ", j)
        print(matrix)

        plt.imshow(matrix, origin="lower", cmap='Purples_r', interpolation='nearest')
        plt.colorbar()
        plt.show()


print("Hello")
simulate(20,25)