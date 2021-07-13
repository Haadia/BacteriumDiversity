# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
from random import randint, uniform
import networkx as nx
import matplotlib as plt

def main(nbre_bact, nb_inter=20,
         rand=uniform, rand_1=400, rand_2=600):
    """Main program."""
    G = nx.DiGraph()
    G.add_nodes_from(range(1, nbre_bact))
    for nodeA in range(1, nbre_bact + 1):
        for _ in range(nb_inter):
            nodeB = randint(1, nbre_bact)
            while nodeB == nodeA:
                nodeB = randint(1, nbre_bact)
            G.add_edge(nodeA, nodeB, weight=int(rand(rand_1, rand_2)))

    A = nx.linalg.graphmatrix.adjacency_matrix(G)
    print(A)
    A = A.toarray()
    A = np.insert(A, 0, np.arange(1, nbre_bact + 1), 1)
    A = np.insert(A, 0, np.arange(0, nbre_bact + 1), 0)
    np.savetxt("data_biais.csv", A, fmt='%d', delimiter=";")


# %% Execution

if __name__ == "__main__":
    # nbre_bact: Number of species to create
    nbre_bact = 3
    # nb_inter: Degrees of each species
    nb_inter = 36
    # rand: Random function to generate interactions graph
    rand = uniform
    # rand_1: First value of rand
    rand_1 = 400
    # rand_2: First value of rand
    rand_2 = 600
    # Creation of matrix
    main(nbre_bact, nb_inter, rand, rand_1, rand_2)
