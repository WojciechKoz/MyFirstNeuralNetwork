import random as rand
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


def create_universe(centroids=[(2,3), (10, 7), (-2, 10)], deviation=2, cluster_size=50):
    def create_object(centroid, deviation):
        return (centroid[0] + rand.uniform(-deviation, deviation), centroid[1] + rand.uniform(-deviation, deviation))

    output = []

    for _ in range(cluster_size):
        for centroid in centroids:
            output.append(create_object(centroid, deviation))

    return np.array(output)


def show_universe(universe):
    plt.scatter(universe.T[0], universe.T[1])
    plt.show()


def show_clusters(clusters):
    colors = ('red', 'green', 'blue', 'cyan', 'gray')
    markers = ('x', 'o', '^', 'v', 's')

    for i, cluster in enumerate(clusters):
        plt.scatter(np.array(cluster).T[0], np.array(cluster).T[1], color=colors[i], marker=markers[i], label=i)
    plt.show()


def init_dist_matrix(univ):
    def dist(A, B):
        return sqrt((A[0] - B[0])**2 + (A[1] - B[1])**2)
    output = []

    for obj in univ:
        output.append([])
        for other in univ:
            output[-1].append(dist(obj, other))
    return np.array(output)

