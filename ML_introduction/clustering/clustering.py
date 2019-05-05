from math import sqrt
from random import choices
import numpy as np

def dist(obj1, obj2):
    return sqrt((obj1[0]-obj2[0])**2 + (obj1[1] - obj2[1])**2)


def make_cluster(med, univ):
    output = [[] for m in med]

    for obj in univ:
        dists = [dist(obj, m) for m in med]
        output[dists.index(min(dists))].append(obj)

    return output


def find_medoid(cluster):
    dist_sum = [sum([dist(obj, diff_obj) for diff_obj in cluster]) for obj in cluster]
    return cluster[dist_sum.index(min(dist_sum))]
        
    
def random_medoids(univ, num):
    output = []

    while len(output) < num:
        rand_elem = choices(univ)[0]
        if not np.any(rand_elem is output):
            output.append(rand_elem)
    return output        


def cluster_loop(univ, n_clus):
    medoids = random_medoids(univ, n_clus)
    clusters = []

    for _ in range(10):
        clusters = make_cluster(medoids, univ)
        medoids = [find_medoid(cluster) for cluster in clusters]
    
    return clusters
