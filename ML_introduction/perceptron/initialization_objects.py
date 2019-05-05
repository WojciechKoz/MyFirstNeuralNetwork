import random as rand
import numpy as np
import matplotlib.pyplot as plt

population_size = 15000

def create_obj(centroid, deviation):
    return np.array([centroid[0] + rand.uniform(-deviation, deviation), centroid[1] + rand.uniform(-deviation, deviation)])

def create_data(centroids=[(1, 4), (5, 2)], deviation=1, population_size=100):
    groups = [[] for _ in centroids]
    
    for _ in range(int(population_size/len(centroids))):
        for centroid, group in zip(centroids, groups):
            group.append(create_obj(centroid, deviation))
        
    return groups 


def show_universe(groups):
    colors=['red', 'green', 'blue', 'lightgreen', 'gray', 'cyan']
    markers=['o', 'x', '^', 'v', 's']    

    for (i, group) in enumerate(groups):
        plt.scatter(np.array(group).T[0], np.array(group).T[1], color=colors[i], marker=markers[i], label=chr(ord('A')+i))
    plt.xlabel("x Attribute")
    plt.ylabel("y Attribute")
    plt.legend(loc="upper right")

    plt.show()

def prepare_data(groups):
    relation = np.array([[groups[i][j], chr(ord('A')+i)] for i in range(len(groups)) for j in range(len(groups[i]))])
    np.random.shuffle(relation)
    return relation.T[0], relation.T[1]    


