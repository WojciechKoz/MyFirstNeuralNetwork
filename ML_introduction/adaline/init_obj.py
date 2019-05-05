import numpy as np
import matplotlib.pyplot as plt
from random import uniform
from matplotlib.colors import ListedColormap


def create_universe(centroids=[(2, 5), (5, 2)], deviation=1, population_size=100):
    def random_obj(centroid, dev):
        return np.array([centroid[0] + uniform(-dev, dev), centroid[1] + uniform(-dev, dev)])
    
    groups = [[] for _ in centroids]

    for _ in range(int(population_size/len(centroids))):
        for centroid, group in zip(centroids, groups):
            group.append(random_obj(centroid, deviation))

    return groups


def prepare_data(groups):
    labels = [[chr(ord('A')+i)] for i in range(len(groups)) for _ in range(len(groups[i]))] 
    x = [obj for group in groups for obj in group]    
    
    seed = np.random.permutation(len(labels))
    return np.array(x)[seed], np.array(labels)[seed].flatten()


def show_universe(groups):
    colors=['red', 'green', 'blue', 'lightgreen', 'gray', 'cyan']
    markers=['o', 'x', '^', 'v', 's']    

    for (i, group) in enumerate(groups):
        plt.scatter(np.array(group).T[0], np.array(group).T[1], color=colors[i], marker=markers[i], label=chr(ord('A')+i))
    plt.xlabel("x Attribute")
    plt.ylabel("y Attribute")
    plt.legend(loc="upper right")

    plt.show()


def set_map(ax):
    ax.set_title("universe")
    ax.set_xlabel("x Attribute")
    ax.set_ylabel("y Attribute")
    ax.legend(loc="upper right")
    
    
       


def show_model(groups, X, model, model2, resolution=0.02):
    colors=['red', 'green', 'blue', 'lightgreen', 'gray', 'cyan']
    markers=['o', 'x', '^', 'v', 's']    
    cols = ListedColormap(colors[:2])

    fig, plots = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    # left plot
    plots[0].plot(range(1, len(model.cost_)+1), np.array(model.cost_), marker='o', label='simple GD')
    plots[0].plot(range(1, len(model2.cost_)+1), np.array(model2.cost_), marker='x', label='stochastic GD')
    plots[0].set_xlabel("Epochs")
    plots[0].set_ylabel("Cost Func")
    plots[0].legend(loc='upper right')
    plots[0].set_title("gradient descent")

    # right plot
    x_min, x_max = X.T[0].min()-1, X.T[0].max()+1
    y_min, y_max = X.T[1].min()-1, X.T[1].max()+1
    x_mesh, y_mesh = np.meshgrid(np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution))
    values = model.predict(np.array([x_mesh.ravel(), y_mesh.ravel()]).T)
    values = values.reshape(x_mesh.shape)
    plots[1].contourf(x_mesh, y_mesh, values, alpha=0.4, cmap=cols)
    plots[1].set_xlim(x_mesh.min(), x_mesh.max())
    plots[1].set_ylim(y_mesh.min(), y_mesh.max())


    for (i, group) in enumerate(groups):
        print(group)
        plots[1].scatter(np.array(group).T[0], np.array(group).T[1], color=colors[i], marker=markers[i], label=chr(ord('A')+i))
    set_map(plots[1])

    plt.show()



def test_model(X, model, model2):
    A = []
    B = []

    for obj in X:
        if model.predict(obj) == 1:
            A.append(obj)
        else:  
            B.append(obj)

    show_model([A, B], X, model, model2)
