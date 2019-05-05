import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from matplotlib.colors import ListedColormap



def plot_decision_regions(X, y, classifier, idx=None, resolution=0.2):
    markers = ('v', 'x', 'o', 's', '^')
    colors = ('red', 'green', 'blue', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x_min, x_max = X.T[0].min() - 1, X.T[0].max() + 1
    y_min, y_max = X.T[1].min() - 1, X.T[1].max() + 1

    mesh_x, mesh_y = np.meshgrid(np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution))

    Z = classifier.predict(np.array([mesh_x.ravel(), mesh_y.ravel()]).T)
    Z = Z.reshape(mesh_x.shape)

    plt.contourf(mesh_x, mesh_y, Z, alpha=0.4, cmap=cmap)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    for num, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.7, c=cmap(num), marker=markers[num], label=cl) 

    if idx:
        plt.scatter(x=X[idx, 0], y=X[idx, 1], c='', edgecolor='black', alpha=1, marker='o', label='test')


    plt.xlabel('długość płatka [standaryzowana]')
    plt.ylabel('szerokość płatka [standaryzowana]')
    plt.legend(loc='upper left')
    plt.show()


X = datasets.load_iris()['data'].T[1:3].T
y = datasets.load_iris()['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
sc.fit(X_train)

X_train_std, X_test_std = sc.transform(X_train), sc.transform(X_test)

model = Perceptron(max_iter=32, eta0=0.1, random_state=0)

model.fit(X_train_std, y_train)

results = model.predict(X_test_std)

print(int(np.where(results == y_test, 1, 0).sum() / len(y_test) * 100), "%")

plot_decision_regions(np.vstack((X_train_std, X_test_std)), np.hstack((y_train, y_test)), model, idx=range(105, 150), resolution=0.01)
