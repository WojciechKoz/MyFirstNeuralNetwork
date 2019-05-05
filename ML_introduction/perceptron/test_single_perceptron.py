from perceptron import Perceptron
from initialization_objects import create_data, prepare_data, create_obj
import numpy as np
import random as rand
import matplotlib.pyplot as plt


def main():
    model = Perceptron(0.1, 5)

    groups = create_data([(10, 0), (0, 5)], 1)

    X_set, Y_set = prepare_data(groups)


    model.fit(X_set, np.where(Y_set == 'A', 1, -1))
    print(model.w_)


    print(model.predict(np.array([1, 7])))
    print(model.predict(np.array([5, 2])))

    A = []
    B = []

    for _ in range(50):
        obj = np.array([rand.uniform(0, 10), rand.uniform(0, 10)])

        if model.predict(obj) == 1:
            A.append(obj)
        else:
            B.append(obj)

    plt.scatter(np.array(A).T[0], np.array(A).T[1], color='red', marker='o', label="new A")
    plt.scatter(np.array(B).T[0], np.array(B).T[1], color='blue', marker='x', label="new B")
    plt.scatter(1 , 7, color='yellow', marker='s', label='A center')
    plt.scatter(5 , 2, color='green', marker='p', label='B center')

    plt.xlabel("X-attribute")
    plt.ylabel("Y-attribute")
    plt.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    main()
