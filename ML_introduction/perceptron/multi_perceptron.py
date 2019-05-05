from perceptron import Perceptron
from initialization_objects import show_universe
import numpy as np
import random as rand


class Classifier:
    def __init__(self, types):
        self.types = types
        self.models = []
        for _ in range(len(types)):
            self.models.append(Perceptron(0.1, 5))

    
    def fit_perceptrons(self, X_set, Y_set):
        for label, model in zip(self.types, self.models):
            y = np.where(Y_set == label, 1, -1)
            print(label, model.fit(X_set, y))


    def show_results(self):
        new_objects = [[] for _ in self.models] 

        for _ in range(1000):
            obj = np.random.rand(2,)*10
            for types, model in zip(new_objects, self.models):
                if model.predict(obj) == 1:
                    types.append(obj)
                    break
        
        show_universe(new_objects)
        
