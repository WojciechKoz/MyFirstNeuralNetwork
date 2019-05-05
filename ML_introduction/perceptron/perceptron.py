# import initialization_objects as ini
import numpy as np

class Perceptron:
    def __init__(self, eta, epochs):
        self.eta = eta
        self.epochs = epochs


    def fit(self, X_set, Y_set):
        self.w_ = np.zeros(1 + len(X_set[0]))
        self.errors_ = []

        for _ in range(self.epochs):
            mistakes = 0
    
            for entry,target in zip(X_set, Y_set):
                error = self.eta*(target - self.predict(entry))

                self.w_[1:] += error * entry
                self.w_[0] += error

                mistakes += int(error != 0.0)
            self.errors_.append(mistakes)
        return self


    def net_input(self, entry):
        return np.dot(entry, self.w_[1:]) + self.w_[0]

  
    def predict(self, entry):
        return np.where(self.net_input(entry) >= 0.0, 1, -1)

