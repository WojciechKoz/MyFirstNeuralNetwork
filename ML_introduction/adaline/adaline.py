import numpy as np


class Adaline:
    def __init__(self, eta, epochs):
        self.eta = eta
        self.epochs = epochs

    
    def fit(self, X, Y, seed=None):
        rgen = np.random.RandomState(seed)
        self.w_ = rgen.normal(loc=0.0, scale=0.1, size=1+X.shape[1])
        self.cost_ = []

        for _ in range(self.epochs):
            output = self.net_input(X)
            errors = (Y - output)

            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()

            cost = np.square(errors).sum() / 2.0
            self.cost_.append(cost)
        print(self.cost_)


    def net_input(self, entry):
        return np.dot(entry, self.w_[1:]) + self.w_[0]

    
    def predict(self, entry):
        return np.where(self.net_input(entry) >= 0.0, 1, -1)
