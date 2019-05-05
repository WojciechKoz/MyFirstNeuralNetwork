import numpy as np

class AdalineSGD:
    def __init__(self, eta, epochs):
        self.eta = eta
        self.epochs = epochs
        self.initialized = False


    def fit(self, X, Y):
        self.initialize_weights(X.shape[1])
        self.cost_ = []

        for _ in range(self.epochs):
            cost = []
            for xi, target in zip(X, Y):
                cost.append(self.update_weights(xi, target))
            self.cost_.append(sum(cost)/len(cost))

        print(self.cost_)


    def update_weights(self, xi, target):
        output = self.net_input(xi)
        error = (target - output)

        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        
        return error**2 / 2.0 # cost
        

    
    def initialize_weights(self, num):
        rgen = np.random.RandomState()
        self.w_ = rgen.normal(loc=0.0, scale=0.1, size=1+num)

        self.initialized = True

    
    def net_input(self, X):
        return np.dot(X, self.w_[1:].T) + self.w_[0] 
            
    
    def predict(self, entry):
        return np.where(self.net_input(entry) >= 0.0, 1, -1)
