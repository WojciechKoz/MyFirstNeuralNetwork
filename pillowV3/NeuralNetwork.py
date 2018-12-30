import numpy as np

def sigmoid_prime(x):
    return np.exp(-x) / ((np.exp(-x) + 1) ** 2)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Network:
    # all list parameters are list of matrices 
    b = []  # biases
    syn = []  # synapses
    LEARN_RATE = 1  
  

    def __init__(self, num_of_neurons: list):  # to create NN you need to give a list how many neurons will be in layers
        np.random.seed(1)

        for i in range(len(num_of_neurons) - 1):
            self.syn.append(2 * np.random.random((num_of_neurons[i], num_of_neurons[i+1])) - 1)  
        
        # biases for respective layers
        for i in range(1, len(num_of_neurons)):
            self.b.append(2 * np.random.random((1, num_of_neurons[i])) - 1)


    def epoch(self, images, y, num_of_iter):  # images : list of batches (batch is a matrix of many inputs)
        # y : list of matrix of desired outputs,  num of iter : numer of backprop for one batch
        
        for i, batch in list(enumerate(images)):
            X = np.array(batch)
            
            print("======================= BATCH {} =======================".format(i))

            for j in range(num_of_iter):
                l = []  # layers
                l.append(X)
                
                for k in range(len(self.b)):
                    l.append(sigmoid(np.dot(l[k], self.syn[k]) + self.b[k]))  # (mozna to bd zmienic)
                
                matrix_error = (y[i] - l[-1])
                
                if j % 20 == 0:
                    error = np.mean(np.abs(matrix_error))
                    print(("[%d] error: " % j) + str(error))

                delta = []
                delta.append(matrix_error * sigmoid_prime(l[-1]))

                # backpropagation
                for k in range(len(self.b)-1, 0, -1):
                    error = delta[-1].dot(self.syn[k].T)
                    delta.append(error * sigmoid_prime(l[k]))

                # grandient descent
                for k in range(len(self.b)):
                    descent = delta.pop()
                    self.syn[k] += l[k].T.dot(descent)
                    self.b[k] += descent.mean(axis=0)


    def run(self, data):
        l = []
        l.append([data])
        
        for k in range(len(self.b)):
            l.append(sigmoid(np.dot(l[k], self.syn[k]) + self.b[k]))

        return np.argmax(l[-1])

