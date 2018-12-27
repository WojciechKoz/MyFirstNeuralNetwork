import numpy as np
from random import uniform
from functions import sigmoid, derivative_of_sigmoid


class NeuralNetwork:
    activations = []  # list of matrices
    z = []  # raw sum of prev layer and weights (input in activation function)
    synapses = []  # weights as matrix
    biases = []
    LEARN_RATE = 1
    Δw = []
    Δb = []
    batchNum = 0  # how many times Δ has calculated since last gradient descent step

    def __init__(self):
        self.activations.append(np.matrix([0, 0]).T)
        self.activations.append(np.matrix([0, 0]).T)
        self.activations.append(np.matrix([0]).T)

        self.z.append(np.matrix([0, 0]).T)
        self.z.append(np.matrix([0, 0]).T)
        self.z.append(np.matrix([0]).T)

        self.synapses.append(np.matrix([[uniform(0, 1), uniform(0, 1)], [uniform(0, 1), uniform(0, 1)]]))
        self.synapses.append(np.matrix([[uniform(0, 1)], [uniform(0, 1)]]))

        self.biases.append(np.matrix([uniform(-1, 1), uniform(-1, 1)]).T)
        self.biases.append(np.matrix([uniform(-1, 1)]).T)

        self.delta_reset()

    def calculate(self):
        for i in range(0, 2):
            self.z[i+1] = self.synapses[i].T * self.activations[i] + self.biases[i]
            self.activations[i+1] = sigmoid(self.z[i+1])

    def preparation(self, input):
        for z, a in zip(self.z, self.activations):  # clear value and input in all neurons
            for rowZ, rowA in zip(z, a):
                for i in range(0, len(rowZ)):
                    rowZ[i] = 0
                    rowA[i] = 0

        self.activations[0] = input

    def show_output(self):
        print(self.activations[-1])

    def cost(self, desired_output):
        cost = 0
        for row, y in zip(self.activations[-1], desired_output):
            for a in row:
                cost += (a - y) * (a - y)
        return 0.5*cost

    def backpropagation(self, desired_output):
        δ1 = np.multiply((self.activations[2] - desired_output), derivative_of_sigmoid(self.z[2]))
        δ2 = np.multiply(self.synapses[1]*δ1, derivative_of_sigmoid(self.z[1]))

        self.Δw[0] = self.Δw[0] + self.activations[0] * δ1.T
        self.Δw[1] = self.Δw[0] + self.activations[1] * δ2.T

        self.Δb[0] = self.Δb[0] + δ1
        self.Δb[1] = self.Δb[1] + δ2

        self.batchNum += 1

    def gradient_descent_step(self):
        self.synapses[0] = self.synapses[0] + self.LEARN_RATE * self.Δw[0] / self.batchNum
        self.synapses[1] = self.synapses[1] + self.LEARN_RATE * self.Δw[1] / self.batchNum

        self.biases[0] = self.biases[0] + self.LEARN_RATE * self.Δb[0] / self.batchNum
        self.biases[1] = self.biases[1] + self.LEARN_RATE * self.Δb[1] / self.batchNum

        self.delta_reset()
        self.batchNum = 0

    def delta_reset(self):
        self.Δw = []
        self.Δb = []

        self.Δw.append(np.matrix([[0, 0], [0, 0]]))
        self.Δw.append(np.matrix([0, 0]).T)

        self.Δb.append(np.matrix([0, 0]).T)
        self.Δb.append(np.matrix([0]).T)
