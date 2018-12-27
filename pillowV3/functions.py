from numpy import exp, multiply


def sigmoid(matrix):
    return 1 / (1 + exp(-matrix))


def derivative_of_sigmoid(matrix):
    return multiply(sigmoid(matrix), 1 - sigmoid(matrix))
