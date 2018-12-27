from NeuralNetwork import *

if __name__ == "__main__":
    nn = NeuralNetwork()
    numOfAttempt = 0
    costAverage = 1

    batches = [np.matrix([0, 0]).T, np.matrix([0, 1]).T, np.matrix([1, 0]).T, np.matrix([1, 1]).T]
    desired_outputs = [0, 1, 1, 0]

    while costAverage > 0.001:
        costAverage = 0

        for i in range(0, 4):
            nn.preparation(batches[i])

            nn.calculate()

            costAverage += nn.cost([desired_outputs[0]])

            nn.backpropagation(desired_outputs[0])

        nn.gradient_descent_step()

        costAverage /= 4

        print(str(numOfAttempt) + str(costAverage))

        numOfAttempt += 1
