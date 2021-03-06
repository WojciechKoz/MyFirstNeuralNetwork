from functions import neuronValuePack, number_to_network_output
from NeuralNetwork import NeuralNetwork
from mnist import MNIST
import random


def test_numbers(images, labels):
    # img = import_picture("/home/wojciech/PycharmProjects/pillow/venv/picture.png")

    NN = NeuralNetwork(4, [784, 16, 16, 10])

    for i in range(0, 100000):
        index = 100  # random.randrange(0, len(images))  # pick a random value as an index of image and label

        img = neuronValuePack(images[index])
        desired_output = number_to_network_output(labels[index])  # for example when we have "2" as a right output
        # it returns list like this [0, 0, 1, 0, 0, 0, ...]

        NN.clear_network()  # every neuron value should equals to 0

        NN.fill_first_layer(img)

        NN.run()

        print(i)
        print(labels[index])

        NN.show_output()
        print()

        NN.backpropagation(desired_output)


def text_xor():
    NN = NeuralNetwork(3, [2, 2, 1])

    for i in range(0, 1000):
        input = [random.randrange(0, 1), random.randrange(1, 2)]

        if input[0] == input[1]:
            desired_output = [0]
        else:
            desired_output = [1]

        NN.clear_network()

        NN.fill_first_layer(input)

        NN.run()

        print(i)
        print(desired_output)

        NN.show_output()
        print()

        NN.backpropagation(desired_output)

    for layer in NN.neurons:
        for neuron in layer:
            print(str(neuron.value) + ": " + str(neuron.bias))
            print(neuron.weights)


def main():
    mndata = MNIST('samples')
    images, labels = mndata.load_testing()

    # test_numbers(images, labels)

    text_xor()


if __name__ == "__main__":
    main()
